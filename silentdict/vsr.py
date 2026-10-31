"""Auto-AVSR visual speech recognition model wrapper (inference + personal fine-tuning helpers)."""
from __future__ import annotations

import argparse
import json
import logging
import math
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import paths  # noqa: F401  (adds third_party to sys.path via package import)
from espnet.asr.asr_utils import get_model_conf, torch_load
from espnet.nets.batch_beam_search import BatchBeamSearch
from espnet.nets.lm_interface import dynamic_import_lm
from espnet.nets.pytorch_backend.e2e_asr_transformer import E2E
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask, pad_list
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.mask import target_mask
from espnet.nets.scorers.length_bonus import LengthBonus

logging.getLogger().setLevel(logging.WARNING)

MEAN, STD = 0.421, 0.165
CROP = 88
DEFAULT_DECODE = {"beam_size": 20, "ctc_weight": 0.1, "lm_weight": 0.3, "penalty": 0.0}


def pick_device(pref: str = "auto") -> torch.device:
    if pref != "auto":
        return torch.device(pref)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_token_list(units_path: Path = paths.SPM_UNITS) -> list[str]:
    units = [line.split()[0] for line in open(units_path, encoding="utf8").read().splitlines()]
    return ["<blank>"] + units + ["<eos>"]


class Tokenizer:
    def __init__(self, model_path: Path = paths.SPM_MODEL, units_path: Path = paths.SPM_UNITS):
        import sentencepiece

        self.spm = sentencepiece.SentencePieceProcessor(model_file=str(model_path))
        units = open(units_path, encoding="utf8").read().splitlines()
        self.ids = {u.split()[0]: int(u.split()[-1]) for u in units}

    def encode(self, text: str) -> list[int]:
        return [self.ids.get(p, self.ids["<unk>"]) for p in self.spm.EncodeAsPieces(text)]


def roi_to_tensor(roi: np.ndarray, offset: tuple[int, int] | None = None) -> torch.Tensor:
    """uint8 (T, 96, 96) -> float (1, T, 88, 88), normalised like Auto-AVSR."""
    x = torch.from_numpy(np.ascontiguousarray(roi)).float() / 255.0
    oy, ox = offset if offset is not None else ((96 - CROP) // 2, (96 - CROP) // 2)
    x = x[:, oy: oy + CROP, ox: ox + CROP]
    return ((x - MEAN) / STD).unsqueeze(0)


@dataclass
class Hypothesis:
    text: str
    score: float                    # what the beam search ranks by
    logprob: float | None = None    # total log-probability, when it differs from score (FLARE)


def load_vsr_model(device: torch.device, personal_weights: Path | None = None) -> E2E:
    conf = json.load(open(paths.VSR_DIR / "model.json"))
    args = argparse.Namespace(**(conf if isinstance(conf, dict) else conf[2]))
    model = E2E(len(load_token_list()), args)
    state = torch.load(paths.VSR_DIR / "model.pth", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    if personal_weights is not None and Path(personal_weights).exists():
        delta = torch.load(personal_weights, map_location="cpu", weights_only=True)
        missing = model.load_state_dict(delta, strict=False)
        if missing.unexpected_keys:
            raise RuntimeError(f"unexpected keys in personal weights: {missing.unexpected_keys[:5]}")
    model.args = args
    _install_cross_attention_cache(model)
    return model.to(device).eval()


def _cached_src_attention(self, query, key, value, mask, rtn_attn=False):
    """Cross-attention that projects the encoder output once per utterance.

    During batched beam search every hypothesis attends to the same encoder output
    (expanded, stride 0 along the batch). ESPnet re-projects it for every hypothesis at
    every output step; caching the projected keys/values makes decoding several times faster.
    """
    if not (self._xattn_enabled and key is value and key.size(0) > 1 and key.stride(0) == 0):
        return type(self).forward(self, query, key, value, mask, rtn_attn)
    tag = (key.data_ptr(), tuple(key.shape[1:]))
    if self._xattn_cache is None or self._xattn_cache[0] != tag:
        base = key[:1]
        k = self.linear_k(base).view(1, -1, self.h, self.d_k).transpose(1, 2)
        v = self.linear_v(base).view(1, -1, self.h, self.d_k).transpose(1, 2)
        self._xattn_cache = (tag, k, v)
    _, k, v = self._xattn_cache
    n = query.size(0)
    q = self.linear_q(query).view(n, -1, self.h, self.d_k).transpose(1, 2)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
    return self.forward_attention(v.expand(n, -1, -1, -1), scores, mask, rtn_attn)


def _install_cross_attention_cache(model: E2E):
    model._xattn_modules = []
    if model.decoder is None:
        return
    for layer in model.decoder.decoders:
        attn = layer.src_attn
        attn._xattn_enabled = False
        attn._xattn_cache = None
        attn.forward = types.MethodType(_cached_src_attention, attn)
        model._xattn_modules.append(attn)


def set_cross_attention_cache(model: E2E, enabled: bool):
    for attn in getattr(model, "_xattn_modules", []):
        attn._xattn_enabled = enabled
        attn._xattn_cache = None


def load_lm(token_list):
    lm_args = get_model_conf(str(paths.LM_DIR / "model.pth"), str(paths.LM_DIR / "model.json"))
    lm_class = dynamic_import_lm(getattr(lm_args, "model_module", "default"), lm_args.backend)
    lm = lm_class(len(token_list), lm_args)
    torch_load(str(paths.LM_DIR / "model.pth"), lm)
    return lm.eval()


class LipReader:
    """Turns a mouth ROI clip into an n-best list of upper-case transcripts."""

    def __init__(self, device: str = "auto", personal: bool = True, decode: dict | None = None,
                 use_lm: bool = True, search_device: str = "cpu"):
        # The visual encoder is one large batched pass (fast on the Apple GPU); beam search is
        # hundreds of tiny sequential steps, which run faster on the CPU than on MPS.
        self.device = pick_device(device)
        self.search_device = torch.device(search_device)
        self.token_list = load_token_list()
        weights = paths.PERSONAL_WEIGHTS if personal else None
        self.personalized = bool(personal and paths.PERSONAL_WEIGHTS.exists())
        self.model = load_vsr_model(self.search_device, weights)
        self.model.encoder.to(self.device)
        self.lm = load_lm(self.token_list).to(self.search_device) if use_lm else None
        cfg = dict(DEFAULT_DECODE)
        if personal and paths.PERSONAL_DECODE.exists():
            cfg.update(json.load(open(paths.PERSONAL_DECODE)))
        if decode:
            cfg.update(decode)
        self.decode_cfg = cfg
        self._build_search()

    def _build_search(self):
        cfg = self.decode_cfg
        scorers = self.model.scorers()
        scorers["lm"] = self.lm if cfg["lm_weight"] > 0 else None
        scorers["length_bonus"] = LengthBonus(len(self.token_list))
        weights = {
            "decoder": 1.0 - cfg["ctc_weight"],
            "ctc": cfg["ctc_weight"],
            "lm": cfg["lm_weight"] if self.lm is not None else 0.0,
            "length_bonus": cfg["penalty"],
        }
        odim = len(self.token_list)
        self.search = BatchBeamSearch(
            beam_size=cfg["beam_size"], vocab_size=odim, weights=weights, scorers=scorers,
            sos=odim - 1, eos=odim - 1, token_list=self.token_list,
            pre_beam_score_key=None if cfg["ctc_weight"] == 1.0 else "decoder",
        ).to(self.search_device).eval()

    def set_decode(self, **kwargs):
        self.decode_cfg.update(kwargs)
        self._build_search()

    def ids_to_text(self, ids) -> str:
        toks = [self.token_list[int(i)] for i in ids if 0 < int(i) < len(self.token_list) - 1]
        return "".join(toks).replace("\u2581", " ").strip()

    name = "auto_avsr"

    @torch.inference_mode()
    def __call__(self, roi, nbest: int = 5) -> list[Hypothesis]:
        roi = getattr(roi, "lip", roi)  # a roi.Clip or a plain (T, 96, 96) array; this model reads lips only
        if len(roi) < 5:
            return []
        self.model.eval()
        x = roi_to_tensor(roi).unsqueeze(0).to(self.device)  # (1, 1, T, 88, 88)
        enc, _ = self.model.encoder(x, None)
        enc = enc.squeeze(0).to(self.search_device)
        set_cross_attention_cache(self.model, True)
        try:
            hyps = self.search(enc, maxlenratio=0.0, minlenratio=0.0)
        finally:
            set_cross_attention_cache(self.model, False)
        out, seen = [], set()
        for h in hyps:
            text = self.ids_to_text(h.yseq[1:-1].tolist())
            if text and text not in seen:
                seen.add(text)
                out.append(Hypothesis(text, float(h.score)))
            if len(out) >= nbest:
                break
        return out


def batch_loss(model: E2E, xs: torch.Tensor, lens: torch.Tensor, ys: list[torch.Tensor],
               ctc_weight: float | None = None):
    """Hybrid CTC/attention loss used to adapt the model to one person's silent speech."""
    device = xs.device
    alpha = model.mtlalpha if ctc_weight is None else ctc_weight
    masks = make_non_pad_mask(lens).to(device).unsqueeze(-2)
    hs, hs_mask = model.encoder(xs, masks)
    hlens = hs_mask.squeeze(1).sum(1)

    # CTC on CPU: robust across PyTorch MPS versions; gradients flow back to the GPU graph.
    logits = model.ctc.ctc_lo(model.ctc.dropout(hs))
    logp = logits.float().log_softmax(-1).transpose(0, 1).cpu()
    targets = torch.cat(ys).cpu()
    olens = torch.tensor([len(y) for y in ys])
    loss_ctc = F.ctc_loss(logp, targets, hlens.cpu(), olens, blank=0, reduction="sum",
                          zero_infinity=True) / len(ys)

    ys_pad = pad_list([y.to(device) for y in ys], model.ignore_id)
    ys_in, ys_out = add_sos_eos(ys_pad, model.sos, model.eos, model.ignore_id)
    ys_mask = target_mask(ys_in, model.ignore_id)
    pred, _ = model.decoder(ys_in, ys_mask, hs, hs_mask)
    loss_att = model.criterion(pred, ys_out)
    loss = alpha * loss_ctc.to(device) + (1 - alpha) * loss_att
    return loss, loss_ctc.item(), loss_att.item()
