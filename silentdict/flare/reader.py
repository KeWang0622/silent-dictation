"""FLARE lip reader with the same interface as vsr.LipReader, plus phrase scoring and fine-tuning helpers."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .. import paths
from ..vsr import Hypothesis, pick_device
from .model import BOS, EOS, PAD, UNK, FlareModel, beam_search

ASSETS = Path(__file__).resolve().parent
MEAN, STD = 0.421, 0.165
DEFAULT_DECODE = {"beam_size": 10, "lenpen": 1.0}


class Tokenizer:
    def __init__(self):
        import sentencepiece

        self.spm = sentencepiece.SentencePieceProcessor(model_file=str(ASSETS / "spm_unigram1000.model"))
        words = [line.split()[0] for line in (ASSETS / "dict.wrd.txt").read_text(encoding="utf8").splitlines() if line.strip()]
        self.symbols = ["<s>", "<pad>", "</s>", "<unk>"] + words
        self.index = {s: i for i, s in enumerate(self.symbols)}

    def encode(self, text: str) -> list[int]:
        return [self.index.get(p, UNK) for p in self.spm.encode(text.lower(), out_type=str)]

    def decode(self, ids) -> str:
        pieces = [self.symbols[i] for i in ids if i not in (BOS, PAD, EOS)]
        return "".join(pieces).replace("\u2581", " ").strip().upper()


def to_tensor(clip: np.ndarray | None) -> torch.Tensor | None:
    """uint8 (T, 96, 96) -> float (1, 1, T, 88, 88), normalised like FLARE's loader."""
    if clip is None:
        return None
    x = torch.from_numpy(np.ascontiguousarray(clip[:, 4:92, 4:92])).float() / 255.0
    return ((x - MEAN) / STD)[None, None]


def load_flare(device: torch.device, personal_weights: Path | None = None) -> FlareModel:
    model = FlareModel()
    state = torch.load(paths.FLARE_WEIGHTS, map_location="cpu", weights_only=True)
    model.load_state_dict({k: v.float() for k, v in state["weights"].items()})
    if personal_weights is not None and Path(personal_weights).exists():
        delta = torch.load(personal_weights, map_location="cpu", weights_only=True)
        model.load_state_dict(delta, strict=False)
    return model.to(device).eval()


class FlareReader:
    """Reads a clip (lip crops, optionally face crops) into an n-best list of upper-case transcripts."""

    name = "flare"

    def __init__(self, device: str = "auto", personal: bool = True, decode: dict | None = None,
                 search_device: str = "cpu"):
        self.device = pick_device(device)
        self.search_device = torch.device(search_device)
        self.tok = Tokenizer()
        weights = paths.PERSONAL_FLARE_WEIGHTS if personal else None
        self.personalized = bool(personal and paths.PERSONAL_FLARE_WEIGHTS.exists())
        self.model = load_flare(self.search_device, weights)
        self.model.encoder.to(self.device)
        self.decode_cfg = dict(DEFAULT_DECODE)
        if personal and paths.PERSONAL_FLARE_DECODE.exists():
            self.decode_cfg.update(json.load(open(paths.PERSONAL_FLARE_DECODE)))
        if decode:
            self.decode_cfg.update({k: v for k, v in decode.items() if k in DEFAULT_DECODE})

    def set_decode(self, **kwargs):
        self.decode_cfg.update({k: v for k, v in kwargs.items() if k in DEFAULT_DECODE})

    @torch.inference_mode()
    def encode(self, lip: np.ndarray, face: np.ndarray | None = None) -> torch.Tensor:
        lip_t, face_t = to_tensor(lip), to_tensor(face)
        enc = self.model.encoder(face_t.to(self.device) if face_t is not None else None, lip_t.to(self.device))
        return enc.to(self.search_device)

    @torch.inference_mode()
    def __call__(self, clip, nbest: int = 5, face: np.ndarray | None = None) -> list[Hypothesis]:
        lip = clip.lip if hasattr(clip, "lip") else clip
        face = clip.face if hasattr(clip, "face") else face
        if len(lip) < 5:
            return []
        enc = self.encode(lip, face)
        hyps = beam_search(self.model.decoder, enc, beam=int(self.decode_cfg["beam_size"]),
                           lenpen=float(self.decode_cfg["lenpen"]))
        out, seen = [], set()
        for h in hyps:
            text = self.tok.decode(h.tokens)
            if text and text not in seen:
                seen.add(text)
                out.append(Hypothesis(text, h.score, h.logprob))
            if len(out) >= nbest:
                break
        return out

    @torch.inference_mode()
    def score_texts(self, enc: torch.Tensor, texts: list[str]) -> np.ndarray:
        """Total log-probability of each text (including </s>) given an encoded clip; for phrase mode."""
        ys = [self.tok.encode(t) + [EOS] for t in texts]
        L = max(len(y) for y in ys)
        out = torch.full((len(ys), L), PAD, dtype=torch.long)
        inp = torch.full((len(ys), L), PAD, dtype=torch.long)
        for i, y in enumerate(ys):
            out[i, :len(y)] = torch.tensor(y)
            inp[i, 0] = EOS
            inp[i, 1:len(y)] = torch.tensor(y[:-1])
        logits = self.model.decoder(inp.to(self.search_device), enc.expand(len(ys), -1, -1))
        lp = logits.float().log_softmax(-1).gather(-1, out.to(self.search_device).unsqueeze(-1)).squeeze(-1)
        return (lp * (out != PAD).to(lp)).sum(-1).cpu().numpy()


def batch_loss(model: FlareModel, faces, lips, lengths, labels, smoothing: float = 0.1):
    """Label-smoothed cross-entropy of the decoder, used for personal fine-tuning."""
    T = lips.size(2)
    padding = torch.arange(T, device=lips.device)[None] >= lengths[:, None]
    enc = model.encoder(faces, lips, padding)
    L = max(len(y) for y in labels) + 1
    inp = torch.full((len(labels), L), PAD, dtype=torch.long, device=lips.device)
    out = torch.full((len(labels), L), PAD, dtype=torch.long, device=lips.device)
    for i, y in enumerate(labels):
        y = torch.as_tensor(y, device=lips.device)
        inp[i, 0] = EOS
        inp[i, 1:len(y) + 1] = y
        out[i, :len(y)] = y
        out[i, len(y)] = EOS
    logits = model.decoder(inp, enc, ~padding[:, None, None, :])
    return F.cross_entropy(logits.float().transpose(1, 2), out, ignore_index=PAD, label_smoothing=smoothing)
