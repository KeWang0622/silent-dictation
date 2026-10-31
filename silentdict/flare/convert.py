"""Convert a released FLARE fairseq checkpoint into a small plain-PyTorch weights file (no fairseq needed).

python -m silentdict.flare.convert checkpoint_last.pt models/flare_large/model.pt
"""
from __future__ import annotations

import pickle
import sys
import types
from pathlib import Path

import torch

KEEP = ("torch", "collections", "numpy", "_codecs", "builtins", "copyreg")


class _Stub:
    """Stands in for fairseq / omegaconf objects stored in the checkpoint; only the tensors are needed."""

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        self.__dict__["_state"] = state

    def __call__(self, *args, **kwargs):
        return _Stub()


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.split(".")[0] in KEEP:
            return super().find_class(module, name)
        return _Stub


_stub_pickle = types.ModuleType("stub_pickle")
_stub_pickle.Unpickler = _Unpickler
_stub_pickle.load = pickle.load
_stub_pickle.__name__ = "stub_pickle"

PREFIXES = {
    "encoder.w2v_model.feature_extractor_video_face.": "encoder.face.",
    "encoder.w2v_model.feature_extractor_video_lip.": "encoder.lip.",
    "encoder.w2v_model.layer_norm.": "encoder.layer_norm.",
    "encoder.w2v_model.post_extract_proj.": "encoder.post_extract_proj.",
    "encoder.w2v_model.encoder.layers.": "encoder.layers.",
    "decoder.embed_tokens.": "decoder.embed_tokens.",
    "decoder.layers.": "decoder.layers.",
    "decoder.layer_norm.": "decoder.layer_norm.",
}


def convert(src: str, dst: str, dtype=torch.float16) -> int:
    state = torch.load(src, map_location="cpu", pickle_module=_stub_pickle, weights_only=False)
    weights = {}
    for key, value in state["model"].items():
        for old, new in PREFIXES.items():
            if key.startswith(old) and torch.is_tensor(value):
                weights[new + key[len(old):]] = value.to(dtype) if value.is_floating_point() else value
    from .model import FlareModel
    missing, unexpected = FlareModel().load_state_dict(weights, strict=False)
    if missing or unexpected:
        raise SystemExit(f"weights do not match the model: missing {missing[:5]}, unexpected {unexpected[:5]}")
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format": "flare-large-v1", "weights": weights}, dst)
    return sum(v.numel() for v in weights.values())


if __name__ == "__main__":
    n = convert(sys.argv[1], sys.argv[2])
    print(f"converted {n / 1e6:.0f}M parameters -> {sys.argv[2]}")
