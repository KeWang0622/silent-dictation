"""Frames -> mouth (and face) crops -> lip-reading n-best -> formatted text."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from . import paths
from .roi import Clip, MouthCropper, frames_to_clip
from .text import sentence_case
from .vsr import Hypothesis

MODELS = {"flare": "FLARE Large", "auto_avsr": "Auto-AVSR"}


def available_models() -> list[str]:
    return (["flare"] if paths.FLARE_WEIGHTS.exists() else []) + ["auto_avsr"]


def resolve_model(choice: str | None) -> str:
    """'auto' picks the most accurate installed model (FLARE when its weights are present)."""
    models = available_models()
    return choice if choice in models else models[0]


def make_reader(model: str = "auto", **kwargs):
    model = resolve_model(model)
    if model == "flare":
        from .flare.reader import FlareReader
        return FlareReader(**kwargs)
    from .vsr import LipReader
    return LipReader(**kwargs)


@dataclass
class Result:
    clip: Clip | None
    hyps: list[Hypothesis] = field(default_factory=list)
    text: str = ""
    error: str | None = None
    timings: dict = field(default_factory=dict)

    @property
    def raw(self) -> str:
        return self.hyps[0].text if self.hyps else ""

    @property
    def roi(self) -> np.ndarray | None:
        return self.clip.lip if self.clip is not None else None


class Recognizer:
    def __init__(self, reader=None, rescorer=None, model: str = "auto", **reader_kwargs):
        self.reader = reader or make_reader(model, **reader_kwargs)
        self.model = getattr(self.reader, "name", "auto_avsr")
        self.uses_face = self.model == "flare"
        self.cropper = MouthCropper()
        self.rescorer = rescorer

    def warm_up(self):
        """Run one dummy pass so the first real sentence does not pay GPU/LLM start-up cost."""
        still = np.full((40, 96, 96), 110, np.uint8)
        self.reader(Clip(still, still if self.uses_face else None), nbest=1)
        if self.rescorer is not None:
            self.rescorer.logprob("HELLO THERE")

    def clip(self, frames) -> Clip:
        return frames_to_clip(frames, self.cropper, face=self.uses_face)

    def recognize_clip(self, clip: Clip | np.ndarray, rescore: bool = True) -> Result:
        if not isinstance(clip, Clip):
            clip = Clip(clip)
        res = Result(clip=clip)
        t = time.perf_counter()
        res.hyps = self.reader(clip, nbest=10)
        res.timings["vsr"] = time.perf_counter() - t
        if not res.hyps:
            res.error = "nothing recognised"
            return res
        best = res.raw
        if rescore and self.rescorer is not None and len(res.hyps) > 1:
            t = time.perf_counter()
            best = self.rescorer.best(res.hyps, self.model)
            res.timings["llm"] = time.perf_counter() - t
        res.text = sentence_case(best)
        return res

    recognize_roi = recognize_clip  # older name

    def recognize_frames(self, frames, rescore: bool = True) -> Result:
        t = time.perf_counter()
        try:
            clip = self.clip(frames)
        except ValueError as exc:
            return Result(clip=None, error=str(exc))
        crop_time = time.perf_counter() - t
        res = self.recognize_clip(clip, rescore=rescore)
        res.timings["crop"] = crop_time
        return res
