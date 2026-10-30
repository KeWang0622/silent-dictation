"""Frames -> mouth ROI -> lip-reading n-best -> formatted text."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .roi import MouthCropper, frames_to_roi
from .text import accept_rewrite, sentence_case
from .vsr import Hypothesis, LipReader

# Only candidates whose beam score is within this margin of the best one are offered to the LLM,
# so it cannot overrule a confident lip-reading result with a less likely candidate.
LLM_SCORE_MARGIN = 2.5


@dataclass
class Result:
    roi: np.ndarray | None
    hyps: list[Hypothesis] = field(default_factory=list)
    text: str = ""
    error: str | None = None
    timings: dict = field(default_factory=dict)

    @property
    def raw(self) -> str:
        return self.hyps[0].text if self.hyps else ""


class Recognizer:
    def __init__(self, reader: LipReader | None = None, rewriter=None, **reader_kwargs):
        self.reader = reader or LipReader(**reader_kwargs)
        self.cropper = MouthCropper()
        self.rewriter = rewriter
        self.context: list[str] = []

    def warm_up(self):
        """Run one dummy pass so the first real sentence does not pay GPU/LLM start-up cost."""
        self.reader(np.full((40, 96, 96), 110, np.uint8), nbest=1)
        if self.rewriter is not None:
            self.rewriter(["hello there"])

    def roi(self, frames) -> np.ndarray:
        return frames_to_roi(frames, self.cropper)

    def recognize_roi(self, roi: np.ndarray, rewrite: bool = True) -> Result:
        res = Result(roi=roi)
        t = time.perf_counter()
        res.hyps = self.reader(roi, nbest=5)
        res.timings["vsr"] = time.perf_counter() - t
        if not res.hyps:
            res.error = "nothing recognised"
            return res
        t = time.perf_counter()
        if rewrite and self.rewriter is not None:
            top = res.hyps[0].score
            offered = [h.text for h in res.hyps if h.score >= top - LLM_SCORE_MARGIN]
            candidate = self.rewriter(offered, context=self.context[-3:])
            res.text = candidate if accept_rewrite(candidate, offered) else sentence_case(res.raw)
            res.timings["llm"] = time.perf_counter() - t
        else:
            res.text = sentence_case(res.raw)
        if res.text:
            self.context.append(res.text)
        return res

    def recognize_frames(self, frames, rewrite: bool = True) -> Result:
        t = time.perf_counter()
        try:
            roi = self.roi(frames)
        except ValueError as exc:
            return Result(roi=None, error=str(exc))
        crop_time = time.perf_counter() - t
        res = self.recognize_roi(roi, rewrite=rewrite)
        res.timings["crop"] = crop_time
        return res
