"""Calibration session used by the app: prompts, quality checks and instant feedback."""
from __future__ import annotations

import threading

from .activity import motion_score
from .dataset import add_item, delete_item, load_prompts
from .roi import MouthCropper, frames_to_clip
from .text import align, sentence_case

MIN_SECONDS = 0.8
MIN_MOTION = 0.05  # spread of mouth opening; real mouthing is ~0.2-0.4


class CalibrationSession:
    def __init__(self, controller, count: int = 60):
        self.controller = controller
        self.prompts = load_prompts(count)
        self.idx = 0
        self.busy = False
        self.saved: list = []
        self.scores: list[tuple[int, int]] = []
        self.status = "ready"      # ready | recording | saving | saved | retry
        self.message = ""
        self.last: dict | None = None
        self.cropper = MouthCropper()
        self.lock = threading.Lock()

    @property
    def done(self) -> bool:
        return self.idx >= len(self.prompts)

    def accepting(self) -> bool:
        return not self.busy and not self.done

    def _retry(self, message: str):
        self.status, self.message = "retry", message

    def handle(self, frames):
        """Runs on the controller's worker thread."""
        self.busy, self.status = True, "saving"
        try:
            if not frames or frames[-1].ts - frames[0].ts < MIN_SECONDS:
                return self._retry("Too short. Hold the key for the whole sentence.")
            if motion_score([f.obs for f in frames]) < MIN_MOTION:
                return self._retry("Your lips barely moved. Mouth the words a little more clearly.")
            try:
                roi = frames_to_clip(frames, self.cropper)
            except ValueError:
                return self._retry("Your face left the frame. Stay centred and try again.")
            prompt = self.prompts[self.idx]
            item = add_item(roi, prompt, "calibration")
            self.saved.append((item, self.idx))
            self.idx += 1
            self.status, self.message = "saved", "Saved"
        finally:
            self.busy = False
        rec = self.controller.recognizer
        if rec is None:
            return
        hyps = rec.reader(roi, nbest=1)
        read = hyps[0].text.lower() if hyps else ""
        ops = align(prompt, read)
        errors = sum(op["op"] != "ok" for op in ops)
        words = max(1, sum(op["op"] != "ins" for op in ops))
        self.scores.append((min(errors, words), words))
        self.last = {"prompt": prompt, "read": read, "align": ops, "errors": errors, "words": words}

    def redo(self):
        with self.lock:
            if self.saved and not self.busy:
                item, idx = self.saved.pop()
                delete_item(item.id)
                self.idx = idx
                if self.scores:
                    self.scores.pop()
                self.last = None
                self.status, self.message = "ready", "Removed. Record it again."

    def skip(self):
        if not self.done and not self.busy:
            self.idx += 1
            self.status, self.message = "ready", ""

    def accuracy(self) -> float | None:
        if not self.scores:
            return None
        e = sum(x for x, _ in self.scores)
        n = sum(y for _, y in self.scores)
        return max(0.0, 1 - e / n)

    @staticmethod
    def display(prompt: str | None) -> str | None:
        if prompt is None:
            return None
        return sentence_case(prompt) if prompt == prompt.lower() else prompt

    def snapshot(self) -> dict:
        return {
            "active": True,
            "idx": self.idx,
            "total": len(self.prompts),
            "prompt": None if self.done else self.display(self.prompts[self.idx]),
            "next": self.display(self.prompts[self.idx + 1]) if self.idx + 1 < len(self.prompts) else None,
            "status": self.status,
            "message": self.message,
            "busy": self.busy,
            "saved": len(self.saved),
            "done": self.done,
            "last": self.last,
            "accuracy": self.accuracy(),
        }
