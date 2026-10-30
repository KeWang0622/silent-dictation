"""Calibration: read prompted sentences silently so the model can adapt to your face and mouthing."""
from __future__ import annotations

import queue
import threading

import cv2

from . import ui
from .capture import CameraEngine
from .dataset import add_item, delete_item, load_items, load_prompts
from .hotkey import PushToTalk, input_monitoring_allowed
from .permissions import CAMERA_HELP, ensure_camera_access
from .pipeline import Recognizer
from .text import wer

WINDOW = "Silent Dictation - Calibration"
MIN_SECONDS = 0.8

TIPS = """
Calibration / 校准
  1. Sit about an arm's length from the camera with light on your face.
  2. Hold the push-to-talk key, mouth the sentence on screen WITHOUT sound, then release.
     Mouth it the way you will in public. Being consistent matters more than exaggerating.
  3. Keys in the calibration window:  r = redo last   s = skip sentence   q = finish
"""


class Calibrator:
    def __init__(self, key="right_option", camera=0, device="auto", count=60, feedback=True):
        self.prompts = load_prompts(count)
        if not self.prompts:
            raise SystemExit("All prompts are already recorded. Add lines to prompts/custom.txt for more.")
        self.idx = 0
        self.busy = False
        self.saved = []  # (item, prompt_index)
        self.feedback_lines = []
        self.message, self.color = "加载中… / loading", ui.GREY
        self.errors = []  # (ref, hyp) pairs for the running baseline score
        self.recognizer = Recognizer(device=device, personal=False)
        self.feedback = feedback
        self.segments: queue.Queue = queue.Queue()
        self.decode_q: queue.Queue = queue.Queue()
        self.engine = CameraEngine(on_segment=self.segments.put, camera_index=camera)
        self.ptt = PushToTalk(key, self._press, self._release)
        self.key = key

    @property
    def done(self) -> bool:
        return self.idx >= len(self.prompts)

    def _press(self):
        if self.busy or self.done:
            return
        self.engine.start_segment()
        self.message, self.color = "录制中… 说完松开 / recording", ui.RED

    def _release(self, cancelled):
        if cancelled:
            self.engine.cancel_segment()
            self.message, self.color = "已取消，请重来 / cancelled", ui.GREY
            return
        if self.engine.recording:
            self.busy = True
            self.engine.stop_segment()

    def _process(self):
        while True:
            frames = self.segments.get()
            if frames is None:
                return
            try:
                if frames[-1].ts - frames[0].ts < MIN_SECONDS:
                    self.message, self.color = "太短，请重录 / too short, try again", ui.AMBER
                    continue
                try:
                    roi = self.recognizer.roi(frames)
                except ValueError as exc:
                    self.message, self.color = f"没看清你的脸，请重录 / {exc}", ui.AMBER
                    continue
                prompt = self.prompts[self.idx]
                item = add_item(roi, prompt, "calibration")
                self.saved.append((item, self.idx))
                self.idx += 1
                self.message, self.color = f"已保存 / saved ({len(self.saved)})", ui.GREEN
                if self.feedback:
                    self.decode_q.put((item, roi))
            finally:
                self.busy = False

    def _decode(self):
        while True:
            job = self.decode_q.get()
            if job is None:
                return
            item, roi = job
            hyps = self.recognizer.reader(roi, nbest=1)
            hyp = hyps[0].text if hyps else ""
            e, n = wer(item.text, hyp)
            self.errors.append((e, n))
            total_e = sum(x for x, _ in self.errors)
            total_n = sum(y for _, y in self.errors)
            self.feedback_lines = [
                f"基础模型读到 / base model read: {hyp.lower() or '(nothing)'}",
                f"这句错词 {e}/{n}；目前整体词错误率 / running WER {total_e / max(total_n, 1):.0%}",
            ]

    def _redo(self):
        if not self.saved or self.busy:
            return
        item, idx = self.saved.pop()
        delete_item(item.id)
        self.idx = idx
        self.message, self.color = "已删除上一句，请重录 / redo", ui.AMBER

    def run(self) -> int:
        print(TIPS)
        if isinstance(self.engine.camera_index, int) and not ensure_camera_access():
            raise SystemExit(CAMERA_HELP)
        if not input_monitoring_allowed(prompt=True):
            print("Waiting for Input Monitoring permission (System Settings > Privacy & Security).")
        self.engine.start()
        self.engine.ready.wait(10)
        if self.engine.error:
            raise SystemExit(self.engine.error)
        self.ptt.start()
        self.ptt.started.wait(5)
        if self.ptt.error:
            raise SystemExit(self.ptt.error)
        threading.Thread(target=self._process, daemon=True).start()
        threading.Thread(target=self._decode, daemon=True).start()
        key_label = self.key.replace("_", " ").title()
        self.message, self.color = f"按住 {key_label} 开始 / hold {key_label}", ui.GREEN
        try:
            while not self.done:
                frame = self.engine.latest
                view = ui.camera_view(frame, self.engine.recording)
                hint, hint_color = ui.face_hint(frame)
                prompt = self.prompts[self.idx]
                items = [
                    (f"校准 {self.idx + 1}/{len(self.prompts)}  ·  {self.message}", (12, 10), 18, self.color),
                    (hint, (12, 38), 15, hint_color),
                ]
                for i, line in enumerate(ui.wrap(prompt, 30, ui.WIDTH - 24)[:3]):
                    items.append((line, (12, 112 + i * 38), 30, ui.WHITE))
                for i, line in enumerate(self.feedback_lines):
                    items.append((line, (12, 232 + i * 22), 14, ui.GREY))
                cv2.imshow(WINDOW, ui.compose(view, None, items, panel_height=290))
                k = cv2.waitKey(30) & 0xFF
                if k in (ord("q"), 27):
                    break
                if k == ord("r"):
                    self._redo()
                if k == ord("s") and not self.busy:
                    self.idx += 1
        except KeyboardInterrupt:
            pass
        finally:
            self.segments.put(None)
            self.decode_q.put(None)
            self.engine.shutdown()
            self.ptt.stop()
            cv2.destroyAllWindows()
        total = len(load_items())
        print(f"Recorded {len(self.saved)} new sentences this session; {total} personal clips in total.")
        if self.errors:
            e = sum(x for x, _ in self.errors)
            n = sum(y for _, y in self.errors)
            print(f"Base model word error rate on your silent speech this session: {e / max(n, 1):.1%}")
        return total
