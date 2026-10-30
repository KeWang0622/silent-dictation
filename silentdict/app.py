"""Everyday silent dictation: hold the push-to-talk key, mouth a sentence, release to type it."""
from __future__ import annotations

import queue
import threading
import time

import cv2

from . import CREDITS, paths, ui
from .capture import CameraEngine
from .dataset import save_history
from .hotkey import PushToTalk, input_monitoring_allowed
from .output import accessibility_allowed, paste_text
from .permissions import CAMERA_HELP, ensure_camera_access
from .pipeline import Recognizer

WINDOW = "Silent Dictation"
MIN_SECONDS = 0.6


class DictationApp:
    def __init__(self, key="right_option", camera=0, device="auto", llm=None, personal=True,
                 paste=True, preview=True, history=True, mirror=True):
        self.key, self.paste, self.preview, self.history, self.mirror = key, paste, preview, history, mirror
        self.status, self.color = "加载模型中… / loading models", ui.GREY
        self.last_text = self.last_raw = ""
        self.last_roi = None
        self.last_ms = 0
        self.pending = 0
        self.queue: queue.Queue = queue.Queue()

        print(CREDITS)
        decode = {}
        if paths.PERSONAL_DECODE.exists():
            import json
            decode = json.load(open(paths.PERSONAL_DECODE))
        use_llm = decode.pop("use_llm", True) if llm is None else llm
        rewriter = None
        if use_llm:
            from .rewrite import DEFAULT_LLM, Rewriter
            if DEFAULT_LLM.exists():
                rewriter = Rewriter()
            else:
                print("Local LLM not downloaded; using raw lip-reading output.")
        self.recognizer = Recognizer(rewriter=rewriter, device=device, personal=personal)
        r = self.recognizer.reader
        print(f"Lip reader: {'personal' if r.personalized else 'base'} model, encoder on {r.device}, "
              f"LLM rewrite {'on' if rewriter else 'off'}")
        self.engine = CameraEngine(on_segment=self.queue.put, camera_index=camera)
        self.ptt = PushToTalk(key, self._press, self._release)
        self.worker = threading.Thread(target=self._work, daemon=True)

    # ---- push-to-talk ----
    def _press(self):
        self.engine.start_segment()
        self.status, self.color = "正在读唇… 松开结束 / reading lips", ui.RED

    def _release(self, cancelled: bool):
        if cancelled:
            self.engine.cancel_segment()
            self.status, self.color = "已取消 / cancelled", ui.GREY
        else:
            self.engine.stop_segment()
            self.pending += 1
            self.status, self.color = "识别中… / decoding", ui.AMBER

    # ---- recognition worker ----
    def _work(self):
        while True:
            frames = self.queue.get()
            if frames is None:
                return
            try:
                self._handle(frames)
            except Exception as exc:  # keep the app alive
                self.status, self.color = f"出错 / error: {exc}", ui.RED
            finally:
                self.pending = max(0, self.pending - 1)

    def _handle(self, frames):
        if frames[-1].ts - frames[0].ts < MIN_SECONDS:
            self.status, self.color = "太短了 / too short", ui.GREY
            return
        t0 = time.perf_counter()
        res = self.recognizer.recognize_frames(frames)
        self.last_ms = int((time.perf_counter() - t0) * 1000)
        if res.error:
            self.status, self.color = f"{res.error}", ui.RED
            return
        self.last_roi, self.last_raw, self.last_text = res.roi, res.raw, res.text
        print(f"[{self.last_ms} ms] raw: {res.raw.lower()}\n           -> {res.text}")
        if self.paste and res.text:
            paste_text(res.text + " ")
        if self.history:
            save_history(res.roi, res.raw, res.text, [h.text for h in res.hyps])
        self.status, self.color = f"完成 / done ({self.last_ms} ms)", ui.GREEN

    # ---- main loop (UI must live on the main thread on macOS) ----
    def run(self):
        if isinstance(self.engine.camera_index, int) and not ensure_camera_access():
            raise SystemExit(CAMERA_HELP)
        threading.Thread(target=self.recognizer.warm_up, daemon=True).start()
        if not input_monitoring_allowed(prompt=True):
            print("Waiting for Input Monitoring permission (System Settings > Privacy & Security).")
        if self.paste and not accessibility_allowed(prompt=True):
            print("Pasting needs Accessibility permission (System Settings > Privacy & Security > Accessibility).")
        self.engine.start()
        self.engine.ready.wait(10)
        if self.engine.error:
            raise SystemExit(self.engine.error)
        self.ptt.start()
        self.ptt.started.wait(5)
        if self.ptt.error:
            raise SystemExit(self.ptt.error)
        self.worker.start()
        key_label = self.key.replace("_", " ").title()
        self.status, self.color = f"就绪：按住 {key_label} 无声说话 / ready", ui.GREEN
        print(f"Ready. Hold {key_label}, mouth your sentence silently, release to type. Ctrl+C to quit.")
        try:
            while True:
                if self.preview:
                    frame = self.engine.latest
                    view = ui.camera_view(frame, self.engine.recording, self.mirror)
                    hint, hint_color = ui.face_hint(frame)
                    items = [
                        (self.status, (116, 10), 20, self.color),
                        (hint + f"   {self.engine.measured_fps:.0f} fps", (116, 40), 16, hint_color),
                    ]
                    if self.last_raw:
                        items.append(("读到 / read: " + self.last_raw.lower(), (116, 70), 15, ui.GREY))
                    for i, line in enumerate(ui.wrap(self.last_text, 22, ui.WIDTH - 24)[:3]):
                        items.append((line, (12, 112 + i * 26), 22, ui.WHITE))
                    cv2.imshow(WINDOW, ui.compose(view, self.last_roi, items))
                    key = cv2.waitKey(30) & 0xFF
                    if key in (ord("q"), 27):
                        break
                    if key == ord("m"):
                        self.mirror = not self.mirror
                else:
                    time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        finally:
            self.queue.put(None)
            self.engine.shutdown()
            self.ptt.stop()
            cv2.destroyAllWindows()
