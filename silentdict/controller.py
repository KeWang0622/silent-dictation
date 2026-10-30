"""App controller: camera, push-to-talk gestures, recognition, calibration and training.

UI-independent. Callbacks (notify) can arrive on any thread; the UI marshals them to the main thread.
Gestures follow Willow Voice: hold the key to dictate, double-tap it for hands-free mode, press it
again to finish.
"""
from __future__ import annotations

import json
import os
import queue
import threading
import time
import traceback
from typing import Callable

from . import devices, paths
from .capture import CameraEngine
from .dataset import save_history
from .hotkey import PushToTalk
from .output import accessibility_allowed, copy_text, paste_text

TAP = 0.3          # a press shorter than this is a tap, not dictation
DOUBLE_TAP = 0.45  # max gap between two taps
MIN_SECONDS = 0.6  # ignore clips shorter than this


class Controller:
    def __init__(self, settings: dict, notify: Callable[[str, object], None]):
        self.settings = settings
        self.notify = notify
        self.state = "loading"   # loading | ready | starting | listening | decoding | training | error
        self.message = "Loading models..."
        self.recognizer = None
        self.camera: CameraEngine | None = None
        self.camera_started = 0.0
        self.camera_lock = threading.Lock()
        self.last_use = 0.0
        self.hands_free = False
        self._consume_up = False
        self._last_tap = 0.0
        self.ptt: PushToTalk | None = None
        self.hotkey_error: str | None = None
        self.calibration = None   # object with .accepting() and .handle(frames)
        self.preview_users = 0
        self.training = False
        self.last_result = None
        self.queue: queue.Queue = queue.Queue()
        self.running = True

    # ------------------------------------------------------------------ lifecycle
    def start(self):
        threading.Thread(target=self._load_models, daemon=True).start()
        threading.Thread(target=self._worker, daemon=True).start()
        threading.Thread(target=self._camera_janitor, daemon=True).start()
        self.start_hotkey()
        if self.settings.get("camera_policy") == "always":
            self.ensure_camera()

    def shutdown(self):
        self.running = False
        self.queue.put(None)
        self.stop_camera()
        if self.ptt:
            self.ptt.stop()

    def _set(self, state: str, message: str = ""):
        self.state, self.message = state, message
        self.notify("state", None)

    def _load_models(self, reload: bool = False):
        try:
            from .pipeline import Recognizer
            decode = {}
            if paths.PERSONAL_DECODE.exists():
                decode = json.load(open(paths.PERSONAL_DECODE))
            choice = self.settings.get("llm", "auto")
            use_llm = decode.pop("use_llm", True) if choice == "auto" else choice == "on"
            rewriter = None
            if use_llm:
                from .rewrite import DEFAULT_LLM, Rewriter
                if DEFAULT_LLM.exists():
                    rewriter = Rewriter()
            rec = Recognizer(rewriter=rewriter, personal=True)
            rec.warm_up()
            self.recognizer = rec
            self._set("ready", self.ready_message())
        except Exception as exc:
            traceback.print_exc()
            self._set("error", f"Could not load models: {exc}")

    def reload_models(self):
        self._set("loading", "Reloading models...")
        threading.Thread(target=self._load_models, args=(True,), daemon=True).start()

    def ready_message(self) -> str:
        r = self.recognizer.reader if self.recognizer else None
        model = "personal model" if (r and r.personalized) else "base model"
        llm = "LLM on" if (self.recognizer and self.recognizer.rewriter) else "LLM off"
        return f"Ready · {model} · {llm}"

    # ------------------------------------------------------------------ hotkey
    def start_hotkey(self):
        if self.ptt:
            self.ptt.stop()
        self.ptt = PushToTalk(self.settings.get("key", "fn"), self._down, self._up, min_hold=0.0)
        self.ptt.start()
        self.ptt.started.wait(3)
        self.hotkey_error = self.ptt.error
        if self.hotkey_error:
            self.notify("state", None)

    def set_key(self, key: str):
        self.settings["key"] = key
        self.start_hotkey()

    def _down(self):
        if self.hands_free:  # second press ends hands-free dictation
            self.hands_free = False
            self._consume_up = True
            self._finish()
            return
        if self.training:
            self.notify("toast", "Training the personal model; dictation is paused.")
            return
        if self.recognizer is None and self.calibration is None:
            self.notify("toast", "Still loading models...")
            return
        if self.calibration is not None and not self.calibration.accepting():
            return
        cam = self.ensure_camera()
        cam.set_max_seconds(30.0)
        cam.start_segment()
        self.last_use = time.monotonic()
        self._set("listening")

    def _up(self, _cancelled: bool):
        if self._consume_up:
            self._consume_up = False
            return
        if self.state != "listening" or self.camera is None:
            return
        held, chorded = self.ptt.last_hold, self.ptt.last_chorded
        if chorded:  # e.g. fn + arrow key: a normal shortcut, not dictation
            self.camera.cancel_segment()
            self._set("ready", self.ready_message())
            return
        if held < TAP:
            now = time.monotonic()
            if now - self._last_tap < DOUBLE_TAP and self.calibration is None:
                self._last_tap = 0.0
                self.hands_free = True
                self.camera.set_max_seconds(60.0)
                self.camera.start_segment()
                self._set("listening", "hands-free")
                return
            self._last_tap = now
            self.camera.cancel_segment()
            self._set("ready", self.ready_message())
            return
        self._finish()

    def _finish(self):
        if self.camera is not None:
            self.camera.stop_segment()
        self.last_use = time.monotonic()
        self._set("decoding")

    # ------------------------------------------------------------------ camera
    def ensure_camera(self) -> CameraEngine:
        with self.camera_lock:
            if self.camera is None or not self.camera.is_alive():
                index = os.environ.get("SILENTDICT_FAKE_CAMERA") or devices.camera_index(
                    self.settings.get("camera_name"))
                self.camera = CameraEngine(on_segment=self.queue.put, camera_index=index)
                self.camera_started = time.monotonic()
                self.camera.start()
            return self.camera

    def stop_camera(self):
        with self.camera_lock:
            if self.camera is not None:
                self.camera.shutdown()
                self.camera = None

    def restart_camera(self):
        self.stop_camera()
        if self.preview_users or self.settings.get("camera_policy") == "always":
            self.ensure_camera()

    def latest_frame(self):
        cam = self.camera
        return cam.latest if cam is not None else None

    def _camera_janitor(self):
        while self.running:
            time.sleep(2)
            policy = self.settings.get("camera_policy", "warm")
            if policy == "always" or self.camera is None or self.preview_users or self.calibration:
                continue
            if self.state in ("listening", "decoding") or self.camera.recording:
                continue
            idle = time.monotonic() - self.last_use
            limit = 0 if policy == "on_demand" else 60 * float(self.settings.get("warm_minutes", 2))
            if idle > limit + 1:
                self.stop_camera()
                self.notify("state", None)

    # ------------------------------------------------------------------ recognition
    def _worker(self):
        while True:
            frames = self.queue.get()
            if frames is None:
                return
            try:
                if self.calibration is not None:
                    self.calibration.handle(frames)
                    self._set("ready", self.ready_message())
                else:
                    self._dictate(frames)
            except Exception as exc:
                traceback.print_exc()
                self._set("ready", self.ready_message())
                self.notify("toast", f"Error: {exc}")

    def _dictate(self, frames):
        if not frames or frames[-1].ts - frames[0].ts < MIN_SECONDS:
            self._set("ready", self.ready_message())
            self.notify("toast", "Too short")
            return
        t0 = time.perf_counter()
        res = self.recognizer.recognize_frames(frames)
        res.timings["total"] = time.perf_counter() - t0
        self.last_result = res
        if res.error:
            self._set("ready", self.ready_message())
            self.notify("toast", res.error)
            return
        print(f"[{res.timings['total']:.2f}s] raw: {res.raw.lower()} -> {res.text}", flush=True)
        res.timings["pasted"] = False
        if res.text and not os.environ.get("SILENTDICT_NO_PASTE"):
            if accessibility_allowed():
                paste_text(res.text + " ")
                res.timings["pasted"] = True
            else:
                copy_text(res.text + " ")
        if self.settings.get("history", True):
            save_history(res.roi, res.raw, res.text, [h.text for h in res.hyps])
        self._set("ready", self.ready_message())
        self.notify("result", res)

    # ------------------------------------------------------------------ training
    def train_async(self, log: Callable[[str], None], done: Callable[[object], None]):
        if self.training:
            return
        self.training = True
        self._set("training", "Training personal model...")

        def job():
            report = None
            try:
                from . import train
                report = train.run(log=log)
            except SystemExit as exc:
                log(str(exc))
            except Exception as exc:
                traceback.print_exc()
                log(f"Training failed: {exc}")
            finally:
                self.training = False
                done(report)
                self.reload_models()

        threading.Thread(target=job, daemon=True).start()
