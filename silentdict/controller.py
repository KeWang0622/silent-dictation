"""App controller: camera, push-to-talk gestures, recognition, calibration and training.

UI-independent. notify(kind, data) can be called from any thread; the UI marshals it to the main thread.
"""
from __future__ import annotations

import csv
import json
import os
import queue
import threading
import time
import traceback
from typing import TYPE_CHECKING, Callable

from . import devices, paths, stats
from .activity import bar_levels
from .dataset import NotEnoughData, history_entries, save_history
from .gestures import CANCEL, HANDS_FREE, START, STOP, Gestures
from .hotkey import PushToTalk, input_monitoring_allowed
from .output import accessibility_allowed, copy_text, paste_text
from .permissions import camera_status

if TYPE_CHECKING:
    from .capture import CameraEngine

MIN_SECONDS = 0.6
BEAMS = {"fast": 6, "balanced": 12, "accurate": 20}


class Controller:
    def __init__(self, settings: dict, notify: Callable[[str, object], None]):
        self.settings = settings
        self.notify = notify
        self.state = "loading"   # loading | ready | listening | decoding | error
        self.message = "Loading models"
        self.hands_free = False
        self.recognizer = None
        self.llm_available = paths.LLM_DIR.exists()
        self.camera: CameraEngine | None = None
        self.camera_started = 0.0
        self.camera_lock = threading.Lock()
        self.last_use = 0.0
        self.gestures = Gestures()
        self.key_down = False
        self.ptt: PushToTalk | None = None
        self.hotkey_error: str | None = None
        self.session = None            # CalibrationSession while calibrating
        self.preview_users = 0
        self.training = {"running": False, "stage": "", "progress": 0.0, "log": [], "report": None, "error": None}
        self.last_result = None
        self.should_paste: Callable[[], bool] = lambda: True
        # the calibration session records the key only while its page is in front; otherwise it's dictation
        self.calibration_visible: Callable[[], bool] = lambda: True
        self.recording_for = "dictation"
        self.queue: queue.Queue = queue.Queue()
        self.running = True

    # ------------------------------------------------------------------ lifecycle
    def start(self):
        threading.Thread(target=self._load_models, daemon=True).start()
        threading.Thread(target=self._worker, daemon=True).start()
        threading.Thread(target=self._janitor, daemon=True).start()
        self.start_hotkey()
        if self.settings.get("camera_policy") == "always" and camera_status() == 3:
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

    # ------------------------------------------------------------------ models
    def decode_settings(self) -> dict:
        decode = {}
        if paths.PERSONAL_DECODE.exists():
            try:
                decode = json.load(open(paths.PERSONAL_DECODE))
            except ValueError:
                decode = {}
        decode["beam_size"] = BEAMS.get(self.settings.get("mode", "balanced"), 12)
        return decode

    def llm_enabled(self) -> bool:
        choice = self.settings.get("llm", "auto")
        if not self.llm_available or choice == "off":
            return False
        if choice == "on":
            return True
        return bool(self.decode_settings().get("use_llm", False))  # auto: only if calibration proved it helps

    def _load_models(self):
        t0 = time.monotonic()
        try:
            from . import capture  # noqa: F401  (load OpenCV and MediaPipe here, not on the UI thread)
            from .pipeline import Recognizer
            decode = self.decode_settings()
            decode.pop("use_llm", None)
            rewriter = None
            if self.llm_enabled():
                from .rewrite import Rewriter
                rewriter = Rewriter()
            rec = Recognizer(rewriter=rewriter, personal=True, decode=decode)
            rec.warm_up()
            self.recognizer = rec
            print(f"models ready in {time.monotonic() - t0:.1f}s (personal={rec.reader.personalized}, "
                  f"ai_cleanup={rewriter is not None})", flush=True)
            self._set("ready")
        except Exception as exc:
            traceback.print_exc()
            self._set("error", f"Could not load models: {exc}")

    def reload_models(self):
        self.recognizer = None
        self._set("loading", "Loading models")
        threading.Thread(target=self._load_models, daemon=True).start()

    # ------------------------------------------------------------------ hotkey + gestures
    def start_hotkey(self):
        if self.ptt:
            self.ptt.stop()
        self.ptt = PushToTalk(self.settings.get("key", "fn"), self._down, self._up, min_hold=0.0)
        self.ptt.start()
        self.ptt.started.wait(3)
        self.hotkey_error = self.ptt.error
        self.notify("state", None)

    def set_key(self, key: str):
        self.settings["key"] = key
        self.gestures.reset()
        self.start_hotkey()

    def _blocked_reason(self) -> str | None:
        if self.training["running"]:
            return "Training your personal model. Dictation resumes when it finishes."
        if self.recognizer is None:
            return "Still loading models"
        if self._calibrating() and not self.session.accepting():
            return "busy"
        if not os.environ.get("SILENTDICT_FAKE_CAMERA") and camera_status() != 3:
            return "camera"
        return None

    def _calibrating(self) -> bool:
        return self.session is not None and self.calibration_visible()

    def _down(self):
        self.key_down = True
        self.notify("key", True)
        action = self.gestures.down()
        if action == STOP:
            self._finish()
        elif action == START:
            reason = self._blocked_reason()
            if reason:
                self.gestures.reset()
                if reason == "camera":
                    self.notify("need_camera", None)
                elif reason != "busy":
                    self.notify("toast", reason)
                return
            cam = self.ensure_camera()
            cam.set_max_seconds(30.0)
            cam.start_segment()
            self.hands_free = False
            self.last_use = time.monotonic()
            self.recording_for = "calibration" if self._calibrating() else "dictation"
            if self.recording_for == "calibration":
                self.session.status = "recording"
            self._set("listening")

    def _up(self, _cancelled: bool):
        self.key_down = False
        self.notify("key", False)
        if self.ptt is None:
            return
        action = self.gestures.up(self.ptt.last_hold, self.ptt.last_chorded)
        if action == CANCEL:
            if self.camera is not None:
                self.camera.cancel_segment()
            if self.session and self.recording_for == "calibration":
                self.session.status = "ready"
            self._set("ready")
        elif action == HANDS_FREE:
            if self.recording_for == "calibration":  # calibration uses press-and-hold only
                self.gestures.reset()
                self.camera.cancel_segment()
                self._set("ready")
                return
            self.hands_free = True
            self.camera.set_max_seconds(60.0)
            self._set("listening", "hands-free")
        elif action == STOP:
            self._finish()

    def _finish(self):
        if self.camera is not None:
            self.camera.stop_segment()
        self.hands_free = False
        self.last_use = time.monotonic()
        self._set("decoding")

    # ------------------------------------------------------------------ camera
    def ensure_camera(self) -> CameraEngine:
        from .capture import CameraEngine
        with self.camera_lock:
            if self.camera is None or not self.camera.is_alive():
                index = os.environ.get("SILENTDICT_FAKE_CAMERA") or devices.camera_index(
                    self.settings.get("camera_name"))
                self.camera = CameraEngine(on_segment=self.queue.put, camera_index=index)
                self.camera.full_rate = bool(self.preview_users or self.session)
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
        if (self.preview_users or self.session) and camera_status() == 3:
            self.ensure_camera()

    def set_preview(self, on: bool):
        self.preview_users = 1 if on else 0
        if on and camera_status() == 3:
            self.ensure_camera()
        if self.camera is not None:
            self.camera.full_rate = bool(on or self.session)
        self.last_use = time.monotonic()

    def latest_frame(self):
        cam = self.camera
        return cam.latest if cam is not None else None

    def camera_problem(self) -> str | None:
        cam = self.camera
        if cam is None:
            return None
        if cam.error:
            return "Can't open this camera"
        frame = cam.latest
        if frame is None:
            return None
        if float(frame.gray.mean()) < 2 and time.monotonic() - self.camera_started > 3:
            return "The image is black. Is the lens covered? Try another camera."
        if cam.last_obs is None:
            return "Can't see your face"
        if cam.last_obs.iod < 30:
            return "Move a little closer"
        if abs(cam.last_obs.yaw) > 0.35:
            return "Face the camera"
        if float(frame.gray.mean()) < 50:
            return "It's dark. More light on your face helps a lot"
        return None

    def levels(self, bars: int = 7) -> list[float]:
        cam = self.camera
        return bar_levels(cam.recent_obs(bars) if cam else [], bars)

    def _janitor(self):
        while self.running:
            time.sleep(2)
            if self.hotkey_error and input_monitoring_allowed():
                self.start_hotkey()
            policy = self.settings.get("camera_policy", "warm")
            if policy == "always" or self.camera is None or self.preview_users or self.session:
                continue
            if self.state in ("listening", "decoding") or self.camera.recording:
                continue
            limit = 0 if policy == "on_demand" else 60 * float(self.settings.get("warm_minutes", 2))
            if time.monotonic() - self.last_use > limit + 1:
                self.stop_camera()
                self.notify("state", None)

    # ------------------------------------------------------------------ recognition
    def _worker(self):
        while True:
            frames = self.queue.get()
            if frames is None:
                return
            try:
                if self.recording_for == "calibration" and self.session is not None:
                    self.session.handle(frames)
                    self._set("ready")
                    self.notify("calibration", None)
                else:
                    self._dictate(frames)
            except Exception as exc:
                traceback.print_exc()
                self._set("ready")
                self.notify("toast", f"Something went wrong: {exc}")

    def _dictate(self, frames):
        if not frames or frames[-1].ts - frames[0].ts < MIN_SECONDS:
            self._set("ready")
            self.notify("toast", "Too short. Hold the key while you mouth the sentence.")
            return
        t0 = time.perf_counter()
        res = self.recognizer.recognize_frames(frames)
        res.timings["total"] = time.perf_counter() - t0
        if res.error:
            self._set("ready")
            self.notify("toast", "Couldn't see your face clearly. Try again facing the camera.")
            return
        print(f"[{res.timings['total']:.2f}s] raw: {res.raw.lower()} -> {res.text}", flush=True)
        res.timings["delivered"] = "app"
        if res.text and not os.environ.get("SILENTDICT_NO_PASTE") and self.should_paste():
            if accessibility_allowed():
                paste_text(res.text + " ")
                res.timings["delivered"] = "pasted"
            else:
                copy_text(res.text + " ")
                res.timings["delivered"] = "copied"
        if res.text:
            stats.record(res.text)
            if self.settings.get("history", True):
                save_history(res.roi, res.raw, res.text, [h.text for h in res.hyps])
        self.last_result = res
        self._set("ready")
        self.notify("result", res)

    # ------------------------------------------------------------------ calibration + training
    def start_calibration(self, count: int):
        from .session import CalibrationSession
        self.session = CalibrationSession(self, count)
        self.gestures.reset()
        if camera_status() == 3 or os.environ.get("SILENTDICT_FAKE_CAMERA"):
            self.ensure_camera().full_rate = True
        self.notify("calibration", None)

    def stop_calibration(self):
        self.session = None
        if self.camera is not None:
            self.camera.full_rate = bool(self.preview_users)
        self.notify("calibration", None)

    def train_async(self):
        if self.training["running"]:
            return
        t = self.training
        t.update(running=True, stage="baseline", progress=0.0, log=[], report=None, error=None)
        self.notify("training", None)

        def log(line: str):
            t["log"] = (t["log"] + [line.strip()])[-40:]
            self.notify("training", None)

        def progress(stage: str, frac: float):
            spans = {"baseline": (0.0, 0.1), "adapt": (0.1, 0.8), "tune": (0.8, 0.95), "llm": (0.95, 0.99),
                     "done": (1.0, 1.0)}
            lo, hi = spans.get(stage, (0.0, 1.0))
            t["stage"], t["progress"] = stage, lo + (hi - lo) * frac
            self.notify("training", None)

        def job():
            try:
                from . import train
                epochs = int(os.environ["SILENTDICT_TRAIN_EPOCHS"]) if os.environ.get("SILENTDICT_TRAIN_EPOCHS") else None
                t["report"] = train.run(log=log, progress=progress, epochs=epochs)
            except NotEnoughData as exc:
                t["error"] = str(exc)
            except Exception as exc:
                traceback.print_exc()
                t["error"] = f"Training failed: {exc}"
            finally:
                t["running"] = False
                self.notify("training", "done")
                self.reload_models()

        threading.Thread(target=job, daemon=True).start()

    # ------------------------------------------------------------------ summaries for the UI
    def recent(self, n: int = 30) -> list[dict]:
        items = history_entries(unreviewed_only=False)[-n:][::-1]
        return [{"id": e["id"], "time": e["created"], "text": e["text"], "raw": e["raw"],
                 "reviewed": e.get("reviewed", False)} for e in items]

    def accuracy_history(self) -> list[dict]:
        progress = paths.REPORTS / "progress.csv"
        if not progress.exists():
            return []
        out = []
        for r in csv.DictReader(open(progress)):
            base = float(r["base_wer"])
            if r.get("active_wer"):
                active = float(r["active_wer"])
            else:
                active = base if r.get("active", "base") == "base" else min(base, float(r["personal_wer"]))
            out.append({"time": r["time"], "base": max(0.0, 1 - base), "personal": max(0.0, 1 - active),
                        "clips": int(r["train_clips"]) + int(r["val_clips"])})
        return out
