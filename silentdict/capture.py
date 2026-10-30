"""Camera capture with per-frame face tracking, push-to-talk segments and pre/post roll."""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

from .face import FaceObs, FaceTracker

WORK_WIDTH = 640  # frames are tracked and cropped at this width; plenty for a laptop camera


@dataclass
class Frame:
    ts: float
    bgr: np.ndarray
    gray: np.ndarray
    obs: FaceObs | None


def to_work_size(bgr: np.ndarray, width: int = WORK_WIDTH) -> np.ndarray:
    h, w = bgr.shape[:2]
    if w <= width:
        return bgr
    return cv2.resize(bgr, (width, int(round(h * width / w))), interpolation=cv2.INTER_AREA)


def make_frame(bgr: np.ndarray, ts: float, tracker: FaceTracker) -> Frame:
    small = to_work_size(bgr)
    obs = tracker(cv2.cvtColor(small, cv2.COLOR_BGR2RGB), ts * 1000.0)
    return Frame(ts, small, cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), obs)


def read_video_file(path: str, start: float = 0.0, end: float | None = None) -> list[Frame]:
    """Decode a video/GIF file into tracked frames (used for testing and offline transcription)."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if not (1.0 < fps < 240.0):
        fps = 25.0
    tracker = FaceTracker(video_mode=True)
    frames, i = [], 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        ts = i / fps
        i += 1
        if ts < start:
            continue
        if end is not None and ts >= end:
            break
        frames.append(make_frame(bgr, ts, tracker))
    cap.release()
    tracker.close()
    return frames


class CameraEngine(threading.Thread):
    """Continuously captures and tracks the face; records segments between start() and stop()."""

    def __init__(self, on_segment: Callable[[list[Frame]], None], camera_index: int | str = 0,
                 width: int = 1280, height: int = 720, fps: int = 30,
                 preroll: float = 0.3, postroll: float = 0.25, max_seconds: float = 30.0):
        super().__init__(daemon=True)
        self.on_segment = on_segment
        self.camera_index = camera_index
        self.size = (width, height, fps)
        self.preroll, self.postroll, self.max_seconds = preroll, postroll, max_seconds
        self.lock = threading.Lock()
        self.ring: deque[Frame] = deque(maxlen=int(fps * 2))
        self.latest: Frame | None = None
        self.segment: list[Frame] | None = None
        self.stop_at: float | None = None
        self.running = True
        self.error: str | None = None
        self.measured_fps = 0.0
        self.ready = threading.Event()

    @property
    def recording(self) -> bool:
        return self.segment is not None

    def start_segment(self):
        with self.lock:
            now = time.monotonic()
            self.segment = [f for f in self.ring if f.ts >= now - self.preroll]
            self.stop_at = None

    def stop_segment(self):
        with self.lock:
            if self.segment is not None and self.stop_at is None:
                self.stop_at = time.monotonic() + self.postroll

    def cancel_segment(self):
        with self.lock:
            self.segment, self.stop_at = None, None

    def run(self):
        w, h, fps = self.size
        replay = isinstance(self.camera_index, str)  # a video file played back as a fake camera
        if replay:
            cap = cv2.VideoCapture(self.camera_index)
            file_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        else:
            cap = cv2.VideoCapture(self.camera_index, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            self.error = ("Cannot open the camera. Allow camera access for this terminal in "
                          "System Settings > Privacy & Security > Camera, then restart it.")
            self.ready.set()
            return
        if not replay:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            cap.set(cv2.CAP_PROP_FPS, fps)
        tracker = FaceTracker(video_mode=True)
        stamps: deque[float] = deque(maxlen=30)
        self.ready.set()
        next_due = time.monotonic()
        try:
            while self.running:
                ok, bgr = cap.read()
                if not ok:
                    if replay:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    time.sleep(0.01)
                    continue
                if replay:  # pace playback like a real camera
                    next_due += 1.0 / file_fps
                    time.sleep(max(0.0, next_due - time.monotonic()))
                ts = time.monotonic()
                frame = make_frame(bgr, ts, tracker)
                stamps.append(ts)
                if len(stamps) > 1:
                    self.measured_fps = (len(stamps) - 1) / (stamps[-1] - stamps[0])
                done = None
                with self.lock:
                    self.latest = frame
                    self.ring.append(frame)
                    if self.segment is not None:
                        self.segment.append(frame)
                        too_long = ts - self.segment[0].ts > self.max_seconds
                        if too_long and self.stop_at is None:
                            self.stop_at = ts
                        if self.stop_at is not None and ts >= self.stop_at:
                            done, self.segment, self.stop_at = self.segment, None, None
                if done:
                    self.on_segment(done)
        finally:
            cap.release()
            tracker.close()

    def shutdown(self):
        self.running = False
        if self.is_alive() and threading.current_thread() is not self:
            self.join(timeout=3)
