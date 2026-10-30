"""MediaPipe Face Landmarker -> the four stable points used by the Auto-AVSR mouth cropper.

MediaPipe 1.x removed the legacy mp.solutions API that Chaplin relied on, so this uses the
Tasks API and derives iBUG-style anchor points from the 478-point face mesh.
"""
from __future__ import annotations

from dataclasses import dataclass

import os

os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import mediapipe as mp  # noqa: E402
import numpy as np  # noqa: E402
from mediapipe.tasks.python import BaseOptions, vision  # noqa: E402

from . import paths  # noqa: E402

# Face-mesh indices that correspond to the iBUG-68 groups used by the reference mean face.
EYE_A = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
EYE_B = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]
NOSE_BASE = [98, 97, 2, 326, 327]  # iBUG 31-35: bottom of the nose
LIPS = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146,
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95,
]
NOSE_TIP = 1
UPPER_INNER_LIP, LOWER_INNER_LIP = 13, 14


@dataclass
class FaceObs:
    points: np.ndarray  # (4, 2) float32: image-left eye, image-right eye, nose base, mouth centre
    bbox: tuple[int, int, int, int]
    mouth_open: float  # inner-lip gap divided by inter-ocular distance
    yaw: float  # nose-tip offset from the eye midpoint, in inter-ocular units
    iod: float  # inter-ocular distance in pixels


def obs_from_landmarks(pts: np.ndarray) -> FaceObs:
    e1 = pts[EYE_A].mean(0)
    e2 = pts[EYE_B].mean(0)
    if e1[0] > e2[0]:
        e1, e2 = e2, e1
    nose = pts[NOSE_BASE].mean(0)
    mouth = pts[LIPS].mean(0)
    four = np.stack([e1, e2, nose, mouth]).astype(np.float32)
    iod = float(np.linalg.norm(e2 - e1)) + 1e-6
    gap = float(np.linalg.norm(pts[UPPER_INNER_LIP] - pts[LOWER_INNER_LIP])) / iod
    yaw = float((pts[NOSE_TIP][0] - (e1[0] + e2[0]) / 2) / iod)
    x0, y0 = pts.min(0)
    x1, y1 = pts.max(0)
    return FaceObs(four, (int(x0), int(y0), int(x1), int(y1)), gap, yaw, iod)


class FaceTracker:
    def __init__(self, model_path=None, video_mode: bool = True):
        model_path = model_path or paths.FACE_MODEL
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO if video_mode else vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.4,
            min_face_presence_confidence=0.4,
            min_tracking_confidence=0.4,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._video = video_mode
        self._last_ts = -1

    def __call__(self, rgb: np.ndarray, ts_ms: float | None = None) -> FaceObs | None:
        h, w = rgb.shape[:2]
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        if self._video:
            ts = int(ts_ms if ts_ms is not None else self._last_ts + 33)
            if ts <= self._last_ts:
                ts = self._last_ts + 1
            self._last_ts = ts
            result = self._landmarker.detect_for_video(image, ts)
        else:
            result = self._landmarker.detect(image)
        if not result.face_landmarks:
            return None
        lm = result.face_landmarks[0]
        pts = np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float32)
        return obs_from_landmarks(pts)

    def close(self):
        self._landmarker.close()
