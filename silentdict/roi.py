"""Mouth region-of-interest extraction compatible with Auto-AVSR / Chaplin preprocessing."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

REF_SIZE = 256
CROP = 96


def stable_reference(mean_face_path: Path | None = None) -> np.ndarray:
    ref = np.load(mean_face_path or Path(__file__).with_name("mean_face_68.npy"))
    return np.vstack([
        ref[36:42].mean(0),  # eye on the image left
        ref[42:48].mean(0),  # eye on the image right
        ref[31:36].mean(0),  # nose base
        ref[48:68].mean(0),  # mouth centre
    ]).astype(np.float32)


def interpolate_points(points: list) -> np.ndarray | None:
    """Fill frames where the face was not found by linear interpolation / edge padding."""
    valid = [i for i, p in enumerate(points) if p is not None]
    if not valid:
        return None
    out = np.zeros((len(points), 4, 2), dtype=np.float32)
    for i in valid:
        out[i] = points[i]
    for a, b in zip(valid, valid[1:]):
        for k in range(a + 1, b):
            t = (k - a) / (b - a)
            out[k] = (1 - t) * out[a] + t * out[b]
    out[: valid[0]] = out[valid[0]]
    out[valid[-1] + 1:] = out[valid[-1]]
    return out


def resample_indices(timestamps, fps: float = 25.0) -> np.ndarray:
    """Pick frames on a uniform 25 fps grid using real capture timestamps.

    Webcams change frame rate in low light; the model was trained on 25 fps video, so
    we resample by time rather than assuming a constant camera rate.
    """
    ts = np.asarray(timestamps, dtype=np.float64)
    if len(ts) < 2:
        return np.arange(len(ts))
    n = int(round((ts[-1] - ts[0]) * fps)) + 1
    grid = ts[0] + np.arange(n) / fps
    right = np.clip(np.searchsorted(ts, grid), 1, len(ts) - 1)
    left = right - 1
    pick_left = (grid - ts[left]) <= (ts[right] - grid)
    return np.where(pick_left, left, right)


class MouthCropper:
    def __init__(self, mean_face_path: Path | None = None, window_margin: int = 12):
        self.ref = stable_reference(mean_face_path)
        self.window_margin = window_margin

    def __call__(self, gray_frames: list, points: list) -> np.ndarray:
        pts = interpolate_points(points)
        if pts is None:
            raise ValueError("no face found in this clip")
        n = len(gray_frames)
        out = np.zeros((n, CROP, CROP), dtype=np.uint8)
        last_m = None
        half = CROP // 2
        for i, frame in enumerate(gray_frames):
            m = min(self.window_margin // 2, i, n - 1 - i)
            smooth = pts[i - m: i + m + 1].mean(0)
            smooth = smooth + (pts[i].mean(0) - smooth.mean(0))
            M, _ = cv2.estimateAffinePartial2D(smooth, self.ref, method=cv2.LMEDS)
            if M is None:
                M = last_m if last_m is not None else cv2.estimateAffinePartial2D(smooth, self.ref)[0]
            last_m = M
            warped = cv2.warpAffine(frame, M, (REF_SIZE, REF_SIZE), flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            cx, cy = M @ np.array([smooth[3, 0], smooth[3, 1], 1.0])
            padded = cv2.copyMakeBorder(warped, half, half, half, half, cv2.BORDER_CONSTANT, value=0)
            x0 = int(round(cx))
            y0 = int(round(cy))
            out[i] = padded[y0: y0 + CROP, x0: x0 + CROP]
        return out


FACE_SCALE = 1.6  # face crop side relative to the landmark box; matches LRS3-style framing best on our benchmark


@dataclass
class Clip:
    """What the lip readers see: grey 96x96 mouth crops and (for FLARE) matching 96x96 face crops."""
    lip: np.ndarray
    face: np.ndarray | None = None

    def __len__(self):
        return len(self.lip)


def face_crops(frames, scale: float = FACE_SCALE, size: int = CROP) -> np.ndarray:
    """Square grey crops around the face (box smoothed over ~0.36 s so the crop does not jitter)."""
    boxes = [None if f.obs is None else f.obs.bbox for f in frames]
    valid = [b for b in boxes if b is not None]
    if not valid:
        raise ValueError("no face found in this clip")
    arr = np.array([((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, max(b[2] - b[0], b[3] - b[1]))
                    for b in (b if b is not None else valid[0] for b in boxes)], dtype=np.float64)
    pad = np.pad(arr, ((4, 4), (0, 0)), mode="edge")
    arr = np.stack([np.convolve(pad[:, i], np.ones(9) / 9, mode="valid") for i in range(3)], 1)
    out = np.empty((len(frames), size, size), np.uint8)
    for i, (f, (cx, cy, side)) in enumerate(zip(frames, arr)):
        k = size / (side * scale)
        M = np.array([[k, 0, size / 2 - cx * k], [0, k, size / 2 - cy * k]], dtype=np.float32)
        out[i] = cv2.warpAffine(f.gray, M, (size, size), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE)
    return out


def frames_to_clip(frames, cropper: MouthCropper, fps: float = 25.0, min_face_ratio: float = 0.5,
                   trim: bool = True, face: bool = True) -> Clip:
    """Mouth (and face) crops on a 25 fps grid, with the still parts at the start and end trimmed."""
    from .activity import active_span

    if not frames:
        raise ValueError("empty clip")
    found = sum(1 for f in frames if f.obs is not None)
    if found / len(frames) < min_face_ratio:
        raise ValueError(f"face visible in only {found}/{len(frames)} frames")
    chosen = [frames[i] for i in resample_indices([f.ts for f in frames], fps)]
    if trim:
        start, end = active_span([f.obs for f in chosen])
        chosen = chosen[start:end]
    lip = cropper([f.gray for f in chosen], [f.obs.points if f.obs is not None else None for f in chosen])
    return Clip(lip, face_crops(chosen) if face else None)


def frames_to_roi(frames, cropper: MouthCropper, fps: float = 25.0, min_face_ratio: float = 0.5,
                  trim: bool = True):
    """frames: objects with .ts, .gray and .obs. Returns uint8 (T, 96, 96).

    With trim, leading/trailing stretches without lip movement are dropped (keeping a margin), so the
    recogniser does not invent words for the moments before and after you mouth the sentence.
    """
    from .activity import active_span

    if not frames:
        raise ValueError("empty clip")
    found = sum(1 for f in frames if f.obs is not None)
    if found / len(frames) < min_face_ratio:
        raise ValueError(f"face visible in only {found}/{len(frames)} frames")
    idx = resample_indices([f.ts for f in frames], fps)
    chosen = [frames[i] for i in idx]
    if trim:
        start, end = active_span([f.obs for f in chosen])
        chosen = chosen[start:end]
    return cropper([f.gray for f in chosen], [f.obs.points if f.obs is not None else None for f in chosen])
