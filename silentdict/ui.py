"""Small OpenCV/PIL preview window helpers (Unicode text via the system PingFang font)."""
from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONTS = [
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/System/Library/Fonts/Helvetica.ttc",
]
WIDTH = 640
PANEL = 190
GREEN, RED, AMBER, GREY, WHITE = (80, 200, 120), (60, 60, 230), (40, 170, 240), (150, 150, 150), (245, 245, 245)


@lru_cache(maxsize=16)
def font(size: int):
    for path in FONTS:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def wrap(text: str, size: int, width: int) -> list[str]:
    f = font(size)
    words, lines, cur = text.split(" "), [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if f.getlength(trial) <= width or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def draw_texts(img: np.ndarray, items) -> np.ndarray:
    """items: iterable of (text, (x, y), size, bgr_color)."""
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    for text, (x, y), size, color in items:
        d.text((x, y), text, font=font(size), fill=(color[2], color[1], color[0]))
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def camera_view(frame, recording: bool, mirror: bool = True) -> np.ndarray:
    if frame is None:
        img = np.zeros((360, WIDTH, 3), np.uint8)
        return img
    img = frame.bgr.copy()
    h, w = img.shape[:2]
    if frame.obs is not None:
        mx, my = frame.obs.points[3]
        r = int(frame.obs.iod * 0.75)
        color = RED if recording else GREEN
        cv2.rectangle(img, (int(mx - r), int(my - r * 0.7)), (int(mx + r), int(my + r * 0.7)), color, 2)
    if mirror:
        img = cv2.flip(img, 1)
    if w != WIDTH:
        img = cv2.resize(img, (WIDTH, int(h * WIDTH / w)))
    if recording:
        cv2.circle(img, (WIDTH - 24, 24), 10, RED, -1)
    return img


def face_hint(frame) -> tuple[str, tuple]:
    if frame is None:
        return "等待摄像头… / waiting for camera", GREY
    if float(frame.gray.mean()) < 2:
        return "画面全黑：镜头被遮挡？可试 --camera 1 / black image", RED
    if frame.obs is None:
        return "没有检测到脸 / no face", RED
    if frame.obs.iod < 40:
        return "离近一点 / move closer", AMBER
    if abs(frame.obs.yaw) > 0.35:
        return "请正对摄像头 / face the camera", AMBER
    bright = float(frame.gray.mean())
    if bright < 60:
        return "光线太暗 / too dark", AMBER
    return "人脸正常 / face OK", GREEN


def compose(view: np.ndarray, roi: np.ndarray | None, items, panel_height: int = PANEL) -> np.ndarray:
    panel = np.full((panel_height, WIDTH, 3), 28, np.uint8)
    canvas = np.vstack([view, panel])
    if roi is not None and len(roi):
        thumb = cv2.resize(roi[len(roi) // 2], (96, 96))
        top = view.shape[0] + 8
        canvas[top:top + 96, 8:104] = cv2.cvtColor(thumb, cv2.COLOR_GRAY2BGR)
    shifted = [(t, (x, y + view.shape[0]), s, c) for t, (x, y), s, c in items]
    return draw_texts(canvas, shifted)
