"""Camera permission handling.

OpenCV cannot show the macOS camera prompt from a background thread, so ask on the main
thread (before the capture thread starts) through AVFoundation.
"""
from __future__ import annotations

import os
import time

STATUS = {0: "not determined", 1: "restricted", 2: "denied", 3: "authorized"}


def camera_status() -> int:
    if os.environ.get("SILENTDICT_FAKE_CAMERA"):  # a video file stands in for the camera (tests, demos)
        return 3
    from AVFoundation import AVCaptureDevice, AVMediaTypeVideo

    return int(AVCaptureDevice.authorizationStatusForMediaType_(AVMediaTypeVideo))


def ensure_camera_access(timeout: float = 120.0) -> bool:
    """Returns True when the camera may be used; shows the system prompt if needed (main thread)."""
    from AVFoundation import AVCaptureDevice, AVMediaTypeVideo
    from Foundation import NSDate, NSRunLoop

    status = camera_status()
    if status == 3:
        return True
    if status in (1, 2):
        return False
    answer = {}
    AVCaptureDevice.requestAccessForMediaType_completionHandler_(
        AVMediaTypeVideo, lambda granted: answer.setdefault("granted", bool(granted)))
    end = time.time() + timeout
    while "granted" not in answer and time.time() < end:
        NSRunLoop.currentRunLoop().runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(0.1))
    return answer.get("granted", False)


def request_camera_async(callback):
    """Show the system camera prompt without blocking the UI; callback(granted) runs on some thread."""
    from AVFoundation import AVCaptureDevice, AVMediaTypeVideo

    AVCaptureDevice.requestAccessForMediaType_completionHandler_(
        AVMediaTypeVideo, lambda granted: callback(bool(granted)))


SETTINGS_URLS = {
    "camera": "x-apple.systempreferences:com.apple.preference.security?Privacy_Camera",
    "input": "x-apple.systempreferences:com.apple.preference.security?Privacy_ListenEvent",
    "accessibility": "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility",
    "keyboard": "x-apple.systempreferences:com.apple.Keyboard-Settings.extension",
}
