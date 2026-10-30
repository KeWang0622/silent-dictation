"""Camera discovery (AVFoundation order matches OpenCV's camera indices on macOS)."""
from __future__ import annotations

import subprocess


def cameras() -> list[str]:
    from AVFoundation import AVCaptureDevice, AVMediaTypeVideo

    return [str(d.localizedName()) for d in AVCaptureDevice.devicesWithMediaType_(AVMediaTypeVideo)]


def camera_index(name: str | None) -> int:
    if not name:
        return 0
    names = cameras()
    return names.index(name) if name in names else 0


def running_apps_matching(*words: str) -> list[str]:
    from AppKit import NSWorkspace

    out = []
    for app in NSWorkspace.sharedWorkspace().runningApplications():
        name = str(app.localizedName() or "")
        if any(w.lower() in name.lower() for w in words):
            out.append(name)
    return out


def globe_key_setting() -> int | None:
    """System Settings > Keyboard > 'Press Globe key to': 0 = Do Nothing, 1 = input source,
    2 = emoji, 3 = dictation. None when the user never changed it."""
    try:
        out = subprocess.run(["defaults", "read", "com.apple.HIToolbox", "AppleFnUsageType"],
                             capture_output=True, text=True, timeout=3)
        return int(out.stdout.strip()) if out.returncode == 0 else None
    except (ValueError, OSError, subprocess.SubprocessError):
        return None
