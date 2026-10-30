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


DICTATION_APPS = {"willow": "Willow Voice", "wispr": "Wispr Flow", "superwhisper": "Superwhisper"}


def running_dictation_apps() -> list[str]:
    """Other push-to-talk dictation apps that are running (they often listen to fn as well)."""
    from AppKit import NSWorkspace

    found = []
    for app in NSWorkspace.sharedWorkspace().runningApplications():
        name = str(app.localizedName() or "").lower()
        for key, product in DICTATION_APPS.items():
            if key in name and product not in found:
                found.append(product)
    return found


def globe_key_setting() -> int | None:
    """System Settings > Keyboard > 'Press Globe key to': 0 = Do Nothing, 1 = input source,
    2 = emoji, 3 = dictation. None when the user never changed it."""
    try:
        out = subprocess.run(["defaults", "read", "com.apple.HIToolbox", "AppleFnUsageType"],
                             capture_output=True, text=True, timeout=3)
        return int(out.stdout.strip()) if out.returncode == 0 else None
    except (ValueError, OSError, subprocess.SubprocessError):
        return None
