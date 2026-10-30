"""Persistent app settings (data/settings.json)."""
from __future__ import annotations

import json

from . import paths

DEFAULTS = {
    "key": "fn",                 # push-to-talk key (see hotkey.KEYS)
    "camera_name": None,         # None = first camera
    "camera_policy": "warm",     # "always" | "warm" (stay on a few minutes after use) | "on_demand"
    "warm_minutes": 2,
    "llm": "auto",               # "auto" (decided by calibration) | "on" | "off"
    "history": True,             # keep mouth clips of dictation for review
    "overlay": True,             # floating indicator at the bottom of the screen
    "language": "auto",          # "auto" | "zh" | "en" (app restart needed)
    "onboarded": False,
}


def load() -> dict:
    data = dict(DEFAULTS)
    if paths.SETTINGS.exists():
        try:
            data.update(json.loads(paths.SETTINGS.read_text()))
        except (OSError, ValueError):
            pass
    return data


def save(data: dict):
    paths.DATA.mkdir(parents=True, exist_ok=True)
    tmp = paths.SETTINGS.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(paths.SETTINGS)
