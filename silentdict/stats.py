"""Usage counters shown on the Home screen."""
from __future__ import annotations

import json
import time

from . import paths


def load() -> dict:
    data = {"dictations": 0, "words": 0, "since": None}
    try:
        data.update(json.loads(paths.STATS.read_text()))
    except (OSError, ValueError):
        pass
    return data


def record(text: str):
    data = load()
    data["dictations"] += 1
    data["words"] += len(text.split())
    data["since"] = data["since"] or time.time()
    paths.DATA.mkdir(parents=True, exist_ok=True)
    paths.STATS.write_text(json.dumps(data))
