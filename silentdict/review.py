"""Review everyday dictation so confirmed sentences become new personal training data."""
from __future__ import annotations

import cv2
import numpy as np

from . import paths
from .dataset import add_item, history_entries, mark_reviewed
from .text import normalize

HELP = """Review recent dictation / 复查最近的输入
  Enter      the shown text is what you meant -> add it to your training data
  type text  the correct sentence -> add the corrected version
  s          skip for now          d   discard (not useful)          q   quit
"""


def _montage(roi: np.ndarray, n: int = 8) -> np.ndarray:
    idx = np.linspace(0, len(roi) - 1, n).astype(int)
    return cv2.resize(np.hstack([roi[i] for i in idx]), (n * 128, 128))


def run() -> int:
    entries = history_entries()
    if not entries:
        print("Nothing to review. Dictate with 'silentdict run' first (history is on by default).")
        return 0
    print(HELP)
    added = 0
    for n, e in enumerate(entries, 1):
        roi = np.load(paths.HISTORY / f"{e['id']}.npz")["roi"]
        print(f"[{n}/{len(entries)}] read:  {e['text']}")
        if normalize(e["raw"]) != normalize(e["text"]):
            print(f"         raw:   {e['raw'].lower()}")
        cv2.imshow("Silent Dictation - review", _montage(roi))
        cv2.waitKey(1)
        answer = input("> ").strip()
        if answer.lower() == "q":
            break
        if answer.lower() == "s":
            continue
        if answer.lower() != "d":
            text = answer if answer else e["text"]
            add_item(roi, text, "review", meta={"read": e["text"], "raw": e["raw"]})
            added += 1
        mark_reviewed(e["id"])
    cv2.destroyAllWindows()
    print(f"Added {added} sentences. Run 'silentdict train' to update your personal model.")
    return added
