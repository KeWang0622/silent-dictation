"""Local personal dataset: mouth ROI clips + transcripts, stored only on this Mac."""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from . import paths
from .text import normalize

VAL_EVERY = 5  # every 5th clip is held out, permanently, to measure real progress


class NotEnoughData(Exception):
    """Too few personal sentences to train or evaluate."""


@dataclass
class Item:
    id: str
    text: str
    split: str
    source: str
    created: float
    frames: int
    meta: dict = field(default_factory=dict)

    @property
    def path(self) -> Path:
        return paths.CLIPS / f"{self.id}.npz"

    def roi(self) -> np.ndarray:
        return np.load(self.path)["roi"]


def load_items(split: str | None = None) -> list[Item]:
    if not paths.MANIFEST.exists():
        return []
    items = []
    for line in paths.MANIFEST.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        if d.get("deleted"):
            continue
        it = Item(**{k: d[k] for k in ("id", "text", "split", "source", "created", "frames")},
                  meta=d.get("meta", {}))
        if it.path.exists() and (split is None or it.split == split):
            items.append(it)
    return items


def _next_split() -> str:
    n = len(load_items())
    return "val" if n % VAL_EVERY == VAL_EVERY - 1 else "train"


def add_item(roi: np.ndarray, text: str, source: str, meta: dict | None = None,
             split: str | None = None) -> Item:
    paths.CLIPS.mkdir(parents=True, exist_ok=True)
    item = Item(id=time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6], text=normalize(text),
                split=split or _next_split(), source=source, created=time.time(), frames=len(roi),
                meta=meta or {})
    np.savez_compressed(item.path, roi=roi.astype(np.uint8))
    with open(paths.MANIFEST, "a") as f:
        f.write(json.dumps(asdict(item)) + "\n")
    return item


def delete_item(item_id: str):
    with open(paths.MANIFEST, "a") as f:
        f.write(json.dumps({"id": item_id, "deleted": True}) + "\n")
    p = paths.CLIPS / f"{item_id}.npz"
    if p.exists():
        p.unlink()


def recorded_texts() -> set[str]:
    return {it.text for it in load_items()}


def _prompt_file(name: str) -> list[str]:
    p = paths.ROOT / "prompts" / name
    if not p.exists():
        return []
    return [s.strip() for s in p.read_text().splitlines() if s.strip() and not s.strip().startswith("#")]


def load_prompts(count: int, skip_recorded: bool = True) -> list[str]:
    """Your own phrases first, then everyday dictation interleaved with phonetically balanced
    Harvard sentences (3:2), skipping anything already recorded."""
    seen = recorded_texts() if skip_recorded else set()
    custom = _prompt_file("custom.txt")
    if paths.PERSONAL_CONTEXT.parent.exists():
        user_custom = paths.PERSONAL / "custom_prompts.txt"
        if user_custom.exists():
            custom = [s.strip() for s in user_custom.read_text().splitlines() if s.strip()] + custom
    dictation, harvard = _prompt_file("calibration_en.txt"), _prompt_file("harvard.txt")
    mixed = []
    while dictation or harvard:
        for _ in range(3):
            if dictation:
                mixed.append(dictation.pop(0))
        for _ in range(2):
            if harvard:
                mixed.append(harvard.pop(0))
    out = []
    for line in custom + mixed:
        if normalize(line) not in seen and line not in out:
            out.append(line)
    return out[:count]


# ---------- history of everyday dictation (for review -> training) ----------

def save_history(roi: np.ndarray, raw: str, text: str, nbest: list[str], max_items: int = 500):
    paths.HISTORY.mkdir(parents=True, exist_ok=True)
    hid = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    np.savez_compressed(paths.HISTORY / f"{hid}.npz", roi=roi.astype(np.uint8))
    (paths.HISTORY / f"{hid}.json").write_text(json.dumps(
        {"id": hid, "raw": raw, "text": text, "nbest": nbest, "created": time.time(), "reviewed": False}))
    entries = sorted(paths.HISTORY.glob("*.json"))
    for old in entries[:-max_items]:
        old.unlink(missing_ok=True)
        old.with_suffix(".npz").unlink(missing_ok=True)


def history_entries(unreviewed_only: bool = True) -> list[dict]:
    out = []
    for p in sorted(paths.HISTORY.glob("*.json")):
        d = json.loads(p.read_text())
        if unreviewed_only and d.get("reviewed"):
            continue
        out.append(d)
    return out


def mark_reviewed(hid: str):
    p = paths.HISTORY / f"{hid}.json"
    d = json.loads(p.read_text())
    d["reviewed"] = True
    p.write_text(json.dumps(d))
