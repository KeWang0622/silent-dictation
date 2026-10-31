"""Text normalisation and word error rate."""
from __future__ import annotations

import re

_KEEP = re.compile(r"[^A-Z0-9' ]+")


def normalize(text: str) -> str:
    """LRS3-style transcript: upper case, letters/digits/apostrophes only."""
    text = text.upper().replace("\u2019", "'")
    text = text.replace("-", " ")
    text = _KEEP.sub(" ", text)
    return " ".join(text.split())


def edit_distance(a: list, b: list) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]


def wer(ref: str, hyp: str) -> tuple[int, int]:
    r = normalize(ref).split()
    h = normalize(hyp).split()
    return edit_distance(r, h), max(len(r), 1)


def corpus_wer(pairs) -> float:
    errs = total = 0
    for ref, hyp in pairs:
        e, n = wer(ref, hyp)
        errs += e
        total += n
    return errs / max(total, 1)


def align(ref: str, hyp: str) -> list[dict]:
    """Word alignment for display: ok / sub / del (missed) / ins (extra)."""
    r, h = normalize(ref).lower().split(), normalize(hyp).lower().split()
    n, m = len(r), len(h)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (r[i - 1] != h[j - 1]))
    out, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (r[i - 1] != h[j - 1]):
            out.append({"op": "ok" if r[i - 1] == h[j - 1] else "sub", "ref": r[i - 1], "hyp": h[j - 1]})
            i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            out.append({"op": "del", "ref": r[i - 1], "hyp": ""})
            i -= 1
        else:
            out.append({"op": "ins", "ref": "", "hyp": h[j - 1]})
            j -= 1
    return out[::-1]


def sentence_case(text: str) -> str:
    """Fallback formatting when no language model is available."""
    words = text.lower().split()
    if not words:
        return ""
    words = ["I" if w == "i" else ("I" + w[1:] if w.startswith("i'") else w) for w in words]
    s = " ".join(words)
    s = s[0].upper() + s[1:]
    return s if s[-1] in ".?!" else s + "."
