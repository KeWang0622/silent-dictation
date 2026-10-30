"""Lip activity from face landmarks: live meter, calibration quality gate and silence trimming."""
from __future__ import annotations

import numpy as np

MARGIN = 10  # frames kept before the first / after the last lip movement (0.4 s at 25 fps)


def mouth_series(obs_list) -> np.ndarray | None:
    """Mouth opening per frame (inner-lip gap / inter-ocular distance), gaps interpolated."""
    vals = np.array([o.mouth_open if o is not None else np.nan for o in obs_list], dtype=np.float64)
    ok = ~np.isnan(vals)
    if ok.sum() < 2:
        return None
    idx = np.arange(len(vals))
    vals[~ok] = np.interp(idx[~ok], idx[ok], vals[ok])
    return vals


def movement(series: np.ndarray) -> np.ndarray:
    d = np.abs(np.diff(series, prepend=series[0]))
    return np.convolve(d, np.ones(5) / 5, mode="same")


def motion_score(obs_list) -> float:
    """Spread of mouth opening; near zero when the lips did not move."""
    s = mouth_series(obs_list)
    if s is None:
        return 0.0
    return float(np.percentile(s, 95) - np.percentile(s, 5))


def active_span(obs_list) -> tuple[int, int]:
    """[start, end) of the part of the clip with lip movement, with margins. Whole clip if unclear."""
    n = len(obs_list)
    s = mouth_series(obs_list)
    if s is None or n < 2 * MARGIN + 25:
        return 0, n
    m = movement(s)
    thr = max(0.004, 0.2 * float(np.percentile(m, 90)))
    active = np.flatnonzero(m > thr)
    if len(active) == 0:
        return 0, n
    start = max(0, int(active[0]) - MARGIN)
    end = min(n, int(active[-1]) + MARGIN + 1)
    if end - start < 25:
        return 0, n
    return start, end


def bar_levels(obs_list, bars: int = 7) -> list[float]:
    """Heights (0..1) for activity bars, newest last."""
    recent = list(obs_list)[-bars:]
    out = [0.08 if o is None else min(1.0, 0.08 + o.mouth_open / 0.45) for o in recent]
    return [0.08] * (bars - len(out)) + [round(v, 3) for v in out]
