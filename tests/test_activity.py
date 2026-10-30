from dataclasses import dataclass

import numpy as np

from silentdict.activity import active_span, bar_levels, motion_score


@dataclass
class Obs:
    mouth_open: float


def series(values):
    return [Obs(float(v)) for v in values]


def test_still_lips_have_no_motion():
    assert motion_score(series([0.05] * 60)) == 0.0


def test_mouthing_has_motion():
    t = np.arange(80)
    assert motion_score(series(0.15 + 0.12 * np.sin(t / 2.0))) > 0.05


def test_missing_faces_are_interpolated():
    obs = series([0.1, 0.3] * 20)
    obs[5] = None
    assert motion_score(obs) > 0.05
    assert motion_score([None] * 30) == 0.0


def test_active_span_trims_silence_with_margin():
    t = np.arange(60)
    values = np.concatenate([np.full(40, 0.05), 0.15 + 0.12 * np.sin(t / 2.0), np.full(40, 0.05)])
    start, end = active_span(series(values))
    assert 20 <= start <= 40
    assert 100 <= end <= 120


def test_active_span_keeps_short_or_still_clips():
    assert active_span(series([0.05] * 20)) == (0, 20)
    assert active_span(series([0.05] * 80)) == (0, 80)


def test_bar_levels():
    levels = bar_levels(series([0.0, 0.45]), bars=7)
    assert len(levels) == 7
    assert levels[-1] == 1.0
    assert all(0.0 <= v <= 1.0 for v in levels)
