from silentdict.gestures import CANCEL, HANDS_FREE, START, STOP, Gestures


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def make():
    clock = Clock()
    return Gestures(tap=0.3, double_tap=0.55, clock=clock), clock


def test_hold_records_until_release():
    g, _ = make()
    assert g.down() == START
    assert g.up(held=1.4, chorded=False) == STOP


def test_single_tap_is_ignored():
    g, _ = make()
    assert g.down() == START
    assert g.up(held=0.1, chorded=False) == CANCEL


def test_double_tap_enters_hands_free_and_next_press_stops():
    g, clock = make()
    g.down()
    assert g.up(held=0.1, chorded=False) == CANCEL
    clock.t += 0.3
    assert g.down() == START
    assert g.up(held=0.1, chorded=False) == HANDS_FREE
    clock.t += 5
    assert g.down() == STOP
    assert g.up(held=0.1, chorded=False) is None  # the stopping press's release is swallowed
    assert g.down() == START  # and the next press starts a new dictation


def test_slow_taps_are_not_a_double_tap():
    g, clock = make()
    g.down()
    g.up(held=0.1, chorded=False)
    clock.t += 1.0
    g.down()
    assert g.up(held=0.1, chorded=False) == CANCEL


def test_key_combination_cancels():
    g, _ = make()
    g.down()
    assert g.up(held=2.0, chorded=True) == CANCEL


def test_key_repeat_while_held_is_ignored():
    g, _ = make()
    assert g.down() == START
    assert g.down() is None
    assert g.up(held=1.0, chorded=False) == STOP
