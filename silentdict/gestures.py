"""Push-to-talk gestures in the style of Willow Voice, independent of any UI or OS API.

* hold the key            -> record while held, transcribe on release
* double-tap the key      -> hands-free: keep recording until the key is pressed again
* key + another key       -> a normal shortcut (e.g. fn + arrow): cancel
* a single short tap      -> ignored
"""
from __future__ import annotations

import time
from typing import Callable

START, STOP, CANCEL, HANDS_FREE = "start", "stop", "cancel", "hands_free"


class Gestures:
    def __init__(self, tap: float = 0.3, double_tap: float = 0.55, clock: Callable[[], float] = time.monotonic):
        self.tap = tap                # presses shorter than this are taps
        self.double_tap = double_tap  # max time between the two tap releases
        self.clock = clock
        self.mode = "idle"            # idle | holding | hands_free
        self._last_tap = float("-inf")
        self._swallow_up = False

    def reset(self):
        self.mode = "idle"
        self._swallow_up = False

    def down(self) -> str | None:
        if self.mode == HANDS_FREE:
            self.mode = "idle"
            self._swallow_up = True  # the release of the stopping press is not a new gesture
            return STOP
        if self.mode == "holding":
            return None
        self.mode = "holding"
        return START

    def up(self, held: float, chorded: bool) -> str | None:
        if self._swallow_up:
            self._swallow_up = False
            return None
        if self.mode != "holding":
            return None
        if chorded:
            self.mode = "idle"
            return CANCEL
        if held < self.tap:
            now = self.clock()
            if now - self._last_tap <= self.double_tap:
                self._last_tap = float("-inf")
                self.mode = HANDS_FREE
                return HANDS_FREE
            self._last_tap = now
            self.mode = "idle"
            return CANCEL
        self.mode = "idle"
        return STOP
