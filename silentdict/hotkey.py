"""Global push-to-talk key for macOS using a listen-only Quartz event tap.

Only modifier-key state changes are inspected, so the tap never needs keyboard-layout APIs
(which crash when called off the main thread on recent macOS versions).
Requires Input Monitoring permission for the app running this process (e.g. Terminal).
"""
from __future__ import annotations

import threading
import time
from typing import Callable

import Quartz

# key name -> (virtual keycode, device-specific flag bit)
KEYS = {
    "right_option": (61, 0x00000040),
    "left_option": (58, 0x00000020),
    "right_command": (54, 0x00000010),
    "right_control": (62, 0x00002000),
    "right_shift": (60, 0x00000004),
    "fn": (63, Quartz.kCGEventFlagMaskSecondaryFn),
}


def input_monitoring_allowed(prompt: bool = False) -> bool:
    try:
        if Quartz.CGPreflightListenEventAccess():
            return True
        if prompt:
            Quartz.CGRequestListenEventAccess()
        return False
    except AttributeError:  # very old pyobjc: assume allowed and let the tap decide
        return True


class PushToTalk(threading.Thread):
    """Calls on_press when the key goes down and on_release(cancelled) when it comes up.

    If another key is typed while holding the push-to-talk key (e.g. Option+E for an accent),
    the press is reported as cancelled so normal keyboard shortcuts keep working.
    """

    def __init__(self, key: str, on_press: Callable[[], None], on_release: Callable[[bool], None],
                 min_hold: float = 0.25):
        super().__init__(daemon=True)
        if key not in KEYS:
            raise ValueError(f"unknown key {key!r}; choose one of {', '.join(KEYS)}")
        self.keycode, self.flag = KEYS[key]
        self.on_press, self.on_release = on_press, on_release
        self.min_hold = min_hold
        self.down = False
        self.chorded = False
        self.pressed_at = 0.0
        self.last_hold = 0.0  # seconds the key was held on the latest release
        self.last_chorded = False  # another key was typed while it was held
        self.error: str | None = None
        self.started = threading.Event()
        self._tap = None
        self._loop = None

    def _callback(self, proxy, etype, event, refcon):
        if etype in (Quartz.kCGEventTapDisabledByTimeout, Quartz.kCGEventTapDisabledByUserInput):
            Quartz.CGEventTapEnable(self._tap, True)
            return event
        if etype == Quartz.kCGEventKeyDown and self.down:
            self.chorded = True
        elif etype == Quartz.kCGEventFlagsChanged:
            code = Quartz.CGEventGetIntegerValueField(event, Quartz.kCGKeyboardEventKeycode)
            if code == self.keycode:
                is_down = bool(Quartz.CGEventGetFlags(event) & self.flag)
                if is_down and not self.down:
                    self.down, self.chorded, self.pressed_at = True, False, time.monotonic()
                    self._safe(self.on_press)
                elif not is_down and self.down:
                    self.down = False
                    self.last_hold = time.monotonic() - self.pressed_at
                    self.last_chorded = self.chorded
                    short = self.last_hold < self.min_hold
                    self._safe(self.on_release, self.chorded or short)
        return event

    @staticmethod
    def _safe(fn, *args):
        try:
            fn(*args)
        except Exception as exc:  # never let an exception kill the event tap
            print("hotkey callback error:", exc)

    def run(self):
        mask = Quartz.CGEventMaskBit(Quartz.kCGEventFlagsChanged) | Quartz.CGEventMaskBit(Quartz.kCGEventKeyDown)
        self._tap = Quartz.CGEventTapCreate(
            Quartz.kCGSessionEventTap, Quartz.kCGHeadInsertEventTap,
            Quartz.kCGEventTapOptionListenOnly, mask, self._callback, None)
        if self._tap is None:
            self.error = ("Could not listen for the push-to-talk key. Allow Input Monitoring for this "
                          "terminal in System Settings > Privacy & Security > Input Monitoring, then restart it.")
            self.started.set()
            return
        source = Quartz.CFMachPortCreateRunLoopSource(None, self._tap, 0)
        self._loop = Quartz.CFRunLoopGetCurrent()
        Quartz.CFRunLoopAddSource(self._loop, source, Quartz.kCFRunLoopCommonModes)
        Quartz.CGEventTapEnable(self._tap, True)
        self.started.set()
        Quartz.CFRunLoopRun()

    def stop(self):
        if self._loop is not None:
            Quartz.CFRunLoopStop(self._loop)
