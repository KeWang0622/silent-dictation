"""Scripted UI runs for development: screenshots and end-to-end checks without touching the real keyboard.

SILENTDICT_AUTOMATION=steps.json python -m silentdict.mac
Each step is a list:  ["wait", 1.5]  ["js", "SD.go({page:'home'})"]  ["cmd", {"cmd": "set", ...}]
["key", "down"] / ["key", "up", 0.8] / ["key", "chord", 0.8]  (simulated dictation key and how long it was held)
["appearance", "dark" | "light"]  ["size", 1060, 720]  ["snapshot", "/path/page.png"] (web content only)
["window_shot", path] / ["overlay_shot", path] (screencapture of that window only)
["until", "c.state == 'ready'", 30] (wait until a Python expression over d=app delegate, c=controller is true)  ["log", "text"]  ["quit"]
"""
from __future__ import annotations

import json
import os
import subprocess
import time

from AppKit import NSAppearance, NSApplication, NSBitmapImageRep, NSPNGFileType
from PyObjCTools import AppHelper


class Automation:
    def __init__(self, delegate, path: str):
        self.delegate = delegate
        self.steps = json.load(open(path))
        self.i = 0

    def start(self, delay: float = 1.0):
        self.delegate.web.evaluateJavaScript_completionHandler_("document.body.classList.add('snap')", None)
        AppHelper.callLater(delay, self.next)

    def next(self, delay: float = 0.25):
        AppHelper.callLater(delay, self._safe_run)

    def _safe_run(self):
        try:
            self._run()
        except Exception:
            import traceback
            traceback.print_exc()
            self.next()

    def _run(self):
        if self.i >= len(self.steps):
            return
        step = self.steps[self.i]
        self.i += 1
        kind, args = step[0], step[1:]
        d, c = self.delegate, self.delegate.controller
        print("automation:", json.dumps(step)[:160], flush=True)
        if kind == "wait":
            return self.next(float(args[0]))
        if kind == "until":  # ["until", "python expression", timeout]: poll until true
            return self._until(args[0], time.monotonic() + float(args[1] if len(args) > 1 else 30))
        if kind == "js":
            d.web.evaluateJavaScript_completionHandler_(args[0], None)
        elif kind == "cmd":
            d.handle(args[0])
        elif kind == "key":
            if args[0] == "down":
                c._down()
            else:  # "up" or "chord" (released after another key was pressed with it)
                c.ptt.last_hold = float(args[1]) if len(args) > 1 else 1.0
                c.ptt.last_chorded = args[0] == "chord"
                c._up(False)
            return self.next(0.0)  # keep key timing exact; only "wait" steps add time
        elif kind == "appearance":
            name = "NSAppearanceNameDarkAqua" if args[0] == "dark" else "NSAppearanceNameAqua"
            NSApplication.sharedApplication().setAppearance_(NSAppearance.appearanceNamed_(name))
        elif kind == "size":
            d.window.setContentSize_((float(args[0]), float(args[1])))
        elif kind == "snapshot":
            return self._snapshot(args[0])
        elif kind in ("window_shot", "overlay_shot"):
            if kind == "window_shot":  # capture with an active title bar
                NSApplication.sharedApplication().activateIgnoringOtherApps_(True)
                d.window.makeKeyAndOrderFront_(None)
            win = d.window if kind == "window_shot" else d.overlay.panel
            AppHelper.callLater(0.3 if kind == "window_shot" else 0.0, self._capture, win, args[0])
            return
        elif kind == "eval":
            def show(result, error):
                print("automation: eval", repr(result)[:400], error or "", flush=True)
                self.next()
            d.web.evaluateJavaScript_completionHandler_(args[0], show)
            return
        elif kind == "py":
            print("automation: py", repr(eval(args[0], {"d": d, "c": c}))[:400], flush=True)
        elif kind == "log":
            print("automation:", args[0], flush=True)
        elif kind == "quit":
            NSApplication.sharedApplication().terminate_(None)
            return
        self.next()

    def _until(self, expr: str, deadline: float):
        d = self.delegate
        if eval(expr, {"d": d, "c": d.controller}) or time.monotonic() > deadline:
            return self.next(0.0)
        AppHelper.callLater(0.05, self._until, expr, deadline)

    def _capture(self, win, path: str):
        subprocess.run(["screencapture", "-x", "-l", str(win.windowNumber()), path], check=False)
        print("automation: saved", path, flush=True)
        self.next()

    def _snapshot(self, path: str):
        def done(image, error):
            if image is not None:
                rep = NSBitmapImageRep.imageRepWithData_(image.TIFFRepresentation())
                os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
                rep.representationUsingType_properties_(NSPNGFileType, {}).writeToFile_atomically_(path, True)
                print("automation: saved", path, flush=True)
            else:
                print("automation: snapshot failed", error, flush=True)
            self.next()

        self.delegate.web.takeSnapshotWithConfiguration_completionHandler_(None, done)
