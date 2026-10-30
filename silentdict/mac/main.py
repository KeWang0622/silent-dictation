"""Silent Dictation for macOS: Dock app with a main window, a menu-bar item and a floating indicator.

Run via the app bundle (bin/build-app), or from a checkout:  PYTHONPATH=. .venv/bin/python -m silentdict.mac
"""
from __future__ import annotations

import base64
import fcntl
import json
import os
import plistlib
import subprocess
import sys
import time
from pathlib import Path

import objc
import Quartz
from AppKit import (NSAlert, NSApplication, NSApplicationActivationPolicyRegular, NSBackingStoreBuffered, NSImage,
                    NSMenu, NSMenuItem, NSStatusBar, NSVariableStatusItemLength, NSView, NSVisualEffectView,
                    NSWindow, NSWindowStyleMaskClosable, NSWindowStyleMaskFullSizeContentView,
                    NSWindowStyleMaskMiniaturizable, NSWindowStyleMaskResizable, NSWindowStyleMaskTitled,
                    NSWorkspace)
from Foundation import NSBundle, NSMakeRect, NSObject, NSRunLoop, NSRunLoopCommonModes, NSTimer, NSURL
from PyObjCTools import AppHelper
from WebKit import WKWebView, WKWebViewConfiguration

from .. import CREDITS, __version__, devices, paths, stats
from .. import settings as settings_mod
from ..dataset import add_item, load_items, mark_reviewed
from ..hotkey import input_monitoring_allowed
from ..output import accessibility_allowed, copy_text
from ..permissions import SETTINGS_URLS, camera_status, request_camera_async
from .overlay import Overlay

WEB = Path(__file__).with_name("web")
REPO_URL = "https://github.com/KeWang0622/silent-dictation"
LOGIN_AGENT = Path.home() / "Library" / "LaunchAgents" / "io.github.kewang0622.silentdictation.plist"
KEY_LABELS = {"fn": "fn", "right_option": "Right ⌥", "right_command": "Right ⌘", "right_control": "Right ⌃"}


def data_url(img, quality: int = 72) -> str:
    import cv2
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode() if ok else ""


def nsimage(img):
    import cv2
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    from Foundation import NSData
    return NSImage.alloc().initWithData_(NSData.dataWithBytes_length_(buf.tobytes(), len(buf))) if ok else None


def mouth_crop(frame, size=96):
    import cv2
    if frame is None or frame.obs is None:
        return None
    cx, cy = frame.obs.points[3]
    r = frame.obs.iod * 0.8
    h, w = frame.gray.shape
    x0, x1 = int(max(0, cx - r)), int(min(w, cx + r))
    y0, y1 = int(max(0, cy - r * 0.8)), int(min(h, cy + r * 0.8))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    return cv2.flip(cv2.resize(frame.bgr[y0:y1, x0:x1], (size, int(size * 0.8))), 1)


def to_json(value) -> str:
    return json.dumps(value, default=lambda o: o.item() if hasattr(o, "item") else str(o))


_reported: set[str] = set()


def report(where: str):
    """Print a traceback once per distinct error; timer callbacks must never raise into AppKit."""
    import traceback
    text = traceback.format_exc()
    key = where + text.strip().splitlines()[-1]
    if key not in _reported:
        _reported.add(key)
        print(f"error in {where}:\n{text}", flush=True)


def bundled() -> bool:
    return NSBundle.mainBundle().bundlePath().endswith(".app")


class Bridge(NSObject):
    def userContentController_didReceiveScriptMessage_(self, _ucc, message):
        try:
            self.owner.handle(json.loads(str(message.body())))
        except Exception:  # never let a UI message crash the app
            report("bridge")


class DragView(NSView):
    """Transparent strip over the web view's top edge so the window can be dragged by its title area."""

    def mouseDownCanMoveWindow(self):
        return True

    def mouseDown_(self, event):
        self.window().performWindowDragWithEvent_(event)


class AppDelegate(NSObject):
    def init(self):
        self = objc.super(AppDelegate, self).init()
        if self is None:
            return None
        self.settings = settings_mod.load()
        self.app_active = False
        self.preview = False
        self.flash = None
        self.flash_until = 0.0
        self.cache = {}
        self.cache_time = {}
        self.lock_file = None
        self.page_ready = False
        self.page = None
        return self

    # ------------------------------------------------------------------ lifecycle
    def applicationDidFinishLaunching_(self, _note):
        app = NSApplication.sharedApplication()
        app.setActivationPolicy_(NSApplicationActivationPolicyRegular)
        if not bundled() and (WEB / "icon.png").exists():
            app.setApplicationIconImage_(NSImage.alloc().initWithContentsOfFile_(str(WEB / "icon.png")))
        self.build_main_menu()
        self.build_status_item()
        self.overlay = Overlay()
        from ..controller import Controller  # OpenCV, MediaPipe and PyTorch load later, off the main thread
        self.controller = Controller(self.settings, self.notify)
        self.controller.should_paste = lambda: not self.app_active
        self.controller.calibration_visible = lambda: self.app_active and self.page == "calibrate"
        self.controller.start()
        self.build_window()
        for interval, sel in ((1 / 15, "tick:"), (0.25, "push:")):
            t = NSTimer.timerWithTimeInterval_target_selector_userInfo_repeats_(interval, self, sel, None, True)
            NSRunLoop.currentRunLoop().addTimer_forMode_(t, NSRunLoopCommonModes)
        print(CREDITS, flush=True)
        print(f"permissions: camera={camera_status()} input={input_monitoring_allowed()} "
              f"accessibility={accessibility_allowed()} bundle={NSBundle.mainBundle().bundlePath()}", flush=True)
        if "--background" not in sys.argv:
            self.show_window()

    def applicationShouldHandleReopen_hasVisibleWindows_(self, _app, _visible):
        self.show_window()
        return True

    def applicationShouldTerminateAfterLastWindowClosed_(self, _app):
        return False

    def applicationDidBecomeActive_(self, _note):
        self.app_active = True

    def applicationDidResignActive_(self, _note):
        self.app_active = bool(os.environ.get("SILENTDICT_AUTOMATION"))  # scripted runs act as if in front

    def applicationShouldTerminate_(self, _app):
        if not self.controller.training["running"] or os.environ.get("SILENTDICT_AUTOMATION"):
            return 1  # NSTerminateNow
        alert = NSAlert.alloc().init()
        alert.setMessageText_("Quit while training?")
        alert.setInformativeText_("Your personal model is still training. If you quit now it won't be saved, "
                                  "and you can train again later from Calibrate.")
        alert.addButtonWithTitle_("Keep Training")
        alert.addButtonWithTitle_("Quit")
        return 1 if alert.runModal() == 1001 else 0  # second button -> NSTerminateNow, else NSTerminateCancel

    def applicationWillTerminate_(self, _note):
        self.controller.shutdown()
        # Worker threads may be in the middle of PyTorch work; skip interpreter teardown, which can abort.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)

    def windowWillClose_(self, _note):
        self.set_preview(False)

    # ------------------------------------------------------------------ menus
    @objc.python_method
    def build_main_menu(self):
        app = NSApplication.sharedApplication()
        main = NSMenu.alloc().init()

        def submenu(title, items):
            holder = NSMenuItem.alloc().init()
            menu = NSMenu.alloc().initWithTitle_(title)
            for spec in items:
                if spec is None:
                    menu.addItem_(NSMenuItem.separatorItem())
                    continue
                label, action, key, mine = spec
                it = menu.addItemWithTitle_action_keyEquivalent_(label, action, key)
                if mine:
                    it.setTarget_(self)
            holder.setSubmenu_(menu)
            main.addItem_(holder)

        submenu("Silent Dictation", [("About Silent Dictation", "showAbout:", "", True), None,
                                     ("Settings…", "showSettings:", ",", True), None,
                                     ("Hide Silent Dictation", "hide:", "h", False),
                                     ("Show All", "unhideAllApplications:", "", False), None,
                                     ("Quit Silent Dictation", "terminate:", "q", False)])
        submenu("Edit", [("Undo", "undo:", "z", False), ("Redo", "redo:", "Z", False), None,
                         ("Cut", "cut:", "x", False), ("Copy", "copy:", "c", False),
                         ("Paste", "paste:", "v", False), ("Select All", "selectAll:", "a", False)])
        submenu("Window", [("Minimize", "performMiniaturize:", "m", False), ("Close", "performClose:", "w", False),
                           None, ("Silent Dictation", "showMain:", "0", True)])
        submenu("Help", [("Silent Dictation on GitHub", "openGitHub:", "", True)])
        app.setMainMenu_(main)

    @objc.python_method
    def build_status_item(self):
        self.status_item = NSStatusBar.systemStatusBar().statusItemWithLength_(NSVariableStatusItemLength)
        self.status_menu = NSMenu.alloc().init()
        self.status_menu.setDelegate_(self)
        self.status_item.setMenu_(self.status_menu)
        self.icon_name = None
        self.set_icon("mouth")

    @objc.python_method
    def set_icon(self, name):
        if name != self.icon_name:
            img = NSImage.imageWithSystemSymbolName_accessibilityDescription_(name, "Silent Dictation")
            img.setTemplate_(True)
            self.status_item.button().setImage_(img)
            self.icon_name = name

    def menuNeedsUpdate_(self, menu):
        menu.removeAllItems()
        c = self.controller
        status = {"loading": "Loading models…", "listening": "Reading lips…", "decoding": "Transcribing…",
                  "error": c.message}.get(c.state, f"Ready — hold {KEY_LABELS.get(self.settings['key'], 'fn')} to dictate")
        menu.addItemWithTitle_action_keyEquivalent_(status, None, "")
        menu.addItem_(NSMenuItem.separatorItem())
        for title, sel in (("Open Silent Dictation", "showMain:"), ("Calibrate…", "showCalibrate:"),
                           ("History…", "showHistory:"), ("Settings…", "showSettings:")):
            menu.addItemWithTitle_action_keyEquivalent_(title, sel, "").setTarget_(self)
        menu.addItem_(NSMenuItem.separatorItem())
        menu.addItemWithTitle_action_keyEquivalent_("Quit Silent Dictation", "terminate:", "q")

    def showAbout_(self, _sender):
        self.show_window("settings", anchor="about")

    def showSettings_(self, _sender):
        self.show_window("settings")

    def showCalibrate_(self, _sender):
        self.show_window("calibrate")

    def showHistory_(self, _sender):
        self.show_window("history")

    def showMain_(self, _sender):
        self.show_window()

    def openGitHub_(self, _sender):
        NSWorkspace.sharedWorkspace().openURL_(NSURL.URLWithString_(REPO_URL))

    # ------------------------------------------------------------------ main window
    @objc.python_method
    def build_window(self):
        style = (NSWindowStyleMaskTitled | NSWindowStyleMaskClosable | NSWindowStyleMaskMiniaturizable
                 | NSWindowStyleMaskResizable | NSWindowStyleMaskFullSizeContentView)
        win = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(0, 0, 1060, 720), style, NSBackingStoreBuffered, False)
        win.setTitle_("Silent Dictation")
        win.setTitlebarAppearsTransparent_(True)
        win.setTitleVisibility_(1)
        win.setMinSize_((920, 640))
        win.setReleasedWhenClosed_(False)
        win.setDelegate_(self)
        win.center()
        win.setFrameAutosaveName_("SilentDictationMain")
        fx = NSVisualEffectView.alloc().initWithFrame_(win.contentView().bounds())
        fx.setMaterial_(7)          # sidebar
        fx.setBlendingMode_(0)      # behind window
        fx.setState_(0)             # follow window active state
        fx.setAutoresizingMask_(18)
        config = WKWebViewConfiguration.alloc().init()
        self.bridge = Bridge.alloc().init()
        self.bridge.owner = self
        config.userContentController().addScriptMessageHandler_name_(self.bridge, "sd")
        if os.environ.get("SILENTDICT_DEVTOOLS"):
            config.preferences().setValue_forKey_(True, "developerExtrasEnabled")
        web = WKWebView.alloc().initWithFrame_configuration_(fx.bounds(), config)
        web.setValue_forKey_(False, "drawsBackground")
        web.setAutoresizingMask_(18)
        fx.addSubview_(web)
        drag = DragView.alloc().initWithFrame_(NSMakeRect(78, fx.bounds().size.height - 30, fx.bounds().size.width - 78, 30))
        drag.setAutoresizingMask_(2 | 8)  # flexible width, stick to top
        fx.addSubview_(drag)
        win.setContentView_(fx)
        index = WEB / "index.html"
        web.loadFileURL_allowingReadAccessToURL_(NSURL.fileURLWithPath_(str(index)), NSURL.fileURLWithPath_(str(WEB)))
        self.window, self.web = win, web

    @objc.python_method
    def show_window(self, page=None, anchor=None):
        self.window.makeKeyAndOrderFront_(None)
        NSApplication.sharedApplication().activateIgnoringOtherApps_(True)
        if page:
            self.js("go", {"page": page, "anchor": anchor})

    @objc.python_method
    def js(self, fn, payload=None):
        if not self.page_ready:
            return
        script = f"window.SD && SD.{fn}({to_json(payload)})"
        self.web.evaluateJavaScript_completionHandler_(script, None)

    # ------------------------------------------------------------------ events from the controller (any thread)
    @objc.python_method
    def notify(self, kind, data):
        AppHelper.callAfter(self.on_event, kind, data)

    @objc.python_method
    def on_event(self, kind, data):
        try:
            self.handle_event(kind, data)
        except Exception:
            report("event " + kind)

    @objc.python_method
    def handle_event(self, kind, data):
        c = self.controller
        if kind == "key":
            self.js("key", bool(data))
        elif kind == "toast":
            self.flash = ("message", str(data), "", "info")
            self.flash_until = time.monotonic() + 2.4
        elif kind == "need_camera":
            self.flash = ("message", "Camera access needed", "Open Silent Dictation to allow it", "warn")
            self.flash_until = time.monotonic() + 3
            self.show_window("setup")
        elif kind == "result":
            delivered = data.timings.get("delivered")
            sub = {"pasted": "", "copied": "Copied — allow Accessibility to paste", "app": ""}.get(delivered, "")
            self.flash = ("message", data.text, sub, "ok")
            self.flash_until = time.monotonic() + 2.6
            self.invalidate("recent", "stats")
            self.js("result", {"text": data.text, "raw": data.raw.lower(), "delivered": delivered,
                               "seconds": round(data.timings.get("total", 0), 2),
                               "nbest": [h.text.lower() for h in data.hyps]})
        elif kind in ("calibration", "training"):
            self.invalidate("clips", "history")
            self.push_(None)
            if kind == "training" and data == "done":
                self.training_finished()
        icon = {"listening": "mouth.fill", "decoding": "ellipsis.circle", "loading": "hourglass",
                "error": "exclamationmark.triangle"}.get(c.state, "mouth")
        if c.training["running"]:
            icon = "hourglass"
        self.set_icon(icon)

    @objc.python_method
    def training_finished(self):
        t = self.controller.training
        if t.get("error"):
            self.flash = ("message", "Training didn't finish", t["error"][:80], "warn")
        else:
            r = t.get("report") or {}
            before = max(0.0, 1 - r.get("base_wer", 1))
            after = max(0.0, 1 - r.get("active_wer", r.get("base_wer", 1)))
            title = "Personal model ready" if r.get("active") == "new" else "Training finished"
            self.flash = ("message", title, f"Accuracy {before:.0%} → {after:.0%} on held-out sentences", "ok")
        self.flash_until = time.monotonic() + 5

    # ------------------------------------------------------------------ periodic work (main thread)
    def tick_(self, _timer):
        try:
            self.update_overlay()
            if self.preview and self.window.isVisible():
                self.push_frame()
        except Exception:
            report("tick")

    @objc.python_method
    def update_overlay(self):
        c = self.controller
        if not self.settings.get("overlay", True) or (c.recording_for == "calibration" and self.app_active):
            self.overlay.hide()
            return
        if c.state == "listening":
            frame = c.latest_frame()
            if frame is None:
                self.overlay.busy("Starting camera…")
                return
            problem = c.camera_problem()
            crop = mouth_crop(frame, 68)
            warning = problem if problem and ("face" in problem.lower() or "black" in problem) else None
            self.overlay.listening(nsimage(crop) if crop is not None else None, c.levels(7), c.hands_free, warning,
                                   KEY_LABELS.get(self.settings.get("key", "fn"), "fn"))
        elif c.state == "decoding":
            self.overlay.busy("Transcribing…")
        elif self.flash and time.monotonic() < self.flash_until:
            _, title, sub, kind = self.flash
            self.overlay.message(title, sub, kind)
        else:
            self.overlay.hide()

    @objc.python_method
    def push_frame(self):
        c = self.controller
        frame = c.latest_frame()
        if frame is None:
            self.js("frame", {"img": None, "problem": c.camera_problem(), "on": c.camera is not None})
            return
        import cv2
        img = cv2.flip(frame.bgr, 1)
        h, w = img.shape[:2]
        box = None
        obs = c.camera.last_obs if c.camera else None
        if obs is not None:
            mx, my = obs.points[3]
            r = obs.iod * 0.75
            box = [float(v) for v in (1 - (mx + r) / w, (my - r * 0.7) / h, 2 * r / w, 1.4 * r / h)]
        crop = mouth_crop(frame, 120)
        self.js("frame", {"img": data_url(img, 70), "box": box, "levels": c.levels(16),
                          "mouth": data_url(crop, 80) if crop is not None else None,
                          "recording": bool(c.camera and c.camera.recording),
                          "problem": c.camera_problem(), "on": True})

    @objc.python_method
    def cached(self, key, ttl, fn):
        now = time.monotonic()
        if key not in self.cache or now - self.cache_time.get(key, 0) > ttl:
            self.cache[key] = fn()
            self.cache_time[key] = now
        return self.cache[key]

    @objc.python_method
    def invalidate(self, *keys):
        for k in keys:
            self.cache_time.pop(k, None)

    def push_(self, _timer):
        try:
            self.push_state()
        except Exception:
            report("push")

    @objc.python_method
    def push_state(self):
        if not self.page_ready or not self.window.isVisible():
            return
        c, s = self.controller, self.settings
        training = {k: v for k, v in c.training.items() if k != "report"}
        report = c.training.get("report")
        if report:
            training["result"] = {"base": max(0.0, 1 - report["base_wer"]),
                                  "personal": max(0.0, 1 - report.get("active_wer", report["personal_wer"])),
                                  "active": report["active"], "strategy": report.get("strategy"),
                                  "clips": report["train_clips"] + report["val_clips"]}
        reader = c.recognizer.reader if c.recognizer else None
        perms = self.cached("perms", 1.0, lambda: {
            "camera": {0: "ask", 1: "denied", 2: "denied", 3: "granted"}.get(camera_status(), "ask"),
            "input": input_monitoring_allowed(), "accessibility": accessibility_allowed()})
        st = stats.load()
        hist = self.cached("history", 5.0, c.accuracy_history)
        state = {
            "version": __version__,
            "state": c.state, "message": c.message, "handsFree": c.hands_free, "keyDown": c.key_down,
            "hotkeyError": bool(c.hotkey_error),
            "personal": bool(reader and reader.personalized),
            "llmActive": bool(c.recognizer and c.recognizer.rewriter), "llmAvailable": c.llm_available,
            "cameraOn": c.camera is not None, "cameraProblem": c.camera_problem(),
            "permissions": perms,
            "settings": {**s, "login": LOGIN_AGENT.exists(), "keyLabel": KEY_LABELS.get(s.get("key", "fn"), "fn")},
            "cameras": self.cached("cameras", 5.0, devices.cameras),
            "globeOk": self.cached("globe", 10.0, devices.globe_key_setting) == 0,
            "conflicts": self.cached("conflicts", 5.0, devices.running_dictation_apps),
            "stats": {"dictations": st["dictations"], "words": st["words"],
                      "accuracy": hist[-1]["personal"] if hist else None},
            "accuracyHistory": hist,
            "clips": self.cached("clips", 3.0, lambda: len(load_items())),
            "recent": self.cached("recent", 3.0, lambda: c.recent(40)),
            "training": training,
            "calibration": c.session.snapshot() if c.session else {"active": False},
        }
        self.js("state", state)

    # ------------------------------------------------------------------ commands from the web UI
    @objc.python_method
    def set_preview(self, on: bool):
        self.preview = on
        self.controller.set_preview(on and camera_status() == 3)

    @objc.python_method
    def handle(self, msg: dict):
        cmd = msg.get("cmd")
        c, s = self.controller, self.settings
        if cmd == "ready":
            first = not self.page_ready
            self.page_ready = True
            page = "home" if s.get("onboarded") else "setup"
            self.js("init", {"page": page})
            self.push_(None)
            if first and os.environ.get("SILENTDICT_AUTOMATION"):
                from .automation import Automation
                self.automation = Automation(self, os.environ["SILENTDICT_AUTOMATION"])
                self.automation.start()
        elif cmd == "preview":
            self.set_preview(bool(msg.get("on")))
        elif cmd == "page":
            self.page = msg.get("page")
            return
        elif cmd == "permission":
            kind = msg.get("kind")
            if kind == "camera":
                if camera_status() == 0:
                    request_camera_async(lambda ok: AppHelper.callAfter(self.after_camera, ok))
                else:
                    self.open_url(SETTINGS_URLS["camera"])
            elif kind == "input":
                if not Quartz.CGRequestListenEventAccess():
                    self.open_url(SETTINGS_URLS["input"])
            elif kind == "accessibility":
                if not Quartz.CGRequestPostEventAccess():
                    self.open_url(SETTINGS_URLS["accessibility"])
            self.invalidate("perms")
        elif cmd == "openSettings":
            self.open_url(SETTINGS_URLS[msg["kind"]])
        elif cmd == "set":
            self.set_setting(msg["key"], msg["value"])
        elif cmd == "calibrate.start":
            c.start_calibration(int(msg.get("count", 60)))
        elif cmd == "calibrate.redo" and c.session:
            c.session.redo()
        elif cmd == "calibrate.skip" and c.session:
            c.session.skip()
        elif cmd == "calibrate.stop":
            c.stop_calibration()
        elif cmd == "train":
            c.stop_calibration()
            c.train_async()
        elif cmd == "review.save":
            self.review_save(msg["id"], msg.get("text", ""))
        elif cmd == "review.discard":
            mark_reviewed(msg["id"])
            self.invalidate("recent")
        elif cmd == "thumb":
            self.send_thumb(msg["id"])
        elif cmd == "copy":
            copy_text(msg.get("text", ""))
        elif cmd == "openData":
            paths.DATA.mkdir(parents=True, exist_ok=True)
            NSWorkspace.sharedWorkspace().openURL_(NSURL.fileURLWithPath_(str(paths.DATA)))
        elif cmd == "openURL":
            self.open_url(msg["url"])
        elif cmd == "restart":
            self.restart()
        elif cmd == "quit":
            NSApplication.sharedApplication().terminate_(None)
        elif cmd == "log":
            print("web:", msg.get("msg"), flush=True)
            return
        self.push_(None)

    @objc.python_method
    def after_camera(self, granted):
        self.invalidate("perms")
        if granted and self.preview:
            self.controller.set_preview(True)
        self.push_(None)

    @objc.python_method
    def open_url(self, url):
        NSWorkspace.sharedWorkspace().openURL_(NSURL.URLWithString_(url))

    @objc.python_method
    def set_setting(self, key, value):
        c, s = self.controller, self.settings
        if key == "login":
            self.set_login(bool(value))
            return
        s[key] = value
        settings_mod.save(s)
        if key == "key":
            c.set_key(value)
        elif key == "camera_name":
            c.restart_camera()
        elif key == "camera_policy" and value == "always" and camera_status() == 3:
            c.ensure_camera()
        elif key in ("mode", "llm"):
            c.reload_models()
        elif key == "onboarded":
            pass

    @objc.python_method
    def set_login(self, on: bool):
        if not on:
            LOGIN_AGENT.unlink(missing_ok=True)
            return
        app_path = NSBundle.mainBundle().bundlePath()
        if not app_path.endswith(".app"):
            return
        LOGIN_AGENT.parent.mkdir(parents=True, exist_ok=True)
        with open(LOGIN_AGENT, "wb") as f:
            plistlib.dump({"Label": "io.github.kewang0622.silentdictation", "RunAtLoad": True,
                           "ProgramArguments": ["/usr/bin/open", "-g", "-a", app_path, "--args", "--background"]}, f)

    @objc.python_method
    def review_save(self, hid, text):
        import numpy as np
        text = text.strip()
        f = paths.HISTORY / f"{hid}.npz"
        if text and f.exists():
            entry = json.loads((paths.HISTORY / f"{hid}.json").read_text())
            add_item(np.load(f)["roi"], text, "review", meta={"read": entry["text"], "raw": entry["raw"]})
        mark_reviewed(hid)
        self.invalidate("recent", "clips")

    @objc.python_method
    def send_thumb(self, hid):
        import cv2
        import numpy as np
        f = paths.HISTORY / f"{hid}.npz"
        if not f.exists():
            return
        roi = np.load(f)["roi"]
        idx = np.linspace(0, len(roi) - 1, 6).astype(int)
        strip = np.hstack([cv2.resize(roi[i], (64, 64)) for i in idx])
        self.js("thumb", {"id": hid, "img": data_url(strip, 75)})

    @objc.python_method
    def restart(self):
        path = NSBundle.mainBundle().bundlePath()
        if not path.endswith(".app"):
            return
        if self.lock_file is not None:
            fcntl.flock(self.lock_file, fcntl.LOCK_UN)
        subprocess.Popen(["/bin/sh", "-c", 'sleep 1; open -n "$0"', path])
        NSApplication.sharedApplication().terminate_(None)


def main() -> int:
    from ..cli import check_models, ensure_user_files

    ensure_user_files()
    paths.DATA.mkdir(parents=True, exist_ok=True)
    lock = open(paths.APP_LOCK, "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("Silent Dictation is already running.", flush=True)
        return 0
    app = NSApplication.sharedApplication()
    missing = check_models()
    if missing:
        app.setActivationPolicy_(NSApplicationActivationPolicyRegular)
        app.activateIgnoringOtherApps_(True)
        alert = NSAlert.alloc().init()
        alert.setMessageText_("Silent Dictation needs its models")
        alert.setInformativeText_("Run ./setup.sh and bin/build-app --install from the source folder.\n\nMissing: "
                                  + ", ".join(missing))
        alert.runModal()
        return 1
    delegate = AppDelegate.alloc().init()
    delegate.lock_file = lock
    app.setDelegate_(delegate)
    AppHelper.runEventLoop(installInterrupt=True)
    return 0
