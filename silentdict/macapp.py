"""Native macOS menu-bar app (Willow Voice style): hold fn and silently mouth a sentence.

Run through the app bundle built by bin/build-app, or directly with
    PYTHONPATH=. .venv/bin/python -m silentdict.macapp
"""
from __future__ import annotations

import csv
import fcntl
import os
import subprocess
import sys
import time

import cv2
import numpy as np
import objc
import Quartz
from AppKit import (
    NSAlert, NSApplication, NSApplicationActivationPolicyAccessory, NSBackingStoreBuffered,
    NSBitmapImageFileTypePNG, NSColor, NSFont, NSImage, NSImageScaleProportionallyUpOrDown, NSImageView,
    NSMenu, NSMenuItem, NSPanel, NSPopUpButton, NSButton, NSScreen, NSStatusBar, NSStatusWindowLevel,
    NSTextField, NSVariableStatusItemLength, NSView, NSVisualEffectView, NSWindow,
    NSWindowCollectionBehaviorCanJoinAllSpaces, NSWindowCollectionBehaviorFullScreenAuxiliary,
    NSWindowCollectionBehaviorStationary, NSWindowStyleMaskBorderless, NSWindowStyleMaskClosable,
    NSWindowStyleMaskMiniaturizable, NSWindowStyleMaskNonactivatingPanel, NSWindowStyleMaskTitled,
    NSWorkspace,
)
from Foundation import NSData, NSLocale, NSMakeRect, NSObject, NSProcessInfo, NSRunLoop, NSRunLoopCommonModes, NSTimer, NSURL
from PyObjCTools import AppHelper

from . import CREDITS, __version__, devices, paths, ui
from . import settings as settings_mod
from .controller import Controller
from .dataset import add_item, delete_item, history_entries, load_items, load_prompts, mark_reviewed
from .hotkey import input_monitoring_allowed
from .output import accessibility_allowed
from .permissions import SETTINGS_URLS, camera_status, request_camera_async
from .roi import MouthCropper, frames_to_roi
from .text import wer

_LANG = settings_mod.load().get("language", "auto")
ZH = _LANG == "zh" or (_LANG == "auto" and str(NSLocale.preferredLanguages()[0]).lower().startswith("zh"))
LOGIN_AGENT = os.path.expanduser("~/Library/LaunchAgents/io.github.kewang0622.silentdictation.plist")


def T(zh: str, en: str) -> str:
    return zh if ZH else en


KEY_LABELS = {"fn": "fn (🌐)", "right_option": "Right Option ⌥", "right_command": "Right Command ⌘",
              "right_control": "Right Control ⌃"}
GREEN = NSColor.systemGreenColor()
RED = NSColor.systemRedColor()
AMBER = NSColor.systemOrangeColor()
GREY = NSColor.secondaryLabelColor()

DELEGATE = None
ACTIONS: dict[int, object] = {}
_TAG = [1]


def _tag(fn) -> int:
    _TAG[0] += 1
    ACTIONS[_TAG[0]] = fn
    return _TAG[0]


# ----------------------------------------------------------------------------- view helpers
def frame_rect(x, top, w, h, height):
    return NSMakeRect(x, height - top - h, w, h)


def label(parent, text, x, top, w, h, height, size=13, bold=False, color=None, wrap=False):
    f = NSTextField.wrappingLabelWithString_(text) if wrap else NSTextField.labelWithString_(text)
    f.setFrame_(frame_rect(x, top, w, h, height))
    f.setFont_(NSFont.boldSystemFontOfSize_(size) if bold else NSFont.systemFontOfSize_(size))
    if color is not None:
        f.setTextColor_(color)
    parent.addSubview_(f)
    return f


def button(parent, title, x, top, w, h, height, fn, key=""):
    b = NSButton.buttonWithTitle_target_action_(title, DELEGATE, "buttonClicked:")
    b.setFrame_(frame_rect(x, top, w, h, height))
    b.setTag_(_tag(fn))
    if key:
        b.setKeyEquivalent_(key)
    parent.addSubview_(b)
    return b


def popup(parent, items, selected, x, top, w, h, height, fn):
    p = NSPopUpButton.alloc().initWithFrame_pullsDown_(frame_rect(x, top, w, h, height), False)
    p.addItemsWithTitles_(items)
    if selected in items:
        p.selectItemWithTitle_(selected)
    p.setTarget_(DELEGATE)
    p.setAction_("popupChanged:")
    p.setTag_(_tag(lambda: fn(int(p.indexOfSelectedItem()))))
    parent.addSubview_(p)
    return p


def image_view(parent, x, top, w, h, height):
    v = NSImageView.alloc().initWithFrame_(frame_rect(x, top, w, h, height))
    v.setImageScaling_(NSImageScaleProportionallyUpOrDown)
    parent.addSubview_(v)
    return v


def to_nsimage(img: np.ndarray, quality: int = 85):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return None
    return NSImage.alloc().initWithData_(NSData.dataWithBytes_length_(buf.tobytes(), len(buf)))


def mouth_thumb(frame, size=96):
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


def camera_problem(controller) -> str | None:
    cam = controller.camera
    frame = controller.latest_frame()
    if cam is None:
        return None
    if cam.error:
        return T("无法打开摄像头", "Cannot open the camera")
    if frame is None:
        return None
    if float(frame.gray.mean()) < 2 and time.monotonic() - controller.camera_started > 3:
        return T("摄像头画面全黑：镜头被遮挡？在设置里换一个摄像头", "Camera image is black: lens covered? Pick another camera")
    if frame.obs is None:
        return T("看不到你的脸，请正对摄像头", "Can't see your face")
    return None


def open_url(url: str):
    NSWorkspace.sharedWorkspace().openURL_(NSURL.URLWithString_(url))


def key_label(settings) -> str:
    return KEY_LABELS.get(settings.get("key", "fn"), settings.get("key", "fn"))


# ----------------------------------------------------------------------------- floating indicator
class Overlay:
    W, H = 480, 66

    def __init__(self):
        style = NSWindowStyleMaskBorderless | NSWindowStyleMaskNonactivatingPanel
        self.panel = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(0, 0, self.W, self.H), style, NSBackingStoreBuffered, False)
        p = self.panel
        p.setLevel_(NSStatusWindowLevel)
        p.setOpaque_(False)
        p.setBackgroundColor_(NSColor.clearColor())
        p.setIgnoresMouseEvents_(True)
        p.setHasShadow_(True)
        p.setHidesOnDeactivate_(False)
        p.setCollectionBehavior_(NSWindowCollectionBehaviorCanJoinAllSpaces
                                 | NSWindowCollectionBehaviorFullScreenAuxiliary
                                 | NSWindowCollectionBehaviorStationary)
        fx = NSVisualEffectView.alloc().initWithFrame_(NSMakeRect(0, 0, self.W, self.H))
        fx.setMaterial_(13)  # HUD window
        fx.setBlendingMode_(0)
        fx.setState_(1)
        fx.setWantsLayer_(True)
        fx.layer().setCornerRadius_(20)
        fx.layer().setMasksToBounds_(True)
        p.setContentView_(fx)
        H = self.H
        self.thumb = image_view(fx, 12, 9, 60, 48, H)
        self.thumb.setWantsLayer_(True)
        self.thumb.layer().setCornerRadius_(8)
        self.thumb.layer().setMasksToBounds_(True)
        self.dot = NSView.alloc().initWithFrame_(frame_rect(84, 15, 12, 12, H))
        self.dot.setWantsLayer_(True)
        self.dot.layer().setCornerRadius_(6)
        fx.addSubview_(self.dot)
        self.title = label(fx, "", 104, 9, self.W - 116, 22, H, size=15, bold=True, color=NSColor.whiteColor())
        self.sub = label(fx, "", 84, 37, self.W - 96, 18, H, size=12, color=NSColor.colorWithWhite_alpha_(0.85, 1))
        self.visible = False

    def show(self, title: str, sub: str, color, image=None):
        self.title.setStringValue_(title if len(title) < 70 else title[:67] + "...")
        self.sub.setStringValue_(sub)
        self.dot.layer().setBackgroundColor_(color.CGColor())
        if image is not None:
            self.thumb.setImage_(image)
        elif not self.visible:
            self.thumb.setImage_(NSImage.imageWithSystemSymbolName_accessibilityDescription_("mouth", None))
        if not self.visible:
            screen = NSScreen.mainScreen().visibleFrame()
            x = screen.origin.x + (screen.size.width - self.W) / 2
            self.panel.setFrameOrigin_((x, screen.origin.y + 28))
            self.panel.orderFrontRegardless()
            self.visible = True

    def hide(self):
        if self.visible:
            self.panel.orderOut_(None)
            self.visible = False


# ----------------------------------------------------------------------------- windows
class BaseWindow:
    def __init__(self, app, title: str, w: int, h: int):
        self.app, self.W, self.H = app, w, h
        style = NSWindowStyleMaskTitled | NSWindowStyleMaskClosable | NSWindowStyleMaskMiniaturizable
        self.window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(0, 0, w, h), style, NSBackingStoreBuffered, False)
        self.window.setTitle_(title)
        self.window.setReleasedWhenClosed_(False)
        self.window.setDelegate_(DELEGATE)
        self.view = self.window.contentView()
        self.is_open = False

    def show(self):
        if not self.is_open:
            self.window.center()
            self.on_open()
        self.is_open = True
        self.window.makeKeyAndOrderFront_(None)
        NSApplication.sharedApplication().activateIgnoringOtherApps_(True)

    def close(self):
        self.window.close()

    def on_open(self):
        pass

    def on_close(self):
        self.is_open = False

    def tick(self):
        pass

    def snapshot(self, path: str):
        v = self.view
        rep = v.bitmapImageRepForCachingDisplayInRect_(v.bounds())
        v.cacheDisplayInRect_toBitmapImageRep_(v.bounds(), rep)
        rep.representationUsingType_properties_(NSBitmapImageFileTypePNG, {}).writeToFile_atomically_(path, True)


class CameraWindowMixin:
    """Windows that show the live camera keep it running while they are open."""

    def on_open(self):
        self.app.controller.preview_users += 1
        if camera_status() == 3:
            self.app.controller.ensure_camera()

    def on_close(self):
        BaseWindow.on_close(self)
        self.app.controller.preview_users = max(0, self.app.controller.preview_users - 1)
        self.app.controller.last_use = time.monotonic()

    def draw_camera(self, view):
        c = self.app.controller
        frame = c.latest_frame()
        recording = bool(c.camera and c.camera.recording)
        img = ui.camera_view(frame, recording)
        if frame is None:
            img = ui.draw_texts(img, [(T("摄像头未开启", "Camera off"), (250, 165), 20, ui.GREY)])
        view.setImage_(to_nsimage(img))


class SetupWindow(CameraWindowMixin, BaseWindow):
    def __init__(self, app):
        BaseWindow.__init__(self, app, T("Silent Dictation 设置", "Silent Dictation Setup"), 660, 640)
        v, H, s = self.view, self.H, app.settings
        label(v, "Silent Dictation", 24, 18, 400, 28, H, size=22, bold=True)
        label(v, T(f"对着摄像头无声做口型就能打字。完成下面几步后，按住 {key_label(s)} 说话，松开就会输入。",
                   f"Type by silently mouthing at your camera. Finish these steps, then hold {key_label(s)} and release to type."),
              24, 52, 612, 36, H, size=13, color=GREY, wrap=True)
        rows = [
            ("camera", T("摄像头权限：读取你的嘴型", "Camera: to see your lips")),
            ("input", T("输入监控：监听按键", "Input Monitoring: to notice the key")),
            ("accessibility", T("辅助功能：把文字粘贴到当前 App", "Accessibility: to paste into the current app")),
        ]
        self.status = {}
        for i, (kind, text) in enumerate(rows):
            top = 100 + i * 36
            self.status[kind] = label(v, "", 24, top + 2, 24, 22, H, size=15)
            label(v, text, 52, top + 3, 340, 22, H, size=13)
            button(v, T("允许", "Allow"), 400, top, 100, 28, H, lambda k=kind: self.request(k))
            button(v, T("打开设置", "Settings"), 506, top, 130, 28, H, lambda k=kind: open_url(SETTINGS_URLS[k]))
        self.restart_hint = label(v, "", 52, 206, 440, 18, H, size=12, color=AMBER)
        self.restart_btn = button(v, T("重启 App", "Restart app"), 506, 202, 130, 26, H, app.restart)
        self.restart_btn.setHidden_(True)

        label(v, T("摄像头", "Camera"), 24, 246, 80, 22, H, size=13, bold=True)
        cams = devices.cameras() or ["(none)"]
        self.cam_popup = popup(v, cams, s.get("camera_name") or cams[0], 110, 242, 290, 28, H, self.pick_camera)
        self.cam_status = label(v, "", 410, 247, 230, 20, H, size=12, color=GREY)
        self.preview = image_view(v, 110, 278, 320, 180, H)

        label(v, T("按键", "Key"), 24, 474, 80, 22, H, size=13, bold=True)
        keys = list(KEY_LABELS)
        popup(v, [KEY_LABELS[k] for k in keys], key_label(s), 110, 470, 200, 28, H,
              lambda i: self.app.set_key(keys[i]))
        self.key_tip = label(v, "", 24, 504, 470, 36, H, size=12, color=GREY, wrap=True)
        self.key_btn = button(v, T("键盘设置", "Keyboard Settings"), 506, 504, 130, 26, H,
                              lambda: open_url(SETTINGS_URLS["keyboard"]))
        self.conflict = label(v, "", 24, 544, 612, 34, H, size=12, color=AMBER, wrap=True)

        button(v, T("开始校准（约 5 分钟）", "Start calibration (about 5 min)"), 300, 592, 220, 32, H, app.open_calibration)
        button(v, T("完成", "Done"), 526, 592, 110, 32, H, self.done, key="\r")
        self._last_slow = 0.0

    def request(self, kind):
        if kind == "camera":
            if camera_status() == 0:
                request_camera_async(lambda granted: AppHelper.callAfter(self.after_camera, granted))
            else:
                open_url(SETTINGS_URLS["camera"])
        elif kind == "input":
            Quartz.CGRequestListenEventAccess()
        else:
            Quartz.CGRequestPostEventAccess()

    def after_camera(self, granted):
        if granted and self.is_open:
            self.app.controller.ensure_camera()

    def pick_camera(self, i):
        names = devices.cameras()
        if i < len(names):
            self.app.settings["camera_name"] = names[i]
            self.app.save_settings()
            self.app.controller.restart_camera()

    def done(self):
        self.app.settings["onboarded"] = True
        self.app.save_settings()
        self.close()

    def tick(self):
        self.draw_camera(self.preview)
        now = time.monotonic()
        if now - self._last_slow < 1.0:
            return
        self._last_slow = now
        c, s = self.app.controller, self.app.settings
        ok = {"camera": camera_status() == 3, "input": input_monitoring_allowed(),
              "accessibility": accessibility_allowed()}
        for k, good in ok.items():
            self.status[k].setStringValue_("✅" if good else "⚠️")
        need_restart = ok["input"] and bool(c.hotkey_error)
        self.restart_hint.setStringValue_(T("已授权输入监控：重启 App 后按键才会生效", "Input Monitoring granted: restart the app to activate the key")
                                          if need_restart else "")
        self.restart_btn.setHidden_(not need_restart)
        problem = camera_problem(c)
        if not ok["camera"]:
            self.cam_status.setStringValue_(T("先允许摄像头权限", "Allow camera access first"))
        elif problem:
            self.cam_status.setStringValue_(problem)
            self.cam_status.setTextColor_(AMBER)
        else:
            self.cam_status.setStringValue_(T("人脸正常 ✓", "Face OK ✓") if c.latest_frame() else "")
            self.cam_status.setTextColor_(GREEN)
        if ok["camera"] and c.camera is None:
            c.ensure_camera()
        key = s.get("key", "fn")
        globe = devices.globe_key_setting()
        if key == "fn" and globe != 0:
            self.key_tip.setStringValue_(T("建议：系统设置 › 键盘 › “按下 🌐 键时”选“无操作”，避免按 fn 时弹出表情或切换输入法。",
                                           "Tip: System Settings › Keyboard › 'Press 🌐 key to' → Do Nothing, so fn does not open emoji or switch input."))
            self.key_btn.setHidden_(False)
        else:
            self.key_tip.setStringValue_(T("双击按键进入免按住模式，再按一次结束。", "Double-tap the key for hands-free mode; press again to finish."))
            self.key_btn.setHidden_(True)
        willow = devices.running_apps_matching("Willow")
        if key == "fn" and willow:
            self.conflict.setStringValue_(T("Willow Voice 也在用 fn：同时运行时按 fn 会两个一起启动。可以把上面的按键换成右 Option，或在公共场合先退出 Willow。",
                                            "Willow Voice also uses fn, so both start together. Choose Right Option above, or quit Willow while in public."))
        else:
            self.conflict.setStringValue_("")


class CalibrationSession:
    """Collects prompted silent sentences. handle() runs on the controller's worker thread."""

    def __init__(self, app, count: int = 60):
        self.app = app
        self.prompts = load_prompts(count)
        self.idx = 0
        self.busy = False
        self.saved = []
        self.errors = []
        self.message = ""
        self.feedback = ""
        self.cropper = MouthCropper()

    @property
    def done(self) -> bool:
        return self.idx >= len(self.prompts)

    def accepting(self) -> bool:
        return not self.busy and not self.done

    def handle(self, frames):
        self.busy = True
        try:
            if not frames or frames[-1].ts - frames[0].ts < 0.8:
                self.message = T("太短了，请重录这一句", "Too short, try again")
                return
            try:
                roi = frames_to_roi(frames, self.cropper)
            except ValueError:
                self.message = T("没看清你的脸，请重录这一句", "Face not visible, try again")
                return
            prompt = self.prompts[self.idx]
            item = add_item(roi, prompt, "calibration")
            self.saved.append((item, self.idx))
            self.idx += 1
            self.message = T(f"已保存 {len(self.saved)} 句", f"Saved {len(self.saved)}")
        finally:
            self.busy = False
        rec = self.app.controller.recognizer
        if rec is not None:
            hyps = rec.reader(roi, nbest=1)
            hyp = hyps[0].text if hyps else ""
            e, n = wer(prompt, hyp)
            self.errors.append((e, n))
            total = sum(x for x, _ in self.errors) / max(sum(y for _, y in self.errors), 1)
            self.feedback = T(f"原模型读到：{hyp.lower() or '（无）'}\n这句错 {e}/{n} 个词 · 目前整体词错误率 {total:.0%}",
                              f"Base model read: {hyp.lower() or '(nothing)'}\n{e}/{n} words wrong · running WER {total:.0%}")

    def redo(self):
        if self.saved and not self.busy:
            item, idx = self.saved.pop()
            delete_item(item.id)
            self.idx = idx
            self.message = T("已删除上一句，请重录", "Removed, record it again")

    def skip(self):
        if not self.done and not self.busy:
            self.idx += 1


class CalibrationWindow(CameraWindowMixin, BaseWindow):
    def __init__(self, app):
        BaseWindow.__init__(self, app, T("校准", "Calibration"), 700, 680)
        v, H = self.view, self.H
        self.preview = image_view(v, 30, 16, 640, 360, H)
        self.progress = label(v, "", 30, 388, 640, 20, H, size=13, color=GREY)
        self.prompt = label(v, "", 30, 414, 640, 96, H, size=28, bold=True, wrap=True)
        self.status = label(v, "", 30, 516, 640, 20, H, size=13)
        self.feedback = label(v, "", 30, 540, 640, 74, H, size=12, color=GREY, wrap=True)
        self.redo_btn = button(v, T("重录上一句", "Redo last"), 30, 630, 130, 32, H, lambda: self.session.redo())
        self.skip_btn = button(v, T("跳过", "Skip"), 166, 630, 100, 32, H, lambda: self.session.skip())
        self.finish_btn = button(v, T("完成并训练", "Finish & train"), 520, 630, 150, 32, H, self.finish)
        self.session = None
        self.training = False
        self.log_lines = []

    def on_open(self):
        CameraWindowMixin.on_open(self)
        self.session = CalibrationSession(self.app)
        self.training = False
        self.app.controller.calibration = self.session
        for b in (self.redo_btn, self.skip_btn, self.finish_btn):
            b.setEnabled_(True)

    def on_close(self):
        CameraWindowMixin.on_close(self)
        self.app.controller.calibration = None

    def finish(self):
        self.app.controller.calibration = None
        total = len(load_items())
        if total < 20:
            self.status.setStringValue_(T(f"至少需要 20 句才能训练（现在 {total} 句）", f"Need at least 20 sentences to train (have {total})"))
            self.app.controller.calibration = self.session
            return
        self.training = True
        for b in (self.redo_btn, self.skip_btn, self.finish_btn):
            b.setEnabled_(False)
        self.prompt.setStringValue_(T("正在训练你的个人模型…", "Training your personal model..."))
        self.log_lines = []
        self.app.start_training(self.add_log, self.training_done)

    def add_log(self, line: str):
        AppHelper.callAfter(self._add_log, line)

    def _add_log(self, line):
        self.log_lines = (self.log_lines + [line.strip()])[-5:]
        self.feedback.setStringValue_("\n".join(self.log_lines))

    def training_done(self, report):
        AppHelper.callAfter(self._training_done, report)

    def _training_done(self, report):
        self.training = False
        self.prompt.setStringValue_(summary_text(report))
        self.finish_btn.setEnabled_(False)

    def tick(self):
        self.draw_camera(self.preview)
        s = self.session
        if s is None or self.training:
            return
        k = key_label(self.app.settings)
        if s.done:
            self.prompt.setStringValue_(T("这一轮的句子都录完了，点“完成并训练”。", "All sentences recorded. Click Finish & train."))
            self.progress.setStringValue_("")
        else:
            self.prompt.setStringValue_(s.prompts[s.idx])
            self.progress.setStringValue_(T(f"第 {s.idx + 1}/{len(s.prompts)} 句 · 按住 {k}，无声说出这句，然后松开",
                                            f"Sentence {s.idx + 1}/{len(s.prompts)} · hold {k}, mouth it silently, release"))
        c = self.app.controller
        if c.state == "listening":
            self.status.setStringValue_(T("● 录制中…", "● Recording..."))
            self.status.setTextColor_(RED)
        elif s.busy:
            self.status.setStringValue_(T("保存中…", "Saving..."))
            self.status.setTextColor_(AMBER)
        else:
            self.status.setStringValue_(camera_problem(c) or s.message)
            self.status.setTextColor_(AMBER if camera_problem(c) else GREEN)
        self.feedback.setStringValue_(s.feedback)


def summary_text(report) -> str:
    if not report:
        return T("训练没有完成，详情见日志。", "Training did not finish; see the log.")
    base, personal = report["base_wer"], report["personal_wer"]
    if report.get("active") == "new":
        return T(f"完成！你的个人模型词错误率 {base:.0%} → {personal:.0%}，已启用。",
                 f"Done! Word error rate {base:.0%} → {personal:.0%}; personal model enabled.")
    return T(f"完成。个人模型 {personal:.0%} 没有比原模型 {base:.0%} 更好，继续使用原来的模型。多录一些句子再训练。",
             f"Done. Personal model {personal:.0%} did not beat {base:.0%}; keeping the current model. Record more and train again.")


class ReviewWindow(BaseWindow):
    def __init__(self, app):
        BaseWindow.__init__(self, app, T("复查最近的输入", "Review recent dictation"), 820, 330)
        v, H = self.view, self.H
        self.montage = image_view(v, 26, 16, 768, 96, H)
        self.counter = label(v, "", 26, 124, 768, 18, H, size=12, color=GREY)
        self.read = label(v, "", 26, 146, 768, 20, H, size=13, color=GREY)
        self.field = NSTextField.alloc().initWithFrame_(frame_rect(26, 174, 768, 30, H))
        self.field.setFont_(NSFont.systemFontOfSize_(16))
        v.addSubview_(self.field)
        label(v, T("如果识别对了，直接保存；不对就改成你当时想说的话再保存。保存的句子会用来训练你的个人模型。",
                   "If it is right, just save; otherwise type what you meant, then save. Saved sentences train your personal model."),
              26, 212, 768, 36, H, size=12, color=GREY, wrap=True)
        button(v, T("保存 ⏎", "Save ⏎"), 26, 270, 120, 32, H, self.save, key="\r")
        button(v, T("跳过", "Skip"), 152, 270, 90, 32, H, self.skip)
        button(v, T("丢弃", "Discard"), 248, 270, 90, 32, H, self.discard)
        button(v, T("训练个人模型", "Train personal model"), 624, 270, 170, 32, H, self.train)
        self.entries = []
        self.i = 0
        self.added = 0

    def on_open(self):
        self.entries = history_entries()
        self.i = 0
        self.added = 0
        self.load()

    def load(self):
        if self.i >= len(self.entries):
            self.montage.setImage_(None)
            self.counter.setStringValue_("")
            self.read.setStringValue_(T(f"没有需要复查的内容了。本次添加了 {self.added} 句。",
                                        f"Nothing left to review. Added {self.added} sentences."))
            self.field.setStringValue_("")
            return
        e = self.entries[self.i]
        roi = np.load(paths.HISTORY / f"{e['id']}.npz")["roi"]
        idx = np.linspace(0, len(roi) - 1, 8).astype(int)
        self.montage.setImage_(to_nsimage(np.hstack([roi[j] for j in idx])))
        self.counter.setStringValue_(f"{self.i + 1} / {len(self.entries)}")
        self.read.setStringValue_(T("原始唇读：", "Lip reading: ") + e["raw"].lower())
        self.field.setStringValue_(e["text"])

    def save(self):
        if self.i < len(self.entries):
            e = self.entries[self.i]
            text = str(self.field.stringValue()).strip()
            if text:
                roi = np.load(paths.HISTORY / f"{e['id']}.npz")["roi"]
                add_item(roi, text, "review", meta={"read": e["text"], "raw": e["raw"]})
                self.added += 1
            mark_reviewed(e["id"])
            self.i += 1
            self.load()

    def skip(self):
        self.i += 1
        self.load()

    def discard(self):
        if self.i < len(self.entries):
            mark_reviewed(self.entries[self.i]["id"])
            self.i += 1
            self.load()

    def train(self):
        self.close()
        self.app.open_training()


class LogWindow(BaseWindow):
    def __init__(self, app):
        BaseWindow.__init__(self, app, T("训练个人模型", "Train personal model"), 560, 300)
        v, H = self.view, self.H
        self.title = label(v, T("正在训练…", "Training..."), 24, 18, 512, 26, H, size=17, bold=True, wrap=True)
        self.log = label(v, "", 24, 60, 512, 220, H, size=12, color=GREY, wrap=True)
        self.lines = []

    def on_open(self):
        self.lines = []
        self.title.setStringValue_(T("正在训练个人模型，需要几分钟；期间暂停输入。", "Training the personal model (a few minutes); dictation is paused."))
        self.log.setStringValue_("")
        self.app.start_training(self.add_log, self.done)

    def add_log(self, line):
        AppHelper.callAfter(self._add, line)

    def _add(self, line):
        self.lines = (self.lines + [line.strip()])[-12:]
        self.log.setStringValue_("\n".join(self.lines))

    def done(self, report):
        AppHelper.callAfter(lambda: self.title.setStringValue_(summary_text(report)))


class PreviewWindow(CameraWindowMixin, BaseWindow):
    def __init__(self, app):
        BaseWindow.__init__(self, app, T("摄像头预览", "Camera preview"), 700, 430)
        v, H = self.view, self.H
        self.preview = image_view(v, 30, 16, 640, 360, H)
        self.status = label(v, "", 30, 392, 640, 20, H, size=13)

    def tick(self):
        self.draw_camera(self.preview)
        c = self.app.controller
        problem = camera_problem(c)
        frame = c.latest_frame()
        fps = c.camera.measured_fps if c.camera else 0
        self.status.setStringValue_(problem or (T(f"人脸正常 · {fps:.0f} fps", f"Face OK · {fps:.0f} fps") if frame else ""))
        self.status.setTextColor_(AMBER if problem else GREEN)


# ----------------------------------------------------------------------------- app delegate
class AppDelegate(NSObject):
    def init(self):
        self = objc.super(AppDelegate, self).init()
        if self is None:
            return None
        global DELEGATE
        DELEGATE = self
        self.settings = settings_mod.load()
        self.windows = {}
        self.menu_actions = {}
        self.flash_until = 0.0
        self.flash = None
        self.lock_file = None
        return self

    # -- lifecycle
    def applicationDidFinishLaunching_(self, _note):
        NSApplication.sharedApplication().setActivationPolicy_(NSApplicationActivationPolicyAccessory)
        self.status_item = NSStatusBar.systemStatusBar().statusItemWithLength_(NSVariableStatusItemLength)
        self.menu = NSMenu.alloc().init()
        self.menu.setDelegate_(self)
        self.status_item.setMenu_(self.menu)
        self.icon = None
        self.set_icon("mouth")
        self.overlay = Overlay()
        self.controller = Controller(self.settings, self.notify)
        self.controller.start()
        timer = NSTimer.timerWithTimeInterval_target_selector_userInfo_repeats_(1 / 15, self, "tick:", None, True)
        NSRunLoop.currentRunLoop().addTimer_forMode_(timer, NSRunLoopCommonModes)
        self._slow = 0.0
        print(CREDITS, flush=True)
        print(f"permissions: camera={camera_status()} input_monitoring={input_monitoring_allowed()} "
              f"accessibility={accessibility_allowed()} bundle={os.environ.get('SILENTDICT_APP_BUNDLE')}", flush=True)
        if os.environ.get("SILENTDICT_SNAPSHOT"):
            AppHelper.callLater(4.0, self.snapshot_and_quit, os.environ["SILENTDICT_SNAPSHOT"])
            return
        if not self.settings.get("onboarded") or not self.permissions_ok():
            self.open_setup()

    def applicationWillTerminate_(self, _note):
        self.controller.shutdown()

    def windowWillClose_(self, note):
        win = note.object()
        for w in self.windows.values():
            if w.window is win:
                w.on_close()

    # -- target/action plumbing
    def buttonClicked_(self, sender):
        fn = ACTIONS.get(int(sender.tag()))
        if fn:
            fn()

    def popupChanged_(self, sender):
        self.buttonClicked_(sender)

    def menuClicked_(self, sender):
        fn = self.menu_actions.get(int(sender.tag()))
        if fn:
            fn()

    def menuNeedsUpdate_(self, menu):
        self.build_menu()

    @objc.python_method
    def notify(self, kind, data):
        AppHelper.callAfter(self.on_event, kind, data)

    @objc.python_method
    def on_event(self, kind, data):
        c = self.controller
        if kind == "toast":
            self.flash = (str(data), "", NSColor.systemGrayColor())
            self.flash_until = time.monotonic() + 2.0
        elif kind == "result":
            pasted = data.timings.get("pasted")
            sub = (T("已粘贴", "Pasted") if pasted else T("已复制到剪贴板（需要辅助功能权限才能自动粘贴）",
                                                           "Copied (allow Accessibility to paste)"))
            self.flash = (data.text, f"{sub} · {data.timings.get('total', 0):.1f}s · {data.raw.lower()[:48]}", GREEN)
            self.flash_until = time.monotonic() + 3.0
        icon = {"listening": "mouth.fill", "decoding": "ellipsis.circle", "training": "hourglass",
                "error": "exclamationmark.triangle", "loading": "hourglass"}.get(c.state, "mouth")
        self.set_icon(icon)

    @objc.python_method
    def set_icon(self, name):
        if name == self.icon:
            return
        img = NSImage.imageWithSystemSymbolName_accessibilityDescription_(name, "Silent Dictation")
        img.setTemplate_(True)
        self.status_item.button().setImage_(img)
        self.icon = name

    # -- periodic UI refresh (main thread)
    def tick_(self, _timer):
        self.update_overlay()
        for w in self.windows.values():
            if w.is_open:
                w.tick()
        now = time.monotonic()
        if now - self._slow > 3:
            self._slow = now
            c = self.controller
            if c.hotkey_error and input_monitoring_allowed():
                c.start_hotkey()

    @objc.python_method
    def update_overlay(self):
        c = self.controller
        if not self.settings.get("overlay", True):
            self.overlay.hide()
            return
        k = key_label(self.settings)
        if c.state == "listening":
            frame = c.latest_frame()
            problem = camera_problem(c)
            if frame is None or (float(frame.gray.mean()) < 2 and not problem):
                self.overlay.show(T("摄像头启动中…", "Starting camera..."),
                                  T("圆点变红后再开始做口型", "Start mouthing when the dot turns red"), AMBER)
                return
            hands_free = c.message == "hands-free"
            title = T("免按住读唇中…", "Hands-free...") if hands_free else T("读唇中…", "Reading lips...")
            sub = problem or (T(f"再按一次 {k} 结束", f"Press {k} again to finish") if hands_free
                              else T(f"松开 {k} 结束", f"Release {k} to finish"))
            thumb = mouth_thumb(frame)
            self.overlay.show(title, sub, AMBER if problem else RED, to_nsimage(thumb) if thumb is not None else None)
        elif c.state == "decoding":
            self.overlay.show(T("识别中…", "Decoding..."), "", AMBER)
        elif self.flash and time.monotonic() < self.flash_until:
            self.overlay.show(*self.flash)
        else:
            self.overlay.hide()

    # -- menu
    @objc.python_method
    def build_menu(self):
        m, c, s = self.menu, self.controller, self.settings
        m.removeAllItems()
        self.menu_actions = {}
        tag = [0]

        def item(title, fn=None, state=None, menu=m):
            it = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(title, "menuClicked:" if fn else None, "")
            if fn:
                tag[0] += 1
                it.setTarget_(self)
                it.setTag_(tag[0])
                self.menu_actions[tag[0]] = fn
            if state is not None:
                it.setState_(1 if state else 0)
            menu.addItem_(it)
            return it

        def submenu(title):
            it = item(title)
            sub = NSMenu.alloc().init()
            it.setSubmenu_(sub)
            it.setEnabled_(True)
            return sub

        status = {"loading": T("正在加载模型…", "Loading models..."), "training": T("正在训练个人模型…", "Training personal model..."),
                  "error": c.message, "listening": T("读唇中…", "Reading lips..."), "decoding": T("识别中…", "Decoding...")}
        if c.recognizer is not None:
            personal = c.recognizer.reader.personalized
            llm_on = c.recognizer.rewriter is not None
            status["ready"] = T(f"就绪 · {'个人模型' if personal else '原始模型'} · 语言模型修正{'开' if llm_on else '关'}",
                                f"Ready · {'personal' if personal else 'base'} model · LLM {'on' if llm_on else 'off'}")
        item(status.get(c.state, c.message))
        k = key_label(s)
        if c.hotkey_error:
            item(T("⚠️ 按键不可用：需要输入监控权限", "⚠️ Key inactive: Input Monitoring needed"), self.open_setup)
        else:
            item(T(f"按住 {k} 说话 · 双击进入免按住模式", f"Hold {k} to dictate · double-tap for hands-free"))
        cam = c.camera is not None
        item(T("摄像头：开启", "Camera: on") if cam else T(f"摄像头：关闭（按 {k} 时打开）", f"Camera: off (opens when you press {k})"))
        acc = self.accuracy_line()
        if acc:
            item(acc)
        m.addItem_(NSMenuItem.separatorItem())
        item(T("摄像头预览…", "Camera Preview..."), self.open_preview)
        item(T("校准…", "Calibrate..."), self.open_calibration)
        item(T("复查最近的输入…", "Review Recent Dictation..."), self.open_review)
        item(T("训练个人模型", "Train Personal Model"), self.open_training)
        m.addItem_(NSMenuItem.separatorItem())

        keys = submenu(T("按键", "Key"))
        for key, name in KEY_LABELS.items():
            item(name, lambda key=key: self.set_key(key), state=s.get("key") == key, menu=keys)
        cams = submenu(T("摄像头", "Camera"))
        current = s.get("camera_name") or (devices.cameras() or [None])[0]
        for name in devices.cameras():
            item(name, lambda name=name: self.set_camera(name), state=name == current, menu=cams)
        keep = submenu(T("摄像头保持开启", "Keep Camera On"))
        for value, name in (("warm", T("用完后 2 分钟", "2 minutes after use")), ("always", T("一直开启（最快）", "Always (fastest)")),
                            ("on_demand", T("只在说话时", "Only while dictating"))):
            item(name, lambda value=value: self.set_setting("camera_policy", value), state=s.get("camera_policy") == value, menu=keep)
        llm = submenu(T("本地语言模型修正", "Local LLM Clean-up"))
        for value, name in (("auto", T("自动（按校准结果）", "Auto (from calibration)")), ("on", T("开", "On")), ("off", T("关", "Off"))):
            item(name, lambda value=value: self.set_llm(value), state=s.get("llm", "auto") == value, menu=llm)
        item(T("显示底部浮窗", "Show Floating Indicator"), lambda: self.toggle("overlay"), state=s.get("overlay", True))
        item(T("保留嘴部片段供复查", "Keep Clips for Review"), lambda: self.toggle("history"), state=s.get("history", True))
        lang = submenu(T("界面语言", "Language"))
        for value, name in (("auto", T("跟随系统", "System")), ("zh", "中文"), ("en", "English")):
            item(name, lambda value=value: self.set_language(value), state=s.get("language", "auto") == value, menu=lang)
        if os.environ.get("SILENTDICT_APP_BUNDLE"):
            item(T("登录时打开", "Open at Login"), self.toggle_login, state=os.path.exists(LOGIN_AGENT))
        m.addItem_(NSMenuItem.separatorItem())
        item(T("设置与权限…", "Setup & Permissions..."), self.open_setup)
        item(T("打开数据文件夹", "Open Data Folder"), lambda: open_url(NSURL.fileURLWithPath_(str(paths.DATA)).absoluteString()))
        item(T("关于 Silent Dictation", "About Silent Dictation"), self.about)
        m.addItem_(NSMenuItem.separatorItem())
        item(T("退出", "Quit"), lambda: NSApplication.sharedApplication().terminate_(None))

    @objc.python_method
    def accuracy_line(self) -> str | None:
        progress = paths.REPORTS / "progress.csv"
        if not progress.exists():
            return T("尚未校准", "Not calibrated yet")
        rows = list(csv.DictReader(open(progress)))
        if not rows:
            return None
        r = rows[-1]
        return T(f"词错误率：原模型 {float(r['base_wer']):.0%} → 个人 {float(r['personal_wer']):.0%}",
                 f"Word error rate: base {float(r['base_wer']):.0%} → personal {float(r['personal_wer']):.0%}")

    # -- settings actions
    @objc.python_method
    def save_settings(self):
        settings_mod.save(self.settings)

    @objc.python_method
    def set_key(self, key):
        self.controller.set_key(key)
        self.save_settings()

    @objc.python_method
    def set_camera(self, name):
        self.settings["camera_name"] = name
        self.save_settings()
        self.controller.restart_camera()

    @objc.python_method
    def set_setting(self, key, value):
        self.settings[key] = value
        self.save_settings()
        if key == "camera_policy" and value == "always":
            self.controller.ensure_camera()

    @objc.python_method
    def set_llm(self, value):
        self.settings["llm"] = value
        self.save_settings()
        self.controller.reload_models()

    @objc.python_method
    def toggle(self, key):
        self.settings[key] = not self.settings.get(key, True)
        self.save_settings()

    @objc.python_method
    def set_language(self, value):
        self.settings["language"] = value
        self.save_settings()
        self.restart()

    @objc.python_method
    def toggle_login(self):
        if os.path.exists(LOGIN_AGENT):
            os.remove(LOGIN_AGENT)
            return
        import plistlib
        os.makedirs(os.path.dirname(LOGIN_AGENT), exist_ok=True)
        with open(LOGIN_AGENT, "wb") as f:
            plistlib.dump({"Label": "io.github.kewang0622.silentdictation", "RunAtLoad": True,
                           "ProgramArguments": ["/usr/bin/open", os.environ["SILENTDICT_APP_BUNDLE"]]}, f)

    @objc.python_method
    def permissions_ok(self) -> bool:
        return camera_status() == 3 and input_monitoring_allowed() and accessibility_allowed()

    # -- windows
    @objc.python_method
    def window(self, name, cls):
        if name not in self.windows:
            self.windows[name] = cls(self)
        self.windows[name].show()
        return self.windows[name]

    @objc.python_method
    def open_setup(self):
        self.window("setup", SetupWindow)

    @objc.python_method
    def open_calibration(self):
        if camera_status() != 3:
            self.open_setup()
            return
        if "setup" in self.windows and self.windows["setup"].is_open:
            self.settings["onboarded"] = True
            self.save_settings()
            self.windows["setup"].close()
        self.window("calibration", CalibrationWindow)

    @objc.python_method
    def open_review(self):
        self.window("review", ReviewWindow)

    @objc.python_method
    def open_preview(self):
        self.window("preview", PreviewWindow)

    @objc.python_method
    def open_training(self):
        if self.controller.training:
            return
        self.window("log", LogWindow)

    @objc.python_method
    def start_training(self, log, done):
        self.controller.train_async(log, done)

    @objc.python_method
    def about(self):
        NSApplication.sharedApplication().activateIgnoringOtherApps_(True)
        alert = NSAlert.alloc().init()
        alert.setMessageText_(f"Silent Dictation {__version__}")
        alert.setInformativeText_(T("对着摄像头无声做口型就能打字，完全在本机运行。\n\n", "Type by silently mouthing at your camera, fully on-device.\n\n")
                                  + CREDITS + "\n\nLip reading: Auto-AVSR (Ma et al.) via Chaplin by Amanvir Parhar. "
                                  "Face tracking: MediaPipe. Language model: Qwen3 on MLX.")
        alert.runModal()

    @objc.python_method
    def restart(self):
        bundle = os.environ.get("SILENTDICT_APP_BUNDLE")
        if not bundle:
            self.flash = (T("请退出后重新打开 App", "Please quit and reopen the app"), "", AMBER)
            self.flash_until = time.monotonic() + 3
            return
        if self.lock_file is not None:
            fcntl.flock(self.lock_file, fcntl.LOCK_UN)
        subprocess.Popen(["/bin/sh", "-c", 'sleep 1; open -n "$0"', bundle])
        NSApplication.sharedApplication().terminate_(None)

    @objc.python_method
    def snapshot_and_quit(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        for name, cls in (("setup", SetupWindow), ("calibration", CalibrationWindow), ("review", ReviewWindow),
                          ("preview", PreviewWindow)):
            w = self.window(name, cls)
            w.tick()
        def capture():
            for name, w in self.windows.items():
                w.tick()
                w.snapshot(os.path.join(out_dir, f"{name}.png"))
            self.overlay.show(T("读唇中…", "Reading lips..."), T("松开 fn 结束", "Release fn to finish"), RED)
            ov = self.overlay.panel.contentView()
            rep = ov.bitmapImageRepForCachingDisplayInRect_(ov.bounds())
            ov.cacheDisplayInRect_toBitmapImageRep_(ov.bounds(), rep)
            rep.representationUsingType_properties_(NSBitmapImageFileTypePNG, {}).writeToFile_atomically_(
                os.path.join(out_dir, "overlay.png"), True)
            self.build_menu()
            titles = []
            for i in range(self.menu.numberOfItems()):
                titles.append(str(self.menu.itemAtIndex_(i).title()))
            with open(os.path.join(out_dir, "menu.txt"), "w") as f:
                f.write("\n".join(titles))
            NSApplication.sharedApplication().terminate_(None)
        AppHelper.callLater(3.0, capture)


def main() -> int:
    from .cli import check_models, ensure_user_files

    ensure_user_files()
    lock = open(paths.APP_LOCK, "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("Silent Dictation is already running.")
        return 0
    NSProcessInfo.processInfo().setProcessName_("Silent Dictation")
    app = NSApplication.sharedApplication()
    missing = check_models()
    if missing:
        app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)
        NSApplication.sharedApplication().activateIgnoringOtherApps_(True)
        alert = NSAlert.alloc().init()
        alert.setMessageText_("Silent Dictation")
        alert.setInformativeText_("Model files are missing. Run ./setup.sh in the project folder first.\n\n" + "\n".join(missing))
        alert.runModal()
        return 1
    delegate = AppDelegate.alloc().init()
    delegate.lock_file = lock
    app.setDelegate_(delegate)
    AppHelper.runEventLoop(installInterrupt=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
