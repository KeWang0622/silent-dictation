"""Floating indicator: a small dark capsule at the bottom of the screen (never takes focus)."""
from __future__ import annotations

import math
import time

from AppKit import (NSAnimationContext, NSBackingStoreBuffered, NSColor, NSFont, NSImage, NSImageView,
                    NSImageScaleProportionallyUpOrDown, NSMakeRect, NSPanel, NSScreen,
                    NSStatusWindowLevel, NSTextField, NSView, NSWindowCollectionBehaviorCanJoinAllSpaces,
                    NSWindowCollectionBehaviorFullScreenAuxiliary, NSWindowCollectionBehaviorStationary,
                    NSWindowStyleMaskBorderless, NSWindowStyleMaskNonactivatingPanel)

H = 46
BAR_W, BAR_GAP, BARS = 4, 4, 7
SPOKES = 8


def _layer_view(x, y, w, h, color, radius):
    v = NSView.alloc().initWithFrame_(NSMakeRect(x, y, w, h))
    v.setWantsLayer_(True)
    v.layer().setBackgroundColor_(color.CGColor())
    v.layer().setCornerRadius_(radius)
    return v


def _symbol(name, color, size=18):
    from AppKit import NSImageSymbolConfiguration
    img = NSImage.imageWithSystemSymbolName_accessibilityDescription_(name, None)
    cfg = NSImageSymbolConfiguration.configurationWithPointSize_weight_(size, 0.3).configurationByApplyingConfiguration_(
        NSImageSymbolConfiguration.configurationWithPaletteColors_([NSColor.whiteColor(), color]))
    return img.imageWithSymbolConfiguration_(cfg)


class Overlay:
    def __init__(self):
        style = NSWindowStyleMaskBorderless | NSWindowStyleMaskNonactivatingPanel
        p = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(0, 0, 160, H), style, NSBackingStoreBuffered, False)
        p.setLevel_(NSStatusWindowLevel)
        p.setOpaque_(False)
        p.setBackgroundColor_(NSColor.clearColor())
        p.setIgnoresMouseEvents_(True)
        p.setHasShadow_(True)
        p.setHidesOnDeactivate_(False)
        p.setAppearance_(__import__("AppKit").NSAppearance.appearanceNamed_("NSAppearanceNameDarkAqua"))
        p.setCollectionBehavior_(NSWindowCollectionBehaviorCanJoinAllSpaces
                                 | NSWindowCollectionBehaviorFullScreenAuxiliary
                                 | NSWindowCollectionBehaviorStationary)
        self.panel = p
        bg = NSView.alloc().initWithFrame_(NSMakeRect(0, 0, 160, H))
        bg.setWantsLayer_(True)
        layer = bg.layer()
        layer.setBackgroundColor_(NSColor.colorWithSRGBRed_green_blue_alpha_(0.075, 0.075, 0.085, 0.94).CGColor())
        layer.setCornerRadius_(H / 2)
        layer.setBorderWidth_(1)
        layer.setBorderColor_(NSColor.colorWithWhite_alpha_(1, 0.10).CGColor())
        p.setContentView_(bg)
        self.bg = bg

        self.thumb = NSImageView.alloc().initWithFrame_(NSMakeRect(6, 6, 34, 34))
        self.thumb.setImageScaling_(NSImageScaleProportionallyUpOrDown)
        self.thumb.setWantsLayer_(True)
        self.thumb.layer().setCornerRadius_(17)
        self.thumb.layer().setMasksToBounds_(True)
        self.thumb.layer().setBackgroundColor_(NSColor.colorWithWhite_alpha_(1, 0.08).CGColor())
        bg.addSubview_(self.thumb)

        self.bars = [_layer_view(0, 0, BAR_W, 6, NSColor.whiteColor(), BAR_W / 2) for _ in range(BARS)]
        for b in self.bars:
            bg.addSubview_(b)
        self.heights = [0.1] * BARS
        self.dot = _layer_view(0, (H - 8) / 2, 8, 8, NSColor.systemRedColor(), 4)
        bg.addSubview_(self.dot)

        self.icon = NSImageView.alloc().initWithFrame_(NSMakeRect(14, (H - 20) / 2, 20, 20))
        bg.addSubview_(self.icon)
        # a small ring of dots that chases around while transcribing (animated from the app's 15 Hz tick)
        self.spokes = []
        for i in range(SPOKES):
            a = math.pi / 2 - 2 * math.pi * i / SPOKES
            cx, cy = 23 + 7 * math.cos(a), H / 2 + 7 * math.sin(a)
            dot = _layer_view(cx - 1.75, cy - 1.75, 3.5, 3.5, NSColor.whiteColor(), 1.75)
            bg.addSubview_(dot)
            self.spokes.append(dot)

        self.label = NSTextField.labelWithString_("")
        self.label.setTextColor_(NSColor.whiteColor())
        self.label.setFont_(NSFont.systemFontOfSize_weight_(13, 0.23))
        self.label.setMaximumNumberOfLines_(2)
        self.label.cell().setWraps_(True)
        self.label.cell().setTruncatesLastVisibleLine_(True)
        bg.addSubview_(self.label)
        self.sub = NSTextField.labelWithString_("")
        self.sub.setTextColor_(NSColor.colorWithWhite_alpha_(1, 0.55))
        self.sub.setFont_(NSFont.systemFontOfSize_(11))
        bg.addSubview_(self.sub)

        self.mode = None
        self.visible = False
        self.width = 160
        self.height = H
        self.t0 = time.monotonic()

    # ------------------------------------------------------------------ layout helpers
    def _hide_all(self):
        for v in [self.thumb, self.dot, self.icon, self.label, self.sub] + self.bars + self.spokes:
            v.setHidden_(True)

    def _place(self, width, height=H):
        width, height = int(width), int(height)
        screen = NSScreen.mainScreen().visibleFrame()
        x = screen.origin.x + (screen.size.width - width) / 2
        y = screen.origin.y + 26
        frame = NSMakeRect(x, y, width, height)
        if (width, height) != (self.width, self.height) and self.visible:
            NSAnimationContext.beginGrouping()
            NSAnimationContext.currentContext().setDuration_(0.18)
            self.panel.animator().setFrame_display_(frame, True)
            NSAnimationContext.endGrouping()
        else:
            self.panel.setFrame_display_(frame, True)
        self.bg.setFrame_(NSMakeRect(0, 0, width, height))
        self.bg.layer().setCornerRadius_(min(height, H) / 2 if height == H else 18)
        self.width, self.height = width, height

    def _show(self):
        if not self.visible:
            self.panel.setAlphaValue_(0.0)
            self.panel.orderFrontRegardless()
            NSAnimationContext.beginGrouping()
            NSAnimationContext.currentContext().setDuration_(0.15)
            self.panel.animator().setAlphaValue_(1.0)
            NSAnimationContext.endGrouping()
            self.visible = True

    def hide(self):
        if self.visible:
            NSAnimationContext.beginGrouping()
            NSAnimationContext.currentContext().setDuration_(0.25)
            self.panel.animator().setAlphaValue_(0.0)
            NSAnimationContext.endGrouping()
            self.visible = False
            self.mode = None

    def _text(self, x, title, subtitle="", max_w=360):
        self.label.setStringValue_(title)
        self.label.setPreferredMaxLayoutWidth_(max_w)
        size = self.label.cell().cellSizeForBounds_(NSMakeRect(0, 0, max_w, 200))
        tw = min(max_w, math.ceil(size.width)) + 2
        th = math.ceil(size.height)
        height = H if th <= 20 and not subtitle else max(H, th + (18 if subtitle else 0) + 16)
        self.label.setHidden_(False)
        if subtitle:
            self.sub.setStringValue_(subtitle)
            sw = min(max_w, math.ceil(self.sub.cell().cellSize().width) + 2)
            self.sub.setHidden_(False)
            tw = max(tw, sw)
            self.label.setFrame_(NSMakeRect(x, height - 10 - th, tw, th))
            self.sub.setFrame_(NSMakeRect(x, 9, sw, 15))
        else:
            self.label.setFrame_(NSMakeRect(x, (height - th) / 2, tw, th))
        return x + tw + 18, height

    # ------------------------------------------------------------------ states
    def listening(self, thumb_image, levels, hands_free=False, warning=None, key_label="fn"):
        if self.mode != "listening":
            self._hide_all()
            self.mode = "listening"
        self.thumb.setHidden_(False)
        if thumb_image is not None:
            self.thumb.setImage_(thumb_image)
        x = 50
        for i, b in enumerate(self.bars):
            target = levels[i] if i < len(levels) else 0.1
            self.heights[i] = 0.55 * self.heights[i] + 0.45 * target
            h = max(4, self.heights[i] * 26)
            b.setHidden_(False)
            b.setFrame_(NSMakeRect(x + i * (BAR_W + BAR_GAP), (H - h) / 2, BAR_W, h))
        x += BARS * (BAR_W + BAR_GAP) + 8
        pulse = 0.55 + 0.45 * (0.5 + 0.5 * math.sin((time.monotonic() - self.t0) * 6))
        self.dot.setHidden_(False)
        self.dot.setAlphaValue_(pulse)
        self.dot.setFrameOrigin_((x, (H - 8) / 2))
        x += 8 + 14
        if hands_free or warning:
            right, height = self._text(x - 4, warning or "Hands-free", "" if warning else f"press {key_label} to finish", 240)
            self.label.setTextColor_(NSColor.systemOrangeColor() if warning else NSColor.whiteColor())
            self._place(right, H)
        else:
            self.label.setHidden_(True)
            self.sub.setHidden_(True)
            self._place(x, H)
        self._show()

    def busy(self, title):
        if self.mode != ("busy", title):
            self._hide_all()
            self.mode = ("busy", title)
            for s in self.spokes:
                s.setHidden_(False)
            self.label.setTextColor_(NSColor.whiteColor())
            right, height = self._text(42, title)
            self._place(right, height)
        head = (time.monotonic() - self.t0) * 12
        for i, s in enumerate(self.spokes):
            s.setAlphaValue_(0.2 + 0.8 * ((i - head) % SPOKES) / SPOKES)
        self._show()

    def message(self, title, subtitle="", kind="ok"):
        key = ("message", title, subtitle, kind)
        if self.mode != key:
            self._hide_all()
            self.mode = key
            symbol, color = {"ok": ("checkmark.circle.fill", NSColor.systemGreenColor()),
                             "warn": ("exclamationmark.circle.fill", NSColor.systemOrangeColor()),
                             "info": ("info.circle.fill", NSColor.colorWithWhite_alpha_(1, 0.7))}[kind]
            self.icon.setImage_(_symbol(symbol, color))
            self.icon.setHidden_(False)
            self.label.setTextColor_(NSColor.whiteColor())
            right, height = self._text(42, title, subtitle, 420)
            self.icon.setFrameOrigin_((14, (height - 20) / 2))
            self._place(right, height)
        self._show()
