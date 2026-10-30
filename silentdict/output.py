"""Insert text into the frontmost app by pasting, then restore the previous clipboard."""
from __future__ import annotations

import threading
import time

import Quartz
from AppKit import NSPasteboard, NSPasteboardItem

V_KEYCODE = 9  # ANSI 'v'


def accessibility_allowed(prompt: bool = False) -> bool:
    try:
        if Quartz.CGPreflightPostEventAccess():
            return True
        if prompt:
            Quartz.CGRequestPostEventAccess()
        return False
    except AttributeError:
        return True


def _snapshot(pb):
    items = []
    for item in pb.pasteboardItems() or []:
        data = {}
        for t in item.types():
            d = item.dataForType_(t)
            if d is not None:
                data[t] = d
        if data:
            items.append(data)
    return items


def _restore(pb, items, change_count):
    if pb.changeCount() != change_count:  # the user copied something new meanwhile
        return
    pb.clearContents()
    if not items:
        return
    objs = []
    for data in items:
        item = NSPasteboardItem.alloc().init()
        for t, d in data.items():
            item.setData_forType_(d, t)
        objs.append(item)
    pb.writeObjects_(objs)


def _cmd_v():
    src = Quartz.CGEventSourceCreate(Quartz.kCGEventSourceStateCombinedSessionState)
    for down in (True, False):
        ev = Quartz.CGEventCreateKeyboardEvent(src, V_KEYCODE, down)
        Quartz.CGEventSetFlags(ev, Quartz.kCGEventFlagMaskCommand)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, ev)
        time.sleep(0.01)


def paste_text(text: str, restore_after: float = 0.6):
    pb = NSPasteboard.generalPasteboard()
    saved = _snapshot(pb)
    pb.clearContents()
    pb.setString_forType_(text, "public.utf8-plain-text")
    count = pb.changeCount()
    _cmd_v()
    if restore_after is not None:
        threading.Timer(restore_after, _restore, args=(pb, saved, count)).start()
