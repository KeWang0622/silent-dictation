"""Render silentdict/mac/web/logo.svg into macos/AppIcon.icns and PNG copies for the web UI and README.

Usage: .venv/bin/python macos/render_icon.py
Follows the macOS icon grid: an 824 px rounded square centred on a 1024 px canvas with a soft shadow.
"""
import shutil
import subprocess
import tempfile
from pathlib import Path

from AppKit import (NSBitmapImageRep, NSColor, NSCompositingOperationSourceOver, NSGraphicsContext, NSImage,
                    NSPNGFileType, NSShadow)
from Foundation import NSMakeRect, NSMakeSize

ROOT = Path(__file__).resolve().parent.parent
LOGO = ROOT / "silentdict" / "mac" / "web" / "logo.svg"


def render(size: int, out: Path, padded: bool = True):
    img = NSImage.alloc().initWithContentsOfFile_(str(LOGO))
    rep = NSBitmapImageRep.alloc().initWithBitmapDataPlanes_pixelsWide_pixelsHigh_bitsPerSample_samplesPerPixel_hasAlpha_isPlanar_colorSpaceName_bytesPerRow_bitsPerPixel_(
        None, size, size, 8, 4, True, False, "NSDeviceRGBColorSpace", 0, 0)
    rep.setSize_(NSMakeSize(size, size))
    NSGraphicsContext.saveGraphicsState()
    NSGraphicsContext.setCurrentContext_(NSGraphicsContext.graphicsContextWithBitmapImageRep_(rep))
    s = size / 1024
    if padded:
        shadow = NSShadow.alloc().init()
        shadow.setShadowOffset_((0, -10 * s))
        shadow.setShadowBlurRadius_(22 * s)
        shadow.setShadowColor_(NSColor.colorWithCalibratedWhite_alpha_(0, 0.32))
        shadow.set()
        rect = NSMakeRect(100 * s, 100 * s, 824 * s, 824 * s)
    else:
        rect = NSMakeRect(0, 0, size, size)
    img.drawInRect_fromRect_operation_fraction_(rect, NSMakeRect(0, 0, 0, 0), NSCompositingOperationSourceOver, 1.0)
    NSGraphicsContext.restoreGraphicsState()
    out.parent.mkdir(parents=True, exist_ok=True)
    rep.representationUsingType_properties_(NSPNGFileType, {}).writeToFile_atomically_(str(out), True)


def main():
    with tempfile.TemporaryDirectory() as tmp:
        iconset = Path(tmp) / "AppIcon.iconset"
        iconset.mkdir()
        for base in (16, 32, 128, 256, 512):
            render(base, iconset / f"icon_{base}x{base}.png")
            render(base * 2, iconset / f"icon_{base}x{base}@2x.png")
        subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", str(ROOT / "macos" / "AppIcon.icns")], check=True)
        shutil.copy(iconset / "icon_256x256@2x.png", ROOT / "silentdict" / "mac" / "web" / "icon.png")
    render(1024, ROOT / "docs" / "assets" / "icon.png")
    print("wrote macos/AppIcon.icns, silentdict/mac/web/icon.png, docs/assets/icon.png")


if __name__ == "__main__":
    main()
