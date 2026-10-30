"""Render the app icon (rounded gradient + SF Symbol 'mouth.fill') to an .icns file."""
import subprocess
import sys
import tempfile
from pathlib import Path

from AppKit import (NSBezierPath, NSBitmapImageFileTypePNG, NSBitmapImageRep, NSColor, NSDeviceRGBColorSpace,
                    NSGradient, NSGraphicsContext, NSImage, NSImageSymbolConfiguration, NSMakeRect)


def render(size: int, path: Path):
    rep = NSBitmapImageRep.alloc().initWithBitmapDataPlanes_pixelsWide_pixelsHigh_bitsPerSample_samplesPerPixel_hasAlpha_isPlanar_colorSpaceName_bytesPerRow_bitsPerPixel_(
        None, size, size, 8, 4, True, False, NSDeviceRGBColorSpace, 0, 0)
    ctx = NSGraphicsContext.graphicsContextWithBitmapImageRep_(rep)
    NSGraphicsContext.saveGraphicsState()
    NSGraphicsContext.setCurrentContext_(ctx)
    s = size / 1024.0
    shape = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
        NSMakeRect(100 * s, 100 * s, 824 * s, 824 * s), 185 * s, 185 * s)
    top = NSColor.colorWithSRGBRed_green_blue_alpha_(0.36, 0.33, 0.95, 1)
    bottom = NSColor.colorWithSRGBRed_green_blue_alpha_(0.10, 0.70, 0.78, 1)
    NSGradient.alloc().initWithStartingColor_endingColor_(top, bottom).drawInBezierPath_angle_(shape, -90)
    config = NSImageSymbolConfiguration.configurationWithPointSize_weight_(430 * s, 0.0).configurationByApplyingConfiguration_(
        NSImageSymbolConfiguration.configurationWithPaletteColors_([NSColor.whiteColor()]))
    symbol = NSImage.imageWithSystemSymbolName_accessibilityDescription_("mouth.fill", None).imageWithSymbolConfiguration_(config)
    sw, sh = symbol.size().width, symbol.size().height
    symbol.drawInRect_(NSMakeRect((size - sw) / 2, (size - sh) / 2, sw, sh))
    NSGraphicsContext.restoreGraphicsState()
    rep.representationUsingType_properties_(NSBitmapImageFileTypePNG, {}).writeToFile_atomically_(str(path), True)


def main(out: str):
    with tempfile.TemporaryDirectory() as tmp:
        iconset = Path(tmp) / "AppIcon.iconset"
        iconset.mkdir()
        for base in (16, 32, 128, 256, 512):
            render(base, iconset / f"icon_{base}x{base}.png")
            render(base * 2, iconset / f"icon_{base}x{base}@2x.png")
        subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", out], check=True)


if __name__ == "__main__":
    main(sys.argv[1])
