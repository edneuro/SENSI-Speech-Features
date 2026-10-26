#!/usr/bin/env python
"""Render assets/icon.svg into raster icon files used by the double-click
launchers: assets/icon.png (Linux .desktop), assets/icon.ico (Windows
shortcut) and assets/icon.icns (macOS SensiExtractor.app). Committed
artifacts — re-run this only if icon.svg changes:

    conda run -p ./env python scripts/make_icons.py

Uses Qt (SVG rasterizer, already a dependency) + Pillow (writes multi-size .ico).
"""
import io
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QBuffer, QByteArray, Qt
from PySide6.QtGui import QGuiApplication, QImage, QPainter
from PySide6.QtSvg import QSvgRenderer
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SVG = ROOT / "assets" / "icon.svg"


def render_png(size: int) -> QImage:
    renderer = QSvgRenderer(str(SVG))
    img = QImage(size, size, QImage.Format.Format_ARGB32)
    img.fill(Qt.GlobalColor.transparent)
    painter = QPainter(img)
    renderer.render(painter)
    painter.end()
    return img


def qimage_to_pil(img: QImage) -> Image.Image:
    ba = QByteArray()
    buf = QBuffer(ba)
    buf.open(QBuffer.OpenModeFlag.WriteOnly)
    # PySide6 6.9 stubs declare format: bytes, but the runtime accepts
    # only str here (verified: passing b"PNG" raises ValueError).
    img.save(buf, "PNG")  # pyright: ignore[reportCallIssue, reportArgumentType]
    buf.close()
    return Image.open(io.BytesIO(ba.data())).convert("RGBA")


def main() -> int:
    if not SVG.exists():
        raise SystemExit(f"missing {SVG}")
    _app = QGuiApplication([])  # required for QImage/QPainter/QSvgRenderer
    pil = qimage_to_pil(render_png(256))
    png_path = ROOT / "assets" / "icon.png"
    ico_path = ROOT / "assets" / "icon.ico"
    pil.save(png_path)
    pil.save(ico_path, sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    # macOS wants sizes up to 1024 px (512@2x); render the SVG at full size so
    # the large Finder/Dock icons are sharp, and Pillow derives the smaller ones.
    icns_path = ROOT / "assets" / "icon.icns"
    qimage_to_pil(render_png(1024)).save(icns_path, format="ICNS")
    print(f"wrote {png_path}")
    print(f"wrote {ico_path}")
    print(f"wrote {icns_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
