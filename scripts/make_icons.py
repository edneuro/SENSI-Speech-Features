#!/usr/bin/env python
"""Render assets/icon.svg into raster icon files used by the double-click
launchers: assets/icon.png (Linux .desktop) and assets/icon.ico (Windows
shortcut). Committed artifacts — re-run this only if icon.svg changes:

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
    img.fill(Qt.transparent)
    painter = QPainter(img)
    renderer.render(painter)
    painter.end()
    return img


def qimage_to_pil(img: QImage) -> Image.Image:
    ba = QByteArray()
    buf = QBuffer(ba)
    buf.open(QBuffer.OpenModeFlag.WriteOnly)
    img.save(buf, "PNG")
    buf.close()
    return Image.open(io.BytesIO(bytes(ba))).convert("RGBA")


def main() -> int:
    if not SVG.exists():
        raise SystemExit(f"missing {SVG}")
    _app = QGuiApplication([])  # required for QImage/QPainter/QSvgRenderer
    pil = qimage_to_pil(render_png(256))
    png_path = ROOT / "assets" / "icon.png"
    ico_path = ROOT / "assets" / "icon.ico"
    pil.save(png_path)
    pil.save(ico_path, sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    print(f"wrote {png_path}")
    print(f"wrote {ico_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
