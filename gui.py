"""SensiExtractor — PySide6 entry point.

Run: python gui.py
"""

import os
import sys
from pathlib import Path

_HERE = Path(__file__).parent.resolve()
os.environ.setdefault("HF_HOME",      str(_HERE / "hf_cache"))
os.environ.setdefault("MFA_ROOT_DIR", str(_HERE / "mfa_data"))
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QPalette, QColor, QIcon

from gui.main_window import MainWindow
from gui.theme import (
    qss_for_appearance,
    BG_PAGE, BG_PANEL, TEXT_PRIMARY, TEXT_SECONDARY,
    ACCENT_COLOR, BORDER_DEFAULT,
)


_ICON_PATH = _HERE / "assets" / "icon.svg"


def _apply_palette(app: "QApplication") -> None:
    """Set a fallback palette so any widget not covered by QSS still renders
    with readable colors. Belt-and-braces against QSS edge cases on Windows."""
    p = app.palette()
    p.setColor(QPalette.Window, QColor(BG_PAGE))
    p.setColor(QPalette.WindowText, QColor(TEXT_PRIMARY))
    p.setColor(QPalette.Base, QColor(BG_PANEL))
    p.setColor(QPalette.AlternateBase, QColor(BG_PAGE))
    p.setColor(QPalette.Text, QColor(TEXT_PRIMARY))
    p.setColor(QPalette.PlaceholderText, QColor(TEXT_SECONDARY))
    p.setColor(QPalette.Button, QColor(BG_PANEL))
    p.setColor(QPalette.ButtonText, QColor(TEXT_PRIMARY))
    p.setColor(QPalette.Highlight, QColor(ACCENT_COLOR))
    p.setColor(QPalette.HighlightedText, QColor("white"))
    p.setColor(QPalette.ToolTipBase, QColor(BG_PANEL))
    p.setColor(QPalette.ToolTipText, QColor(TEXT_PRIMARY))
    p.setColor(QPalette.Link, QColor(ACCENT_COLOR))
    p.setColor(QPalette.Mid, QColor(BORDER_DEFAULT))
    app.setPalette(p)


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("SENSI-FeatureExtractor")
    if _ICON_PATH.exists():
        app.setWindowIcon(QIcon(str(_ICON_PATH)))
    # Palette fallback first, then QSS on top. If a widget escapes the QSS
    # cascade for any reason, the palette still gives it readable colors.
    _apply_palette(app)
    app.setStyleSheet(qss_for_appearance())
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
