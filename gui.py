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
# MKL on AMD CPUs (Windows/Ryzen) raises a fatal exception (0xc06d007f) during
# BLAS matmul (e.g. librosa's mel-filterbank einsum in the acoustic level) under
# the default Intel threading layer. Force sequential threading before any
# numpy/MKL initialisation. Mirrors tests/conftest.py + main.py.
os.environ.setdefault("MKL_THREADING_LAYER", "sequential")

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QPalette, QColor, QIcon

from gui.main_window import MainWindow
from gui.theme import (
    qss_for_appearance,
    BG_PAGE, BG_PANEL, TEXT_PRIMARY, TEXT_SECONDARY,
    ACCENT_COLOR, BORDER_DEFAULT,
)


_ICON_PATH = _HERE / "assets" / "icon.svg"

# Windows picks the taskbar button's icon by the process's AppUserModelID, not
# by the window. Without an explicit ID the GUI runs as plain python.exe, gets
# grouped with Python, and the taskbar can show Python's icon instead of ours.
_APP_USER_MODEL_ID = "edneuro.SensiExtractor.GUI"


def _set_windows_app_id() -> None:
    """Give the process its own taskbar identity. Must run before QApplication."""
    if sys.platform != "win32":
        return
    import ctypes
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            _APP_USER_MODEL_ID)
    except (AttributeError, OSError):
        pass  # cosmetic only; never block startup


def _apply_palette(app: "QApplication") -> None:
    """Set a fallback palette so any widget not covered by QSS still renders
    with readable colors. Belt-and-braces against QSS edge cases on Windows."""
    p = app.palette()
    p.setColor(QPalette.ColorRole.Window, QColor(BG_PAGE))
    p.setColor(QPalette.ColorRole.WindowText, QColor(TEXT_PRIMARY))
    p.setColor(QPalette.ColorRole.Base, QColor(BG_PANEL))
    p.setColor(QPalette.ColorRole.AlternateBase, QColor(BG_PAGE))
    p.setColor(QPalette.ColorRole.Text, QColor(TEXT_PRIMARY))
    p.setColor(QPalette.ColorRole.PlaceholderText, QColor(TEXT_SECONDARY))
    p.setColor(QPalette.ColorRole.Button, QColor(BG_PANEL))
    p.setColor(QPalette.ColorRole.ButtonText, QColor(TEXT_PRIMARY))
    p.setColor(QPalette.ColorRole.Highlight, QColor(ACCENT_COLOR))
    p.setColor(QPalette.ColorRole.HighlightedText, QColor("white"))
    p.setColor(QPalette.ColorRole.ToolTipBase, QColor(BG_PANEL))
    p.setColor(QPalette.ColorRole.ToolTipText, QColor(TEXT_PRIMARY))
    p.setColor(QPalette.ColorRole.Link, QColor(ACCENT_COLOR))
    p.setColor(QPalette.ColorRole.Mid, QColor(BORDER_DEFAULT))
    app.setPalette(p)


def main() -> int:
    _set_windows_app_id()
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
