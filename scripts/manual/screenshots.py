"""Screenshots of the real app for the "Using the app" chapter.

Runs the GUI offscreen in a child process with a scratch config (every
feature on, folders pointing at the manual's examples) — never the user's
config/ — and grabs each sidebar page. Output: dev/manual/screens/*.png
(committed, so a rebuild doesn't need Qt).
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from sensi.paths import PROJECT_ROOT

SCREENS_DIR = PROJECT_ROOT / "dev" / "manual" / "screens"
PAGES = ["inputs", "word", "document", "subword", "sentence", "acoustic", "run", "logs"]
SIZE = (1280, 1000)

_CHILD = r"""
import os, sys, json, time
sys.path[:0] = [{root!r}]
os.environ["QT_QPA_PLATFORM"] = "offscreen"
if sys.platform == "win32":                     # offscreen Qt finds no fonts by itself
    os.environ.setdefault("QT_QPA_FONTDIR", os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts"))
os.environ.setdefault("MKL_THREADING_LAYER", "sequential")
from sensi.utils.local_caches import use_project_caches; use_project_caches()
from PySide6.QtWidgets import QApplication
from sensi.gui.theme import qss_for_appearance
from sensi.gui.main_window import MainWindow
app = QApplication(sys.argv)
from PySide6.QtGui import QFont
app.setFont(QFont("Segoe UI", 9))      # what a normal Windows launch uses
app.setStyleSheet(qss_for_appearance())
win = MainWindow(); win.resize({w}, {h}); win.show()
def settle(n=60):
    for _ in range(n):
        app.processEvents(); time.sleep(0.02)
settle(120)
names = {pages!r}
for i, name in enumerate(names):
    win.sidebar.setCurrentRow(i); settle()
    win.grab().save(os.path.join({out!r}, f"{{i + 1:02d}}-{{name}}.png"))
print("captured", len(names))
"""


def capture(out_dir: Path = SCREENS_DIR) -> list[Path]:
    from manual.examples import EXAMPLES_DIR, _env, _prefs, _write_config
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="manual_screens_") as tmp:
        tmp = Path(tmp)
        out = EXAMPLES_DIR / "output"          # tidy path in the screenshots; git-ignored (output/)
        out.mkdir(exist_ok=True)
        prefs = _prefs("base", out)
        prefs["shared"]["analyzer_name"] = "Your Name"
        _write_config(prefs, tmp / "cfg")
        code = _CHILD.format(root=str(PROJECT_ROOT), w=SIZE[0], h=SIZE[1], pages=PAGES, out=str(out_dir))
        env = _env() | {"SENSI_CONFIG_DIR": str(tmp / "cfg")}
        proc = subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT, env=env,
                              capture_output=True, text=True, timeout=600)
        if proc.returncode != 0:
            raise RuntimeError("screenshot capture failed:\n" + proc.stderr[-3000:])
    return sorted(out_dir.glob("*.png"))
