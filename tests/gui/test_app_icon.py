"""The GUI entry point gives Windows its own taskbar identity, so the taskbar
shows the SensiExtractor icon instead of grouping the window under Python."""
import ctypes
import importlib.util
import sys
from pathlib import Path

import pytest

_GUI_PY = Path(__file__).resolve().parents[2] / "gui.py"


def _load_entry_point():
    # gui.py shares its name with the sensi/gui/ package, so load it by path.
    spec = importlib.util.spec_from_file_location("sensi_gui_entry", _GUI_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(sys.platform != "win32", reason="Windows taskbar identity")
def test_sets_explicit_app_user_model_id():
    entry = _load_entry_point()
    entry._set_windows_app_id()
    shell32 = ctypes.windll.shell32
    ole32 = ctypes.windll.ole32
    out = ctypes.c_wchar_p()
    hr = shell32.GetCurrentProcessExplicitAppUserModelID(ctypes.byref(out))
    try:
        assert hr == 0
        assert out.value == entry._APP_USER_MODEL_ID
    finally:
        ole32.CoTaskMemFree(out)


def test_app_id_is_skipped_off_windows(monkeypatch):
    entry = _load_entry_point()
    monkeypatch.setattr(entry.sys, "platform", "darwin")
    entry._set_windows_app_id()  # must be a no-op, not raise
