"""GUI test configuration: redirect the two-YAML config store to a temp dir
so tests do not pollute config/features.yaml + config/run.yaml."""

import os
import pytest


@pytest.fixture(autouse=True)
def _isolated_prefs(tmp_path, monkeypatch):
    """Redirect all save_preferences calls to a per-test temp dir."""
    prefs_file = tmp_path / "test_preferences.json"
    monkeypatch.setenv("SENSI_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("SENSI_PREFS_PATH", str(prefs_file))
    yield


@pytest.fixture(autouse=True)
def _join_run_estimate_workers():
    """Any RunSection starts background estimate workers (gui/section_run);
    wait for them so no test leaves a thread running into the next one."""
    yield
    import sys
    mod = sys.modules.get("sensi.gui.section_run")
    if mod is not None:
        assert mod.wait_for_estimate_threads(10.0), "estimate worker left running"


@pytest.fixture
def prefs_tmp(tmp_path, monkeypatch):
    """Named alias used by tests that want an explicit handle on prefs isolation.

    The autouse fixture above already redirects the two-YAML config store for
    every test, so this fixture is mostly a semantic marker — it returns the
    legacy prefs.json path in case a test needs it as a scratch prefs location.
    """
    prefs_file = tmp_path / "test_preferences.json"
    monkeypatch.setenv("SENSI_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("SENSI_PREFS_PATH", str(prefs_file))
    return prefs_file
