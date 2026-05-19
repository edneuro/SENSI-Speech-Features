"""GUI test configuration: redirect preferences writes to a temp file
so tests do not pollute config/preferences.json."""

import os
import pytest


@pytest.fixture(autouse=True)
def _isolated_prefs(tmp_path, monkeypatch):
    """Redirect all save_preferences calls to a per-test temp file."""
    prefs_file = tmp_path / "test_preferences.json"
    monkeypatch.setenv("SENSI_PREFS_PATH", str(prefs_file))
    yield


@pytest.fixture
def prefs_tmp(tmp_path, monkeypatch):
    """Named alias used by tests that want an explicit handle on prefs isolation.

    The autouse fixture above already redirects SENSI_PREFS_PATH for every
    test, so this fixture is mostly a semantic marker — it returns the prefs
    file path in case a test needs it.
    """
    prefs_file = tmp_path / "test_preferences.json"
    monkeypatch.setenv("SENSI_PREFS_PATH", str(prefs_file))
    return prefs_file
