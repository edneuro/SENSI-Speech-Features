"""Integration test configuration: isolate preferences.

Batch runs load preferences internally (e.g. main.py's session-units sidecar
builds a plan from `load_preferences()`), so without this an integration test
would read — and migrate/rewrite — the developer's real two-YAML config store
(config/features.yaml + config/run.yaml), making outcomes depend on that
machine's saved GUI state. Redirect SENSI_CONFIG_DIR (and, for the legacy-json
import path, SENSI_PREFS_PATH) to a per-test temp dir so every integration
test sees the shipped defaults, hermetically.
"""

import pytest


@pytest.fixture(autouse=True)
def _isolated_prefs(tmp_path, monkeypatch):
    monkeypatch.setenv("SENSI_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("SENSI_PREFS_PATH", str(tmp_path / "test_preferences.json"))
    yield
