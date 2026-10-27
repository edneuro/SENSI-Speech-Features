"""Add the project root to sys.path so tests can import extractors/utils."""
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Tests that load models use the project's caches, not the home directory.
from sensi.utils.local_caches import use_project_caches  # noqa: E402
use_project_caches()

# MKL on AMD CPUs crashes when using the default Intel-thread layer for
# floating-point BLAS calls (Windows, Ryzen). Force sequential threading so
# matrix-multiply in librosa/numpy works without a fatal exception (0xc06d007f).
os.environ.setdefault("MKL_THREADING_LAYER", "sequential")

import pytest


@pytest.fixture(autouse=True)
def _isolated_config_dir(tmp_path, monkeypatch):
    """Point the two-YAML config store at a per-test temp dir so no test can
    read or migrate/rewrite the repo's real config/. Tests that exercise the
    legacy-json path additionally set SENSI_PREFS_PATH (same tmp_path, so the
    json and the YAMLs it migrates into share a directory)."""
    monkeypatch.setenv("SENSI_CONFIG_DIR", str(tmp_path))
    yield
