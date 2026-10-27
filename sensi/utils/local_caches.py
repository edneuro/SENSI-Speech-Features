"""Keep every model download inside the project folder (data/).

Called at each entry point (gui.py, run.py, tests/conftest.py,
scripts/download_models.py) and from sensi/extractors/__init__.py, so any
script that loads a model uses the project's caches instead of the libraries'
defaults in the home directory (~/.cache/huggingface, ~/Documents/MFA,
~/gensim-data).

setdefault: a location the user already set (e.g. HF_HOME on Sherlock, or the
container's MFA_ROOT_DIR) wins. Must run before huggingface_hub / gensim are
imported -- both read their cache location once, at import time.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from sensi.paths import GENSIM_DIR, HF_CACHE_DIR, LEGACY_MODEL_DIRS, MFA_DIR

CACHE_DIRS = {
    "HF_HOME":         HF_CACHE_DIR,
    "MFA_ROOT_DIR":    MFA_DIR,
    "GENSIM_DATA_DIR": GENSIM_DIR,
}


def _resolve(path: Path) -> Path:
    """data/<x>, or the pre-data/ top-level folder while it hasn't been moved."""
    legacy = LEGACY_MODEL_DIRS.get(path)
    if legacy is not None and not path.exists() and legacy.exists():
        return legacy
    return path


def use_project_caches() -> None:
    for var, path in CACHE_DIRS.items():
        os.environ.setdefault(var, str(_resolve(path)))
    os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")  # no unauthenticated warning


def move_legacy_model_dirs() -> None:
    """Move model folders from the old top-level layout into data/, once.

    Called by the launchers only (gui.py, run.py, scripts/download_models.py),
    before use_project_caches(); importing the package never moves anything.
    A same-disk rename is instant. If it fails (a file is open), the old
    folder keeps being used -- use_project_caches() falls back to it.
    """
    for new, old in LEGACY_MODEL_DIRS.items():
        if old.exists() and not new.exists():
            try:
                new.parent.mkdir(parents=True, exist_ok=True)
                old.rename(new)
                print(f"[setup] Moved {old.name}/ to data/{new.name}/ "
                      "(new folder layout).", file=sys.stderr)
            except OSError as exc:
                print(f"[setup] Could not move {old.name}/ to data/ ({exc}); "
                      f"still using {old}.", file=sys.stderr)
