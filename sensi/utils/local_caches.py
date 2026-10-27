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

from sensi.paths import (DICT_DIR, GENSIM_DIR, HF_CACHE_DIR, LEGACY_DICT_DIR,
                         LEGACY_MODEL_DIRS, MFA_DIR)

CACHE_DIRS = {
    "HF_HOME":         HF_CACHE_DIR,
    "MFA_ROOT_DIR":    MFA_DIR,
    "GENSIM_DATA_DIR": GENSIM_DIR,
}


def resolve_cache_dir(path: Path) -> Path:
    """data/<x>, or the pre-data/ top-level folder while it hasn't been moved."""
    legacy = LEGACY_MODEL_DIRS.get(path)
    if legacy is not None and not path.exists() and legacy.exists():
        return legacy
    return path


def use_project_caches() -> None:
    for var, path in CACHE_DIRS.items():
        os.environ.setdefault(var, str(resolve_cache_dir(path)))
    os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")  # no unauthenticated warning


def _say(msg: str) -> None:
    print(f"[setup] {msg}", file=sys.stderr)


def _move_legacy_dictionaries() -> None:
    """Move files left in the old top-level dictionaries/ into data/dictionaries/.

    After a pull, the tracked tables are already in data/dictionaries/, but
    local-only files (a LIWC .dic, the EWFG parquet, self-built norm tables)
    stay behind in dictionaries/. Move each one that isn't in data/ yet; never
    overwrite. The folder is removed once empty.
    """
    old = LEGACY_DICT_DIR
    if not old.is_dir():
        return
    for f in sorted(old.iterdir()):
        target = DICT_DIR / f.name
        if not f.is_file():
            continue
        if target.exists():
            _say(f"dictionaries/{f.name} also exists in data/dictionaries/; "
                 "using the data/ copy. Delete the old one when you're sure.")
            continue
        try:
            DICT_DIR.mkdir(parents=True, exist_ok=True)
            f.rename(target)
            _say(f"Moved dictionaries/{f.name} to data/dictionaries/ (new folder layout).")
        except OSError as exc:
            _say(f"Could not move dictionaries/{f.name} to data/dictionaries/ ({exc}); "
                 "it is not used from the old folder -- move it by hand.")
    try:
        old.rmdir()             # only succeeds when nothing is left
    except OSError:
        pass


def move_legacy_model_dirs() -> None:
    """Move model folders (and leftover dictionary files) from the old top-level
    layout into data/, once.

    Called by the launchers (gui.py, run.py, scripts/download_models.py) and at
    the very start of tests/conftest.py, always before use_project_caches();
    importing the package never moves anything. A same-disk rename is instant.
    If it fails (a file is open), the old folder keeps being used --
    use_project_caches() falls back to it. A folder this process already points
    a cache variable at is left alone, so a running session never loses it.
    """
    pinned = {os.path.normcase(os.path.abspath(v))
              for v in (os.environ.get(k) for k in CACHE_DIRS) if v}
    for new, old in LEGACY_MODEL_DIRS.items():
        if not old.exists():
            continue
        if new.exists():
            _say(f"Both {old.name}/ and data/{new.name}/ exist; using data/{new.name}/. "
                 f"The old {old.name}/ is no longer used -- delete it to free the space.")
            continue
        if os.path.normcase(str(old)) in pinned:
            continue
        try:
            new.parent.mkdir(parents=True, exist_ok=True)
            old.rename(new)
            _say(f"Moved {old.name}/ to data/{new.name}/ (new folder layout).")
        except OSError as exc:
            _say(f"Could not move {old.name}/ to data/ ({exc}); still using {old}.")
    _move_legacy_dictionaries()
