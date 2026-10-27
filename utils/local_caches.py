"""Keep every model download inside the project folder.

Called at each entry point (main.py, gui.py, run.py, tests/conftest.py,
utils/download_models.py) and from extractors/__init__.py, so any script that
loads a model uses the project's caches instead of the libraries' defaults in
the home directory (~/.cache/huggingface, ~/Documents/MFA, ~/gensim-data).

setdefault: a location the user already set (e.g. HF_HOME on Sherlock, or the
container's MFA_ROOT_DIR) wins. Must run before huggingface_hub / gensim are
imported -- both read their cache location once, at import time.
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CACHE_DIRS = {
    "HF_HOME":         PROJECT_ROOT / "hf_cache",     # GPT-2, BERT, Llama
    "MFA_ROOT_DIR":    PROJECT_ROOT / "mfa_data",     # MFA models + config
    "GENSIM_DATA_DIR": PROJECT_ROOT / "gensim_data",  # Word2Vec Google News
}


def use_project_caches() -> None:
    for var, path in CACHE_DIRS.items():
        os.environ.setdefault(var, str(path))
    os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")  # no unauthenticated warning
