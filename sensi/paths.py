"""Where the project's folders are. The one place that knows the layout.

Every module that needs a folder imports it from here instead of computing
`Path(__file__).parent.parent / ...`, so moving a folder is a one-line change.
Model caches can be relocated with the usual environment variables (HF_HOME,
MFA_ROOT_DIR, GENSIM_DATA_DIR -- see sensi/utils/local_caches.py); the
functions below read them at call time.
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ASSETS_DIR = PROJECT_ROOT / "assets"
CONFIG_DIR_DEFAULT = PROJECT_ROOT / "config"
DOCS_FEATURES_DIR = PROJECT_ROOT / "docs" / "features"

DATA_DIR = PROJECT_ROOT / "data"
DICT_DIR = DATA_DIR / "dictionaries"          # lexical norms, LIWC .dic, EWFG
HF_CACHE_DIR = DATA_DIR / "hf_cache"          # GPT-2, BERT, Llama
MFA_DIR = DATA_DIR / "mfa_data"               # MFA models + config
GENSIM_DIR = DATA_DIR / "gensim_data"         # Word2Vec Google News

# Top-level folders from before the data/ layout (2026-10-27).
LEGACY_DICT_DIR = PROJECT_ROOT / "dictionaries"
LEGACY_MODEL_DIRS = {
    HF_CACHE_DIR: PROJECT_ROOT / "hf_cache",
    MFA_DIR: PROJECT_ROOT / "mfa_data",
    GENSIM_DIR: PROJECT_ROOT / "gensim_data",
}


def mfa_dir() -> Path:
    """The MFA root actually in use: MFA_ROOT_DIR when set, else data/mfa_data."""
    return Path(os.environ.get("MFA_ROOT_DIR") or MFA_DIR)


def mfa_pretrained(kind: str, filename: str) -> Path:
    """A file under MFA's pretrained_models/<kind>/ (acoustic, dictionary, g2p)."""
    return mfa_dir() / "pretrained_models" / kind / filename
