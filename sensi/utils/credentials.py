"""HuggingFace token-file handling shared by run.py, the GUI dispatcher, and
token detection.

The conventional location is <config_dir>/hf_token.txt (gitignored), named by
prefs `shared.hf_token_file`. A relative value resolves against the config
dir (sensi.utils.preferences.get_config_dir), so the shipped default "hf_token.txt"
is correct on every machine; absolute paths and ~/$VARS also work.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

HF_TOKEN_BASENAME = "hf_token.txt"


def resolve_hf_token_path(shared: dict[str, Any]) -> Path | None:
    """Path named by shared.hf_token_file, or None when unset/empty."""
    raw = (shared or {}).get("hf_token_file") or ""
    if not raw:
        return None
    expanded = os.path.expanduser(os.path.expandvars(str(raw)))
    path = Path(expanded)
    if not path.is_absolute():
        from sensi.utils.preferences import get_config_dir
        path = get_config_dir() / path
    return path


def read_hf_token(shared: dict[str, Any]) -> str:
    """Token text from the configured file; '' when missing/unreadable/empty."""
    path = resolve_hf_token_path(shared)
    if path is None or not path.is_file():
        return ""
    try:
        return path.read_text(encoding="utf-8-sig").strip()
    except OSError as exc:
        print(f"[warn] could not read hf_token_file {path}: {exc}", file=sys.stderr)
        return ""


def export_hf_token(shared: dict[str, Any]) -> bool:
    """Set HF_TOKEN from the configured file. Never overwrites an existing
    HF_TOKEN (a session-pasted token wins). Returns True iff it set the var."""
    if os.environ.get("HF_TOKEN"):
        return False
    token = read_hf_token(shared)
    if not token:
        return False
    os.environ["HF_TOKEN"] = token
    return True


def save_hf_token(token: str) -> Path:
    """Persist a token to <config_dir>/hf_token.txt and point the pref at it."""
    from sensi.utils.preferences import get_config_dir, save_preferences
    path = get_config_dir() / HF_TOKEN_BASENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(token.strip() + "\n", encoding="utf-8")
    save_preferences({"shared": {"hf_token_file": HF_TOKEN_BASENAME}})
    return path
