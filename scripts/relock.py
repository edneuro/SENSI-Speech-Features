#!/usr/bin/env python
"""Regenerate conda-lock.yml from the two source specs.

Source of truth is split by installer:
  - environment.yml   -> conda deps (python, Kaldi/MFA, compiled sci stack)
  - requirements.txt  -> pip deps

conda-lock only locks pip deps listed *inline* in environment.yml's
dependencies.pip section (it cannot read `-r requirements.txt`). So we splice
requirements.txt into a temporary merged env file, lock that, then delete it.
The committed artifact is conda-lock.yml (all platforms, exact + hashed).

Usage:   python scripts/relock.py
Requires: conda-lock on PATH  (conda install -n base -c conda-forge conda-lock)
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml  # PyYAML is a project dependency

ROOT = Path(__file__).resolve().parent.parent
ENV_YML = ROOT / "environment.yml"
REQS = ROOT / "requirements.txt"
LOCKFILE = ROOT / "conda-lock.yml"


def pip_requirements() -> list[str]:
    """Non-comment, non-blank lines from requirements.txt (markers preserved)."""
    lines = []
    for raw in REQS.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        lines.append(line)
    return lines


def build_merged_env() -> str:
    """environment.yml with requirements.txt spliced into dependencies.pip."""
    env = yaml.safe_load(ENV_YML.read_text(encoding="utf-8"))
    deps = env.setdefault("dependencies", [])
    if "pip" not in [d for d in deps if isinstance(d, str)]:
        deps.append("pip")
    deps.append({"pip": pip_requirements()})
    return yaml.safe_dump(env, sort_keys=False)


def main() -> int:
    if shutil.which("conda-lock") is None:
        sys.exit(
            "conda-lock not found. Install:\n"
            "  conda install -n base -c conda-forge conda-lock"
        )
    merged = build_merged_env()
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", suffix=".yml", delete=False, dir=ROOT, encoding="utf-8"
        ) as fh:
            fh.write(merged)
            tmp = Path(fh.name)
        print(f"Locking {ENV_YML.name} + {REQS.name} -> {LOCKFILE.name} ...")
        subprocess.run(
            ["conda-lock", "lock", "-f", str(tmp), "--lockfile", str(LOCKFILE)],
            check=True,
            cwd=ROOT,
        )
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)
    print(f"Wrote {LOCKFILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
