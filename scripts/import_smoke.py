#!/usr/bin/env python
"""Import every module of the sensi package + PySide6 + the run.py launcher.

Used by CI to prove the locked env is installable AND importable on each OS,
without needing the ~1.5 GB vendored models (a fresh checkout lacks them).
Exits nonzero if any import fails. Imports must be side-effect-free (no model
loading at import time — the codebase loads models lazily).
"""
import importlib
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # headless PySide6
os.environ.setdefault("MKL_THREADING_LAYER", "sequential")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Run-by-hand helpers live in scripts/, outside the package; nothing in the
# app imports them, so they're outside the import surface this test covers.
targets = ["run", "PySide6.QtWidgets"]
for path in sorted((ROOT / "sensi").rglob("*.py")):
    parts = path.relative_to(ROOT).with_suffix("").parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    targets.append(".".join(parts))

failed = []
for mod in targets:
    try:
        importlib.import_module(mod)
        print(f"ok    {mod}")
    except Exception as exc:  # noqa: BLE001 - report every failure
        print(f"FAIL  {mod}: {type(exc).__name__}: {exc}")
        failed.append(mod)

if failed:
    raise SystemExit(f"{len(failed)} import(s) failed: {failed}")
print("all imports ok")
