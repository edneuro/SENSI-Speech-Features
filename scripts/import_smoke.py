#!/usr/bin/env python
"""Import every extractor/util module + PySide6 + the CLI entry point.

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

# Run-only helper scripts: users invoke them directly; nothing in the app
# imports them, so they're outside the import surface this smoke test covers.
# (download_models guards its work under __main__, so importing it is now
# side-effect-free — it stays here only because import-testing it adds nothing.)
SKIP = {"sensi.utils.download_models"}

targets = ["main", "run", "PySide6.QtWidgets"]
for pkg in ("extractors", "utils", "gui", "gui/planner"):
    for path in sorted((ROOT / pkg).glob("*.py")):
        if path.stem == "__init__":
            continue
        mod = f"{pkg.replace('/', '.')}.{path.stem}"
        if mod in SKIP:
            continue
        targets.append(mod)

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
