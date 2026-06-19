"""Guardrail: importing the planner must not pull in the heavy ML stack.

Regression test for the ~51 s GUI startup caused by registry.py eagerly
importing torch/transformers/spaCy. Planning needs only Stage metadata; the
extractor callables import lazily on first run.
"""
import importlib
import sys


def test_importing_planner_does_not_load_heavy_ml():
    # Start from a clean slate for the modules we care about.
    for mod in ("torch", "transformers", "spacy"):
        sys.modules.pop(mod, None)
    for name in [m for m in sys.modules if m.startswith("gui.planner")]:
        del sys.modules[name]

    importlib.import_module("gui.planner")

    for mod in ("torch", "transformers", "spacy"):
        assert mod not in sys.modules, f"{mod} was imported at planner import time"


def test_stages_metadata_available_without_running():
    from gui.planner.registry import STAGES
    assert len(STAGES) > 0
    # Every stage exposes its metadata and a callable run, but calling it is
    # what would trigger the import — we never call it here.
    for stage in STAGES:
        assert stage.name
        assert callable(stage.run)
