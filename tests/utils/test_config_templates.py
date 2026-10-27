"""Drift guard: the committed config templates are build artifacts of
DEFAULT_PREFS. If this fails, rerun: python scripts/export_default_config.py"""
from pathlib import Path

from sensi.utils.preferences import DEFAULT_PREFS
from sensi.utils.config_yaml import read_yaml, split_prefs

_CONFIG = Path(__file__).resolve().parents[2] / "config"
_REGEN = "templates out of sync with DEFAULT_PREFS — rerun: python scripts/export_default_config.py"


def test_features_template_matches_code_defaults():
    features_data, _ = split_prefs(DEFAULT_PREFS)
    assert read_yaml(_CONFIG / "features_defaults.yaml") == features_data, _REGEN


def test_run_template_matches_code_defaults():
    _, run_data = split_prefs(DEFAULT_PREFS)
    assert read_yaml(_CONFIG / "run_defaults.yaml") == run_data, _REGEN


def test_templates_carry_the_template_banner():
    for name in ("features_defaults.yaml", "run_defaults.yaml"):
        text = (_CONFIG / name).read_text(encoding="utf-8")
        assert text.startswith("# ======"), name
        assert "TEMPLATE" in text.splitlines()[0], name
