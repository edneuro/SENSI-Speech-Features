"""YAML-pair I/O layer for the two-file config store (features.yaml + run.yaml).

features.yaml holds schema_version/features/output/predictability/semantic/
prosody/... (global, shareable, reset-able by the GUI); run.yaml holds
`shared` (machine-specific folders, analyzer name, theme, token-file paths).
utils.preferences reads/writes prefs through split_prefs/join_yaml_pair and
the write_*_yaml helpers below; run.py validates raw files with
validate_features_yaml/validate_run_yaml before loading them.
"""
from __future__ import annotations

import os
import yaml
from copy import deepcopy


def _type_ok(default_val, val) -> bool:
    if isinstance(default_val, bool):
        return isinstance(val, bool)
    if isinstance(default_val, (int, float)):
        return isinstance(val, (int, float)) and not isinstance(val, bool)
    if isinstance(default_val, str):
        return isinstance(val, str)
    if isinstance(default_val, list):
        return isinstance(val, list)
    return True


FEATURES_YAML = "features.yaml"
RUN_YAML = "run.yaml"
_RUN_SECTIONS = ("shared",)

_FEATURES_HEADER = (
    "# SensiExtractor feature config (features.yaml) — what to extract.\n"
    "# Global/shareable: the GUI's \"Reset settings\" rewrites this file, and a\n"
    "# headless run (run.py) consumes it as-is.\n"
    "# Comments are regenerated on every save; hand-added comments do not survive.\n"
)
_RUN_HEADER = (
    "# SensiExtractor run config (run.yaml) — machine/run-specific.\n"
    "# Folders, analyzer name, theme, token-file paths. Never touched by the\n"
    "# GUI's \"Reset settings\".\n"
    "# Comments are regenerated on every save; hand-added comments do not survive.\n"
)
_SECTION_COMMENTS = {
    "schema_version": "Config schema version — managed by the app; do not edit.",
    "features": "Per-feature toggles (feat_* / master_* / freq_* / sem_* / lexling_*).",
    "output": "Output artifacts: per-level CSV/.mat toggles + doc-level column groups.",
    "predictability": "Predictability roster: models (slugs), metrics, per-model context caps.",
    "semantic": "Semantic-indicator backend + category multi-selects.",
    "prosody": "Praat pitch/pause parameters + p-center k.",
    "acoustic_ts": "Acoustic time-series (TRF) parameters.",
    "phoneme_surprisal": "Phoneme surprisal / cohort entropy parameters.",
    "lexical_norms": "Lexical-norm table paths + late-AoA threshold.",
    "shared": "Machine-specific: folders, analyzer name, theme, token-file paths.",
}


def split_prefs(prefs: dict) -> tuple[dict, dict]:
    """Split a merged prefs dict into (features_data, run_data). Deep copies."""
    run = {k: deepcopy(prefs[k]) for k in _RUN_SECTIONS if k in prefs}
    features = {k: deepcopy(v) for k, v in prefs.items() if k not in _RUN_SECTIONS}
    return features, run


def join_yaml_pair(features_data: dict, run_data: dict) -> dict:
    """Combine the two YAML payloads back into one prefs-shaped user dict."""
    user = deepcopy(features_data) if isinstance(features_data, dict) else {}
    if isinstance(run_data, dict):
        for section in _RUN_SECTIONS:
            if isinstance(run_data.get(section), dict):
                user[section] = deepcopy(run_data[section])
    return user


def read_yaml(path) -> dict:
    """{} for a missing/empty file; raises yaml.YAMLError on malformed YAML."""
    p = os.path.expanduser(str(path))
    if not os.path.exists(p):
        return {}
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def _dump_with_sections(header: str, data: dict) -> str:
    parts = [header]
    for key, value in data.items():
        comment = _SECTION_COMMENTS.get(key)
        if comment:
            parts.append(f"\n# {comment}\n")
        parts.append(yaml.safe_dump({key: value}, sort_keys=False,
                                    allow_unicode=True, default_flow_style=False))
    return "".join(parts)


def _write_text(path, text: str) -> None:
    from pathlib import Path
    p = Path(os.path.expanduser(str(path)))
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def write_features_yaml(path, data: dict) -> None:
    _write_text(path, _dump_with_sections(_FEATURES_HEADER, data))


def write_run_yaml(path, data: dict) -> None:
    _write_text(path, _dump_with_sections(_RUN_HEADER, data))


def validate_features_yaml(data: dict) -> list[str]:
    """Errors for a features.yaml payload; [] means valid. Derived live from
    DEFAULT_PREFS (lazy import: preferences imports this module's writers)."""
    from utils.preferences import DEFAULT_PREFS as _defaults
    if not isinstance(data, dict):
        return ["features.yaml root must be a mapping"]
    errors: list[str] = []
    known_sections = set(_defaults) - set(_RUN_SECTIONS)
    for key in data:
        if key not in known_sections:
            errors.append(f"unknown top-level section: {key!r}")
    for section in ("features", "output"):
        valid = set(_defaults[section])
        for key, val in (data.get(section) or {}).items():
            if key not in valid:
                errors.append(f"unknown {section} toggle: {key!r}")
            elif not isinstance(val, bool):
                errors.append(f"toggle {key!r} must be true/false, got {type(val).__name__}")
    typed_sections = known_sections - {"schema_version", "features", "output", "semantic"}
    for section in typed_sections:
        default_section = _defaults[section]
        values = data.get(section)
        if not (isinstance(default_section, dict) and isinstance(values, dict)):
            continue
        for k, v in values.items():
            if k not in default_section:
                errors.append(f"unknown setting: {section}.{k}")
            elif not _type_ok(default_section[k], v):
                errors.append(
                    f"setting {section}.{k} must be {type(default_section[k]).__name__}, "
                    f"got {type(v).__name__}")
    for k in (data.get("semantic") or {}):
        if k not in _defaults["semantic"]:
            errors.append(f"unknown semantic key: {k!r}")
    return errors


def validate_run_yaml(data: dict) -> list[str]:
    from utils.preferences import DEFAULT_PREFS as _defaults
    if not isinstance(data, dict):
        return ["run.yaml root must be a mapping"]
    errors: list[str] = []
    for key in data:
        if key not in _RUN_SECTIONS:
            errors.append(f"unknown top-level section: {key!r} (expected 'shared')")
    shared_defaults = _defaults["shared"]
    for k, v in (data.get("shared") or {}).items():
        if k not in shared_defaults:
            errors.append(f"unknown shared key: {k!r}")
        elif not _type_ok(shared_defaults[k], v):
            errors.append(f"shared.{k} must be {type(shared_defaults[k]).__name__}, "
                          f"got {type(v).__name__}")
    return errors
