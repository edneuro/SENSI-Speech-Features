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
    if isinstance(default_val, dict):
        return isinstance(val, dict)
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
    "alignment": "Forced-alignment options (MFA G2P for out-of-dictionary words).",
    "shared": "Machine-specific: folders, analyzer name, theme, token-file paths.",
}


# Feature/output toggle keys that existed in a past schema version and have
# since been removed or renamed by a utils.preferences migration step.
# validate_features_yaml() tolerates these (warns, does not error) so a
# pre-migration features.yaml — which WILL contain them, since each was a
# default at some point — does not fail run.py's raw-shape validation before
# load_preferences() gets a chance to migrate it away. A genuinely unknown
# key (typo or hand-edit) is still an error.
RETIRED_FEATURE_KEYS = frozenset({
    "feat_phoneme_label",   # v20: split into 4 feat_phoneme_* column toggles
    "pred_surprisal_gpt2",  # v18: unified into predictability.models/metrics
    "pred_surprisal_llm",   # v18
    "pred_cloze",           # v18
    "pred_topmatch",        # v18
    "feat_struct_term_id",  # v14: renamed to feat_struct_token_id
})

# predictability.* settings retired the same way (section-scoped, so tracked
# separately from the features/output toggle keys above).
RETIRED_PREDICTABILITY_KEYS = frozenset({
    "llm_model",  # v18: superseded by predictability.models (a slug list)
})


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


def _replace_with_retry(src, dst, attempts: int = 10, delay_s: float = 0.05) -> None:
    """os.replace, retried briefly: on Windows another program (antivirus,
    OneDrive, an editor) holding `dst` open for an instant makes the rename
    fail with PermissionError; waiting ~0.5 s total covers that."""
    import time
    for i in range(attempts):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if i == attempts - 1:
                raise
            time.sleep(delay_s)


def _write_text(path, text: str) -> None:
    """Write `text` to `path` atomically (temp file in the same dir + os.replace).

    Prevents a crash mid-write (or a torn write) from leaving `path` truncated
    or unparseable — readers always see either the old content or the new
    content, never a partial file.
    """
    from pathlib import Path
    import tempfile
    p = Path(os.path.expanduser(str(path)))
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=p.name + ".", suffix=".tmp", dir=str(p.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        _replace_with_retry(tmp_name, p)
    except BaseException:
        try:
            os.remove(tmp_name)
        except OSError:
            pass
        raise


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
        raw = data.get(section)
        if raw is not None and not isinstance(raw, dict):
            errors.append(f"section {section!r} must be a mapping")
            continue
        valid = set(_defaults[section])
        for key, val in (raw or {}).items():
            if key in RETIRED_FEATURE_KEYS:
                continue  # tolerated; see retired_keys_present() for the warning
            if key not in valid:
                errors.append(f"unknown {section} toggle: {key!r}")
            elif not isinstance(val, bool):
                errors.append(f"toggle {key!r} must be true/false, got {type(val).__name__}")
    typed_sections = known_sections - {"schema_version", "features", "output", "semantic"}
    for section in typed_sections:
        default_section = _defaults[section]
        values = data.get(section)
        if values is not None and not isinstance(values, dict):
            errors.append(f"section {section!r} must be a mapping")
            continue
        if not (isinstance(default_section, dict) and isinstance(values, dict)):
            continue
        for k, v in values.items():
            if section == "predictability" and k in RETIRED_PREDICTABILITY_KEYS:
                continue  # tolerated; see retired_keys_present() for the warning
            if k not in default_section:
                errors.append(f"unknown setting: {section}.{k}")
            elif not _type_ok(default_section[k], v):
                errors.append(
                    f"setting {section}.{k} must be {type(default_section[k]).__name__}, "
                    f"got {type(v).__name__}")
    sem_raw = data.get("semantic")
    if sem_raw is not None and not isinstance(sem_raw, dict):
        errors.append("section 'semantic' must be a mapping")
    else:
        for k in (sem_raw or {}):
            if k not in _defaults["semantic"]:
                errors.append(f"unknown semantic key: {k!r}")
    from utils.config_checks import value_errors
    errors.extend(value_errors(data))
    return errors


def retired_keys_present(data: dict) -> list[str]:
    """Non-fatal warning strings for retired keys found in a features.yaml
    payload — call this only after validate_features_yaml(data) returns [].

    A retired key is not a validation error (see RETIRED_FEATURE_KEYS /
    RETIRED_PREDICTABILITY_KEYS above): it means the file predates a
    utils.preferences migration and load_preferences() will drop it on the
    next save. Surfacing it as a warning still tells the user their config
    is stale, without blocking a headless run.py the way an error would.
    """
    warnings: list[str] = []
    if not isinstance(data, dict):
        return warnings
    for section in ("features", "output"):
        raw = data.get(section)
        if isinstance(raw, dict):
            for key in raw:
                if key in RETIRED_FEATURE_KEYS:
                    warnings.append(
                        f"retired toggle {key!r} in {section!r} — migrated "
                        "away; it will be dropped from features.yaml on the "
                        "next save")
    pred = data.get("predictability")
    if isinstance(pred, dict):
        for key in pred:
            if key in RETIRED_PREDICTABILITY_KEYS:
                warnings.append(
                    f"retired setting 'predictability.{key}' — migrated "
                    "away; it will be dropped from features.yaml on the "
                    "next save")
    return warnings


def validate_run_yaml(data: dict) -> list[str]:
    from utils.preferences import DEFAULT_PREFS as _defaults
    if not isinstance(data, dict):
        return ["run.yaml root must be a mapping"]
    errors: list[str] = []
    for key in data:
        if key not in _RUN_SECTIONS:
            errors.append(f"unknown top-level section: {key!r} (expected 'shared')")
    shared_raw = data.get("shared")
    if shared_raw is not None and not isinstance(shared_raw, dict):
        errors.append("section 'shared' must be a mapping")
    else:
        shared_defaults = _defaults["shared"]
        for k, v in (shared_raw or {}).items():
            if k not in shared_defaults:
                errors.append(f"unknown shared key: {k!r}")
            elif not _type_ok(shared_defaults[k], v):
                errors.append(f"shared.{k} must be {type(shared_defaults[k]).__name__}, "
                              f"got {type(v).__name__}")
    return errors
