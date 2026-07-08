"""Portable YAML config <-> prefs mapping for the Sherlock run workflow.

Projects the GUI's nested prefs dict (utils.preferences.DEFAULT_PREFS, active
schema) into a flat, human-editable YAML config and back. Schema-agnostic: valid
keys and settings sections are derived from the live DEFAULT_PREFS at call time,
never hardcoded, so the mapping survives schema changes.
"""
from __future__ import annotations

from copy import deepcopy

from utils.preferences import DEFAULT_PREFS

# Prefs top-level sections with dedicated handling; every other section is
# treated as a "settings" block in the YAML.
_NON_SETTINGS = frozenset({"schema_version", "shared", "features", "output", "semantic"})


def _settings_sections() -> list[str]:
    """Prefs sections that map into the YAML `settings:` block (derived live)."""
    return [k for k in DEFAULT_PREFS if k not in _NON_SETTINGS]


def prefs_to_config(prefs: dict) -> dict:
    """Project a prefs dict into the portable config shape.

    features + output booleans -> flat `features`; predictability/prosody/... ->
    `settings.<section>`; semantic -> `semantic_categories`. `shared` folders and
    any secret are intentionally NOT copied (they are Sherlock-specific).
    """
    features: dict = {}
    features.update(prefs.get("features", {}))
    features.update(prefs.get("output", {}))
    settings = {s: deepcopy(prefs[s]) for s in _settings_sections() if s in prefs}
    return {
        "run": {
            "transcripts_folder": "",
            "audio_folder": "",
            "output_folder": "",
            "analyzer": "",
            "hf_token_file": "~/hf_token.txt",
            "openai_api_key_file": "~/openai_key.txt",
        },
        "features": features,
        "settings": settings,
        "semantic_categories": deepcopy(prefs.get("semantic", {})),
    }


def config_to_prefs(config: dict) -> dict:
    """Merge a config dict onto a fresh DEFAULT_PREFS copy -> prefs dict.

    `features` keys route to prefs['features']/['output'] by membership; settings
    sections and semantic categories overlay their prefs sections; `run` folders
    map into prefs['shared']. Unknown keys are ignored here (validate_config
    reports them).
    """
    prefs = deepcopy(DEFAULT_PREFS)
    feat_keys = set(prefs["features"])
    out_keys = set(prefs["output"])
    for key, val in (config.get("features") or {}).items():
        if key in feat_keys:
            prefs["features"][key] = val
        elif key in out_keys:
            prefs["output"][key] = val
    for section, values in (config.get("settings") or {}).items():
        if isinstance(prefs.get(section), dict) and isinstance(values, dict):
            prefs[section].update(values)
    sem = config.get("semantic_categories") or {}
    if isinstance(sem, dict):
        prefs["semantic"].update(sem)
    run = config.get("run") or {}
    shared = prefs["shared"]
    shared["last_transcripts_folder"] = run.get("transcripts_folder") or ""
    shared["last_audio_folder"] = run.get("audio_folder") or ""
    shared["last_output_folder"] = run.get("output_folder") or ""
    shared["analyzer_name"] = run.get("analyzer") or ""
    return prefs
