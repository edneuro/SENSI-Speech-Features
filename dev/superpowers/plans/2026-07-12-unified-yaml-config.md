# Unified YAML config (Project C) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `config/preferences.json` with two YAML files (`config/features.yaml` global/reset-able + `config/run.yaml` machine-specific) consumed identically by GUI, CLI, and headless runs; promote the headless runner to a top-level `run.py`; retire `sherlock/export_config.py` + `sherlock/config.yaml`.

**Architecture:** `utils/preferences.py` keeps its public API (`load_preferences` / `save_preferences` / `reset_to_defaults`, same merged-dict shape) but stores to the YAML pair, with a one-time legacy-json import through the EXISTING v1→v19 migration chain. `utils/config_yaml.py` is repurposed from the flat sherlock-config mapper into the YAML-pair I/O layer (split/join/read/write/validate). `run.py` is `sherlock/run_extract.py` reading the pair via the same loader.

**Tech Stack:** Python 3, pyyaml 6.x (already a dependency), pytest. All commands below assume `conda activate ./env` from the project root (path-based conda env — see CLAUDE.md).

**Spec (user-approved decisions, do not re-litigate):** `docs/superpowers/specs/2026-07-12-unified-yaml-config-design.md`

## Global Constraints

- Work on a feature branch `yaml-config` off `main` (spec handoff note; Projects A/B precedent). One commit per task.
- **No prefs schema version bump** — keys/shapes stay 1:1 with the v19 schema; the file split happens at the storage layer (spec §6). The single additive key `shared.hf_token_file` (default `""`) is filled by the existing deep-merge, no migration step.
- Public API signatures unchanged: `load_preferences() -> dict`, `save_preferences(updates: dict) -> None`, `reset_to_defaults(*, keep_shared: bool = True) -> dict`. GUI/planner/extractors/main.py keep consuming the same merged dict (spec §3).
- Env overrides: new `SENSI_CONFIG_DIR` (dir holding both YAMLs); `SENSI_PREFS_PATH` kept as the legacy-json path override. Fallback rule (plan decision): when `SENSI_CONFIG_DIR` is unset but `SENSI_PREFS_PATH` is set, the config dir is `Path(SENSI_PREFS_PATH).parent` — this keeps the ~40 existing per-test `setenv("SENSI_PREFS_PATH", tmp…)` isolation sites hermetic without editing them all.
- Hermeticity rule (plan decision): the legacy json is looked up at `SENSI_PREFS_PATH` if set, else **`get_config_dir() / "preferences.json"`** — never at the project path when an override is active. Otherwise a test isolated only by `SENSI_CONFIG_DIR` would read (and rename to `.bak`!) the developer's real `config/preferences.json` during the one-time import. Production behavior (no env vars) is unchanged.
- `config/features.yaml` holds: `schema_version`, `features`, `output`, `predictability`, `semantic`, `prosody`, `acoustic_ts`, `phoneme_surprisal`, `lexical_norms`. `config/run.yaml` holds exactly the `shared` section (under a top-level `shared:` key — 1:1 with the prefs schema): `analyzer_name`, `last_*_folder` ×3, `theme`, and the new `hf_token_file`.
- GUI "Reset settings" rewrites ONLY `features.yaml`; `run.yaml` is never touched by reset (spec §2.2).
- YAML files are written with regenerated header + per-section comments; hand-added comments do not survive a programmatic save — documented behavior (spec §4).
- TDD for the utils layer (loader/saver/split/join/migration); test-after for `run.py` and docs (spec §5 + memory `feedback-tdd-scope`).
- Import-cycle rule: `utils/config_yaml.py` must NOT import `utils.preferences` at module level (preferences imports config_yaml's writers at module level). Its validators import `DEFAULT_PREFS` inside the function body.
- Suite baseline at handoff: **728 passed**. Full suite green at the end of every task (`python -m pytest`). Task 6 deletes ~13 legacy tests and adds ports, so the final count will differ — green is the requirement, not the count.

---

### Task 1: YAML-pair I/O layer in `utils/config_yaml.py`

**Files:**
- Modify: `utils/config_yaml.py` (ADD new functions; do NOT remove the existing flat-config functions yet — `sherlock/run_extract.py`, `sherlock/export_config.py`, and their tests still import them until Task 6)
- Test: `tests/utils/test_config_yaml_pair.py` (new file; the old `tests/utils/test_config_yaml.py` keeps guarding the old functions until Task 6)

**Interfaces:**
- Consumes: `utils.preferences.DEFAULT_PREFS` (lazily, inside validators only).
- Produces (used by Tasks 3–6):
  - `FEATURES_YAML = "features.yaml"`, `RUN_YAML = "run.yaml"` (module constants)
  - `split_prefs(prefs: dict) -> tuple[dict, dict]` — `(features_data, run_data)`
  - `join_yaml_pair(features_data: dict, run_data: dict) -> dict` — combined user dict in the prefs shape
  - `read_yaml(path) -> dict` — `{}` if missing/empty; raises `yaml.YAMLError` if malformed
  - `write_features_yaml(path, data) -> None`, `write_run_yaml(path, data) -> None` — create parent dirs, write header + per-section comments
  - `validate_features_yaml(data: dict) -> list[str]`, `validate_run_yaml(data: dict) -> list[str]` — human-readable errors, empty list = valid

- [ ] **Step 1: Write the failing tests** (`tests/utils/test_config_yaml_pair.py`)

```python
import pytest
import yaml
from copy import deepcopy
from utils.preferences import DEFAULT_PREFS
from utils.config_yaml import (
    FEATURES_YAML, RUN_YAML, split_prefs, join_yaml_pair, read_yaml,
    write_features_yaml, write_run_yaml,
    validate_features_yaml, validate_run_yaml,
)


def test_split_prefs_routes_shared_to_run_and_rest_to_features():
    features, run = split_prefs(DEFAULT_PREFS)
    assert set(run) == {"shared"}
    assert run["shared"] == DEFAULT_PREFS["shared"]
    assert "shared" not in features
    assert features["schema_version"] == DEFAULT_PREFS["schema_version"]
    # every non-shared section lands in features.yaml
    assert set(features) == set(DEFAULT_PREFS) - {"shared"}


def test_split_prefs_returns_copies():
    features, run = split_prefs(DEFAULT_PREFS)
    run["shared"]["analyzer_name"] = "mutated"
    features["features"]["master_structural"] = "mutated"
    assert DEFAULT_PREFS["shared"]["analyzer_name"] != "mutated"
    assert DEFAULT_PREFS["features"]["master_structural"] != "mutated"


def test_join_yaml_pair_round_trips_defaults():
    features, run = split_prefs(DEFAULT_PREFS)
    assert join_yaml_pair(features, run) == DEFAULT_PREFS


def test_join_yaml_pair_tolerates_missing_parts():
    features, run = split_prefs(DEFAULT_PREFS)
    assert "shared" not in join_yaml_pair(features, {})
    only_run = join_yaml_pair({}, run)
    assert only_run == {"shared": DEFAULT_PREFS["shared"]}


def test_write_read_round_trip(tmp_path):
    features, run = split_prefs(DEFAULT_PREFS)
    fp, rp = tmp_path / FEATURES_YAML, tmp_path / RUN_YAML
    write_features_yaml(fp, features)
    write_run_yaml(rp, run)
    assert read_yaml(fp) == features
    assert read_yaml(rp) == run


def test_written_files_have_regenerated_comments(tmp_path):
    features, run = split_prefs(DEFAULT_PREFS)
    fp, rp = tmp_path / FEATURES_YAML, tmp_path / RUN_YAML
    write_features_yaml(fp, features)
    write_run_yaml(rp, run)
    ftext = fp.read_text(encoding="utf-8")
    rtext = rp.read_text(encoding="utf-8")
    assert ftext.startswith("#") and rtext.startswith("#")
    # a per-section comment precedes each top-level section
    assert "\n# " in ftext and "predictability" in ftext
    assert "Reset settings" in ftext  # reset-able is documented in the header
    assert "Never touched" in rtext


def test_write_creates_parent_dirs(tmp_path):
    nested = tmp_path / "a" / "b" / FEATURES_YAML
    features, _ = split_prefs(DEFAULT_PREFS)
    write_features_yaml(nested, features)
    assert read_yaml(nested) == features


def test_read_yaml_missing_or_empty_returns_empty_dict(tmp_path):
    assert read_yaml(tmp_path / "nope.yaml") == {}
    p = tmp_path / "empty.yaml"
    p.write_text("", encoding="utf-8")
    assert read_yaml(p) == {}


def test_read_yaml_malformed_raises(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("features: [unclosed", encoding="utf-8")
    with pytest.raises(yaml.YAMLError):
        read_yaml(p)


def test_validate_features_yaml_accepts_default_split():
    features, _ = split_prefs(DEFAULT_PREFS)
    assert validate_features_yaml(features) == []


def test_validate_features_yaml_flags_unknown_and_bad_type():
    features, _ = split_prefs(DEFAULT_PREFS)
    features["features"]["not_a_real_toggle"] = True
    key = next(iter(DEFAULT_PREFS["features"]))
    features["output"]["word_level_csv"] = "yes"          # not a bool
    features["prosody"]["not_a_setting"] = 1
    features["bogus_section"] = {}
    errs = validate_features_yaml(features)
    assert any("not_a_real_toggle" in e for e in errs)
    assert any("word_level_csv" in e and "true/false" in e for e in errs)
    assert any("prosody.not_a_setting" in e for e in errs)
    assert any("bogus_section" in e for e in errs)


def test_validate_run_yaml_accepts_default_and_flags_unknown():
    _, run = split_prefs(DEFAULT_PREFS)
    assert validate_run_yaml(run) == []
    run["shared"]["bogus_key"] = "x"
    run["not_shared"] = {}
    errs = validate_run_yaml(run)
    assert any("bogus_key" in e for e in errs)
    assert any("not_shared" in e for e in errs)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/utils/test_config_yaml_pair.py -v`
Expected: FAIL — `ImportError: cannot import name 'FEATURES_YAML'`

- [ ] **Step 3: Implement in `utils/config_yaml.py`**

Add below the existing code (keep the old functions untouched). Note the module currently does `from utils.preferences import DEFAULT_PREFS` at module level — that stays for now (the old functions use it); Task 6 removes it. The NEW validators must still do their own lazy import so they survive Task 6's cleanup:

```python
# ---------------------------------------------------------------------------
# Two-file YAML store (features.yaml + run.yaml) — Project C.
# features.yaml = what to extract (global, reset-able); run.yaml = machine-
# specific (the prefs `shared` section). utils.preferences reads/writes prefs
# through these helpers; run.py validates raw files with the validators.
# ---------------------------------------------------------------------------

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
```

(The existing `_type_ok`, `os`, `yaml`, `deepcopy` imports are already in the module.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/utils/test_config_yaml_pair.py tests/utils/test_config_yaml.py -v`
Expected: all PASS (old flat-config tests still green too).

Note: `test_validate_features_yaml_accepts_default_split` requires `shared.hf_token_file` to NOT be in features data — trivially true (it's in `shared`). But `validate_run_yaml`'s default-accept test will only pass once `hf_token_file` exists in `DEFAULT_PREFS["shared"]` **if the test injects it** — it doesn't; it validates the split of the CURRENT defaults, so it is green both before and after Task 3 adds the key.

- [ ] **Step 5: Commit**

```bash
git add utils/config_yaml.py tests/utils/test_config_yaml_pair.py
git commit -m "feat(config): YAML-pair I/O layer (split/join/read/write/validate)"
```

---

### Task 2: Extract the migration chain into a pure function

**Files:**
- Modify: `utils/preferences.py` (the `load_preferences` body, lines ~883–1102)
- Test: existing `tests/utils/test_preferences_migration.py` + `tests/utils/test_preferences.py` are the regression guard — no new tests.

**Interfaces:**
- Produces: `_merged_from_user(user: dict) -> tuple[dict, bool]` — pure (no disk I/O): runs the full v1→v19 migration/merge/finalize logic on an already-parsed user dict against a fresh `deepcopy(DEFAULT_PREFS)`; returns `(merged, changed)` where `changed` is True iff the source `schema_version` differed from `SCHEMA_VERSION` (i.e. the caller should persist).

- [ ] **Step 1: Refactor**

Move everything in `load_preferences` from `version = user.get('schema_version', 1)` (line ~898) through the final version-stamp block (line ~1100) into `_merged_from_user(user)`. Inside it:
- Start with `merged: dict[str, Any] = deepcopy(DEFAULT_PREFS)` (moved from the old body).
- Delete every interior `_write(path, merged)` call (11 sites — one per version branch; they were all overwritten by the final write anyway).
- Replace the tail
  ```python
  if merged.get("schema_version") != SCHEMA_VERSION:
      merged["schema_version"] = SCHEMA_VERSION
      _write(path, merged)
  return merged
  ```
  with
  ```python
  changed = merged.get("schema_version") != SCHEMA_VERSION
  merged["schema_version"] = SCHEMA_VERSION
  return merged, changed
  ```
  (Unconditional stamp is safe: when nothing changed the stamp is a no-op.)

`load_preferences` becomes (json behavior unchanged for this task):

```python
def load_preferences() -> dict[str, Any]:
    """Load preferences; auto-migrate from any earlier schema version if needed."""
    path = get_preferences_path()
    if not path.exists():
        return deepcopy(DEFAULT_PREFS)
    try:
        with path.open('r', encoding='utf-8') as f:
            user = json.load(f)
    except (json.JSONDecodeError, OSError):
        return deepcopy(DEFAULT_PREFS)
    if not isinstance(user, dict):
        return deepcopy(DEFAULT_PREFS)
    merged, changed = _merged_from_user(user)
    if changed:
        _write(path, merged)
    return merged
```

- [ ] **Step 2: Run the full suite**

Run: `python -m pytest`
Expected: 728 passed, 2 skipped (identical to baseline — pure refactor).

- [ ] **Step 3: Commit**

```bash
git add utils/preferences.py
git commit -m "refactor(prefs): extract pure _merged_from_user from load_preferences"
```

---

### Task 3: YAML-first load/save in `utils/preferences.py`

**Files:**
- Modify: `utils/preferences.py`
- Test: `tests/utils/test_preferences_yaml.py` (new)

**Interfaces:**
- Consumes (Task 1): `split_prefs`, `join_yaml_pair`, `read_yaml`, `write_features_yaml`, `write_run_yaml`, `FEATURES_YAML`, `RUN_YAML` from `utils.config_yaml` (module-level import is safe — config_yaml never imports preferences at module level after Task 6; until then the old module-level `from utils.preferences import DEFAULT_PREFS` in config_yaml makes the import order matter: **import config_yaml lazily inside the preferences functions for Tasks 3–5**, then Task 6 lifts it to module level when the cycle is gone. Each function below does `from utils.config_yaml import …` in its body.)
- Produces (used by Tasks 4–6 and run.py):
  - `get_config_dir() -> Path`
  - `get_features_yaml_path() -> Path`, `get_run_yaml_path() -> Path`
  - `load_preferences()` — YAML-first (legacy-json import is Task 4)
  - `save_preferences(updates)` — routes writes by section
  - New default `DEFAULT_PREFS["shared"]["hf_token_file"] == ""`

- [ ] **Step 1: Write the failing tests** (`tests/utils/test_preferences_yaml.py`)

```python
"""Two-file YAML store behavior: paths/env precedence, save routing,
round-trips, tolerance. Legacy-json import is covered in
test_preferences_migration.py; reset in test_preferences.py."""
import yaml
from copy import deepcopy
from pathlib import Path

from utils.preferences import (
    DEFAULT_PREFS, SCHEMA_VERSION, get_config_dir,
    get_features_yaml_path, get_run_yaml_path,
    load_preferences, save_preferences,
)
from utils.config_yaml import (
    FEATURES_YAML, RUN_YAML, read_yaml, split_prefs,
    write_features_yaml, write_run_yaml,
)


def test_config_dir_env_precedence(tmp_path, monkeypatch):
    a, b = tmp_path / "a", tmp_path / "b"
    monkeypatch.setenv("SENSI_CONFIG_DIR", str(a))
    monkeypatch.setenv("SENSI_PREFS_PATH", str(b / "p.json"))
    assert get_config_dir() == a                       # CONFIG_DIR wins
    monkeypatch.delenv("SENSI_CONFIG_DIR")
    assert get_config_dir() == b                       # falls back to PREFS_PATH parent
    monkeypatch.delenv("SENSI_PREFS_PATH")
    assert get_config_dir().name == "config"           # project default
    assert get_features_yaml_path().name == FEATURES_YAML
    assert get_run_yaml_path().name == RUN_YAML


def test_legacy_json_path_follows_config_dir(tmp_path, monkeypatch):
    # Hermeticity: with only SENSI_CONFIG_DIR set, the legacy json is looked
    # up INSIDE that dir — never at the repo's real config/preferences.json.
    from utils.preferences import get_preferences_path
    monkeypatch.delenv("SENSI_PREFS_PATH", raising=False)
    monkeypatch.setenv("SENSI_CONFIG_DIR", str(tmp_path))
    assert get_preferences_path() == tmp_path / "preferences.json"


def test_load_returns_defaults_when_no_files(tmp_path, monkeypatch):
    monkeypatch.setenv("SENSI_CONFIG_DIR", str(tmp_path))
    prefs = load_preferences()
    assert prefs == DEFAULT_PREFS
    assert not (tmp_path / FEATURES_YAML).exists()     # load never creates files


def test_default_shared_has_hf_token_file():
    assert DEFAULT_PREFS["shared"]["hf_token_file"] == ""


def test_save_then_load_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("SENSI_CONFIG_DIR", str(tmp_path))
    save_preferences({"features": {"master_structural": False},
                      "shared": {"analyzer_name": "Alice"}})
    prefs = load_preferences()
    assert prefs["features"]["master_structural"] is False
    assert prefs["shared"]["analyzer_name"] == "Alice"
    # sections landed in the right files
    fdata = read_yaml(tmp_path / FEATURES_YAML)
    rdata = read_yaml(tmp_path / RUN_YAML)
    assert fdata["features"]["master_structural"] is False
    assert fdata["schema_version"] == SCHEMA_VERSION
    assert "shared" not in fdata
    assert rdata["shared"]["analyzer_name"] == "Alice"
    assert set(rdata) == {"shared"}


def test_first_save_creates_both_files(tmp_path, monkeypatch):
    monkeypatch.setenv("SENSI_CONFIG_DIR", str(tmp_path))
    save_preferences({"features": {"master_structural": False}})
    assert (tmp_path / FEATURES_YAML).exists()
    assert (tmp_path / RUN_YAML).exists()              # pair completed on first save


def test_save_routes_writes_by_section(tmp_path, monkeypatch):
    monkeypatch.setenv("SENSI_CONFIG_DIR", str(tmp_path))
    save_preferences({"features": {"master_structural": False}})
    features_before = (tmp_path / FEATURES_YAML).read_text(encoding="utf-8")
    save_preferences({"shared": {"analyzer_name": "Bob"}})
    # a shared-only save must not rewrite features.yaml
    assert (tmp_path / FEATURES_YAML).read_text(encoding="utf-8") == features_before
    assert read_yaml(tmp_path / RUN_YAML)["shared"]["analyzer_name"] == "Bob"


def test_yaml_pair_takes_priority_over_legacy_json(tmp_path, monkeypatch):
    import json
    monkeypatch.setenv("SENSI_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("SENSI_PREFS_PATH", str(tmp_path / "preferences.json"))
    (tmp_path / "preferences.json").write_text(
        json.dumps({"schema_version": SCHEMA_VERSION,
                    "shared": {"analyzer_name": "FromJson"}}), encoding="utf-8")
    features, run = split_prefs(deepcopy(DEFAULT_PREFS))
    run["shared"]["analyzer_name"] = "FromYaml"
    write_features_yaml(tmp_path / FEATURES_YAML, features)
    write_run_yaml(tmp_path / RUN_YAML, run)
    assert load_preferences()["shared"]["analyzer_name"] == "FromYaml"
    assert (tmp_path / "preferences.json").exists()    # json untouched when pair exists


def test_missing_schema_version_in_yaml_assumes_current(tmp_path, monkeypatch):
    # A hand-made features.yaml without schema_version must NOT be run through
    # the v1 flat-file migration (which would mangle its sections).
    monkeypatch.setenv("SENSI_CONFIG_DIR", str(tmp_path))
    features, run = split_prefs(deepcopy(DEFAULT_PREFS))
    features.pop("schema_version")
    features["features"]["master_structural"] = False
    write_features_yaml(tmp_path / FEATURES_YAML, features)
    write_run_yaml(tmp_path / RUN_YAML, run)
    prefs = load_preferences()
    assert prefs["features"]["master_structural"] is False
    assert prefs["schema_version"] == SCHEMA_VERSION


def test_corrupted_features_yaml_falls_back_to_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("SENSI_CONFIG_DIR", str(tmp_path))
    (tmp_path / FEATURES_YAML).write_text("features: [unclosed", encoding="utf-8")
    prefs = load_preferences()
    assert prefs["features"] == DEFAULT_PREFS["features"]
    # load never overwrites a corrupted file
    assert (tmp_path / FEATURES_YAML).read_text(encoding="utf-8") == "features: [unclosed"


def test_unknown_keys_survive_save_load(tmp_path, monkeypatch):
    # forward-compat: same tolerance the json store had
    monkeypatch.setenv("SENSI_CONFIG_DIR", str(tmp_path))
    features, run = split_prefs(deepcopy(DEFAULT_PREFS))
    features["future_section"] = {"x": 1}
    write_features_yaml(tmp_path / FEATURES_YAML, features)
    write_run_yaml(tmp_path / RUN_YAML, run)
    save_preferences({"features": {"master_structural": False}})
    assert read_yaml(tmp_path / FEATURES_YAML)["future_section"] == {"x": 1}
```

- [ ] **Step 2: Run to verify failures**

Run: `python -m pytest tests/utils/test_preferences_yaml.py -v`
Expected: FAIL — `ImportError: cannot import name 'get_config_dir'`

- [ ] **Step 3: Implement in `utils/preferences.py`**

(a) Add the new shared default to the ACTIVE defaults only (v19 literal; additive, filled by deep-merge for existing stores — no version bump per spec §6):

```python
DEFAULT_PREFS_V19: dict[str, Any] = {
    **DEFAULT_PREFS_V18,
    "schema_version": SCHEMA_VERSION_V19,
    # hf_token_file added 2026-07-12 (Project C, run.yaml): path to a file
    # holding a HuggingFace token for headless runs. Additive — no bump.
    "shared": {**DEFAULT_PREFS_V18["shared"], "hf_token_file": ""},
    "predictability": {
        **DEFAULT_PREFS_V18["predictability"],
        "context": {},
    },
}
```

(b) Path helpers. `get_preferences_path()` is now the LEGACY json path and must follow the config dir (see the hermeticity rule in Global Constraints):

```python
def get_preferences_path() -> Path:
    """LEGACY json path — only read for the one-time import to the YAML pair."""
    override = os.environ.get('SENSI_PREFS_PATH')
    if override:
        return Path(override)
    return get_config_dir() / 'preferences.json'


def get_config_dir() -> Path:
    """Directory holding features.yaml + run.yaml.

    Precedence: SENSI_CONFIG_DIR > parent of SENSI_PREFS_PATH (so the many
    existing test isolation sites that only set the legacy var stay hermetic)
    > <project_root>/config.
    """
    override = os.environ.get('SENSI_CONFIG_DIR')
    if override:
        return Path(override)
    legacy = os.environ.get('SENSI_PREFS_PATH')
    if legacy:
        return Path(legacy).parent
    return Path(__file__).resolve().parent.parent / 'config'


def get_features_yaml_path() -> Path:
    from utils.config_yaml import FEATURES_YAML
    return get_config_dir() / FEATURES_YAML


def get_run_yaml_path() -> Path:
    from utils.config_yaml import RUN_YAML
    return get_config_dir() / RUN_YAML
```

(c) Replace `load_preferences` / `save_preferences` (json branch shown here returns defaults — Task 4 fills in the real legacy import; keep a `# TODO(Task 4)` marker only until then):

```python
def _read_yaml_tolerant(path: Path) -> dict[str, Any]:
    import yaml as _yaml
    from utils.config_yaml import read_yaml
    try:
        return read_yaml(path)
    except (_yaml.YAMLError, OSError):
        return {}


def _write_yaml_pair(merged: dict[str, Any]) -> None:
    from utils.config_yaml import split_prefs, write_features_yaml, write_run_yaml
    features_data, run_data = split_prefs(merged)
    write_features_yaml(get_features_yaml_path(), features_data)
    write_run_yaml(get_run_yaml_path(), run_data)


def load_preferences() -> dict[str, Any]:
    """Load preferences from the YAML pair; one-time import from legacy json."""
    from utils.config_yaml import join_yaml_pair
    f_path, r_path = get_features_yaml_path(), get_run_yaml_path()
    if f_path.exists() or r_path.exists():
        user = join_yaml_pair(_read_yaml_tolerant(f_path), _read_yaml_tolerant(r_path))
        # The YAML store began at v19; a missing/hand-stripped schema_version
        # must not fall into the v1 flat-json migration.
        user.setdefault('schema_version', SCHEMA_VERSION)
        merged, changed = _merged_from_user(user)
        if changed:
            _write_yaml_pair(merged)
        return merged
    return _load_legacy_json()


def _load_legacy_json() -> dict[str, Any]:
    # Task 4 replaces this stub with the one-time json import.
    return deepcopy(DEFAULT_PREFS)


def save_preferences(updates: dict[str, Any]) -> None:
    """Merge `updates` into current prefs; write only the touched YAML file(s)."""
    from utils.config_yaml import split_prefs, write_features_yaml, write_run_yaml
    current = load_preferences()
    _deep_merge_current(current, updates)
    current['schema_version'] = SCHEMA_VERSION
    features_data, run_data = split_prefs(current)
    f_path, r_path = get_features_yaml_path(), get_run_yaml_path()
    touched_run = 'shared' in updates
    touched_features = any(k != 'shared' for k in updates)
    if touched_features or not f_path.exists():
        write_features_yaml(f_path, features_data)
    if touched_run or not r_path.exists():
        write_run_yaml(r_path, run_data)
```

Keep `_write` (json) — the legacy machinery and Task 4 still use the json reader; update the module docstring's "Default location" note to describe the two-file store + both env vars.

- [ ] **Step 4: Run the new tests, then the full suite**

Run: `python -m pytest tests/utils/test_preferences_yaml.py -v` → all PASS.
Run: `python -m pytest` → expect FAILURES ONLY in tests that assert json-on-disk after `load_preferences`/`save_preferences` (the ~8 sites listed in Task 5) and the two `reset_to_defaults` tests. **Do not fix them here** — record the failing list for Task 5; everything else must be green. If anything OUTSIDE that set fails, stop and investigate before committing.

- [ ] **Step 5: Commit**

```bash
git add utils/preferences.py tests/utils/test_preferences_yaml.py
git commit -m "feat(prefs): two-file YAML store (features.yaml + run.yaml) behind the same API"
```

---

### Task 4: One-time legacy-json import (+ `.bak`)

**Files:**
- Modify: `utils/preferences.py` (`_load_legacy_json`)
- Test: append to `tests/utils/test_preferences_yaml.py`

**Interfaces:**
- Consumes: `_merged_from_user` (Task 2), `_write_yaml_pair` (Task 3), `get_preferences_path()`.
- Produces: on first load with only a legacy json present — YAML pair written, json renamed `preferences.json.bak`, migrated dict returned.

- [ ] **Step 1: Write the failing tests** (append to `tests/utils/test_preferences_yaml.py`)

```python
def test_legacy_json_imports_once_and_renames_to_bak(tmp_path, monkeypatch):
    import json
    monkeypatch.delenv("SENSI_CONFIG_DIR", raising=False)
    p = tmp_path / "preferences.json"
    monkeypatch.setenv("SENSI_PREFS_PATH", str(p))
    legacy = deepcopy(DEFAULT_PREFS)
    legacy["shared"]["analyzer_name"] = "Legacy"
    legacy["features"]["master_structural"] = False
    p.write_text(json.dumps(legacy), encoding="utf-8")

    prefs = load_preferences()
    assert prefs["shared"]["analyzer_name"] == "Legacy"
    assert prefs["features"]["master_structural"] is False
    assert not p.exists()
    assert (tmp_path / "preferences.json.bak").exists()
    assert (tmp_path / FEATURES_YAML).exists() and (tmp_path / RUN_YAML).exists()

    # second load reads the YAMLs, not the .bak: mutate features.yaml, reload
    fdata = read_yaml(tmp_path / FEATURES_YAML)
    fdata["features"]["master_structural"] = True
    write_features_yaml(tmp_path / FEATURES_YAML, fdata)
    assert load_preferences()["features"]["master_structural"] is True


def test_legacy_v2_json_runs_full_migration_chain(tmp_path, monkeypatch):
    import json
    monkeypatch.delenv("SENSI_CONFIG_DIR", raising=False)
    p = tmp_path / "preferences.json"
    monkeypatch.setenv("SENSI_PREFS_PATH", str(p))
    v2 = {"schema_version": 2,
          "shared": {"analyzer_name": "OldTimer", "dictionary_backend": "liwc"},
          "word_level": {"feature_toggles": {"zipf_freq": False}}}
    p.write_text(json.dumps(v2), encoding="utf-8")
    prefs = load_preferences()
    assert prefs["schema_version"] == SCHEMA_VERSION
    assert prefs["shared"]["analyzer_name"] == "OldTimer"
    assert prefs["semantic"]["backend"] == "empath"      # v11 liwc coercion
    assert prefs["features"]["freq_subtlex"] is False    # v2→v3 toggle mapping
    assert (tmp_path / "preferences.json.bak").exists()
    on_disk = read_yaml(tmp_path / FEATURES_YAML)
    assert on_disk["schema_version"] == SCHEMA_VERSION


def test_corrupted_legacy_json_returns_defaults_without_bak(tmp_path, monkeypatch):
    monkeypatch.delenv("SENSI_CONFIG_DIR", raising=False)
    p = tmp_path / "preferences.json"
    monkeypatch.setenv("SENSI_PREFS_PATH", str(p))
    p.write_text("{not json", encoding="utf-8")
    assert load_preferences() == DEFAULT_PREFS
    assert p.exists()                                   # left in place for inspection
    assert not (tmp_path / "preferences.json.bak").exists()
    assert not (tmp_path / FEATURES_YAML).exists()
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/utils/test_preferences_yaml.py -v`
Expected: the three new tests FAIL (stub returns defaults, no files written); prior tests PASS.

- [ ] **Step 3: Implement `_load_legacy_json`**

```python
def _load_legacy_json() -> dict[str, Any]:
    """One-time import: legacy preferences.json → migration chain → YAML pair.

    The original json is renamed to preferences.json.bak (pristine — the
    migration runs in memory), so the user can inspect what was imported.
    A corrupted json is left in place and defaults are returned, matching the
    old loader's tolerance.
    """
    path = get_preferences_path()
    if not path.exists():
        return deepcopy(DEFAULT_PREFS)
    try:
        with path.open('r', encoding='utf-8') as f:
            user = json.load(f)
    except (json.JSONDecodeError, OSError):
        return deepcopy(DEFAULT_PREFS)
    if not isinstance(user, dict):
        return deepcopy(DEFAULT_PREFS)
    merged, _ = _merged_from_user(user)
    _write_yaml_pair(merged)
    try:
        os.replace(path, path.with_name(path.name + '.bak'))
    except OSError:
        pass  # YAML pair now exists and takes priority; a stuck json is inert
    return merged
```

- [ ] **Step 4: Run the file, then the suite**

Run: `python -m pytest tests/utils/test_preferences_yaml.py -v` → all PASS.
Run: `python -m pytest tests/utils -v` → same known-failure set as end of Task 3 (json-on-disk assertions + reset tests), nothing new.

- [ ] **Step 5: Commit**

```bash
git add utils/preferences.py tests/utils/test_preferences_yaml.py
git commit -m "feat(prefs): one-time legacy preferences.json import with .bak rename"
```

---

### Task 5: Reset semantics, conftest fixtures, and existing-test updates

**Files:**
- Modify: `utils/preferences.py` (`reset_to_defaults`)
- Modify: `tests/conftest.py`, `tests/gui/conftest.py`, `tests/integration/conftest.py`
- Modify: `tests/utils/test_preferences.py`, `tests/utils/test_preferences_migration.py`
- Test: append reset tests to `tests/utils/test_preferences_yaml.py`

**Interfaces:**
- Consumes: everything from Tasks 3–4.
- Produces: `reset_to_defaults(*, keep_shared=True)` — same signature; rewrites `features.yaml` with `DEFAULT_PREFS` (minus `shared`); leaves `run.yaml` untouched when `keep_shared=True`, resets it too when `False`. The GUI caller (`gui/main_window.py:189`) needs no change.

- [ ] **Step 1: Write the failing reset tests** (append to `tests/utils/test_preferences_yaml.py`)

```python
def test_reset_rewrites_features_yaml_only(tmp_path, monkeypatch):
    from utils.preferences import reset_to_defaults
    monkeypatch.setenv("SENSI_CONFIG_DIR", str(tmp_path))
    save_preferences({"features": {"master_structural": False},
                      "shared": {"analyzer_name": "Keep Me"}})
    run_before = (tmp_path / RUN_YAML).read_text(encoding="utf-8")
    fresh = reset_to_defaults()                        # keep_shared=True default
    assert fresh["features"]["master_structural"] is True
    assert fresh["shared"]["analyzer_name"] == "Keep Me"
    assert (tmp_path / RUN_YAML).read_text(encoding="utf-8") == run_before
    fdata = read_yaml(tmp_path / FEATURES_YAML)
    assert fdata["features"]["master_structural"] is True


def test_reset_full_wipe_resets_run_yaml_too(tmp_path, monkeypatch):
    from utils.preferences import reset_to_defaults
    monkeypatch.setenv("SENSI_CONFIG_DIR", str(tmp_path))
    save_preferences({"shared": {"analyzer_name": "Wipe Me"}})
    fresh = reset_to_defaults(keep_shared=False)
    assert fresh["shared"]["analyzer_name"] == ""
    assert read_yaml(tmp_path / RUN_YAML)["shared"]["analyzer_name"] == ""
```

Run: `python -m pytest tests/utils/test_preferences_yaml.py -v` → the two new tests FAIL (reset still writes json).

- [ ] **Step 2: Implement `reset_to_defaults`**

```python
def reset_to_defaults(*, keep_shared: bool = True) -> dict[str, Any]:
    """Rewrite features.yaml with the shipped defaults.

    run.yaml (folders, analyzer, theme, token paths) is never touched by a
    default reset — pass keep_shared=False to reset it too. Files are local
    (gitignored), so this only affects this machine. Returns the fresh prefs.
    """
    from utils.config_yaml import split_prefs, write_features_yaml, write_run_yaml
    current = load_preferences()          # also triggers the legacy-json import
    fresh = deepcopy(DEFAULT_PREFS)
    if keep_shared and isinstance(current.get('shared'), dict):
        fresh['shared'] = deepcopy(current['shared'])
    features_data, run_data = split_prefs(fresh)
    write_features_yaml(get_features_yaml_path(), features_data)
    if not keep_shared:
        write_run_yaml(get_run_yaml_path(), run_data)
    return fresh
```

- [ ] **Step 3: Update the three conftests**

`tests/conftest.py` — add (below the MKL guard; keep the sys.path block):

```python
import pytest


@pytest.fixture(autouse=True)
def _isolated_config_dir(tmp_path, monkeypatch):
    """Point the two-YAML config store at a per-test temp dir so no test can
    read or migrate/rewrite the repo's real config/. Tests that exercise the
    legacy-json path additionally set SENSI_PREFS_PATH (same tmp_path, so the
    json and the YAMLs it migrates into share a directory)."""
    monkeypatch.setenv("SENSI_CONFIG_DIR", str(tmp_path))
    yield
```

`tests/gui/conftest.py` — update both fixtures to set BOTH vars (docstrings updated to say "two-YAML config store"; `prefs_tmp` keeps returning the legacy json path since its consumers use it as a scratch prefs location):

```python
@pytest.fixture(autouse=True)
def _isolated_prefs(tmp_path, monkeypatch):
    monkeypatch.setenv("SENSI_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("SENSI_PREFS_PATH", str(tmp_path / "test_preferences.json"))
    yield
```

(same two lines in `prefs_tmp`, which then returns the json path). `tests/integration/conftest.py` — same two `setenv` lines in its `_isolated_prefs`.

- [ ] **Step 4: Update the json-on-disk assertions in existing tests**

Add one helper at the top of `tests/utils/test_preferences_migration.py`:

```python
def _yaml_on_disk(json_path):
    """The store the migrated prefs actually landed in (YAML pair next to the
    legacy json path)."""
    from pathlib import Path
    from utils.config_yaml import read_yaml, join_yaml_pair, FEATURES_YAML, RUN_YAML
    d = Path(json_path).parent
    return join_yaml_pair(read_yaml(d / FEATURES_YAML), read_yaml(d / RUN_YAML))
```

Then update these sites (each currently does `json.loads(<prefs path>.read_text(...))` after a `load_preferences()`/`save_preferences()` call — replace with `_yaml_on_disk(<prefs path>)`; the surrounding assertions on the resulting dict stay as-is):
- `tests/utils/test_preferences_migration.py:126` (save writes schema_version)
- `tests/utils/test_preferences_migration.py:300`, `:441`, `:703`, `:752`, `:805`, `:832`
- `tests/utils/test_preferences.py` — the two `reset_to_defaults` tests (lines ~14–37): assert against `_yaml_on_disk`-style reads or just the returned dict; drop any assertion that `p.json` itself was rewritten. Check the rest of the file for other on-disk json assertions and port them the same way (15 tests total, small file).

Any migration test that asserts the legacy json still EXISTS after load must be flipped to assert the `.bak` exists instead (`Path(str(p) + ".bak")`).

- [ ] **Step 5: Run the FULL suite and fix stragglers**

Run: `python -m pytest`
Expected: green. Known risk areas if not: (a) a test asserting a json file appears after `save_preferences` — port to YAML reads; (b) a gui test relying on writing a prefs json BEFORE first load — still works via the legacy import, but the file becomes `.bak` after the first `load_preferences()`, so a second direct `json.loads` of it fails — port to `_yaml_on_disk`. Nothing in production code should need changes here; if a production change seems required, stop and flag it in the task report.

- [ ] **Step 6: Commit**

```bash
git add utils/preferences.py tests/
git commit -m "feat(prefs): reset rewrites features.yaml only; test isolation via SENSI_CONFIG_DIR"
```

---

### Task 6: Promote the headless runner to `run.py`; retire the sherlock config machinery

**Files:**
- Create: `run.py` (repo root)
- Create: `tests/test_run_py.py` (port of `tests/test_run_extract.py`)
- Delete: `sherlock/run_extract.py`, `sherlock/export_config.py`, `sherlock/__init__.py`, `tests/test_run_extract.py`, `tests/test_export_config.py`
- Modify: `utils/config_yaml.py` (remove the flat-config layer), `tests/utils/test_config_yaml.py` (delete file — the pair tests from Task 1 are the replacement), `.gitignore`

**Interfaces:**
- Consumes: `load_preferences`, `get_features_yaml_path`, `get_run_yaml_path` (Task 3); `read_yaml`, `validate_features_yaml`, `validate_run_yaml`, `split_prefs`, `write_features_yaml`, `write_run_yaml` (Task 1); `planner.plan_from_prefs` / `to_namespace` / `to_acoustic_namespace` and `main.run_*_batch` (existing, unchanged).
- Produces: `python run.py [--config-dir DIR]` — validates the YAML pair, expands `~`/`$VARS` in folders, sets `HF_TOKEN` from `shared.hf_token_file`, dispatches word/doc/acoustic levels exactly like the GUI. Module-level `run.load_credentials(shared: dict)` and `run.dispatch(prefs: dict)` for tests.

- [ ] **Step 1: Create `run.py`**

Behavior parity notes vs the old `sherlock/run_extract.py`: `dispatch` is a straight port with `prefs` instead of the flat config; required-field checks keep requiring `analyzer_name`, `last_transcripts_folder`, `last_output_folder` (same as before — acoustic-only runs unchanged in this project); `openai_api_key_file` handling is dropped (OpenAI backend removed in v17); the MKL guard is REQUIRED at the top (memory `project-mkl-threading-entrypoints`: entry points must set it or real runs crash on AMD/Windows).

```python
#!/usr/bin/env python3
"""Headless SensiExtractor runner — reads config/features.yaml + config/run.yaml.

Uses the same planner + batch runners as the GUI, so behavior matches the app
exactly. features.yaml says WHAT to extract (shareable; the GUI edits it);
run.yaml says WHERE on this machine (folders, analyzer, token-file paths).
Usage:
    python run.py [--config-dir DIR]    # DIR defaults to <repo>/config
"""
import argparse
import datetime as _dt
import os
import sys

# MKL on AMD CPUs crashes with the default Intel-thread layer (0xc06d007f);
# every entry point must set this before numpy does BLAS work (cf. main.py).
os.environ.setdefault("MKL_THREADING_LAYER", "sequential")

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def load_credentials(shared: dict) -> None:
    """Set HF_TOKEN from shared.hf_token_file (gated-Llama downloads)."""
    path = shared.get("hf_token_file")
    if not path:
        return
    resolved = os.path.expanduser(os.path.expandvars(path))
    if not os.path.isfile(resolved):
        return
    try:
        token = open(resolved, encoding="utf-8").read().strip()
    except OSError as exc:
        print(f"[warn] could not read hf_token_file {resolved}: {exc}", file=sys.stderr)
        return
    if token:
        os.environ["HF_TOKEN"] = token


def _prepare(ns):
    # Mirror run_dispatch._prepare_runtime_args: edu_bands list -> comma string.
    if isinstance(getattr(ns, "edu_bands", None), list):
        ns.edu_bands = ",".join(ns.edu_bands) if ns.edu_bands else None
    return ns


def dispatch(prefs: dict) -> None:
    """Run the enabled levels from a prefs dict (mirrors gui.run_dispatch.run)."""
    from gui import planner
    plan = planner.plan_from_prefs(prefs)
    run_ts = _dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    outputs = plan.output_csvs
    align_cache: dict = {}

    word_outputs = {"features", "syllables", "phonemes"} & outputs
    if word_outputs:
        from main import run_word_level_batch
        ns = _prepare(planner.to_namespace(plan, prefs, level="word"))
        ns.run_ts = run_ts
        ns.skip_no_match = False
        ns._align_cache = align_cache
        run_word_level_batch(ns)
    if "documents_summary" in outputs:
        from main import run_doc_level_batch
        ns = _prepare(planner.to_namespace(plan, prefs, level="doc"))
        ns.run_ts = run_ts
        ns.skip_no_match = bool(word_outputs)
        ns._align_cache = align_cache
        run_doc_level_batch(ns)
    if "acoustic" in outputs:
        from main import run_acoustic_level_batch
        ns = planner.to_acoustic_namespace(prefs)
        ns.run_ts = run_ts
        run_acoustic_level_batch(ns)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config-dir", default=None,
                    help="directory containing features.yaml + run.yaml "
                         "(default: <repo>/config, or $SENSI_CONFIG_DIR)")
    args = ap.parse_args()
    if args.config_dir:
        os.environ["SENSI_CONFIG_DIR"] = args.config_dir

    import yaml as _yaml
    from utils.preferences import (load_preferences, get_features_yaml_path,
                                   get_run_yaml_path)
    from utils.config_yaml import (read_yaml, validate_features_yaml,
                                   validate_run_yaml)

    errors: list[str] = []
    for path, validate in ((get_features_yaml_path(), validate_features_yaml),
                           (get_run_yaml_path(), validate_run_yaml)):
        try:
            data = read_yaml(path)
        except _yaml.YAMLError as exc:
            errors.append(f"{path.name}: not valid YAML ({exc})")
            continue
        errors.extend(f"{path.name}: {e}" for e in validate(data))
    if errors:
        print("Invalid config:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        sys.exit(1)

    prefs = load_preferences()
    shared = prefs["shared"]
    # Expand ~ and $VARS so HPC paths like $SCRATCH/... resolve.
    for key in ("last_transcripts_folder", "last_audio_folder",
                "last_output_folder"):
        val = shared.get(key) or ""
        shared[key] = os.path.expanduser(os.path.expandvars(val)) if val else ""

    if not (shared.get("analyzer_name") or "").strip():
        print("run.yaml: shared.analyzer_name is required (name for this run).",
              file=sys.stderr)
        sys.exit(1)
    missing = [k for k in ("last_transcripts_folder", "last_output_folder")
               if not (shared.get(k) or "").strip()]
    if missing:
        print(f"run.yaml: missing required folder(s): {', '.join(missing)}",
              file=sys.stderr)
        sys.exit(1)

    load_credentials(shared)
    dispatch(prefs)
    print(f"Done. Outputs under {shared['last_output_folder']}.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Port the runner tests → `tests/test_run_py.py`** (test-after per TDD-scope memory: CLI wiring, not extractor math)

```python
import os
from copy import deepcopy
import pytest

from utils.preferences import DEFAULT_PREFS
from utils.config_yaml import (FEATURES_YAML, RUN_YAML, split_prefs,
                               write_features_yaml, write_run_yaml)


def _write_pair(tmp_path, monkeypatch, features_overrides=None, **shared_overrides):
    """Write a features.yaml + run.yaml pair into tmp_path and point the app at it."""
    monkeypatch.setenv("SENSI_CONFIG_DIR", str(tmp_path))
    prefs = deepcopy(DEFAULT_PREFS)
    prefs["shared"].update(
        last_transcripts_folder=str(tmp_path / "t"),
        last_audio_folder=str(tmp_path / "a"),
        last_output_folder=str(tmp_path / "o"),
        analyzer_name="tester",
    )
    prefs["shared"].update(shared_overrides)
    for section, values in (features_overrides or {}).items():
        prefs[section].update(values)
    features, run_data = split_prefs(prefs)
    write_features_yaml(tmp_path / FEATURES_YAML, features)
    write_run_yaml(tmp_path / RUN_YAML, run_data)
    return prefs


def test_load_credentials_sets_hf_token_from_file(tmp_path, monkeypatch):
    import run
    monkeypatch.delenv("HF_TOKEN", raising=False)
    hf = tmp_path / "hf.txt"; hf.write_text("hf_secret\n", encoding="utf-8")
    run.load_credentials({"hf_token_file": str(hf)})
    assert os.environ["HF_TOKEN"] == "hf_secret"


def test_load_credentials_missing_file_is_noop(tmp_path, monkeypatch):
    import run
    monkeypatch.delenv("HF_TOKEN", raising=False)
    run.load_credentials({"hf_token_file": str(tmp_path / "nope.txt")})
    assert "HF_TOKEN" not in os.environ


def test_dispatch_runs_word_level_from_output_toggle(tmp_path, monkeypatch):
    import run
    calls = {}
    def fake_word(ns, **kw):
        calls["word"] = (ns.transcripts_folder, ns.analyzer)
    monkeypatch.setattr("main.run_word_level_batch", fake_word, raising=False)
    monkeypatch.setattr("main.run_doc_level_batch",
                        lambda *a, **k: calls.setdefault("doc", True), raising=False)
    prefs = _write_pair(tmp_path, monkeypatch, features_overrides={
        "output": {"word_level_csv": True, "document_summary": False}})
    run.dispatch(prefs)
    assert calls["word"] == (str(tmp_path / "t"), "tester")
    assert "doc" not in calls


def test_main_exits_when_required_folder_missing(tmp_path, monkeypatch, capsys):
    import run
    _write_pair(tmp_path, monkeypatch, last_transcripts_folder="")
    monkeypatch.setattr("sys.argv", ["run.py"])
    with pytest.raises(SystemExit) as ei:
        run.main()
    assert ei.value.code == 1
    assert "last_transcripts_folder" in capsys.readouterr().err


def test_main_exits_on_invalid_features_yaml(tmp_path, monkeypatch, capsys):
    import run
    _write_pair(tmp_path, monkeypatch)
    (tmp_path / FEATURES_YAML).write_text(
        "features:\n  not_a_real_toggle: true\n", encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["run.py"])
    with pytest.raises(SystemExit) as ei:
        run.main()
    assert ei.value.code == 1
    assert "not_a_real_toggle" in capsys.readouterr().err
```

- [ ] **Step 3: Delete the retired files and strip `utils/config_yaml.py`**

```bash
git rm sherlock/run_extract.py sherlock/export_config.py sherlock/__init__.py tests/test_run_extract.py tests/test_export_config.py tests/utils/test_config_yaml.py
```

In `utils/config_yaml.py`, delete the flat-config layer: `_NON_SETTINGS`, `_settings_sections`, `prefs_to_config`, `config_to_prefs`, `_KNOWN_TOP`, `validate_config`, `_HEADER`, `read_config`, `write_config`, and the module-level `from utils.preferences import DEFAULT_PREFS` (the pair validators import it lazily). Update the module docstring: it is now the YAML-pair I/O layer for the two-file config store. Keep `_type_ok`. Optionally lift the lazy `from utils.config_yaml import …` imports in `utils/preferences.py` to module level now that the cycle is gone (verify with `python -c "import utils.preferences, utils.config_yaml"` both orders).

Also delete the untracked local `sherlock/config.yaml` if present (`rm sherlock/config.yaml`) — it is dead machinery now.

- [ ] **Step 4: Update `.gitignore`**

Replace the runtime-config block:

```
# Runtime config (license-bound or user-specific)
config/preferences.json
config/preferences.json.bak
config/features.yaml
config/run.yaml
```

and delete the sherlock generated-config lines at the bottom:

```
# Generated run config (machine-specific paths; regenerate with export_config.py)
sherlock/config.yaml
```

- [ ] **Step 5: Run the full suite**

Run: `python -m pytest`
Expected: green (old flat-config/export/run_extract tests removed; `tests/test_run_py.py` + `tests/utils/test_config_yaml_pair.py` cover the replacements). Also smoke the entry point by hand:

```bash
python run.py --config-dir <some-empty-tmp-dir>
```

Expected: exit 1 with `run.yaml: shared.analyzer_name is required (name for this run).` — proves import wiring without needing stimuli.

- [ ] **Step 6: Commit**

```bash
git add run.py tests/test_run_py.py utils/config_yaml.py .gitignore
git commit -m "feat(run): promote headless runner to top-level run.py; retire sherlock config machinery"
```

---

### Task 7: Docs — RUNBOOK, sbatch, CLAUDE.md, README, guide, master plan

**Files:**
- Modify: `sherlock/run_container.sbatch`, `sherlock/RUNBOOK.md`, `CLAUDE.md`, `README.md`, `Feature_Extraction_Guide.md`, `docs/master_plan.md`

**Interfaces:**
- Consumes: the shipped behavior of Tasks 1–6 (describe what IS, not what was planned).

- [ ] **Step 1: `sherlock/run_container.sbatch`**

Replace the config plumbing (currently `CONFIG="${CONFIG:-$REPO/sherlock/config.yaml}"`, the `[ -f "$CONFIG" ]` guard, and the `python "$REPO/sherlock/run_extract.py" "$CONFIG"` invocation) with:

```bash
CONFIG_DIR="${CONFIG_DIR:-$REPO/config}"
...
[ -f "$CONFIG_DIR/features.yaml" ] || { echo "[job] ERROR: features.yaml not found in $CONFIG_DIR (rsync the repo's config/ over, or edit one here)"; exit 1; }
[ -f "$CONFIG_DIR/run.yaml" ] || { echo "[job] ERROR: run.yaml not found in $CONFIG_DIR (create it on Sherlock — machine-specific folders/token paths)"; exit 1; }
...
  python "$REPO/run.py" --config-dir "$CONFIG_DIR"
```

Update the header comments in the same file (they mention `sherlock/config.yaml` and `export_config.py`).

- [ ] **Step 2: `sherlock/RUNBOOK.md`**

Rewrite the config-flow sections to the new model (the container/rsync/HPC sections stay):
- The GUI's saved settings ARE the shareable config now: `config/features.yaml` (no export step — delete the `export_config.py` instructions at lines ~64–68 and the mention at ~34).
- `config/run.yaml` is machine-specific: create/fill it ON Sherlock (folders with `$SCRATCH`/`~`, `analyzer_name`, `hf_token_file`). Give a 7-line example `run.yaml` (top-level `shared:` with the six keys).
- rsync guidance: the code rsync now carries `config/features.yaml`; add `--exclude 'config/run.yaml'` (and `--exclude 'config/preferences.json*'`) so a local machine's paths never clobber Sherlock's.
- Run invocation: `python $GROUP_HOME/SensiExtractor/run.py` (default `--config-dir` is the repo's `config/`); the troubleshooting bullet at ~235 becomes "a bad features.yaml/run.yaml makes run.py print the errors and exit before running".
- HF token: `run.yaml`'s `shared.hf_token_file` points at the token file (section at ~135 updated).

- [ ] **Step 3: `CLAUDE.md`**

- Project-structure tree: add `run.py  # Headless entry point (reads config/features.yaml + config/run.yaml)` next to `main.py`; add a `config/` note (`features.yaml` global/reset-able, `run.yaml` machine-specific, both gitignored; legacy `preferences.json` auto-imported once → `.bak`); update the `utils/` description if it mentions config_yaml's old role.
- "Running" section: add `python run.py` as the headless option.

- [ ] **Step 4: `README.md:192` and `Feature_Extraction_Guide.md:323`**

- README: `config/preferences.json`'s `predictability` section → `config/features.yaml`'s `predictability` section. Grep README for any other `preferences.json` mention.
- Guide line 323: the prefs file is now the `config/features.yaml` + `config/run.yaml` pair (gitignored); override the directory via `SENSI_CONFIG_DIR`. (The guide has known pre-existing drift elsewhere — do not expand scope beyond this line.)

- [ ] **Step 5: `docs/master_plan.md`**

- §2: remove the "Next up: Unified YAML config (Project C)" block (§3 gets the completed row); prefs-version line: note storage moved to the YAML pair, schema still v19.
- §3: add the Project C row (date 2026-07-12): two-file YAML store behind the unchanged `utils/preferences.py` API; `SENSI_CONFIG_DIR` (+`SENSI_PREFS_PATH`-parent fallback); one-time json import → `.bak`; Reset rewrites features.yaml only; `run.py` promoted; sherlock export machinery retired; final suite count.
- §6 durable conventions: prefs bullet — add one line that storage is the YAML pair since 2026-07-12 (schema/migration chain unchanged).

- [ ] **Step 6: Full suite + commit**

```bash
python -m pytest
git add sherlock/ CLAUDE.md README.md Feature_Extraction_Guide.md docs/master_plan.md
git commit -m "docs: two-file YAML config — RUNBOOK/sbatch, CLAUDE.md, README, master plan"
```

---

## Post-plan checklist (for the coordinating session, after all tasks)

- Merge `yaml-config` → `main` per superpowers:finishing-a-development-branch.
- Manual-validation note for the user (add to the §2 table if desired): launch `python gui.py` once on this machine — confirm `config/preferences.json` becomes `preferences.json.bak` and the two YAMLs appear with the analyzer/folders intact; then "Reset settings" and confirm folders survive.
