# Sherlock file-driven config workflow — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a user configure in the local GUI, export a portable `config.yaml`, and run the pipeline on Sherlock via `run_extract.py config.yaml` — reusing the app's prefs schema + planner + batch runners.

**Architecture:** A tracked, schema-agnostic mapping module (`utils/config_yaml.py`) converts between the live `DEFAULT_PREFS` dict and a flat YAML config. Two thin repo-root entry scripts sit on top: `export_config.py` (GUI prefs → YAML, local) and `run_extract.py` (YAML → prefs → planner + batch runners, Sherlock). Glue wires the container sbatch to it.

**Tech Stack:** Python 3.10, PyYAML (`safe_load`/`safe_dump`), pytest.

## Global Constraints

- **Schema-agnostic:** derive valid keys and settings sections from the live
  `utils.preferences.DEFAULT_PREFS` at runtime — never hardcode a version's key list.
- **Active schema is v16**; sections: `schema_version`, `shared`, `features`,
  `output`, `predictability`, `semantic`, `prosody`, `acoustic_ts`, `phoneme_surprisal`.
- **No secret in the config:** `hf_token_file` / `openai_api_key_file` are *paths*
  in the `run:` block; the run script reads the file and sets `HF_TOKEN` /
  `OPENAI_API_KEY`. Export never copies secrets or local `shared` folder paths.
- Scripts live at **repo root** (`sherlock/` is git-untracked). Run tests with
  `./env/python.exe -m pytest` from the repo root.
- Config top-level sections: `run`, `features`, `settings`, `semantic_categories`.
  The YAML `features:` block holds BOTH feature and output toggles, routed on load.

Spec: `docs/superpowers/specs/2026-07-08-sherlock-config-workflow-design.md`.

## File structure

- `utils/config_yaml.py` (new) — mapping + validation + YAML I/O. One responsibility: config↔prefs.
- `export_config.py` (new, repo root) — thin CLI: prefs → YAML.
- `run_extract.py` (new, repo root) — thin CLI: YAML → run.
- `requirements.txt` (modify) — declare `pyyaml`.
- `sherlock/run_container.sbatch` (modify) — call `run_extract.py` (untracked; edited on disk).
- Tests: `tests/utils/test_config_yaml.py`, `tests/test_run_extract.py`, `tests/test_export_config.py`.

---

### Task 1: config↔prefs mapping (`prefs_to_config` / `config_to_prefs`)

**Files:**
- Create: `utils/config_yaml.py`
- Test: `tests/utils/test_config_yaml.py`

**Interfaces:**
- Consumes: `utils.preferences.DEFAULT_PREFS`.
- Produces: `_settings_sections() -> list[str]`, `prefs_to_config(prefs: dict) -> dict`, `config_to_prefs(config: dict) -> dict`.

- [ ] **Step 1: Write the failing tests**

Create `tests/utils/test_config_yaml.py`:
```python
from copy import deepcopy
from utils.preferences import DEFAULT_PREFS
from utils.config_yaml import prefs_to_config, config_to_prefs, _settings_sections


def test_settings_sections_excludes_structural_sections():
    s = set(_settings_sections())
    assert "predictability" in s and "prosody" in s and "acoustic_ts" in s
    assert not (s & {"schema_version", "shared", "features", "output", "semantic"})


def test_prefs_to_config_shape_and_omits_secrets_and_paths():
    cfg = prefs_to_config(DEFAULT_PREFS)
    assert set(cfg) == {"run", "features", "settings", "semantic_categories"}
    # features block merges features + output toggles
    assert set(cfg["features"]) == set(DEFAULT_PREFS["features"]) | set(DEFAULT_PREFS["output"])
    # run block has empty placeholders + token-file paths, no real shared values
    assert cfg["run"]["transcripts_folder"] == ""
    assert cfg["run"]["analyzer"] == ""
    assert "hf_token_file" in cfg["run"] and "openai_api_key_file" in cfg["run"]
    assert "hf_token" not in cfg["run"]  # never a raw secret


def test_config_to_prefs_round_trips_toggles_settings_categories():
    cfg = prefs_to_config(DEFAULT_PREFS)
    prefs = config_to_prefs(cfg)
    assert prefs["features"] == DEFAULT_PREFS["features"]
    assert prefs["output"] == DEFAULT_PREFS["output"]
    assert prefs["semantic"] == DEFAULT_PREFS["semantic"]
    for s in _settings_sections():
        assert prefs[s] == DEFAULT_PREFS[s]


def test_config_to_prefs_routes_toggles_and_run_folders():
    cfg = prefs_to_config(DEFAULT_PREFS)
    # pick one real feature key and one real output key
    feat_key = next(iter(DEFAULT_PREFS["features"]))
    out_key = next(iter(DEFAULT_PREFS["output"]))
    cfg["features"][feat_key] = False
    cfg["features"][out_key] = False
    cfg["run"].update(transcripts_folder="/scratch/t", audio_folder="/scratch/a",
                       output_folder="/scratch/o", analyzer="me")
    prefs = config_to_prefs(cfg)
    assert prefs["features"][feat_key] is False
    assert prefs["output"][out_key] is False
    assert prefs["shared"]["last_transcripts_folder"] == "/scratch/t"
    assert prefs["shared"]["last_audio_folder"] == "/scratch/a"
    assert prefs["shared"]["last_output_folder"] == "/scratch/o"
    assert prefs["shared"]["analyzer_name"] == "me"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./env/python.exe -m pytest tests/utils/test_config_yaml.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'utils.config_yaml'`.

- [ ] **Step 3: Implement the mapping**

Create `utils/config_yaml.py`:
```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./env/python.exe -m pytest tests/utils/test_config_yaml.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add utils/config_yaml.py tests/utils/test_config_yaml.py
git commit -m "feat(config-yaml): schema-agnostic prefs<->config mapping"
```

---

### Task 2: `validate_config`

**Files:**
- Modify: `utils/config_yaml.py`
- Test: `tests/utils/test_config_yaml.py`

**Interfaces:**
- Produces: `validate_config(config: dict) -> list[str]` (empty list = valid).

- [ ] **Step 1: Write the failing tests**

Append to `tests/utils/test_config_yaml.py`:
```python
from utils.config_yaml import validate_config


def test_validate_config_accepts_exported_default():
    cfg = prefs_to_config(DEFAULT_PREFS)
    assert validate_config(cfg) == []


def test_validate_config_flags_unknown_toggle():
    cfg = prefs_to_config(DEFAULT_PREFS)
    cfg["features"]["not_a_real_toggle"] = True
    errs = validate_config(cfg)
    assert any("not_a_real_toggle" in e for e in errs)


def test_validate_config_flags_bad_toggle_type():
    cfg = prefs_to_config(DEFAULT_PREFS)
    key = next(iter(DEFAULT_PREFS["features"]))
    cfg["features"][key] = "yes"  # not a bool
    errs = validate_config(cfg)
    assert any(key in e and "true/false" in e for e in errs)


def test_validate_config_flags_unknown_section_and_setting():
    cfg = prefs_to_config(DEFAULT_PREFS)
    cfg["bogus_section"] = {}
    cfg["settings"].setdefault("prosody", {})["not_a_setting"] = 1
    errs = validate_config(cfg)
    assert any("bogus_section" in e for e in errs)
    assert any("not_a_setting" in e for e in errs)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./env/python.exe -m pytest tests/utils/test_config_yaml.py -k validate -v`
Expected: FAIL — `cannot import name 'validate_config'`.

- [ ] **Step 3: Implement `validate_config`**

Append to `utils/config_yaml.py`:
```python
_KNOWN_TOP = frozenset({"run", "features", "settings", "semantic_categories"})


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


def validate_config(config: dict) -> list[str]:
    """Return human-readable errors for a config; empty list means valid."""
    if not isinstance(config, dict):
        return ["config root must be a mapping (a YAML object)"]
    errors: list[str] = []
    for key in config:
        if key not in _KNOWN_TOP:
            errors.append(f"unknown top-level section: {key!r} (expected {sorted(_KNOWN_TOP)})")
    valid_toggles = set(DEFAULT_PREFS["features"]) | set(DEFAULT_PREFS["output"])
    for key, val in (config.get("features") or {}).items():
        if key not in valid_toggles:
            errors.append(f"unknown feature/output toggle: {key!r}")
        elif not isinstance(val, bool):
            errors.append(f"toggle {key!r} must be true/false, got {type(val).__name__}")
    settings_sections = set(_settings_sections())
    for section, values in (config.get("settings") or {}).items():
        if section not in settings_sections:
            errors.append(f"unknown settings section: {section!r} (expected {sorted(settings_sections)})")
            continue
        default_section = DEFAULT_PREFS[section]
        if isinstance(default_section, dict) and isinstance(values, dict):
            for k, v in values.items():
                if k not in default_section:
                    errors.append(f"unknown setting: {section}.{k}")
                elif not _type_ok(default_section[k], v):
                    errors.append(
                        f"setting {section}.{k} must be {type(default_section[k]).__name__}, "
                        f"got {type(v).__name__}")
    sem = config.get("semantic_categories")
    if isinstance(sem, dict):
        for k in sem:
            if k not in DEFAULT_PREFS["semantic"]:
                errors.append(f"unknown semantic_categories key: {k!r}")
    return errors
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./env/python.exe -m pytest tests/utils/test_config_yaml.py -v`
Expected: all pass (Task 1 + Task 2 tests).

- [ ] **Step 5: Commit**

```bash
git add utils/config_yaml.py tests/utils/test_config_yaml.py
git commit -m "feat(config-yaml): validate_config with key/type checks"
```

---

### Task 3: YAML read/write helpers

**Files:**
- Modify: `utils/config_yaml.py`
- Test: `tests/utils/test_config_yaml.py`

**Interfaces:**
- Produces: `read_config(path: str) -> dict`, `write_config(config: dict, path: str) -> None`.

- [ ] **Step 1: Write the failing test**

Append to `tests/utils/test_config_yaml.py`:
```python
def test_write_then_read_config_round_trips(tmp_path):
    from utils.config_yaml import read_config, write_config
    cfg = prefs_to_config(DEFAULT_PREFS)
    p = tmp_path / "config.yaml"
    write_config(cfg, str(p))
    text = p.read_text(encoding="utf-8")
    assert text.startswith("#")            # header comment present
    back = read_config(str(p))
    assert back == cfg                      # structural round-trip


def test_read_config_empty_file_returns_empty_dict(tmp_path):
    from utils.config_yaml import read_config
    p = tmp_path / "empty.yaml"
    p.write_text("", encoding="utf-8")
    assert read_config(str(p)) == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./env/python.exe -m pytest tests/utils/test_config_yaml.py -k "round_trips or empty_file" -v`
Expected: FAIL — `cannot import name 'read_config'`.

- [ ] **Step 3: Implement the helpers**

Add near the top of `utils/config_yaml.py` (after the imports, add `import os` and `import yaml`), and append:
```python
_HEADER = (
    "# SensiExtractor run config — generated by export_config.py.\n"
    "# Edit the `run:` section on Sherlock (folders, analyzer, token-file paths).\n"
    "# `features:` are on/off toggles; `settings:` are typed values;\n"
    "# `semantic_categories:` are the multi-select category lists.\n"
)


def read_config(path: str) -> dict:
    with open(os.path.expanduser(path), "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def write_config(config: dict, path: str) -> None:
    with open(os.path.expanduser(path), "w", encoding="utf-8") as f:
        f.write(_HEADER)
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True, default_flow_style=False)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./env/python.exe -m pytest tests/utils/test_config_yaml.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add utils/config_yaml.py tests/utils/test_config_yaml.py
git commit -m "feat(config-yaml): YAML read/write helpers with header"
```

---

### Task 4: `export_config.py` (repo root)

**Files:**
- Create: `export_config.py`
- Test: `tests/test_export_config.py`

**Interfaces:**
- Consumes: `utils.preferences.load_preferences`, `utils.config_yaml.prefs_to_config`/`write_config`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_export_config.py`:
```python
import subprocess, sys, os
from pathlib import Path
from utils.config_yaml import read_config, validate_config


def test_export_config_writes_valid_yaml(tmp_path, monkeypatch):
    # Point the app's prefs at a temp path so we export whatever defaults load.
    monkeypatch.setenv("SENSI_PREFS_PATH", str(tmp_path / "prefs.json"))
    out = tmp_path / "config.yaml"
    repo = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [sys.executable, str(repo / "export_config.py"), "--out", str(out)],
        capture_output=True, text=True, cwd=str(repo),
        env={**os.environ, "SENSI_PREFS_PATH": str(tmp_path / "prefs.json")},
    )
    assert result.returncode == 0, result.stderr
    assert out.exists()
    cfg = read_config(str(out))
    assert set(cfg) == {"run", "features", "settings", "semantic_categories"}
    assert validate_config(cfg) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./env/python.exe -m pytest tests/test_export_config.py -v`
Expected: FAIL — `export_config.py` does not exist (non-zero return / file-not-found).

- [ ] **Step 3: Implement `export_config.py`**

Create `export_config.py` at the repo root:
```python
#!/usr/bin/env python3
"""Export the GUI's saved settings into a portable config.yaml for Sherlock.

Reads the app's preferences (respecting SENSI_PREFS_PATH) and writes a config the
Sherlock runner (run_extract.py) consumes. Fill the `run:` section on Sherlock.
"""
import argparse

from utils.preferences import load_preferences
from utils.config_yaml import prefs_to_config, write_config


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="config.yaml", help="output path (default: config.yaml)")
    args = ap.parse_args()
    config = prefs_to_config(load_preferences())
    write_config(config, args.out)
    print(f"Wrote {args.out}. Before running on Sherlock, fill the `run:` section "
          f"(transcripts/audio/output folders, analyzer, and token-file paths).")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./env/python.exe -m pytest tests/test_export_config.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add export_config.py tests/test_export_config.py
git commit -m "feat: export_config.py — GUI prefs to portable config.yaml"
```

---

### Task 5: `run_extract.py` (repo root)

**Files:**
- Create: `run_extract.py`
- Test: `tests/test_run_extract.py`

**Interfaces:**
- Consumes: `utils.config_yaml.{read_config,validate_config,config_to_prefs}`; `gui.planner.{plan_from_prefs,to_namespace,to_acoustic_namespace}`; `main.{run_word_level_batch,run_doc_level_batch,run_acoustic_level_batch}`.
- Produces: module functions `load_credentials(run: dict) -> None`, `dispatch(config: dict) -> None`, `main()`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_run_extract.py`:
```python
import os
from utils.preferences import DEFAULT_PREFS
from utils.config_yaml import prefs_to_config


def _base_config(tmp_path):
    cfg = prefs_to_config(DEFAULT_PREFS)
    cfg["run"].update(
        transcripts_folder=str(tmp_path / "t"),
        audio_folder=str(tmp_path / "a"),
        output_folder=str(tmp_path / "o"),
        analyzer="tester",
    )
    return cfg


def test_load_credentials_sets_env_from_files(tmp_path, monkeypatch):
    import run_extract
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    hf = tmp_path / "hf.txt"; hf.write_text("hf_secret\n", encoding="utf-8")
    run_extract.load_credentials({"hf_token_file": str(hf)})
    assert os.environ["HF_TOKEN"] == "hf_secret"


def test_load_credentials_missing_file_is_noop(tmp_path, monkeypatch):
    import run_extract
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    run_extract.load_credentials({"openai_api_key_file": str(tmp_path / "nope.txt")})
    assert "OPENAI_API_KEY" not in os.environ


def test_dispatch_runs_word_level_from_output_toggle(tmp_path, monkeypatch):
    import run_extract
    calls = {}
    def fake_word(ns, **kw):
        calls["word"] = (ns.transcripts_folder, ns.analyzer)
    monkeypatch.setattr("main.run_word_level_batch", fake_word, raising=False)
    monkeypatch.setattr("main.run_doc_level_batch", lambda *a, **k: calls.setdefault("doc", True), raising=False)
    cfg = _base_config(tmp_path)
    # Ensure a word-level output is on and doc is off.
    cfg["features"]["word_level_csv"] = True
    cfg["features"]["document_summary"] = False
    run_extract.dispatch(cfg)
    assert calls["word"] == (str(tmp_path / "t"), "tester")
    assert "doc" not in calls
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./env/python.exe -m pytest tests/test_run_extract.py -v`
Expected: FAIL — `No module named 'run_extract'`.

- [ ] **Step 3: Implement `run_extract.py`**

Create `run_extract.py` at the repo root:
```python
#!/usr/bin/env python3
"""Run the SensiExtractor pipeline on Sherlock from a portable config.yaml.

Maps the config onto the app's prefs schema and reuses the same planner + batch
runners the GUI uses, so behavior matches the app exactly. Usage:
    python run_extract.py [config.yaml]
"""
import argparse
import datetime as _dt
import os
import sys

from utils.config_yaml import read_config, validate_config, config_to_prefs
from gui import planner


def load_credentials(run: dict) -> None:
    """Set HF_TOKEN / OPENAI_API_KEY from the key-file paths in the run block."""
    for field, env in (("hf_token_file", "HF_TOKEN"),
                       ("openai_api_key_file", "OPENAI_API_KEY")):
        path = run.get(field)
        if not path:
            continue
        resolved = os.path.expanduser(os.path.expandvars(path))
        if not os.path.isfile(resolved):
            continue
        try:
            token = open(resolved, encoding="utf-8").read().strip()
        except OSError as exc:
            print(f"[warn] could not read {field} {resolved}: {exc}", file=sys.stderr)
            continue
        if token:
            os.environ[env] = token


def _prepare(ns):
    # Mirror run_dispatch._prepare_runtime_args: edu_bands list -> comma string.
    if isinstance(getattr(ns, "edu_bands", None), list):
        ns.edu_bands = ",".join(ns.edu_bands) if ns.edu_bands else None
    return ns


def dispatch(config: dict) -> None:
    """Build prefs from config and run the enabled levels (mirrors run_dispatch.run)."""
    prefs = config_to_prefs(config)
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
    ap.add_argument("config", nargs="?", default="config.yaml",
                    help="path to the YAML config (default: config.yaml)")
    args = ap.parse_args()

    config = read_config(args.config)
    errors = validate_config(config)
    if errors:
        print("Invalid config:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        sys.exit(1)

    run = config.get("run") or {}
    if not (run.get("analyzer") or "").strip():
        print("run.analyzer is required (name for this run).", file=sys.stderr)
        sys.exit(1)

    load_credentials(run)
    dispatch(config)
    out = run.get("output_folder") or "(output_folder unset)"
    print(f"Done. Outputs under {out}.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./env/python.exe -m pytest tests/test_run_extract.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add run_extract.py tests/test_run_extract.py
git commit -m "feat: run_extract.py — run pipeline from config.yaml via planner + batch runners"
```

---

### Task 6: Glue — declare PyYAML, wire the container sbatch

**Files:**
- Modify: `requirements.txt`
- Modify: `sherlock/run_container.sbatch` (git-untracked — edited on disk, not committed)

**Interfaces:** none.

- [ ] **Step 1: Declare PyYAML in requirements.txt**

Add under an appropriate section of `requirements.txt` (it is already installed in the container; this is a correctness declaration):
```
# Config file parsing for the Sherlock run workflow (run_extract.py / export_config.py)
pyyaml>=6
```

- [ ] **Step 2: Verify the import resolves in the env**

Run: `./env/python.exe -c "import yaml; print('pyyaml', yaml.__version__)"`
Expected: prints a 6.x version (already present — no rebuild).

- [ ] **Step 3: Point run_container.sbatch at run_extract.py**

In `sherlock/run_container.sbatch`, replace the `python "$REPO/main.py" \ …flags…` invocation with:
```bash
apptainer exec --nv \
  --bind "$SCRATCH:$SCRATCH" \
  --bind "$REPO:$REPO" \
  --env HF_HOME="$HF_HOME" \
  --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
  "$SIF" \
  python "$REPO/run_extract.py" "$REPO/config.yaml"
```
Add a CONFIG line `CONFIG="${CONFIG:-$REPO/config.yaml}"` near the other config vars and use `"$CONFIG"` in place of the literal path. (This file is git-untracked, so it is edited on disk only — do not attempt to `git add` it.)

- [ ] **Step 4: Commit the tracked change**

```bash
git add requirements.txt
git commit -m "chore: declare pyyaml for the Sherlock config workflow"
```

---

## Self-Review

**Spec coverage:** Component A (config.yaml shape) → Tasks 1+3 (mapping produces it, write_config emits it). B (config_yaml.py: prefs_to_config/config_to_prefs/validate_config/read_config/write_config) → Tasks 1–3. C (export_config.py) → Task 4. D (run_extract.py incl. credential loading + dispatch) → Task 5. E (glue: sbatch + pyyaml) → Task 6. Testing requirements → tests in Tasks 1–5. All covered.

**Placeholder scan:** Every code step contains complete code; every run step has an exact command + expected result. No TBD/TODO.

**Type consistency:** `prefs_to_config`/`config_to_prefs`/`_settings_sections` (Task 1) reused verbatim in Tasks 2–5. `validate_config` (Task 2) signature matches its call in `run_extract.main` (Task 5). `read_config`/`write_config` (Task 3) match calls in Tasks 4–5. `load_credentials`/`dispatch` (Task 5) names match the Task 5 tests. Output-set membership names (`{"features","syllables","phonemes"}`, `"documents_summary"`, `"acoustic"`) copied verbatim from `gui/run_dispatch.py:77-92`.

**Risk note:** `run_extract.py` imports `gui.planner` and `main`. Both are pure-logic imports (planner has no Qt dependency; main sets MKL defaults at import). The Task 5 tests import `run_extract` headlessly and will surface any import-time Qt/display issue immediately.

**Deliberate scope decision (spec deviation):** the spec's testing section mentioned a real end-to-end word-level run producing a CSV. The plan instead tests `dispatch` with monkeypatched batch runners (Task 5). Rationale: `run_extract` only maps config→prefs and delegates to the existing, already-tested batch runners; a real e2e would re-exercise the whole pipeline (spaCy/GPT-2/etc.) slowly for little incremental coverage of *this* code. The genuine end-to-end validation is the user's Sherlock dry-run. Not a silent gap — flagged here for the final review to accept or override.
