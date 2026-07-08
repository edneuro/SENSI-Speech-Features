# Design: file-driven config workflow for Sherlock (GUI → YAML → run)

**Date:** 2026-07-08
**Status:** approved (design), pending implementation plan

## Problem

SensiExtractor is configured through a PySide6 GUI that cannot run on Sherlock's
headless compute nodes, and the CLI exposes ~80 flags — too many to drive a real
run by hand. Users need a way to configure locally (where the GUI works) and run
on Sherlock without re-specifying everything as flags.

## Goal

A portable, human-readable config file that captures a configuration and drives a
Sherlock run through the **same prefs schema, planner, and batch runners** the GUI
already uses — so it inherits the app's dependency resolution, column dropping,
MFA-once caching, etc. rather than re-deriving flags.

Workflow: GUI (local) → `export_config.py` → `config.yaml` → rsync → Sherlock →
`run_extract.py config.yaml` → feature CSVs.

## Non-goals

- No new feature-extraction logic; this is orchestration over existing runners.
- No GUI code changes — export is a standalone script reading the GUI's saved prefs.
- Updating the Sherlock RUNBOOK/docs is a **separate issue** (out of scope here).
- No secret (HF token or OpenAI key) is ever stored in the config; each is read
  from a plain key-file the config only references by path — the config stays safe
  to share/commit/rsync.
- Not exposing a curated subset — every prefs toggle/setting is representable.

## Existing anchors this builds on (verified)

- `utils/preferences.py`: `load_preferences()` resolves the prefs file
  (`SENSI_PREFS_PATH` override, else default) and returns the nested prefs dict;
  `DEFAULT_PREFS_V3` / `_FEATURES_DEFAULT` / `_OUTPUT_DEFAULT` define the valid keys
  and defaults; `schema_version` is present.
- `gui/planner/emit.py:156-169` `to_namespace(plan, prefs, level)` reads
  `transcripts_folder`/`audio_folder`/`output_folder`/`analyzer` from
  `prefs["shared"]` (`last_transcripts_folder`, `last_audio_folder`,
  `last_output_folder`, `analyzer_name`).
- `gui/run_dispatch.py:64-102` `run()` is the dispatch template: `plan_from_prefs`
  → for each enabled output, `to_namespace(...)` → `run_word_level_batch` /
  `run_doc_level_batch` / `run_acoustic_level_batch`, sharing one `_align_cache`.
- Prefs sections to map: `features` (booleans), `output` (booleans),
  `predictability.llm_model`, `semantic.{backend,categories_liwc,categories_empath}`,
  `prosody.*`, `acoustic_ts.*`, `shared.{analyzer_name,last_*_folder}`.

## Components

### A. `config.yaml` — the portable handoff

One commented YAML file, four top-level sections:
```yaml
run:                     # Sherlock-specific; the GUI can't supply these
  transcripts_folder: ""
  audio_folder: ""
  output_folder: ""
  analyzer: ""
  # Secrets stay OUT of this file — point at plain files you create in $HOME.
  # Only the one matching your chosen llm_model is needed:
  hf_token_file: ~/hf_token.txt          # for local Llama (meta-llama/*) gated download
  openai_api_key_file: ~/openai_key.txt  # for OpenAI models (gpt-*); needs internet
features:                # booleans — feat_*/lexling_*/freq_*/pred_*/sem_*/acoustic_*
  ...
  word_level_csv: true   # output toggles live here too (what to produce)
  document_summary: false
settings:
  llm_model: meta-llama/Meta-Llama-3-8B
  prosody: {pitch_units: hz, pitch_floor_hz: 60, pitch_ceiling_hz: 600, pause_threshold_s: 0.2}
  acoustic_ts: {analysis_rate: 100, mel_preset: speech, n_mels: 40, n_mfcc: 13}
semantic_categories:
  backend: auto
  liwc:   [...]
  empath: [...]
```
"Which levels run" is not a separate field — it follows from the `output` toggles
(word_level_csv / document_summary / acoustic), mirroring the GUI.

### B. `utils/config_yaml.py` — the tracked, tested mapping module

Pure functions, no I/O side effects beyond file read/write helpers:
- `prefs_to_config(prefs: dict) -> dict` — project a prefs dict into the config
  shape (features+output → `features`; predictability/prosody/acoustic_ts →
  `settings`; semantic → `semantic_categories`). Emits a `run:` block with empty
  placeholders (does NOT copy `shared.last_*_folder` — local paths — or any token).
- `config_to_prefs(config: dict) -> dict` — deep-merge the config onto a fresh
  `DEFAULT_PREFS` copy (so schema_version + any unspecified keys keep defaults);
  map the `run:` folders/analyzer into `prefs["shared"]`
  (`last_transcripts_folder` etc., `analyzer_name`). Each key under the YAML
  `features:` block is **routed by membership**: keys defined in
  `_FEATURES_DEFAULT` go to `prefs["features"]`, keys defined in `_OUTPUT_DEFAULT`
  go to `prefs["output"]` (a key in neither is an error, caught by
  `validate_config`). `prefs_to_config` performs the inverse — it flattens both
  prefs sections into the single YAML `features:` block.
- `validate_config(config: dict) -> list[str]` — return human-readable errors for
  unknown feature/output keys (typo protection, checked against DEFAULT_PREFS
  keys), wrong scalar types, and unknown top-level sections. Empty list = valid.
- `read_config(path) / write_config(config, path)` — thin YAML load/dump wrappers
  (PyYAML `safe_load` / `safe_dump`), writing a header comment block.

### C. `export_config.py` (repo root, runs locally)

Thin CLI: `load_preferences()` → `prefs_to_config(prefs)` → `write_config(...,
out_path)`. Default output `config.yaml`; `--prefs PATH` and `--out PATH` optional.
Prints where it wrote and a one-line reminder to fill the `run:` section on Sherlock.

### D. `run_extract.py` (repo root, runs on Sherlock inside the container)

Thin CLI: `run_extract.py [config.yaml]` (default `config.yaml` in CWD).
1. `read_config` → `validate_config` (exit non-zero with the error list if invalid).
2. Load credentials from whichever key-file is present (expand `~`/env): if
   `run.hf_token_file` exists, set `os.environ["HF_TOKEN"]` (local-Llama download;
   the offline batch job with `HF_HUB_OFFLINE=1` won't use it); if
   `run.openai_api_key_file` exists, set `os.environ["OPENAI_API_KEY"]` (OpenAI
   backends — requires internet, so not usable on Sherlock's offline compute
   nodes). Both are optional; only the one matching the selected `llm_model` matters.
3. `config_to_prefs(config)` → prefs dict.
4. Mirror `run_dispatch.run()`'s dispatch: `plan_from_prefs(prefs)`; for each enabled
   output, `to_namespace(plan, prefs, level)`, share one `_align_cache`, call the
   matching batch runner. Fold the `edu_bands` list→str fix (`_prepare_runtime_args`).
5. Print per-level progress + final output locations.

### E. Glue

- `sherlock/run_container.sbatch`: replace the `python main.py --flags…` invocation
  with `python "$REPO/run_extract.py" "$REPO/config.yaml"` (config path configurable).
- `requirements.txt`: add `pyyaml>=6` (already present in the container — no rebuild).

## Data flow

```
GUI ─save─▶ prefs.json ─load_preferences()─▶ export_config.py ─prefs_to_config─▶ config.yaml
                                                                                     │ rsync
config.yaml ─read_config─▶ validate_config ─config_to_prefs (merge on DEFAULT_PREFS,
            run:→shared)─▶ prefs ─plan_from_prefs+to_namespace─▶ batch runners ─▶ CSVs
                              hf_token_file ─▶ HF_TOKEN (download-time only)
```

## Error handling

- Invalid config (unknown key / bad type / missing `run` folders when a matching
  level is enabled) → `run_extract.py` prints the collected errors and exits non-zero
  before any extraction starts.
- A missing `hf_token_file` / `openai_api_key_file` is not an error (offline local
  runs need no token); a path that is set but unreadable is a warning. If the
  selected `llm_model` needs a credential that isn't available (no key-file and no
  env var), the backend's own actionable error fires at load time.

## Testing

Per project TDD scope (mapping/validation = logic → TDD; script glue = test-after):
- `tests/utils/test_config_yaml.py` (TDD): `prefs_to_config` omits `shared` paths and
  token; `config_to_prefs` round-trips `prefs_to_config` output back to the same
  effective prefs (features/settings/categories); `config_to_prefs` maps `run:`
  folders into `shared.last_*_folder`; `validate_config` flags an unknown feature
  key, a bad scalar type, and an unknown section, and passes a good config.
- `tests/test_run_extract.py` (test-after): drives `run_extract.py`'s dispatch on a
  tiny text-only config (monkeypatched batch runners) — asserts the right levels
  fire from the output toggles and that `run:` folders reach the namespaces; one
  real end-to-end word-level run on a fixture transcript producing a CSV. Also
  assert credential loading: a present `hf_token_file`/`openai_api_key_file` sets
  the matching env var, a missing one is a silent no-op.
- `export_config.py` / `run_extract.py` entry points: a smoke test each.

## Risks / notes

- **Schema drift:** `config_to_prefs` always merges onto the current `DEFAULT_PREFS`,
  so an old config missing new keys still runs with new defaults; unknown keys are
  caught by `validate_config`. `config.yaml` need not carry `schema_version`.
- **Scripts live at repo root, not `sherlock/`** — `sherlock/` is git-untracked, and
  this is tracked/tested code. The container bind-mounts the repo, so the sbatch can
  call `$REPO/run_extract.py`.
- **Path expansion:** expand `~` and `$VARS` in `run:` paths and `hf_token_file`.
- `run_dispatch.run()` and `run_extract.py` now share dispatch shape; if that glue
  grows, consider extracting a shared helper — not needed for v1 (keep the small
  duplication over premature abstraction).
