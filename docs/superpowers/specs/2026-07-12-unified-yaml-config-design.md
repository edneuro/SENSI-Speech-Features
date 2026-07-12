# Unified YAML config (Project C) — design

**Date:** 2026-07-12
**Status:** approved by user — READY FOR PLANNING/IMPLEMENTATION in a fresh session
**Handoff note:** brainstorming is COMPLETE (all decisions below are user-approved
verbatim). The next step is the superpowers **writing-plans** skill → an
implementation plan at `docs/superpowers/plans/2026-07-12-unified-yaml-config.md`
→ execute task-by-task (subagent-driven, review gate per task, TDD for logic)
on a feature branch, exactly like Projects A/B (see the master_plan §3 rows of
2026-07-10/11 for the precedent).

---

## 1. Goal

Replace `config/preferences.json` with **two YAML files** that are the single
source of config everywhere — GUI, CLI batches, and Sherlock/headless runs —
and promote the headless runner out of `sherlock/` to a global entry point.

## 2. User-approved decisions (do not re-litigate)

1. **YAML fully replaces preferences.json** as the live store. The GUI reads
   and writes YAML directly. No json↔yaml sync/export step remains.
2. **Two files:**
   - `config/features.yaml` — the global, shareable "what to extract" file:
     `schema_version`, `features`, `output`, `predictability`
     (models/metrics/context), `semantic`, `prosody`, `acoustic_ts`,
     `phoneme_surprisal`, `lexical_norms`. **GUI Reset-to-defaults rewrites
     exactly this file.** This is also what Sherlock consumes, so a Reset
     applies everywhere.
   - `config/run.yaml` — machine/run-specific: input/output folders, analyzer
     name, token-file paths (`hf_token_file`, currently sherlock-only).
     **Never touched by Reset**; per machine.
3. The user explicitly flagged that the current sherlock config mixes run
   info with feature selection — the split above is the fix.

## 3. Architecture

- **Same internal API, new storage.** `utils/preferences.py`'s
  `load_preferences()` / `save_preferences()` keep their signatures and the
  same merged-dict shape every consumer already uses. Internally they read the
  two YAMLs (merging into one dict: `shared` section ← run.yaml, the rest ←
  features.yaml) and route writes by section. GUI/planner/extractors/main.py
  need no changes to how they consume prefs.
- **One-time migration.** On load: if the YAML pair exists, use it. Else if
  `config/preferences.json` exists, run it through the EXISTING v1→v19
  migration chain, split into the two YAMLs, write them, and rename the json
  to `preferences.json.bak`. The whole json migration machinery is retained
  as the legacy import path (its tests keep guarding it).
- **Env overrides for tests/tooling:** new `SENSI_CONFIG_DIR` (directory
  containing both YAMLs) for isolation fixtures; keep `SENSI_PREFS_PATH` as
  the legacy-json path override used by migration tests.
- **Headless runnable:** promote `sherlock/run_extract.py` → top-level
  `run.py` reading the same two YAMLs (word/document/acoustic levels via the
  planner's emit path, like today). Retire `sherlock/export_config.py` and
  `sherlock/config.yaml`. `sherlock/` keeps only container/HPC-specific
  pieces (Apptainer.def, submit scripts). Update `utils/config_yaml.py`
  accordingly (its `config_to_prefs` logic becomes/feeds the YAML loader).
- **YAML lib:** pyyaml — already a dependency (used by utils/config_yaml.py).

## 4. Judgment calls already accepted by the user

- YAML keys/shapes are 1:1 with the current prefs schema (format change +
  section split only), so versioning/migration logic carries over and diffs
  stay reviewable. `schema_version` lives in features.yaml.
- Files are written with regenerated section-header comments (like the old
  sherlock export); hand-added comments do NOT survive a programmatic save —
  documented behavior.
- Reset keeps folders/analyzer (run.yaml untouched) — matches today's Reset.

## 5. Testing expectations

- TDD for the loader/saver/split/migration logic (utils layer).
- Update the three prefs-isolation conftest fixtures (gui/integration/root)
  from a json path to `SENSI_CONFIG_DIR`; migration tests keep exercising the
  legacy json path deliberately.
- Round-trip tests: save→load equality per section; Reset rewrites
  features.yaml only; json→yaml conversion once + .bak; run.py smoke (reuse
  the existing sherlock run_extract tests, relocated).
- Full suite green at the end (baseline at handoff: **728 passed**).

## 6. Out of scope

- No prefs schema version bump needed if keys are unchanged (the storage
  format is not part of the schema); bump only if a key actually moves in a
  way the migration chain must know about (the shared→run.yaml split happens
  at the storage layer, not in the dict shape).
- No GUI layout changes (the config pages just keep working).

## 7. Context a fresh session needs

- Read `docs/master_plan.md` first (per CLAUDE.md); its §3 top rows document
  Projects A (predictability unification, prefs v18), B (per-model context
  size, v19), and the release-hardening sweep — this project builds on them.
- Current baseline: branch `main` @ b0fcbca+, suite 728 passed. Recent
  user-driven GUI fixes: HF-token box always visible; context spinners always
  editable with adaptive steps; metric ⓘ doc icons restored; no horizontal
  scrollbar (regression tests in tests/gui/test_page_widths.py).
- Known pending items (unrelated to this project): real gated-Llama
  end-to-end validation on user hardware; doc drift in
  `docs/feature_dependencies.md` + `Feature_Extraction_Guide.md` (still
  describe pre-unification Surprisal/LLM_* columns).
