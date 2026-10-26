# Output Switches (section switches + Run-tab levels) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the confusing "Enable all …" master toggles with a uniform section switch
on every card, and replace the Run tab's seven mixed output toggles with one switch per
level, without changing what any existing config produces.

**Architecture:** Storage changes once (prefs schema v20 → v21: five `output.level_*` keys
replace the per-file output keys; new section keys for cards that had none). The planner
translates level keys into the *same* internal artifact ids it already uses (`features`,
`syllables`, …), so `emit.py`, `main.py`, `run.py`, `run_dispatch.py`, preflight and
`session_units.py` keep consuming `plan.output_csvs` unchanged. The GUI gets one new card
API (switch + tag + groups + shared settings) that every feature page adopts, and a new
Run-tab level list.

**Tech Stack:** Python 3.10, PySide6 (Qt 6), pytest + pytest-qt, YAML config
(`utils/preferences.py` + `utils/config_yaml.py`).

**Spec:** `docs/superpowers/specs/2026-10-26-section-switch-design.md` (owner-approved
2026-10-26). Read it before starting; this plan argues from it.

**Branch:** create `feat/output-switches` from the current tip of `fix/mfa-reporting`
(unmerged at the time of writing; it carries `utils/config_checks.py` and the Run-tab
readiness changes this work builds on). If the owner has merged it, branch from `main`.

## Global Constraints

- Golden rule (CLAUDE.md): GUI pages only read/write prefs; **which stages run is decided in
  `gui/planner/`**. No "if toggle then run" logic in `section_*.py`.
- Schema: `SCHEMA_VERSION` 20 → **21**, with a finalize block `if version < SCHEMA_VERSION_V21`
  in `utils/preferences.py:_merged_from_user` (pattern: the v20 block). Add
  `SCHEMA_VERSION_V21` to the "already-sectioned" version tuple (`preferences.py:~986`).
- New section keys default **True** (included); level defaults match today's outputs:
  `level_word: True`, every other level `False`.
- Existing master key names are kept (old configs stay valid).
- Visible copy (verbatim from the spec): tag "Included" / "Included — nothing selected" /
  "Left out of outputs"; links "All · None"; level names "Word level", "Document level",
  "Sub-word level", "Sentence level", "Acoustic level"; level line
  "Everything ticked on the <Page> features page." (Acoustic: "Audio-only time series from
  the Acoustic features page; doesn't need transcripts."). The words "Enable all" / "Emit …
  columns" must not appear in the GUI.
- Switch colours: on `#00B2D0` (= `gui/theme.py:ACCENT_COLOR`), off `#b9c0c5`; 4 px left card
  edge in the same colours; header background `#f4f5f6` when left out.
- After any prefs change: `python scripts/export_default_config.py` (drift-guard test).
- Test-after for GUI/prefs (project TDD scope); planner logic gets tests in the same task.
- Python for commands: `./env/python.exe` with `env`, `env/Scripts`, `env/Library/bin`
  prepended to `PATH` (memory: mfa.exe / libsndfile DLLs). Suite: `pytest -q -p no:cacheprovider`.

## Review Focus

1. **Old configs must produce identical files.** A v20 config with e.g. syllables on /
   phonemes off, or embeddings on / word CSV off, must yield the same `plan.output_csvs`
   and `plan.stages_to_run` after migration (Task 1 snapshots + Task 3 equivalence test).
2. **Document predictability means with Word Predictability off** must compute (owner
   decision) — today they silently don't (Task 3 test).
3. **Greying must reach every control in a card**, including nested rows (Predictability's
   model rows, the HF-token row excluded), on first paint *and* after a toggle (Task 4/5 tests).
4. **A section switch never changes feature ticks**, and "All · None" never changes the
   switch (Task 4 test).
5. **A hand-edited v20 `features.yaml` must still pass `run.py` validation** (old output keys
   tolerated as retired with a warning, not rejected) (Task 2 test).

---

### Task 1: Snapshot today's planner behaviour for representative v20 configs

**Files:**
- Create: `tests/fixtures/output_switch_snapshots/make_snapshots.py`
- Create: `tests/fixtures/output_switch_snapshots/v20_cases.json` (generated, committed)

**Interfaces:**
- Produces: `v20_cases.json` = list of `{"name": str, "prefs_overrides": dict, "output_csvs":
  list[str] (sorted), "stages_to_run": list[str]}` computed by the **current, unmodified**
  `gui.planner.plan_from_prefs(merged)` where `merged` = `deepcopy(DEFAULT_PREFS)` deep-merged
  with `prefs_overrides`.

- [ ] **Step 1:** Write `make_snapshots.py` with these cases (names exact):
  `defaults`; `all_on` (every `features`/`output` bool True, `predictability.models=["gpt2"]`,
  metrics surprisal+cloze True); `embeddings_without_word_csv` (`word_level_csv` False,
  `word2vec_mat`/`bert_word_mat` True, `master_word2vec`/`master_bert_word` True);
  `syllables_not_phonemes` (`syllables_csv` True, `phonemes_csv` False, every `feat_syllable_*`
  and `feat_phoneme_*` True); `psr_mat_without_phonemes` (`phonemes_csv` False,
  `phoneme_surprisal_mat` True, `feat_phoneme_surprisal` True); `bert_sentence_without_csv`
  (`sentences_csv` False, `bert_sentence_mat` True, `master_bert_sentence` True, every
  `feat_sentence_*` True); `acoustic_only` (`word_level_csv` False, `acoustic_ts_mat` True,
  `feat_acoustic_ts_envelope` True); `doc_only_with_pred_means` (`word_level_csv` False,
  `document_summary` True, `doc_agg_predictability` True, gpt2+surprisal).
- [ ] **Step 2:** Run it on the unmodified branch tip; commit the JSON.
  Run: `./env/python.exe tests/fixtures/output_switch_snapshots/make_snapshots.py`
  Expected: `v20_cases.json` with 8 entries; `doc_only_with_pred_means` has no `"predictability"`
  in `stages_to_run` (today's bug — Task 3 changes exactly this case).
- [ ] **Step 3: Commit** — `test: snapshot v20 planner outputs for the output-switch migration`

### Task 2: Prefs schema v21 — levels, new section keys, migration

**Files:**
- Modify: `utils/preferences.py` (v21 constants after the v20 block ~line 896; finalize block
  after the v20 finalize ~line 1171; version tuple ~986; active aliases ~921)
- Modify: `utils/config_yaml.py` (`RETIRED_FEATURE_KEYS` gains the retired output keys)
- Regenerate: `config/features_defaults.yaml`, `config/run_defaults.yaml`
- Test: `tests/utils/test_preferences_v21.py`

**Interfaces:**
- Produces (exact keys):
  - `output`: add `level_word`, `level_document`, `level_subword`, `level_sentence`,
    `level_acoustic` (bool). Remove `word_level_csv`, `document_summary`, `syllables_csv`,
    `phonemes_csv`, `sentences_csv`, `acoustic_ts_mat`, `phoneme_surprisal_mat`,
    `word2vec_mat`, `bert_word_mat`, `bert_sentence_mat`.
  - `output` (doc-page section switches, default True): `master_doc_aggregates`,
    `master_doc_lex_diversity`, `master_doc_readability`, `master_doc_category_proportions`.
  - `features` (default True): `master_semantic_indicators`, `master_semantic_similarity`,
    `master_sentence_similarity`; (default False) `feat_phoneme_surprisal_mat`.
  - `master_syllable_pcenter` (exists) becomes the P-center card's switch.
  - `_migrate_outputs_to_levels(merged: dict, old_output: dict) -> None` in `preferences.py`.

- [ ] **Step 1: Implement the constants and `_migrate_outputs_to_levels`.** Rules (exact;
  `wl`=word_level_csv, `w2v`=word2vec_mat, `bw`=bert_word_mat, `ds`=document_summary,
  `syl`=syllables_csv, `ph`=phonemes_csv, `psrm`=phoneme_surprisal_mat, `sent`=sentences_csv,
  `bs`=bert_sentence_mat, `ac`=acoustic_ts_mat, missing → that key's v20 default):
  - `level_word = wl or w2v or bw`; `level_document = ds`; `level_subword = syl or ph or psrm`;
    `level_sentence = sent or bs`; `level_acoustic = ac`.
  - `master_word2vec &= w2v`; `master_bert_word &= bw`; `master_bert_sentence &= bs`;
    `feat_phoneme_surprisal_mat = psrm`.
  - If `level_word and not wl`: set False `master_structural`, `master_lexling`,
    `master_frequency`, `master_predictability`, `master_lexical_norms`, `master_acoustic`,
    `master_word_prosody`, `master_semantic_indicators`.
  - If `level_subword and not syl`: set False `master_syllable_intervals`,
    `master_syllable_labels`, `master_syllable_prosody`, `master_syllable_pcenter`.
  - If `level_subword and not ph`: set False `master_phoneme_intervals`,
    `feat_phoneme_surprisal`, `feat_phoneme_cohort_entropy`.
  - If `level_sentence and not sent`: set False `master_sentence_timing`,
    `master_sentence_prosody`.
  - Then delete the retired output keys. Runs only for sources `< SCHEMA_VERSION_V21`.
- [ ] **Step 2:** Add the retired output keys to `RETIRED_FEATURE_KEYS` (section `output`)
  so `validate_features_yaml` warns instead of erroring on a v20 file.
- [ ] **Step 3: Write tests** in `tests/utils/test_preferences_v21.py`:
  - `test_defaults_v21_levels`: `DEFAULT_PREFS["output"]["level_word"] is True` and the other
    four levels False; no retired key in `DEFAULT_PREFS["output"]`.
  - `test_v20_file_migrates_to_levels` (parametrize over the 8 Task-1 cases): load a v20
    `features.yaml` built from the case into a tmp `SENSI_CONFIG_DIR`; after
    `load_preferences()`, `schema_version == 21` and no retired key remains.
  - `test_v20_yaml_validates_with_warning`: `validate_features_yaml(v20_payload) == []` and
    `retired_keys_present(v20_payload)` names `word_level_csv`.
- [ ] **Step 4:** Run `python scripts/export_default_config.py`; run
  `pytest -q -p no:cacheprovider tests/utils` — Expected: PASS (fix the drift-guard /
  schema tests that pin v20 numbers to v21).
- [ ] **Step 5: Commit** — `feat(prefs): v21 output levels and section switches (migration keeps outputs identical)`

### Task 3: Planner — levels → artifacts, section gating, document predictability means

**Files:**
- Modify: `gui/planner/plan.py` (`_OUTPUT_CSV_MAP` → `LEVEL_ARTIFACTS`; `_active_output_csvs`;
  `OUTPUT_KEYED_TOGGLES`; `_predictability_active`; `plan_from_prefs`)
- Modify: `gui/planner/registry.py` (clusters listed below)
- Modify: `gui/planner/emit.py` (only if Step 4's test shows the doc namespace lacks predictability)
- Test: `tests/planner/test_output_levels.py`

**Interfaces:**
- Consumes: Task 2 keys.
- Produces: `LEVEL_ARTIFACTS: dict[str, tuple[str, ...]]` (public, used by Task 6) =
  `{"level_word": ("features", "word2vec", "bert_word"), "level_document":
  ("documents_summary",), "level_subword": ("syllables", "phonemes", "phoneme_surprisal_mat"),
  "level_sentence": ("sentences", "bert_sentence"), "level_acoustic": ("acoustic",)}` in
  `gui/planner/plan.py`; `_active_output_csvs(prefs) -> frozenset[str]` = union of artifacts
  of levels that are on. `plan.output_csvs` keeps its meaning (live artifacts).

- [ ] **Step 1: Registry changes** (cluster `name` → change):
  `word2vec` → `toggle_keys=("master_word2vec",)`, `master_toggle="master_semantic_similarity"`;
  `bert word embeddings` → `toggle_keys=("master_bert_word",)`,
  `master_toggle="master_semantic_similarity"`; `bert sentence embeddings` →
  `toggle_keys=("master_bert_sentence",)`, `master_toggle="master_sentence_similarity"`;
  `phoneme surprisal mat` → `toggle_keys=("feat_phoneme_surprisal_mat",)`,
  `master_toggle="master_phoneme_surprisal"`; `semantic indicators` →
  `master_toggle="master_semantic_indicators"`; `p-center` →
  `master_toggle="master_syllable_pcenter"`; `acoustic time-series` →
  `master_toggle="level_acoustic"`; `doc agg zipf freq`, `doc agg edu zipf`,
  `doc agg predictability`, `doc lexical norms` → `master_toggle="master_doc_aggregates"`;
  `doc lex diversity` → `"master_doc_lex_diversity"`; `doc readability` →
  `"master_doc_readability"`; `doc category proportions` →
  `"master_doc_category_proportions"`. Add every new `output`-section key (`level_acoustic`,
  the four `master_doc_*`) to `OUTPUT_KEYED_TOGGLES`.
- [ ] **Step 2: `_predictability_active(prefs, active_csvs) -> tuple[bool, bool]`** returning
  `(word_path, doc_path)`: `word_path` = today's rule; `doc_path` = `"documents_summary"` in
  `active_csvs` and the `doc agg predictability` cluster is active and models+metrics are
  non-empty (independent of `master_predictability`). Update every caller to the tuple
  (`git grep -n _predictability_active`). In `plan_from_prefs`: stage
  `predictability` (+`spacy_parse`) is required if either path; `predictability_columns` and
  adding `"features"` to `live_csvs` happen only for `word_path`.
- [ ] **Step 3: Write tests** in `tests/planner/test_output_levels.py`:
  - `test_migrated_v20_cases_plan_identically` (parametrize over `v20_cases.json` except
    `doc_only_with_pred_means`): migrate the case (via `_merged_from_user` on a v20 dict) →
    `sorted(plan.output_csvs) == case["output_csvs"]` and
    `list(plan.stages_to_run) == case["stages_to_run"]`.
  - `test_doc_means_compute_with_word_predictability_off`: the migrated
    `doc_only_with_pred_means` case plus `master_predictability: False` →
    `"predictability" in plan.stages_to_run`, `"features" not in plan.output_csvs`,
    `plan.predictability_columns == ()`.
  - `test_section_off_drops_artifact`: Sub-word level on, `master_syllable_intervals`,
    `_labels`, `_prosody`, `master_syllable_pcenter` False → `"syllables" not in output_csvs`.
  - `test_level_off_drops_all_its_artifacts`: everything ticked, `level_word` False →
    none of `features`, `word2vec`, `bert_word` in `output_csvs`.
- [ ] **Step 4:** Add `test_doc_namespace_carries_predictability`: for the doc-means case,
  `to_namespace(plan, prefs, level="doc").predictability["models"] == ["gpt2"]` and
  `no_doc_predictability is False`. Fix `emit.py` only if it fails.
- [ ] **Step 5:** Run `pytest -q -p no:cacheprovider tests/planner tests/gui/test_run_dispatch.py`
  — Expected: PASS after updating tests that set the retired output keys to the level keys.
- [ ] **Step 6: Commit** — `feat(planner): output levels, uniform section gating, doc predictability means`

### Task 4: Card widget — switch, tag, groups, shared settings, full-depth greying

**Files:**
- Modify: `gui/widgets.py` (`ClusterCard`; new `SectionSwitch`)
- Modify: `gui/theme.py` (QSS for `SectionTag`, card edge states, `All · None` link buttons)
- Test: `tests/gui/test_section_card.py`

**Interfaces:**
- Produces:
  - `class SectionSwitch(QAbstractButton)` — checkable pill, `sizeHint` 38×20, painted per
    Global Constraints colours; emits `toggled(bool)`.
  - `ClusterCard(title: str, subtitle: str, master_key: str, *, master_section: str =
    "features", available: bool = True, disabled_subtitle: str | None = None, parent=None)` —
    `master_label` parameter removed. Header row: title `QLabel`, `SectionSwitch`, tag
    `QLabel` (objectName `"SectionTag"`), stretch.
  - `ClusterCard.add_child(widget)` / `add_child_layout(layout)` — gate **recursively** (walk
    nested layouts and child widgets).
  - `ClusterCard.add_group(label: str, content: QWidget | QLayout) -> None` — group label +
    two flat link buttons "All" and "None" (separated by " · ") that check/uncheck every
    `ToggleRow`/`QCheckBox` inside `content`; content and links are gated.
  - `ClusterCard.add_shared(widget: QWidget, note: str) -> None` — never gated; shown in a
    bordered sub-box with `note` as a small label.
  - `ClusterCard.tag_text() -> str` — `"Left out of outputs"` if the switch is off, else
    `"Included"` if any gated checkable is checked, else `"Included — nothing selected"`;
    refreshed on the switch and on every gated checkable's toggle.
  - `ClusterCard.set_master_checked(v: bool)` — programmatic set without re-saving (used by
    Task 6 to sync the Acoustic card with the Run tab).
  - Existing `master_toggle()` accessor returns the `SectionSwitch` (tests use it).
- [ ] **Step 1: Implement** the above. Persist the switch as today (`save_preferences(
  {master_section: {master_key: v}})`) and emit `settings_changed`. `_perma_disabled`
  widgets stay disabled regardless.
- [ ] **Step 2: Write tests** (`qtbot`, tmp prefs as in `tests/gui/test_section_run.py`):
  - `test_switch_off_greys_nested_rows`: a card with a grid added via `add_child_layout`
    containing an `QHBoxLayout` row with a `QCheckBox` → off ⇒ the checkbox `isEnabled()` is
    False; on ⇒ True; also on first paint with the key saved False.
  - `test_switch_does_not_change_ticks`: tick state of children identical before/after
    off→on.
  - `test_all_none_links_set_group_only`: "None" unchecks every group checkbox and leaves
    the switch on and the prefs key unchanged.
  - `test_shared_widget_never_greyed`: `add_shared` widget enabled while off.
  - `test_tag_states`: the three tag strings exactly as in Global Constraints.
- [ ] **Step 3:** Run `pytest -q -p no:cacheprovider tests/gui/test_section_card.py` — PASS.
- [ ] **Step 4: Commit** — `feat(gui): section switch card (switch, tag, groups, shared settings)`

### Task 5: Feature pages adopt the new card everywhere

**Files:**
- Modify: `gui/section_word_features.py`, `gui/section_doc_features.py`,
  `gui/section_subword_features.py`, `gui/section_sentence_features.py`,
  `gui/section_acoustic_features.py`
- Modify: `gui/main_window.py` (only if a new cross-page refresh is needed — the Acoustic card
  ↔ Run-tab sync goes through the existing `settings_changed → refresh` wiring)
- Test: `tests/gui/test_feature_pages_switches.py` + update existing page tests

**Interfaces:**
- Consumes: Task 4 card API, Task 2 keys.

- [ ] **Step 1: Convert every card** (spec "Scope of cards"): every `ClusterCard(...)` drops
  `master_label`; every plain `Card` that holds features becomes a `ClusterCard` with its
  Task-2 key — Semantic indicators (`master_semantic_indicators`), Semantic similarity
  (`master_semantic_similarity`; the Word2Vec/BERT rows become plain ticks bound to
  `master_word2vec`/`master_bert_word` only — they no longer write `output.*_mat`),
  Aggregates / Lexical diversity / Readability / Category proportions (`master_doc_*`,
  `master_section="output"`), P-center (`master_syllable_pcenter`), Sentence similarity
  (`master_sentence_similarity`; `master_bert_sentence` becomes a tick). **G2P stays a plain
  `Card`** (no switch). The Acoustic card's switch is `output.level_acoustic`
  (`master_section="output"`), refreshed from the Run tab via `settings_changed`.
- [ ] **Step 2: Groups and shared settings.** Use `add_group` for every labelled checkbox
  group (Predictability "Models"/"Metrics", frequency bands, prosody columns, …) — this also
  gates Predictability's rows that are ungated today. Word Prosody's pitch units / floor /
  ceiling go in `add_shared(widget, "Pitch settings (used by all prosody sections)")`. The
  HuggingFace token box stays outside the gated area. Section-only settings (pause threshold,
  P-center k, phoneme analysis rate, late-AoA threshold, acoustic params, context spins) are
  gated with their card.
- [ ] **Step 3: Phoneme surprisal card** gets a tick "Impulse-train .mat
  (`<stim>_phoneme_surprisal.mat`)" bound to `feat_phoneme_surprisal_mat`; its subtitle drops
  "when enabled on the Run page".
- [ ] **Step 4: Write tests** in `tests/gui/test_feature_pages_switches.py`:
  - `test_no_enable_all_wording`: across all five pages, no widget text contains
    `"Enable all"` or `"Emit "`.
  - `test_predictability_models_greyed_when_left_out`: Word page, `master_predictability`
    False ⇒ every model checkbox and context spin disabled; HF-token field enabled.
  - `test_pitch_settings_stay_editable`: `master_word_prosody` False ⇒ pitch floor spin
    enabled.
  - `test_g2p_card_has_no_switch`.
  - `test_acoustic_card_switch_is_level_acoustic`: toggling the card writes
    `output.level_acoustic`.
- [ ] **Step 5:** Run `pytest -q -p no:cacheprovider tests/gui` — PASS (update page tests that
  asserted on `master_label` text or `output.*_mat`).
- [ ] **Step 6: Commit** — `feat(gui): every feature card uses the section switch`

### Task 6: Run tab — one switch per level with its file list

**Files:**
- Modify: `gui/section_run.py` (replace the seven `ToggleRow`s and `_ARTIFACT_DESC`)
- Modify: `gui/preflight.py` (empty-plan message)
- Test: `tests/gui/test_section_run.py` (update + add)

**Interfaces:**
- Consumes: `LEVEL_ARTIFACTS` (Task 3), `SectionSwitch` (Task 4).
- Produces: `_LEVEL_ROWS: list[tuple[str, str, str]]` = (level key, name, description) in
  display order Word, Document, Sub-word, Sentence, Acoustic, and `_ARTIFACT_PATHS: dict[str,
  str]` artifact id → path template (reuse today's strings without the "(Word page)" notes).

- [ ] **Step 1: Implement** one row per level: `SectionSwitch` bound to `output.<level key>`,
  bold name, count tag, description line (Global Constraints copy), then one monospace line
  per artifact in `LEVEL_ARTIFACTS[key]`: plain if in `plan.output_csvs`, struck through
  otherwise. Count tag: `"Left out"` (level off); `"Nothing selected"` (on, no live artifact);
  else `"N file(s) per stimulus"` with correct singular/plural (`"1 file per stimulus"`, `"3 files per stimulus"`), except Document `"1 file for the whole run"` and Acoustic
  `"1 file per audio file"`. Remove the "Will produce" label (the rows replace it); keep
  readiness/warnings below. Toggling a row saves, calls `refresh()`, emits `settings_changed`.
- [ ] **Step 2:** `preflight.py` empty-plan message → `"Nothing will be written — turn on at
  least one level on the Run page, and tick a feature on that level's page."`
- [ ] **Step 3: Tests:** `test_run_tab_has_five_level_switches` (names exact, order exact);
  `test_level_row_strikes_unproduced_files` (Sub-word on, only syllable features ticked ⇒
  phoneme paths struck, count tag `"1 file per stimulus"`); `test_acoustic_level_syncs_card` (Run-tab toggle updates
  `output.level_acoustic`, the Acoustic card reflects it after `settings_changed`). Update
  the existing tests that used `word_level_csv`/`acoustic_ts_mat`.
- [ ] **Step 4:** Run `pytest -q -p no:cacheprovider` (full suite) — Expected: all pass.
- [ ] **Step 5: Commit** — `feat(gui): Run tab shows one switch per output level`

### Task 7: Docs, real-GUI verification, bookkeeping

**Files:**
- Modify: any doc quoting the old Run-tab toggles or "Enable all" (find with
  `git grep -n "word_level_csv\|syllables_csv\|phonemes_csv\|sentences_csv\|acoustic_ts_mat\|phoneme_surprisal_mat\|Enable all\|Run page\|Run tab" -- README.md docs sherlock/RUNBOOK.md CLAUDE.md`)
- Modify: `docs/master_plan.md` (§3 row; §4 mark "Gating simplification" as shipped with this
  design's scope)
- Modify: `config/*_defaults.yaml` only via `scripts/export_default_config.py`

- [ ] **Step 1:** Update the docs found by the grep (README headless example keys,
  `docs/features/phoneme_surprisal.md` "Run page" wording, RUNBOOK config keys).
- [ ] **Step 2: Real headless check** — copy the owner's `config/` to a scratch dir (never
  edit the live one), run `run.py --config-dir <scratch>` on `validation/data` with the
  pre-change branch and the new branch into two output folders; every CSV / `.mat` must be
  identical (`numpy.array_equal(..., equal_nan=True)` for `.mat`; `DataFrame.equals` for CSVs,
  ignoring `logs/`).
- [ ] **Step 3: Real GUI check** (`python gui.py`, offscreen is not enough): on every page,
  flip each card's switch and confirm greying, tag text, ticks preserved, and persistence
  after restart; on the Run tab flip each level and confirm the file lists; toggle Acoustic
  in both places; run once with Sub-word on and phonemes left out and confirm no
  `phoneme/` CSV is written.
- [ ] **Step 4:** Add the §3 row to `docs/master_plan.md`; run the full suite once more.
- [ ] **Step 5: Commit** — `docs: output switches; master_plan row`
