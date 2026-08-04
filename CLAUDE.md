Hi Claude.

This is **SensiExtractor**, a word-level stimulus feature extraction pipeline for EEG naturalistic studies. It is part of a larger data curation effort managed one level up in `../Data Curation/`.

## Start here at every new session

1. Read **[`docs/master_plan.md`](docs/master_plan.md)** — single entry point for project state: mission, what's active now, completed batches, roadmap, conventions.
2. Check `memory/MEMORY.md` — durable user preferences and feedback that override defaults.

`master_plan.md` owns living state; this CLAUDE.md is stable orientation. If they conflict, master_plan wins.

## What it is

SensiExtractor is the standardized SOP tool for the SENSI naturalistic-EEG pipeline (PI: Blair Kaneshiro, ISC Wu Tsai dataset), part of the larger `../Data Curation/` effort. Takes audio (.wav) + transcript (.txt) → word-level feature CSV + document-level summary CSV. Features are toggled via the GUI or `config/features.yaml` (down to per-column granularity); CLI flags cover most but are group-level only, and some newer features (e.g. predictability) are prefs/GUI-driven with no CLI flag. Feature inventory is captured implicitly by `extractors/` filenames and the completed-batches table in `master_plan.md`.

## Project structure

```
SensiExtractor/
├── gui.py                 # Entry point (launches PySide6 GUI)
├── main.py                # CLI entry point + the three batch runners (word/doc/acoustic) + column-drop maps that ALL entry points share
├── run.py                 # Headless entry point (reads config/features.yaml + config/run.yaml)
├── requirements.txt       # pip deps (source; spliced into conda-lock.yml, never installed directly)
├── environment.yml        # conda deps (source; Python, Kaldi/MFA, compiled sci stack + platforms: key)
├── conda-lock.yml         # GENERATED lock (all 3 platforms, exact+hashed) — never hand-edit; regenerate via scripts/relock.py
├── setup.bat / setup.sh   # One-time setup scripts (install from conda-lock.yml)
├── config/                # features.yaml (GITIGNORED, global "what to extract"; GUI "Reset settings" rewrites exactly this file) + run.yaml (GITIGNORED, machine-specific: shared.analyzer_name / last_*_folder / theme / hf_token_file)
│   ├── features_defaults.yaml / run_defaults.yaml  # COMMITTED templates generated from code defaults (scripts/export_default_config.py) — copy to features.yaml / run.yaml to skip the GUI for a headless setup
│   └── hf_token.txt       # GITIGNORED — HuggingFace token file; shared.hf_token_file defaults to this name, resolved against config/
├── gui/
│   ├── __init__.py
│   ├── main_window.py        # Main QMainWindow + 8-item sidebar (Inputs, Word / Document / Sub-word / Sentence / Acoustic features, Run, Logs) + status strip; owns the settings_changed → refresh wiring between pages
│   ├── theme.py              # QSS stylesheets (SENSI cyan accent, light only)
│   ├── widgets.py            # Reusable widgets (SectionHeader, Card, FolderRow, ToggleRow, AmberDotIndicator); ToggleRow auto-renders ⓘ info-icon when a docs/features/<key>.md exists
│   ├── section_inputs.py     # Inputs page
│   ├── section_word_features.py     # Word-level features page (Structural × 8, Lex-ling × 4, Freq, Predictability, Semantic indicators, Semantic similarity (Word2Vec + BERT word), Prosody × 10); also hosts the HF-token box + Save button
│   ├── section_doc_features.py      # Document-level features page (Counts × 4, Aggregates × 4, Lex diversity, Readability, Category proportions, Composite psych × 4, Doc prosody × 16)
│   ├── section_subword_features.py  # Sub-word page (Syllable timing/labels/prosody, P-center, Phonemes, Phoneme surprisal)
│   ├── section_sentence_features.py # Sentence-level page (timing, prosody × 11, BERT sentence similarity)
│   ├── section_acoustic_features.py # Acoustic page (audio-only TRF time series: envelope, onset-env, log-mel, MFCC; shared analysis-rate grid)
│   ├── planner/              # SINGLE SOURCE OF TRUTH for which stages run — sits between prefs and main.py (see "How the pieces fit together")
│   │   ├── stages.py         # Stage + FeatureCluster dataclasses
│   │   ├── registry.py       # STAGES + CLUSTERS registry (lazy-bound extractor callables); add a feature here
│   │   ├── plan.py           # plan_from_prefs: DAG closure + topo sort over enabled clusters
│   │   └── emit.py           # to_namespace: plan → the argparse.Namespace main.py consumes
│   ├── section_run.py        # Run page (output checkboxes, readiness, Run button)
│   ├── section_logs.py       # Logs page (live viewer, past runs)
│   ├── status_strip.py       # Persistent bottom status strip
│   ├── category_picker.py    # Category picker widget (level-aware: word vs document)
│   ├── category_dialog.py    # Modal dialog wrapping CategoryPicker (one per level)
│   ├── preflight.py          # Continuous warning engine
│   ├── run_dispatch.py       # Background-thread orchestrator
│   ├── feature_docs.py       # Per-feature docs loader + _SHARED_DOC_MAP (toggle key → docs/features/*.md)
│   └── info_popup.py         # InfoIcon button + FeatureInfoDialog (renders the .md via QTextBrowser)
├── assets/
│   ├── icon.svg              # App icon (title bar, taskbar) — SENSI cyan glyph
│   ├── info.svg              # Info-icon glyph used by gui/info_popup.py
│   ├── check.svg, dash.svg, dot.svg  # Checkbox / radio indicator graphics
├── docs/
│   ├── master_plan.md        # Single entry point for project state (read first) — one line per completed batch
│   ├── changelog.md          # Archived full detail of every completed batch (NOT read each session; open for deep history)
│   ├── reproducibility.md    # Env lock runbook: two-tier model, dep changes, platform support, archival ritual
│   ├── features/             # Per-feature .md registry — one file per documented toggle (or one shared file for related toggles), surfaced in the GUI via gui/info_popup.py
│   └── superpowers/          # Batch specs + plans (one per feature batch)
├── extractors/            # One module per feature family; each is a pure(-ish) function the planner calls. Full inventory = these filenames.
│   ├── text_parser.py     # spaCy annotation (the token DataFrame every text feature builds on)
│   ├── aligner.py         # MFA forced alignment + persistence/reuse (ensure_alignments, freshness sidecars)
│   ├── align_merge.py     # Joins spaCy tokens ↔ MFA words (monotonic difflib); source of Onset/End on every word row
│   ├── audio_features.py  # Parselmouth pitch/volume/HNR Contours object reused by word/syllable/sentence prosody
│   ├── word_freq.py       # wordfreq Zipf; educator_freq.py = EWFG grade-band freqs
│   ├── predictability.py  # GPT-2/Llama surprisal, cloze, top-match (roster in predictability_models.py)
│   ├── predictability_models.py  # Model roster (slugs, hf_ids, native_ctx) + effective_context clamp — no torch
│   ├── semantic_domain.py # Empath/LIWC category indicators;  psych_composite.py = doc Authentic/Analytic/Tone/Clout
│   ├── lexical_norms.py   # AoA/concreteness/MRC/VAD lookups (dictionaries/*.parquet)
│   ├── lexical_diversity.py / readability.py     # doc-level TTR/MTLD, Flesch/FK
│   ├── prosody.py / sentence_prosody.py / syllable_prosody.py  # per-level prosody columns
│   ├── timing.py          # Word_Duration / Word_Gap (derived from Onset/End)
│   ├── syllabify.py / pcenter.py                 # Maximum-Onset syllabification; P-center
│   ├── phoneme_surprisal.py       # context-independent cohort surprisal + entropy
│   ├── word2vec_embed.py / bert_embed.py         # embedding .mat sidecars
│   └── acoustic_ts.py     # audio-only TRF time series (envelope/onset/log-mel/MFCC)
├── utils/                 # Cross-cutting infrastructure (not feature extractors)
│   ├── preferences.py     # Prefs schema + versioned migration chain + load/save API — the config contract
│   ├── config_yaml.py     # features.yaml + run.yaml split/merge/IO behind preferences.py
│   ├── credentials.py     # HF token resolution (resolve/read/export/save_hf_token)
│   ├── doc_summary.py     # Word-level frame → document-summary row (incl. dynamic predictability-column aggregation)
│   ├── file_pairing.py    # Matches transcripts ↔ audio into work items for the batch runners
│   ├── run_logger.py / session_units.py / system_info.py  # run-log artifacts, column→unit map, env capture
│   ├── transcript_prep.py # Number spelling (num2words), whitespace normalization
│   ├── textgrid_parser.py # MFA TextGrid → DataFrame
│   ├── download_models.py # One-time MFA/spaCy model download helper (run: python utils/download_models.py)
│   ├── download_lexical_norms.py # Builds dictionaries/*.parquet lexical-norm tables
│   └── validate_cloze.py  # LLM-cloze validation against Provo Corpus
├── sherlock/              # Apptainer build/run artifacts only (Apptainer.def, run_container.sbatch, RUNBOOK.md, .sif — git-ignored) + validate_llama_predictability.py
└── scripts/
    ├── relock.py          # Regenerates conda-lock.yml from environment.yml + requirements.txt
    ├── export_default_config.py  # Regenerates config/features_defaults.yaml + run_defaults.yaml from utils.preferences.DEFAULT_PREFS
    ├── import_smoke.py     # CI import smoke test (env-verify.yml)
    └── make_icons.py       # Regenerates assets/*.svg
```

CI: `.github/workflows/env-verify.yml` installs from `conda-lock.yml` on all three platforms (install+import smoke; merge gate) — see `docs/reproducibility.md`.

**Do not touch or analyze**: `env/` (path-based conda env, OS-specific binaries including Kaldi), `hf_cache/` (GPT-2 weights), `mfa_data/` (MFA acoustic models).

## How the pieces fit together (read before ANY change)

Most breakage here comes from editing one layer and forgetting its coupled layer — the GUI stops matching what runs, or an extractor change silently drops columns. The layers are decoupled **on purpose** so you edit in one place, but a *feature* spans several. Know which.

**The data flow (one direction, GUI / CLI / headless all converge on the same middle):**

```
utils/preferences.py  (the prefs dict — schema + migrations; stored as config/features.yaml + run.yaml)
        │  GUI section_*.py pages READ/WRITE prefs (they never decide what runs)
        ▼
gui/planner/  plan_from_prefs()  →  to_namespace()      ← THE hub: prefs → which stages run → args.Namespace
        │  (CLI build_parser() and run.py produce the SAME Namespace shape)
        ▼
main.py  run_word_level_batch / run_doc_level_batch / run_acoustic_level_batch
        │  calls extractors in planner-topo order; MFA is pre-aligned ONCE per dispatch (_prealign_batch)
        ▼
extractors/*  →  a pandas DataFrame (+ .mat sidecars)
        │  main.py applies per-column DROP maps (_word_columns_to_drop / _doc_columns_to_drop / _syllable_ / _phoneme_ / _sentence_)
        ▼
output CSVs / .mat under word/ · syllable/ · phoneme/ · sentence/ · document/ · acoustic/
```

**The golden rule:** the GUI decides *nothing* about execution. It only edits prefs. `gui/planner/` (fed by `docs/feature_dependencies.md`) is the single source of truth for which stages run and what `main.py` receives. If you find yourself adding "if this toggle then run that" logic in a `section_*.py` file, stop — it belongs in the planner registry.

**When you touch X, also touch Y (the checklist that prevents the breakage you've seen):**

| If you change… | You must also update… | Skip it and… |
|---|---|---|
| **An extractor's output columns** (rename/add/remove) | the matching `_*_COLUMN_MAP` in `main.py`, `utils/doc_summary.py` (doc aggregation; predictability aggregates via `_PREDICTABILITY_COL_RE`), the `docs/features/<key>.md`, and tests | GUI per-column toggles silently drop the wrong column, or doc summary omits it |
| **Add a feature / toggle** | (1) extractor module, (2) one `Stage` + one `FeatureCluster` in `gui/planner/registry.py`, (3) prefs key + version bump + migration in `utils/preferences.py`, (4) toggle widget in `gui/section_*.py`, (5) column-drop map in `main.py` if per-column droppable, (6) `docs/features/<key>.md`, (7) tests + `python scripts/export_default_config.py` | the schema-drift guard test fails, or the toggle exists in the GUI but never runs |
| **Prefs schema** (any key) | bump `SCHEMA_VERSION` + add a migration step; rerun `scripts/export_default_config.py` (drift-guard test enforces); check the planner registry schema-drift test | old configs crash on load, or CI fails on stale templates |
| **A `section_*.py` page layout/wiring** | the `settings_changed → refresh` connections in `gui/main_window.py` (pages don't refresh each other directly) | a change on one page leaves Run-tab readiness or another page stale |
| **A GUI numeric default / range** (e.g. a spinbox) | keep it equal to the prefs default in `utils/preferences.py` AND the extractor's clamp (e.g. `predictability_models.effective_context`) | the GUI and the actual run disagree on the value |

**Two things easy to forget:**
- **Word and Document tabs each re-run the full pipeline** per document. MFA is the expensive step and must run **at most once per doc** — it's cached across both passes via `_prealign_batch` / `ensure_alignments`. Never add a code path that re-aligns per pass.
- **`MKL_THREADING_LAYER` must be set at every entry point** (`main.py`, `gui.py`, `run.py`, `tests/conftest.py`) — a real run crashes on AMD/Windows without it (the mel matmul), and passing tests do **not** prove a real run works. See `memory/project-mkl-threading-entrypoints.md`.

**Verifying a change:** the suite passing ≠ the feature works. For anything touching an extractor or the GUI, run the real path (`python gui.py` or `python main.py …`) on a real stimulus — the fixtures in this repo are empty placeholders, so MFA-dependent output can't be exercised by tests alone.

## Environment

The Python environment is a **path-based conda env at `env/`** (a "prefix" env — identified by its directory path, not a registered name). It is NOT a Python venv, and there is no env named `sensi`. MFA + Kaldi are installed via conda inside it.

Activate with `conda activate ./env` from the project root. `./env` is built by running `setup.sh` / `setup.bat`, which installs it from the committed `conda-lock.yml` (a generated, exact-pinned lock for Windows/Linux/Apple-Silicon macOS) — not from `pip install -r requirements.txt` directly. To change a dependency, see **[`docs/reproducibility.md`](docs/reproducibility.md)**: edit `environment.yml`/`requirements.txt`, run `python scripts/relock.py`, commit the regenerated `conda-lock.yml`.

## Running

```bash
conda activate ./env

# GUI (PySide6)
python gui.py

# CLI — full pipeline
python main.py --audio stimulus.wav --transcript stimulus.txt

# Text features only (no audio)
python main.py --transcript stimulus.txt --no-align

# Headless — reads config/features.yaml + config/run.yaml (same planner as the GUI)
python run.py [--config-dir DIR]   # DIR defaults to <repo>/config
```

## Timeless design decisions

These are stable across all batches. Batch-specific decisions live in `master_plan.md`'s completed-batches table.

- **MFA for forced alignment**: Montreal Forced Aligner (Kaldi-based), chosen for accuracy and active maintenance.
- **Sliding window for GPT-2**: handles naturalistic texts that exceed the 1024-subword limit.
- **Models stored locally**: `hf_cache/` and `mfa_data/` ship with the project so collaborators skip ~1.5 GB of downloads.
- **Two coordinate systems, easy to confuse**: `Onset`/`End` are acoustic word boundaries in seconds (MFA); lowercase `start`/`end` are character offsets within the sentence string (spaCy). Never conflate.

## Relationship to the parent project

The SENSI naturalistic stimuli (audio + transcripts) live in `../Data Curation/`. This tool produces the feature CSVs that feed the publication-ready stimulus+features datasets. The parent project tracks which features are in-use in EEG analyses and whether new extractors are needed — refer there for feature scope decisions.
