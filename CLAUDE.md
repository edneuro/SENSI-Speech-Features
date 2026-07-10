Hi Claude.

This is **SensiExtractor**, a word-level stimulus feature extraction pipeline for EEG naturalistic studies. It is part of a larger data curation effort managed one level up in `../Data Curation/`.

## Start here at every new session

1. Read **[`docs/master_plan.md`](docs/master_plan.md)** — single entry point for project state: mission, what's active now, completed batches, roadmap, conventions.
2. Check `memory/MEMORY.md` — durable user preferences and feedback that override defaults.

`master_plan.md` owns living state; this CLAUDE.md is stable orientation. If they conflict, master_plan wins.

## What it is

SensiExtractor is the standardized SOP tool for the SENSI naturalistic-EEG pipeline (PI: Blair Kaneshiro, ISC Wu Tsai dataset), part of the larger `../Data Curation/` effort. Takes audio (.wav) + transcript (.txt) → word-level feature CSV + document-level summary CSV. All features toggleable via GUI or CLI flags. Feature inventory is captured implicitly by `extractors/` filenames and the completed-batches table in `master_plan.md`.

## Project structure

```
SensiExtractor/
├── gui.py                 # Entry point (launches PySide6 GUI)
├── main.py                # CLI entry point
├── requirements.txt       # pip deps (source; spliced into conda-lock.yml, never installed directly)
├── environment.yml        # conda deps (source; Python, Kaldi/MFA, compiled sci stack + platforms: key)
├── conda-lock.yml         # GENERATED lock (all 3 platforms, exact+hashed) — never hand-edit; regenerate via scripts/relock.py
├── setup.bat / setup.sh   # One-time setup scripts (install from conda-lock.yml)
├── gui/
│   ├── __init__.py
│   ├── main_window.py        # Main QMainWindow + 7-item sidebar (Inputs, Word/Doc/Sub-word/Sentence features, Run, Logs) + status strip
│   ├── theme.py              # QSS stylesheets (SENSI cyan accent, light only)
│   ├── widgets.py            # Reusable widgets (SectionHeader, Card, FolderRow, ToggleRow, AmberDotIndicator); ToggleRow auto-renders ⓘ info-icon when a docs/features/<key>.md exists
│   ├── section_inputs.py     # Inputs page
│   ├── section_word_features.py     # Word-level features page (Structural × 8, Lex-ling × 4, Freq, Predictability, Semantic indicators, Semantic similarity (Word2Vec + BERT word), Acoustic × 4, Prosody × 10)
│   ├── section_doc_features.py      # Document-level features page (Counts × 4, Aggregates × 4, Lex diversity, Readability, Category proportions, Composite psych × 4, Doc prosody × 16)
│   ├── section_subword_features.py  # Sub-word page (Syllable timing/labels/prosody, P-center, Phonemes)
│   ├── section_sentence_features.py # Sentence-level page (timing, prosody × 11, BERT sentence similarity)
│   ├── section_run.py        # Run page (output checkboxes, readiness, cost panel, Run button)
│   ├── section_logs.py       # Logs page (live viewer, past runs)
│   ├── status_strip.py       # Persistent bottom status strip
│   ├── category_picker.py    # Category picker widget (level-aware: word vs document)
│   ├── category_dialog.py    # Modal dialog wrapping CategoryPicker (one per level)
│   ├── preflight.py          # Continuous warning engine
│   ├── cost_estimate.py      # OpenAI cost estimation via tiktoken
│   ├── run_dispatch.py       # Background-thread orchestrator
│   ├── feature_docs.py       # Per-feature docs loader + _SHARED_DOC_MAP (toggle key → docs/features/*.md)
│   └── info_popup.py         # InfoIcon button + FeatureInfoDialog (renders the .md via QTextBrowser)
├── assets/
│   ├── icon.svg              # App icon (title bar, taskbar) — SENSI cyan glyph
│   ├── info.svg              # Info-icon glyph used by gui/info_popup.py
│   ├── check.svg, dash.svg, dot.svg  # Checkbox / radio indicator graphics
├── docs/
│   ├── master_plan.md        # Single entry point for project state (read first)
│   ├── reproducibility.md    # Env lock runbook: two-tier model, dep changes, platform support, archival ritual
│   ├── features/             # Per-feature .md registry — one file per documented toggle (or one shared file for related toggles), surfaced in the GUI via gui/info_popup.py
│   └── superpowers/          # Batch specs + plans (one per feature batch)
├── extractors/
│   ├── text_parser.py     # spaCy annotation
│   ├── aligner.py         # MFA forced alignment
│   ├── word_freq.py       # wordfreq
│   └── surprisal.py       # GPT-2 surprisal
├── utils/
│   ├── transcript_prep.py # Number spelling (num2words), whitespace normalization
│   ├── textgrid_parser.py # MFA TextGrid → DataFrame
│   ├── download_models.py # One-time MFA/spaCy model download helper (run: python utils/download_models.py)
│   ├── download_lexical_norms.py # Builds dictionaries/*.parquet lexical-norm tables
│   └── validate_cloze.py  # LLM-cloze validation against Provo Corpus
└── scripts/
    └── relock.py          # Regenerates conda-lock.yml from environment.yml + requirements.txt
```

CI: `.github/workflows/env-verify.yml` installs from `conda-lock.yml` on all three platforms (install+import smoke; merge gate) — see `docs/reproducibility.md`.

**Do not touch or analyze**: `env/` (path-based conda env, OS-specific binaries including Kaldi), `hf_cache/` (GPT-2 weights), `mfa_data/` (MFA acoustic models).

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
```

## Timeless design decisions

These are stable across all batches. Batch-specific decisions live in `master_plan.md`'s completed-batches table.

- **MFA over GENTLE**: Montreal Forced Aligner (Kaldi-based) for accuracy and active maintenance.
- **Sliding window for GPT-2**: handles naturalistic texts that exceed the 1024-subword limit.
- **Models stored locally**: `hf_cache/` and `mfa_data/` ship with the project so collaborators skip ~1.5 GB of downloads.
- **Two coordinate systems, easy to confuse**: `Onset`/`End` are acoustic word boundaries in seconds (MFA); lowercase `start`/`end` are character offsets within the sentence string (spaCy). Never conflate.

## Relationship to the parent project

The SENSI naturalistic stimuli (audio + transcripts) live in `../Data Curation/`. This tool produces the feature CSVs that feed the publication-ready stimulus+features datasets. The parent project tracks which features are in-use in EEG analyses and whether new extractors are needed — refer there for feature scope decisions.
