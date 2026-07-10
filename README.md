# SensiExtractor

Portable word-level stimulus feature extraction pipeline for EEG naturalistic studies.

**Input**: audio (.wav) + transcript (.txt)  
**Output**: one CSV with one row per token, with linguistic and acoustic features

---

## What's new (prosody batch, 2026-05-12)

- **Word-level acoustic prosody** (10 columns): `Pitch_Mean/Min/Max/Std`, `Volume_Mean/Min/Max/Std`, `Pitch_Delta`, `Volume_Delta`. Min/Max use p5/p95 percentiles (robust to pitch-tracker errors); deltas are Brain Treebank-style word-to-word changes.
- **Document-level prosody** (16 columns) in `documents_summary_*.csv`: same pitch/volume aggregates over the whole stimulus, mean absolute deltas, HNR, articulation rate (syllables/sec excluding pauses), and 4 pause statistics.
- **Praat-Parselmouth backend** — autocorrelation pitch tracking + HNR. Single pass per stimulus, contours cached on the args namespace and reused at the doc-summary stage.
- **Pitch units toggle** — choose Hz or semitones (relative to per-stimulus voiced-pitch median) via the GUI or `--pitch-units`. Pitch floor/ceiling and pause threshold are also configurable.
- **GUI cards** on both word and document feature pages; preflight warns when prosody is enabled without alignment or without the `praat-parselmouth`/`pyphen` deps installed.
- **Prefs schema bumped to v5** — additive only; existing v1-v4 prefs auto-migrate.
- **New deps**: `praat-parselmouth>=0.4`, `pyphen>=0.14`.

See `Feature_Extraction_Guide.md` § "Acoustic Prosody (added 2026-05-12)" for column definitions and `docs/superpowers/specs/2026-05-12-prosody-design.md` for design rationale.

---

## What's new (batch 1.5, 2026-05-10)

- **Tabbed GUI**: word-level and document-level workflows are now fully independent tabs with their own feature toggles, category selections, and Run buttons.
- **New doc-level features**: lexical diversity (TTR, MTLD), readability (Flesch reading ease, FK-grade), LIWC summary scores (Authentic, Analytic, Tone, Clout — LIWC backend only).
- **Master doc-level CSV**: `documents_summary_<ts>.csv` with one row per matched pair, replacing the per-stim summary sidecar.
- **CLI `--level word|document` flag**: defaults to `word` for backward compatibility.
- **Categories dialog**: closing via the window X now saves the selection (previously discarded).
- **`preferences.json` schema**: migrated from a flat layout to `shared / word_level / document_level` sections. Auto-migration runs on first load.

See `Feature_Extraction_Guide.md` for full documentation.

---

## What's new (batch 1, 2026-05-10)

- **Folder-pair batch mode**: select a transcripts folder and an audio folder; the app pairs files by base filename and processes them all in one run.
- **Run logs**: every batch emits a `.log`, a structured `manifest.json`, and a `no-match.csv` in the output folder.
- **Analyzer field**: required for batch runs; recorded in the manifest.
- **EWFG grade-band frequencies** (`Edu_Zipf_*`): bring your own EWFG lookup parquet; pipeline emits NaN until you do.
- **Semantic-domain indicators** (`Sem_*`): LIWC-primary / Empath-fallback, configurable category set via GUI modal.
- **LLM-cloze** (`LLM_Cloze`, `LLM_Surprisal`, `LLM_TopMatch`): modern LLM-based predictability alongside the existing GPT-2 `Surprisal` column.
- **Derived timing** (`Word_Duration`, `Word_Gap`): cheap aggregations of MFA outputs.
- **Document-level sidecar**: `<stimulus>_summary.csv` per batch run.
- **GUI overhaul**: grouped feature toggles, LLM model dropdown, dictionary backend radio, semantic category modal, progress + cancellation.

See `Feature_Extraction_Guide.md` for full documentation, and
`docs/superpowers/specs/2026-05-10-feature-batch-1-design.md` for design rationale.

---

## Features extracted

| Column | Source | Description |
|--------|--------|-------------|
| `doc_id`, `paragraph_id`, `sentence_id` | spaCy | Document/sentence location |
| `sentence` | spaCy | Full sentence text |
| `start`, `end` | spaCy | Character offsets within `sentence` |
| `token_id`, `token_in_sentence_id` | spaCy | Global and per-sentence token index |
| `token` | spaCy | Original word form |
| `lemma` | spaCy | Dictionary base form |
| `upos`, `xpos` | spaCy | Universal and Penn Treebank POS tags |
| `feats` | spaCy | Morphological features |
| `dep_rel`, `head_token_id` | spaCy | Syntactic dependency |
| `deps`, `misc` | spaCy | Enhanced deps; space/punctuation flags |
| `Zipf_Freq` | wordfreq | Log word frequency (SUBTLEX-US, Zipf scale) |
| `Surprisal` | GPT-2 | Contextual predictability in bits |
| `Transcript` | MFA | Word form sent to aligner (numbers spelled out) |
| `Onset` | MFA | Word start time in audio (seconds) |
| `End` | MFA | Word end time in audio (seconds) |

All features are optional and can be toggled in the GUI or via CLI flags.

---

## Sharing with colleagues

| Folder | Share it? | Why |
|--------|-----------|-----|
| `env/` | No | Contains OS-specific compiled binaries and absolute paths — must be built per machine |
| `mfa_data/` | Yes | Model weight files — no OS-specific code, work on any platform |
| `hf_cache/` | Yes | GPT-2 weights in platform-agnostic format |

**Recommended**: zip the folder including `mfa_data/` and `hf_cache/`. Recipients skip ~1.5 GB of downloads and only need to run `setup.bat` / `setup.sh` to build their local `env/`.

---

## Setup

### Step 1 — Install Miniconda (only needed for MFA alignment)

Miniconda is a minimal conda installer (~100 MB). You only need it because MFA
depends on native Kaldi libraries that cannot be installed via pip.

- **Windows**: download the `.exe` from https://docs.conda.io/en/latest/miniconda.html  
  Run the installer — default options are fine. Use **Anaconda Prompt** (added
  to your Start Menu) to run the commands below.
- **Mac**: download the `.pkg` or `.sh` installer for your chip (Intel or Apple Silicon).  
  Run it, then open a new **Terminal** window.

If you don't need forced alignment (Onset/End columns), skip this step entirely
and use a regular Python virtual environment instead.

### Step 2 — Run the setup script

Navigate to the SensiExtractor folder in your terminal, then:

**Windows** (Anaconda Prompt):
```
setup.bat
```

**Mac / Linux** (Terminal):
```bash
bash setup.sh
```

This creates the conda environment, installs the Python dependencies from
`requirements.txt`, downloads MFA models (~1 GB), and installs the spaCy model.
You only do this once.

### Step 3 — Launch the app

```bash
conda activate ./env
python gui.py
```

---

## Running the GUI

```bash
conda activate ./env   # (or your venv if skipping alignment)
python gui.py
```

The GUI lets you:
- Browse for an audio file and a transcript file
- Check/uncheck which features to extract
- Choose an output folder and filename
- Watch progress in a live log panel

---

## Running from the command line

```bash
# Full pipeline (all features)
python main.py --audio stimulus.wav --transcript stimulus.txt

# Text features only — no audio needed, no conda needed
python main.py --transcript stimulus.txt --no-align

# Skip GPT-2 surprisal (it downloads ~500 MB on first run)
python main.py --audio stimulus.wav --transcript stimulus.txt --no-surprisal

# Custom output path
python main.py --audio stimulus.wav --transcript stimulus.txt --output my_features.csv

# Batch mode (recommended for multiple stimuli)
python main.py \
  --transcripts-folder path/to/transcripts \
  --audio-folder path/to/audio \
  --output-folder path/to/output \
  --analyzer "Your Name"
```

### CLI flags

| Flag | Default | Effect |
|------|---------|--------|
| `--audio PATH` | — | Path to .wav audio file |
| `--transcript PATH` | required | Path to plain-text transcript |
| `--output PATH` | `features.csv` | Output CSV path |
| `--no-align` | off | Skip MFA forced alignment |
| `--no-parse` | off | Skip spaCy annotation |
| `--no-freq` | off | Skip word frequency |
| `--no-surprisal` | off | Skip GPT-2 surprisal |

---

## Notes on accuracy

- **Onset/End** values are true acoustic word boundaries from MFA (Kaldi-based), suitable for EEG word-onset analysis.
- **Surprisal** uses GPT-2 base model with full left-context. For texts longer than 1024 GPT-2 subword tokens, a sliding context window is applied automatically.
- **`start`/`end`** (lowercase) are character offsets in the sentence string — not audio times. Do not confuse with `Onset`/`End` (uppercase), which are seconds in the audio file.

---

## Project structure

```
SensiExtractor/
├── gui.py                     # Graphical interface (recommended entry point)
├── main.py                    # CLI entry point
├── requirements.txt
├── extractors/
│   ├── text_parser.py         # spaCy linguistic annotation
│   ├── aligner.py             # MFA forced alignment
│   ├── word_freq.py           # wordfreq (Zipf frequency)
│   └── surprisal.py           # GPT-2 surprisal
└── utils/
    ├── transcript_prep.py     # Number spelling, whitespace normalization
    └── textgrid_parser.py     # MFA TextGrid → DataFrame
```
