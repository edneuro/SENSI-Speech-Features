# SensiExtractor

Turn a spoken-word **audio file + its transcript** into publication-ready,
**word-level feature tables** for naturalistic EEG analysis (e.g. N400 or TRF
regressors). Part of the SENSI naturalistic-EEG pipeline.

Every feature is optional and toggleable from either a desktop **GUI** or the
**command line**.

---

## What it does

**Input:** one `.wav` audio file + one plain-text transcript (`.txt`/`.tsv`/`.md`),
or folders of matched pairs for batch processing. (Text-only and audio-only runs
are also supported — see [Usage](#usage).)

**Output:** CSV tables with one row per word (and optional sub-word, sentence,
and document-level tables), each column a linguistic or acoustic feature aligned
to the audio, ready to time-lock against EEG.

A word row ties together three things for each token: **where it is in the text**
(sentence/position), **what it is linguistically** (lemma, POS, frequency,
predictability, meaning), and **where it is in the audio** (onset/offset in
seconds, from forced alignment).

## What it extracts

All optional; grouped by level.

- **Word-level**
  - *Structural / linguistic* (spaCy): sentence & token indices, lemma, POS, morphology, dependency parse
  - *Frequency*: `Zipf_Freq` (SUBTLEX-US); optional grade-band frequencies (EWFG)
  - *Predictability*: `Surprisal` (GPT-2); LLM-cloze (`LLM_Cloze`, `LLM_Surprisal`, `LLM_TopMatch`)
  - *Semantics*: semantic-domain category flags (Empath/LIWC); Word2Vec & BERT embeddings
  - *Lexical norms* (opt-in, bring-your-own data): age-of-acquisition, concreteness, imageability, familiarity, valence/arousal/dominance
  - *Acoustic timing* (MFA forced alignment): `Onset`, `End`, `Word_Duration`, `Word_Gap`
  - *Prosody* (Praat/Parselmouth): pitch & volume mean/min/max/std/delta
- **Sub-word**
  - *Syllable*: timing, ARPABET label, prosody, P-center
  - *Phoneme*: timing, label, phoneme surprisal / cohort entropy
- **Sentence-level**: timing, prosody (pitch/volume/HNR), BERT sentence embedding
- **Document-level** (one row per stimulus): token/sentence counts, mean frequency/surprisal, lexical diversity (TTR, MTLD), readability (Flesch), category proportions, composite psycholinguistic scores, document prosody & pause statistics
- **Acoustic level** (audio-only, no transcript): continuous TRF predictors — broadband & onset envelopes, log-mel spectrogram, MFCCs — written as `.mat` time series

See **[`Feature_Extraction_Guide.md`](Feature_Extraction_Guide.md)** for exact
column definitions.

---

## Requirements

- **A supported platform:** Windows 11+, Linux, or Apple-Silicon macOS (M1/M2/M3/M4).
  - **Intel Macs are not supported** — PyTorch dropped Intel-macOS builds after 2.2.2, which is incompatible with this project's NumPy 2.x. 
- **Miniconda** (a minimal conda installer). The whole environment is conda-based: forced alignment needs native Kaldi libraries, and the compiled scientific stack (NumPy, pandas, SciPy, librosa, …) comes from conda-forge. A plain pip `virtualenv` will **not** work.
- **~10 GB free disk**: the environment (~2 GB) plus bundled models (~1.5 GB, already in the repo).

---

## Installation

### Step 1 — Install Miniconda and hook it into your shell

Download the installer for your OS from
<https://docs.conda.io/en/latest/miniconda.html>.

**Windows**
- Run the `.exe` (default options are fine).
- Use the **Anaconda Prompt** (added to your Start Menu) for all commands below — it already has `conda` initialized.

**macOS (Apple Silicon)**
- Run the Apple-Silicon `.pkg` (or `.sh`) installer.
- Initialize conda for your shell, then **open a fresh Terminal**:
  ```bash
  conda init zsh          # zsh is the macOS default; use your shell name if different
  ```
- Close the Terminal and open a new one before continuing.

**Linux**
- Run the `.sh` installer.
- Initialize conda for your shell, then open a fresh shell:
  ```bash
  conda init bash         # or zsh/fish — whatever you use
  ```

> **Troubleshooting — `Run 'conda init' before 'conda activate'`:** you skipped
> or haven't reloaded the init step. Run `conda init <your-shell>`, open a new
> terminal, and retry. To activate in the *current* shell without reopening it:
> ```bash
> source "$(conda info --base)/etc/profile.d/conda.sh"
> ```

### Step 2 — Run the setup script

From the SensiExtractor project folder:

**Windows** (Anaconda Prompt):
```
setup.bat
```

**macOS / Linux** (Terminal):
```bash
bash setup.sh
```

This installs [conda-lock](https://github.com/conda/conda-lock) into your base
conda environment, builds `./env` from `conda-lock.yml` (an exact,
pinned dependency set for your platform), downloads the MFA models, and installs
the spaCy model. You only do this once.

### Step 3 — Launch

Setup creates a **`runGUI`** launcher tailored to your machine — just
double-click it in your file browser (no terminal needed):

- **Windows:** `runGUI.bat`
- **macOS:** `runGUI.command`
- **Linux:** `runGUI.sh`

Or launch from a terminal:
```bash
conda activate ./env
python gui.py
```

(If `conda activate` errors, see the Step 1 troubleshooting note.)

---

## Usage

### GUI (recommended)

Double-click the **`runGUI`** launcher created by setup (`runGUI.bat` on
Windows, `runGUI.command` on macOS, `runGUI.sh` on Linux), or from a terminal:

```bash
conda activate ./env
python gui.py
```

The window has a sidebar — **Inputs, Word-level, Document-level, Sub-word,
Sentence, Acoustic, Run, Logs**. Pick your input files/folders, toggle the
features you want on each page, then go to **Run** to choose which output files
to write and start the job. The Run page shows readiness warnings and (for the
LLM features) a cost estimate; the **Logs** page live-tails the run.

### Command line

Single file:
```bash
# Full pipeline (audio + transcript)
python main.py --audio stimulus.wav --transcript stimulus.txt --output features.csv

# Text features only (no audio, alignment skipped) — conda env still required
python main.py --transcript stimulus.txt --no-align
```

Batch (recommended for multiple stimuli — pairs files by shared basename, e.g.
`story1.txt` ↔ `story1.wav`):
```bash
python main.py \
  --transcripts-folder path/to/transcripts \
  --audio-folder       path/to/audio \
  --output-folder      path/to/output \
  --analyzer "Your Name" \
  --level word          # word (default) | document | acoustic
```

Commonly used flags (run `python main.py --help` for the complete list — there
are many per-feature toggles):

| Flag | Effect |
|------|--------|
| `--no-align` | Skip MFA forced alignment (`Onset`/`End`) |
| `--no-surprisal` | Skip GPT-2 surprisal |
| `--no-llm-cloze` | Skip LLM-cloze metrics |
| `--no-prosody` / `--no-doc-prosody` | Skip word- / document-level prosody |
| `--pitch-units hz\|semitones` | Pitch unit (default `semitones`) |
| `--sem-dict liwc\|empath\|auto\|<path>` | Semantic-domain backend (default `auto`) |
| `--level word\|document\|acoustic` | Batch runner: word CSV, document summary, or audio-only TRF `.mat` |
| `--analyzer NAME` | **Required for batch mode**; recorded in the run manifest |

### Outputs

A batch run writes, under your output folder:

```
output/
├── word/       <stim>_features.csv          # one row per word
│                <stim>_word2vec.mat, <stim>_bert_word.mat
├── syllable/   <stim>_syllables.csv
├── phoneme/    <stim>_phonemes.csv, <stim>_phoneme_surprisal.mat
├── sentence/   <stim>_sentences.csv, <stim>_bert_sentence.mat
├── document/   documents_summary_<timestamp>.csv   # one row per stimulus (--level document)
├── acoustic/   <stim>_acoustic.mat                 # audio-only TRF features (--level acoustic)
├── Montreal Forced Alignment/                      # cached MFA alignments, reused across runs
└── logs/       run_<ts>.log, run_<ts>_manifest.json, ...
```

Single-file mode writes only the one CSV given by `--output`.

---

## Running on Sherlock (HPC)

The GUI can't run on Sherlock (it's headless), and a local `./env` is built for
one OS so it won't run on the cluster's nodes. Instead, the whole environment is
packaged into a portable **Apptainer container** (`sensiextractor.sif`) and run
there via a config file.

Everything for that workflow — building the container, configuring the run,
transferring files, submitting the GPU job, and collecting results — is in
**[`sherlock/RUNBOOK.md`](sherlock/RUNBOOK.md)**.

---

## Changing dependencies

Dependencies are locked in `conda-lock.yml` (generated — **never hand-edit it**).
To add or change a package: edit `environment.yml` (conda packages) or
`requirements.txt` (pip packages), run `python scripts/relock.py`, and commit the
regenerated `conda-lock.yml`. Full details, platform support, and the archival
policy are in **[`docs/reproducibility.md`](docs/reproducibility.md)**.

---

## Project structure

```
SensiExtractor/
├── gui.py                     # GUI entry point (recommended)
├── main.py                    # CLI entry point
├── environment.yml            # conda deps (source)
├── requirements.txt           # pip deps (source)
├── conda-lock.yml             # GENERATED exact lock (all platforms) — never hand-edit
├── setup.bat / setup.sh       # one-time setup (installs ./env from conda-lock.yml)
├── gui/                       # PySide6 GUI (one module per sidebar page)
├── extractors/                # feature extractors (spaCy, MFA, GPT-2, prosody, …)
├── utils/                     # transcript prep, TextGrid parsing, model download
├── scripts/relock.py          # regenerates conda-lock.yml from the source specs
├── dictionaries/              # lexical-norm lookup tables
├── mfa_data/                  # bundled MFA models (do not edit)
├── hf_cache/                  # bundled GPT-2/BERT weights (do not edit)
├── sherlock/                  # HPC container + run workflow (see sherlock/RUNBOOK.md)
└── docs/
    ├── reproducibility.md     # dependency/lock model + archival policy
    ├── master_plan.md         # project state & roadmap
    └── features/              # per-feature documentation
```

---

## A note on coordinates (don't confuse these)

- **`Onset` / `End`** (capitalized) — acoustic word boundaries **in seconds**, from MFA. Use these for EEG time-locking.
- **`start` / `end`** (lowercase) — character offsets **within the sentence string**, from spaCy. Not time.
