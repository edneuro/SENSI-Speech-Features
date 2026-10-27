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
  - *Frequency*: `Zipf_Freq` (wordfreq combined English list — SUBTLEX subtitles + Wikipedia/books/news/web); optional grade-band frequencies (EWFG)
  - *Predictability*: context-conditional word probability from a multi-select roster of **local** language models — GPT-2 (free, no token, CPU-capable) plus three gated Llama-3 variants (3.2-3B / 8B / 70B, need a free HuggingFace token + accepted license). Pick any combination of models and metrics (surprisal, cloze probability, top-1 match); each selected model×metric pair writes a namespaced `<slug>_<Metric>` column, e.g. `gpt2_Surprisal`, `llama3_8b_Cloze` — see [`docs/features/pred_surprisal.md`](docs/features/pred_surprisal.md)
  - *Semantics*: semantic-domain category flags (Empath/LIWC); Word2Vec & BERT embeddings
  - *Lexical norms* (opt-in, bring-your-own data): age-of-acquisition, concreteness, imageability, familiarity, valence/arousal/dominance
  - *Acoustic timing* (MFA forced alignment): `Onset`, `End`, `Word_Duration`, `Word_Gap`
  - *Prosody* (Praat/Parselmouth): pitch & volume mean/min/max/std/delta
- **Sub-word**
  - *Syllable*: timing, ARPABET label, prosody, P-center
  - *Phoneme*: timing, `ARPABET` + `Stress` + `IPA` + `Pron_Source`, phoneme surprisal / cohort entropy
- **Sentence-level**: timing, prosody (pitch/volume/HNR), BERT sentence embedding
- **Document-level** (one row per stimulus): token/sentence counts, mean frequency/surprisal, lexical diversity (TTR, MTLD), readability (Flesch), category proportions, composite psycholinguistic scores, document prosody & pause statistics
- **Acoustic level** (audio-only, no transcript): continuous TRF predictors — broadband & onset envelopes, log-mel spectrogram, MFCCs — written as `.mat` time series

See **[`docs/features/`](docs/features/)** for exact column definitions,
one file per feature.

---

## Requirements

- **A supported platform:** Windows 11+, Linux, or Apple-Silicon macOS (M1/M2/M3/M4).
  - **Intel Macs are not supported** — PyTorch dropped Intel-macOS builds after 2.2.2, which is incompatible with this project's NumPy 2.x. 
- **Miniconda** (a minimal conda installer). The whole environment is conda-based: forced alignment needs native Kaldi libraries, and the compiled scientific stack (NumPy, pandas, SciPy, librosa, …) comes from conda-forge. A plain pip `virtualenv` will **not** work.
- **~10 GB free disk**: the environment (~6 GB — full ML stack: PyTorch, transformers, spaCy, gensim, plus Kaldi/MFA and the PySide6 GUI) plus downloaded models (~1.5 GB: the MFA models during setup, GPT-2/BERT on first use) plus the one-time Word2Vec download (~1.7 GB, on first use of the Word2Vec embedding feature). All models stay inside the project's `data/` folder (`data/hf_cache/`, `data/mfa_data/`, `data/gensim_data/`), not in your home directory; a cache location you already set (`HF_HOME`, `MFA_ROOT_DIR`, `GENSIM_DATA_DIR`) takes precedence.

---

## Installation

### Step 1 — Install Miniconda and hook it into your shell

Download the installer for your OS from
<https://www.anaconda.com/docs/getting-started/installation>.

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

Setup creates a double-click launcher tailored to your machine. A small
console window opens alongside the GUI showing live progress logs (that's
intentional — close the GUI and press a key to dismiss it).

- **Windows:** double-click **`SensiExtractor`** (a shortcut with the app icon; it runs `runGUI.bat`).
- **Linux:** double-click **`SensiExtractor`** (a `.desktop` entry with the app icon; it runs `runGUI.sh`).
- **macOS:** double-click **`SensiExtractor`** (`SensiExtractor.app`, with the app icon). It opens no Terminal window; its log goes to `~/Library/Logs/SensiExtractor.log`. For live logs in Terminal, double-click **`runGUI.command`** instead.

Or launch from a terminal:
```bash
conda activate ./env
python gui.py
```

(If `conda activate` errors, see the Step 1 troubleshooting note.)

---

## Usage

### GUI (recommended)

Double-click the launcher created by setup (**`SensiExtractor`**, the
app-icon shortcut on Windows/Linux and `SensiExtractor.app` on macOS), or from a
terminal:

```bash
conda activate ./env
python gui.py
```

The window has a sidebar — **Inputs, Word-level, Document-level, Sub-word,
Sentence, Acoustic, Run, Logs**. Pick your input files/folders, then tick the
features you want on each page. Each feature section has an on/off switch next to
its title: switched off, the section is **left out of outputs** (its controls
grey out, your ticks are kept for when you switch it back on). Then go to
**Run**, switch on the output levels you want (Word, Document, Sub-word,
Sentence, Acoustic — one switch per feature page; each lists the files it will
write) and start the job. The Run page also shows readiness warnings (e.g. a
missing HuggingFace token for a selected gated Llama model); the **Logs** page
live-tails the run.

#### Run-tab estimates

Before you start, the Run page shows three rough figures for the current
inputs and settings:

- **Disk space for outputs** — the total size of the files the run will write,
  plus a size next to each output level. Worked out from word counts, audio
  length and the columns you have switched on; it does not count the saved MFA
  alignments or the run logs.
- **Peak memory (RAM)** — the most memory the run should need at once, mostly
  set by the largest model you selected, next to the memory this computer has.
  A warning appears if it may not fit.
- **Expected time** — a range for the whole run. Stimuli that already have a
  fresh saved alignment in the output folder are not charged for MFA.

The time range starts as a wide first estimate (marked as such on the page).
Each run records how long each step took on this computer, and once a step has
a few timings the range narrows to one learned for this machine. MFA is
counted as a fixed start-up each time it runs plus a cost per second of audio;
that part is learned once three runs have aligned clearly different amounts of
audio, and the range stays wide for a corpus much larger than any batch aligned
so far. Model loads are timed too (the spaCy model loads during the first
parse and is timed as part of it). Starting the run itself (loading Python
libraries, writing files) is not timed; the estimate adds a fixed 10 seconds
for it, measured on a real run. The timings are kept in
`config/timing_history.json` (next to `features.yaml` / `run.yaml`, or in the
folder `SENSI_CONFIG_DIR` points to). The file is specific to this computer and
is not shared through git; delete it to reset the time estimates to first
estimates.

### Headless (no GUI): `run.py`

The other way to run is `python run.py`, which does exactly what the Run button
does, with the settings saved in `config/features.yaml` (what to extract) and
`config/run.yaml` (folders, analyzer name). Set things up once in the GUI and
the files are already there; or, without the GUI, copy the committed templates
and edit them:

```bash
cp config/features_defaults.yaml config/features.yaml
cp config/run_defaults.yaml config/run.yaml
# edit config/run.yaml: analyzer_name, last_transcripts_folder, last_output_folder, ...
python run.py
```

Folder paths in `run.yaml` can be absolute (recommended) or relative; a
relative path is resolved against the folder holding `run.yaml`, not the
folder you launch `run.py` from. `run.py` checks the config before doing any
work and stops with a plain-language message for an unknown key, a wrong type,
an out-of-range value (e.g. a pitch floor above the ceiling) or a missing
input folder.

`run.py` also prints this hint whenever `config/features.yaml` is missing.
Both templates are regenerated from the code defaults by
`python scripts/export_default_config.py` and are pinned by a drift-guard
test, so they never go stale relative to the shipped defaults.

`python run.py --config-dir DIR` reads the two files from another folder
(Sherlock does this). Predictability models and metrics are set in the
`predictability` block of `features.yaml` (or the GUI's Predictability card).
A gated Llama-3 model needs a HuggingFace token: the card's **Save** button
writes it to `config/hf_token.txt` (git-ignored) and points
`shared.hf_token_file` at it, so `run.py` picks it up too — see
[`docs/features/llm_credentials.md`](docs/features/llm_credentials.md).

(The old `python main.py --flags` command line was retired on 2026-10-27; its
flags covered only part of the features. Use the GUI or `run.py`.)

### Outputs

A batch run writes, under your output folder:

```
output/
├── word/       <stim>_features.csv          # one row per word
│                <stim>_word2vec.mat, <stim>_bert_word.mat
├── syllable/   <stim>_syllables.csv
├── phoneme/    <stim>_phonemes.csv, <stim>_phoneme_surprisal.mat
├── sentence/   <stim>_sentences.csv, <stim>_bert_sentence.mat
├── document/   documents_summary_<timestamp>.csv   # one row per stimulus
├── acoustic/   <stim>_acoustic.mat                 # audio-only TRF features
├── Montreal Forced Alignment/                      # cached MFA alignments, reused across runs
└── logs/       run_<ts>.log, run_<ts>_manifest.json, ...
```

Each folder is written only when its level is
switched on in `config/features.yaml`'s `output` block (`level_word`,
`level_document`, `level_subword`, `level_sentence`, `level_acoustic`; the
Run page's level switches) and at least one included section ticks something
for that file.

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
├── gui.py  run.py             # the two ways to start the app (GUI / headless)
├── setup.bat / setup.sh       # one-time setup (installs ./env from conda-lock.yml)
├── environment.yml            # conda deps (source)
├── requirements.txt           # pip deps (source)
├── conda-lock.yml             # GENERATED exact lock (all platforms) — never hand-edit
├── sensi/                     # all application code
│   ├── pipeline.py            #   the batch runners both launchers call
│   ├── paths.py               #   where every folder is (one place)
│   ├── gui/                   #   PySide6 GUI (one module per sidebar page) + planner/
│   ├── extractors/            #   feature extractors (spaCy, MFA, GPT-2, prosody, …)
│   └── utils/                 #   preferences, transcript prep, TextGrid parsing, logging
├── scripts/                   # maintainer + one-time helpers (download_models, relock, …)
├── config/                    # settings templates (+ your features.yaml / run.yaml)
├── data/
│   ├── dictionaries/          #   lexical-norm lookup tables
│   └── hf_cache/ mfa_data/ gensim_data/   # downloaded models (git-ignored)
├── assets/                    # app icons
├── docs/
│   ├── features/              # per-feature documentation (also shown in the app)
│   └── reproducibility.md     # dependency/lock model + archival policy
├── dev/                       # project history: plans, specs, changelog, validation
├── sherlock/                  # HPC container + run workflow (see sherlock/RUNBOOK.md)
└── tests/
```

---

## A note on coordinates (don't confuse these)

- **`Onset` / `End`** (capitalized) — acoustic word boundaries **in seconds**, from MFA. Use these for EEG time-locking.
- **`start` / `end`** (lowercase) — character offsets **within the sentence string**, from spaCy. Not time.
