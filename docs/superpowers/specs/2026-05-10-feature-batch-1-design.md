yes# Stimulus Feature Extraction — Batch 1 Design

**Date:** 2026-05-10
**Status:** Draft for review
**Authors:** Amilcar Malave (PI: Blair Kaneshiro), Claude
**Scope:** SensiExtractor project (`SensiExtractor/`)
**Parent project:** `../Data Curation/`

---

## 1. Background

SensiExtractor takes audio (.wav) + transcript (.txt) and produces a word-level feature CSV used in EEG analyses of naturalistic speech. The current pipeline emits one CSV per stimulus with these columns per word:

- **spaCy**: `doc_id, paragraph_id, sentence_id, sentence, start, end, term_id, token_id, token, lemma, upos, xpos, feats, dep_rel, head_token_id, deps, misc`
- **Frequency**: `Zipf_Freq` (SUBTLEX-US via `wordfreq`)
- **Predictability**: `Surprisal` (GPT-2 base, sliding window for >1024 subword tokens)
- **Alignment**: `Transcript, Onset, End` (Montreal Forced Aligner)

The parent `../Data Curation/` project tracks which features are in-use across studies and which need to be added.

This spec defines **Batch 1** of feature additions. Subsequent batches (prosody, BERT embeddings, p-center, document/syllable/phoneme levels) are listed in the roadmap appendix and each gets its own spec → plan → implement cycle.

## 2. Goals

Add the following to the existing word-level CSV without changing the data model:

1. Child-appropriate word frequency at four grade bands (EWFG-derived)
2. Semantic-domain category indicators (LIWC-primary, Empath-fallback)
3. LLM-derived predictability metrics (cloze, surprisal, top-match)
4. Two derived timing columns (word duration, inter-word gap)
5. A document-level summary sidecar CSV per stimulus (semantic-domain proportions plus simple aggregates)
6. A GUI overhaul that switches from per-file selection to folder-pair batch processing, with run logs, an analyzer field, and improved selectors (see §10)

Keep all existing columns and behavior unchanged. Each new feature is independently togglable via a CLI flag.

## 3. Non-goals (deferred to later batches)

- Phoneme-level, syllable-level, or sentence-level CSVs
- Acoustic prosody (pitch, intensity, voicing) — batch 2
- BERT contextual embeddings
- P-center extraction
- Human-rated cloze norms
- Multi-level output architecture (currently we emit one per-word CSV + one per-stimulus summary CSV; we are not committing to a generalized `level=...` schema yet)

## 4. Feature designs

### 4.1 Educator's Word Frequency Guide (child frequency)

**Source:** Zeno, Ivens, Millard, Duvvuri (1995), *The Educator's Word Frequency Guide*. ISBN 0-89172-307-9. Touchstone Applied Science Associates. The book ships with a CD-ROM containing the full lookup data; this is the data we need.

**Acquisition:** User to acquire EWFG via the [acquisition checklist](#appendix-a-ewfg-acquisition-checklist) (ask around lab → buy used book + CD → extract data files). Implementation is unblocked on data — see pluggable interface below.

**Output columns:**

| Column | Meaning |
|---|---|
| `Edu_Zipf_K2` | Zipf-scale frequency in K-grade 2 educational texts |
| `Edu_Zipf_3to5` | Zipf-scale frequency in grades 3-5 |
| `Edu_Zipf_6to8` | Zipf-scale frequency in grades 6-8 |
| `Edu_Zipf_9to12` | Zipf-scale frequency in grades 9-12 |
| `Edu_Zipf_Overall` | Zipf-scale aggregate across all grades |

Function-word handling matches existing `Zipf_Freq` behavior (lookup applies to all tokens; OOV → NaN).

**Normalization:** EWFG natively reports Standard Frequency Index (SFI) which is `log10(per-million-frequency × 10) + 1`. Convert to Zipf scale via `Zipf = SFI - 1` (approximately — exact transformation depends on EWFG's definition of "per million" — verify against documentation when CD data arrives). The goal is direct numerical comparability with existing `Zipf_Freq`.

**Pluggable interface:** `extractors/educator_freq.py` reads from a lookup file at a configurable path (default `dictionaries/edu_zipf.parquet`). The lookup is a dataframe with columns `word, edu_zipf_k2, edu_zipf_3to5, edu_zipf_6to8, edu_zipf_9to12, edu_zipf_overall`. A one-time `tools/build_edu_zipf_lookup.py` script converts the EWFG CD data into this format. If the lookup file is missing, the extractor logs a warning and emits NaN columns rather than crashing.

**CLI flags:** `--no-edu-freq` to skip; `--edu-freq-lookup PATH` to override location.

### 4.2 Semantic-domain indicators

Same shape as semantic-domain (#5) and LIWC (#8) from the original feature wishlist; collapsed into one extractor with pluggable dictionary backends.

**Dictionary backends:**

- **LIWC** (primary) — auto-detected if `dictionaries/liwc-2022.dic` is present. Parsed via the `liwc` Python package (or equivalent). The dictionary file is gitignored (license-bound).
- **Empath** (open fallback) — used when no LIWC `.dic` is found. Installed via `pip install empath`. No license required.

CLI override: `--sem-dict liwc|empath|<path>`.

**Default category set** (15 categories chosen for cognitive/EEG research relevance):
`social, cognition, emotion_positive, emotion_negative, body, motion, perception, time, space, math, science, work, family, communication, money`

Definitions of each category are documented in [Appendix C](#appendix-c-default-semantic-category-descriptions).

**Category selection — GUI checklist with persistence:**

Selection is settable via three mechanisms, in order of precedence:

1. CLI flag `--sem-categories cat1,cat2,...` (highest priority; one-shot override)
2. Preferences file `config/preferences.json` (key: `semantic_categories`); persists between runs
3. 15-category SOP default (used if neither 1 nor 2 is set)

The preferences file path is overridable via the `SENSI_PREFS_PATH` environment variable. The file is gitignored. Schema is extensible JSON; batch 1 stores only the `semantic_categories` key, but later batches may add `llm_model`, `dictionary_backend`, etc. without breaking compatibility:

```json
{
  "semantic_categories": ["social", "cognition", "emotion_positive", ...]
}
```

The GUI (`gui.py`) gains a **"Configure categories…"** button on the main panel. Clicking it opens a modal dialog containing:

- A scrollable, searchable checklist of all categories available in the active dictionary backend (LIWC has ~80; Empath has 200+)
- Checkboxes pre-checked from `preferences.json`, or from the SOP default if no prefs file exists yet
- A **"Reset to SOP default"** button that re-checks the 15 default categories
- **OK** (saves selection to `preferences.json` and uses for the current run) and **Cancel** (discards changes)

CLI users get the same persistence: `--sem-categories` selections are not auto-saved (one-shot semantics), but if a user wants to persist a CLI choice, they can edit `preferences.json` directly or use a `--save-prefs` flag.

**Output columns (word-level):**

- One binary column per category, named `Sem_<Dict>_<category>` where `<Dict>` ∈ `{LIWC, Empath}`. Value ∈ `{0, 1, NaN}`.
- One roll-up column `Sem_Categories` with comma-separated category names that the word matched (e.g., `social,communication`), or empty string if no matches, or NaN if function word/punct.
- POS filter: only `NOUN`, `VERB`, `ADJ`, `ADV` get scored. All other tokens get NaN across all `Sem_*` columns.

**Output columns (document-level sidecar `<stimulus>_summary.csv`):**

One row per stimulus. Columns:

- `stimulus_id`, `n_tokens`, `n_content_tokens`, `n_sentences`, `n_paragraphs`
- `mean_zipf_freq`, `mean_edu_zipf_overall`, `mean_surprisal`, `mean_llm_surprisal`, `mean_llm_cloze`
- For each category in the active set: `prop_<category>` (proportion of content tokens in that category) and `count_<category>` (count of content tokens in that category)
- `dictionary_used`: `LIWC` or `Empath`

This sidecar is emitted only if the word-level run produced semantic-domain columns. Aggregations are one-line `groupby` operations on the per-word DataFrame; no new processing logic.

**CLI flags:** `--no-sem-domain` to skip word-level columns AND sidecar; `--no-doc-summary` to skip just the sidecar.

### 4.3 LLM-derived predictability

Three metrics per word, computed from one forward pass per token:

| Column | Definition |
|---|---|
| `LLM_Surprisal` | `-log2 P(word \| context)` in bits |
| `LLM_Cloze` | `P(word \| context)` raw probability |
| `LLM_TopMatch` | `1` if target is the model's top-1 next-token prediction, else `0` |

Existing `Surprisal` (GPT-2) column is **kept unchanged** for backward comparability with prior analyses.

**Model selection — auto-detect with override:**

```
Default selection order:
  if OPENAI_API_KEY env var set        → gpt-4o-mini
  else if local CUDA GPU ≥48GB VRAM    → llama-3-70b
  else if local CUDA GPU ≥12GB VRAM    → llama-3-8b
  else if local CUDA GPU ≥ 8GB VRAM    → llama-3.2-3b
  else                                 → llama-3.2-3b on CPU (slow but works)
```

CLI override: `--llm-model gpt-4o-mini|gpt-4o|llama-3-70b|llama-3-8b|llama-3.2-3b|<hf-model-id>`.

**Sliding window:** Not used. Modern LLMs have ≥8K context; naturalistic stimuli of a few thousand words fit entirely. Each token's prediction sees the full preceding stimulus context.

**Documentation requirements:** The `Feature_Extraction_Guide.md` MUST include this table for the LLM-cloze section:

| Model | Approx cloze correlation (Provo) | Cost per ~3000-word stimulus |
|---|---|---|
| `gpt-4o` | ~0.85 | $1–3 (API) |
| `gpt-4o-mini` | ~0.83 | $0.10–0.30 (API) |
| `llama-3-70b` (local) | ~0.83–0.85 | Free (electricity); needs ≥48GB VRAM |
| `llama-3-8b` (local) | ~0.78–0.81 | Free; needs ≥12GB VRAM |
| `llama-3.2-3b` (local) | ~0.75–0.78 | Free; needs ≥8GB VRAM |
| GPT-2 (existing `Surprisal`) | ~0.55–0.60 | Free; ~500MB; baseline only |

These numbers are estimates from published cloze validation work; users should run `validate_cloze.py` to obtain exact values for their installation.

**Validation script (`validate_cloze.py`):**

A standalone, one-time script that:

1. Downloads Provo Corpus (Luke & Christianson 2018) into `data/provo/` (one-time, ~few MB)
2. Runs the currently-configured LLM over each Provo paragraph
3. Computes Pearson r between extracted `LLM_Cloze` and Provo human cloze norms
4. Prints model name, N words tested, observed correlation, and expected range. Exits 0 on PASS (within expected range), 1 on FAIL.

Output example:

```
Validation against Provo Corpus
Model:  meta-llama/Llama-3-8B
N words tested: 2,689
Pearson r vs human cloze: 0.79
Expected range for this model: 0.75–0.81
PASS ✓
```

The script is a sanity check, not a per-stimulus runtime cost.

**CLI flags:** `--no-llm-cloze` to skip; `--llm-model MODEL` to override; `--llm-batch-size N` for memory tuning.

### 4.4 Derived timing columns

Computed from existing MFA `Onset`/`End` columns. No new dependencies.

| Column | Definition |
|---|---|
| `Word_Duration` | `End - Onset`; NaN if either is NaN |
| `Word_Gap` | `Onset[i] - End[i-1]`; NaN for first word in stimulus or if either neighbor is NaN |

Standard in naturalistic-stimulus EEG literature (Brain Treebank, Hamilton & Huth 2020, etc.).

CLI flag: tied to alignment — skipped if `--no-align` is used (since they're derived from MFA columns).

## 5. Output structure

After Batch 1, each stimulus produces:

```
<stimulus>_features.csv   # one row per word, all word-level columns
<stimulus>_summary.csv    # one row, document-level aggregates
```

This is a deliberately minimal extension — no `level=...` schema, no nested directory structure. When future batches add phoneme/syllable levels, we revisit the data model in a separate spec.

## 6. CLI surface (final)

Existing flags retained: `--no-align, --no-parse, --no-freq, --no-surprisal, --audio, --transcript, --output`.

New flags:

```
--no-edu-freq               Skip Edu_Zipf_* columns
--edu-freq-lookup PATH      Override EWFG lookup file location

--no-sem-domain             Skip Sem_* columns and document summary
--sem-dict liwc|empath|PATH Override dictionary backend
--sem-categories LIST       Comma-separated category names; one-shot override
                            (precedence: this flag > preferences.json > SOP default)
--save-prefs                Persist current --sem-categories selection to preferences.json

--no-llm-cloze              Skip LLM_* columns
--llm-model MODEL           Override LLM choice (default = auto-detect)
--llm-batch-size N          Forward-pass batch size for memory tuning

--no-doc-summary            Skip <stimulus>_summary.csv emission
```

## 7. Final word-level column order

```
doc_id, paragraph_id, sentence_id, sentence, start, end, term_id, token_id,
token, lemma, upos, xpos, feats, dep_rel, head_token_id, deps, misc,
Zipf_Freq,
Edu_Zipf_K2, Edu_Zipf_3to5, Edu_Zipf_6to8, Edu_Zipf_9to12, Edu_Zipf_Overall,
Sem_<Dict>_<cat1>, Sem_<Dict>_<cat2>, ..., Sem_Categories,
Surprisal,
LLM_Surprisal, LLM_Cloze, LLM_TopMatch,
Transcript, Onset, End, Word_Duration, Word_Gap
```

## 8. Documentation deliverables

- Update `Feature_Extraction_Guide.md` (project-local) with new feature sections, CLI flags, the LLM model cost/quality table, and the **per-category descriptions from Appendix C** with example words for each of the 15 SOP-default categories.
- Document the GUI "Configure categories…" workflow and the `preferences.json` schema in `Feature_Extraction_Guide.md`.
- Update `README.md` with new flag list and a one-paragraph "What's new in batch 1" note.
- Create `../Data Curation/FEATURES_PROVENANCE.md` (parent-project, separate task — see roadmap) covering all features across the curation effort with tools, versions, licenses, and open alternatives.

## 9. Validation and testing

- Unit tests per extractor with small fixture stimuli (1-2 sentences, known expected output).
- Integration test running the full pipeline on a 30-second toy stimulus.
- The `validate_cloze.py` script described in §4.3.
- Manual spot-check on at least one ISC-Wu Tsai SENSI stimulus end-to-end before merging.

## 10. GUI overhaul (batch 1 scope)

The current `gui.py` selects audio and transcript files individually, which is inefficient for the lab's typical workload of dozens of stimuli per study. Batch 1 replaces this with a folder-pair batch mode and improves every selector on the main panel.

### 10.1 Workflow change

User flow:

1. Pick **Transcripts folder** (text files)
2. Pick **Audio folder** (audio files)
3. Pick **Output folder** (where features CSVs, summary CSVs, and run log artifacts are written)
4. Enter **Analyzer name** (required)
5. Configure features (checkboxes), LLM model, dictionary backend, semantic categories
6. Click **Run** — progress bar shows files processed
7. On completion, a summary dialog reports `N pairs processed, N errors, N unmatched`, with a link/path to the run log

The Run button is disabled until all four required inputs (three folders + analyzer) are filled.

### 10.2 File pairing rules

- **Top-level only** scan of each folder (no recursion into subfolders)
- **Text candidates**: `.txt`, `.tsv`, `.md` (case-insensitive)
- **Audio candidates**: `.wav`, `.mp3`, `.flac` (case-insensitive)
- **Pairing**: text and audio files with the same **base filename** (case-insensitive, ignoring extension) are paired. `story1.txt` ↔ `story1.WAV` matches; `Story1.txt` ↔ `story1.wav` matches.
- **Ambiguous matches**: when a base name has multiple candidates (e.g., `story1.wav` and `story1.mp3`), pick by extension priority — text: `txt > tsv > md`; audio: `wav > flac > mp3` — and emit a warning to the run log.
- **Unmatched files**: text without a matching audio (or vice versa) is appended to `no-match.csv` with the reason.
- **Sequential processing**: pairs are processed one at a time. Errors on a single pair do not halt the run; the next pair proceeds, and the error is captured in the manifest.

### 10.3 Run log artifacts

Three artifacts in the output folder, all sharing the run timestamp:

```
output/
  run_2026-05-10_14-32-05.log              # human-readable, grep-able
  run_2026-05-10_14-32-05_manifest.json    # machine-readable structured manifest
  run_2026-05-10_14-32-05_no-match.csv     # files that didn't pair
  story1_features.csv
  story1_summary.csv
  story2_features.csv
  story2_summary.csv
  ...
```

**`run_<ts>.log`** — plain text, one line per event with timestamp, level (`INFO`/`WARN`/`ERROR`), and message. Used for ad-hoc grepping and quick visual review.

**`run_<ts>_manifest.json`** — structured fields:

| Field | Description |
|---|---|
| `start_time, end_time, total_runtime_seconds` | ISO-8601 timestamps and total duration |
| `analyzer` | Name from the GUI input or `--analyzer` flag |
| `sensi_extractor_version` | Read from a `__version__` string; git commit hash if available |
| `system_info` | OS, Python version, GPU detected (name + VRAM), system RAM |
| `config` | All feature flags, LLM model used, dictionary backend used, semantic categories selected, EWFG lookup file path (if any) |
| `input_transcripts_folder, input_audio_folder, output_folder` | Absolute paths |
| `text_extensions_searched, audio_extensions_searched` | Extension lists actually used |
| `n_files_scanned, n_pairs_matched, n_pairs_processed, n_unmatched_text, n_unmatched_audio` | Summary counts |
| `pairs[]` | Per-stimulus array; each entry has `stimulus_id, transcript_path, audio_path, runtime_seconds, status` (`success`/`error`), `output_csv_path, output_summary_path, error_message` (if any) |
| `warnings[]` | Non-fatal issues (e.g., "EWFG lookup file missing → Edu_Zipf_* will be NaN", "ambiguous match for story1: chose .wav over .mp3") |
| `errors[]` | Fatal per-file errors with traceback excerpt |

**`run_<ts>_no-match.csv`** — columns: `filename, found_in_folder, missing_counterpart, reason`.

Example row: `story7.txt, transcripts/, audio/, no audio file with matching base name`.

### 10.4 Improved selectors

Reorganized into clear sections using `ttk` widgets (no new dependencies; `ttk` ships with tkinter):

**Section 1 — Inputs**

- Three folder-picker rows (Transcripts / Audio / Output). Each row: `Browse…` button + a label showing the truncated path with the full path on hover. `initialdir` for the picker is the last-used path from `preferences.json`.
- "Analyzer name" entry field. Persisted to `preferences.json`; auto-filled on next launch.

**Section 2 — Model & dictionary**

- **LLM model**: `ttk.Combobox` with descriptive entries:
  ```
  gpt-4o-mini       ($0.10–0.30/stim, r≈0.83, requires OPENAI_API_KEY)
  gpt-4o            ($1–3/stim, r≈0.85, requires OPENAI_API_KEY)
  Llama-3-70B local (free, r≈0.84, needs ≥48GB VRAM)
  Llama-3-8B local  (free, r≈0.79, needs ≥12GB VRAM)
  Llama-3.2-3B local(free, r≈0.76, needs ≥8GB VRAM)
  ```
  The auto-detected default for the user's hardware is highlighted/preselected.

- **Dictionary backend**: `ttk.Radiobutton` group — `LIWC (auto-detected)` / `Empath` / `Custom path…` — with an inline status indicator that updates live based on what dictionary files are present.

- **"Configure categories…"** button (already specced in §4.2): opens the modal checklist.

**Section 3 — Feature toggles**

Grouped checkboxes with section headers, replacing the existing flat list:

- **Linguistic**: spaCy parse
- **Frequency**: SUBTLEX Zipf, EWFG grade-band frequencies
- **Semantic**: semantic-domain indicators
- **Predictability**: GPT-2 surprisal (legacy baseline), LLM-cloze
- **Alignment**: MFA forced alignment + derived timing columns (`Word_Duration`, `Word_Gap`)
- **Outputs**: Document-level summary sidecar

Each toggle has a tooltip explaining what it does and which columns it emits.

**Section 4 — Run controls**

- **Run** button (large, primary). Disabled until required inputs are filled.
- **Progress bar**: `ttk.Progressbar` showing files processed / total.
- **Status label**: "Processing story3.wav (4 / 12)…"
- **Cancel** button: disabled until running. Cancellation terminates *after* the current file completes (avoids partial output).

### 10.5 Persistence — extending `preferences.json`

The `preferences.json` schema introduced in §4.2 is extended to:

```json
{
  "analyzer_name": "Amilcar Malave",
  "last_transcripts_folder": "C:/.../transcripts",
  "last_audio_folder": "C:/.../audio",
  "last_output_folder": "C:/.../output",
  "llm_model": "gpt-4o-mini",
  "dictionary_backend": "liwc",
  "semantic_categories": ["social", "cognition", "..."],
  "feature_toggles": {
    "spacy_parse": true,
    "zipf_freq": true,
    "edu_freq": true,
    "sem_domain": true,
    "gpt2_surprisal": true,
    "llm_cloze": true,
    "alignment": true,
    "doc_summary": true
  }
}
```

Schema is forward-compatible: unknown keys are ignored on load; missing keys fall back to the SOP default. CLI users who set explicit flags get those; users who don't set flags get values from `preferences.json` if present, else SOP defaults.

### 10.6 CLI parity with the GUI

Every GUI control has a CLI flag, so headless runs are equivalent. New CLI flags:

```
--transcripts-folder PATH   Batch mode: process all paired files in folder
--audio-folder PATH         Batch mode: paired-audio folder
--output-folder PATH        Where to write CSVs and run log artifacts
--analyzer NAME             Required for batch runs; recorded in manifest
--text-extensions LIST      Override default txt,tsv,md
--audio-extensions LIST     Override default wav,mp3,flac
```

When `--transcripts-folder` and `--audio-folder` are both provided, the existing `--audio` and `--transcript` single-file flags are ignored.

### 10.7 Backward compatibility

- The existing single-file flags (`--audio`, `--transcript`) remain functional for one-shot runs and existing scripts.
- Single-file runs do **not** emit a run log (it's a batch-mode artifact). They just write the per-stimulus CSV as today.
- `gui.py`'s old single-file UI is replaced. There's no transitional dual-mode UI; the folder-pair workflow becomes the only GUI workflow.

## 11. Roadmap (post-batch 1)

Items deferred to future batches, each with its own brainstorm → spec:

- **Batch 2: Acoustic prosody** — volume, pitch, delta volume, delta pitch (per Brain Treebank), word-level summarization. Adds librosa or Praat-Parselmouth dependency.
- **Batch 3: Phoneme/syllable level** — phoneme identity/onset/duration from MFA TextGrid; syllable boundaries; introduces the multi-level data model.
- **Batch 4: P-center** — perceptual onset of syllables; requires acoustic algorithm (Scott 1993 / Villing 2010). Syllable level.
- **Batch 5: BERT embeddings** — 768-dim contextual vectors per word; written to a separate sidecar to avoid CSV bloat.
- **Batch 6: Human-rated cloze norms** — not an extractor; data acquisition / collaboration with cloze-norming corpora.
- **Batch 7: Document-level enrichment** — readability scores, lexical diversity, type-token ratio, beyond what's in the §4.2 sidecar.
- **Cross-cutting: Correctness scoring** — clarify with collaborator whether this means MFA alignment confidence (low cost) or ASR-vs-transcript agreement (medium cost). Park until clarified.
- **Cross-cutting: Brain Treebank parity** — `Index_in_Sentence` (already have via `token_id`), `Word_Head` (already have via `head_token_id`). Verified during this brainstorm.

## Appendix A: EWFG acquisition checklist

For the user to follow, in order of cheapest effort first:

1. **Ask around the lab** — Blair Kaneshiro (PI), lab members, and adjacent reading/psycholinguistics groups. EWFG and TASA-corpus data have circulated in research labs since the 90s.
2. **Email Coh-Metrix group** at U Memphis (Arthur Graesser) or ASU (Danielle McNamara). They incorporate TASA frequency data and may share or point to a source.
3. **Buy used** if step 1-2 fail. AbeBooks / eBay / Amazon for ISBN 0-89172-307-9. **Confirm with seller that the CD-ROM is included** — many used copies have lost it. Budget ~$30-150.
4. **Read the disc** with an external USB CD drive if the laptop lacks one (~$20). Mount on Windows; copy entire disc contents to a local folder. Don't run the bundled lookup app.
5. **Send the data files** (extension `.dat`, `.txt`, `.csv`, `.dbf`, or no extension; tens of MB likely). I'll write a parser that produces `dictionaries/edu_zipf.parquet`.

Implementation is unblocked on this — the extractor uses a pluggable lookup interface and emits NaN columns until the data file is dropped in.

## Appendix B: Decision log

- 2026-05-10: Driver = open SOP growth, no specific deadline. Optimization target = feature value × cost.
- 2026-05-10: Three-feature batch 1 chosen (EWFG, semantic-domain, LLM-cloze). Brain Treebank cross-check confirmed batch 1 is additive vs prior literature.
- 2026-05-10: EWFG locked at 4 grade bands + overall, pluggable lookup interface. Acquisition path = lab → buy used → parse CD data.
- 2026-05-10: Semantic-domain locked at LIWC primary / Empath fallback (lab has LIWC license; Empath ships with repo for unlicensed users). 15-category SOP default. Word-level binary columns + document-level sidecar in same batch.
- 2026-05-10: LLM-cloze locked at gpt-4o-mini default with auto-detected local fallback (Llama-3-70b/8b/3.2-3b based on VRAM). Three metrics shipped. Provo Corpus validation script bundled. Cost/correlation table in docs.
- 2026-05-10: Word_Duration and Word_Gap added as cheap derived columns per Brain Treebank standard.
- 2026-05-10: Semantic-category selection gets a GUI checklist + persistence in `config/preferences.json`. CLI flag overrides the file; file overrides the SOP default. Each category's meaning + example words documented in Appendix C and the user-facing `Feature_Extraction_Guide.md`.
- 2026-05-10: GUI overhaul added to batch 1 scope (§10). Switch from per-file selection to folder-pair batch processing (top-level scan, permissive extensions: txt/tsv/md ↔ wav/mp3/flac, ambiguous → priority + warn, unmatched → no-match.csv). Required `analyzer` field. Run log artifacts: human `.log`, `manifest.json`, `no-match.csv`. Improved selectors for file/folder pickers, LLM model, dictionary backend, and feature toggles. `preferences.json` schema extended to persist analyzer name, last folder paths, model, dictionary, and feature toggles. Single-file CLI flags retained for backward compatibility.

## Appendix C: Default semantic category descriptions

These are the 15 SOP-default categories. Each definition is **dictionary-agnostic** — the actual word membership comes from whichever backend (LIWC or Empath) is active, and may differ slightly between them. The descriptions describe what the category is *intended* to capture and give example words. When LIWC is active, the LIWC manual's category definitions are authoritative for that run; when Empath is active, the Empath documentation's seed-word lists are authoritative.

| Category | What it captures | Example words |
|---|---|---|
| `social` | References to people, relationships, and social interaction | friend, neighbor, partner, group, together |
| `cognition` | Thinking, reasoning, knowledge, memory, decision-making | think, know, understand, believe, decide |
| `emotion_positive` | Positive affect — joy, love, hope, satisfaction | happy, love, hope, smile, glad |
| `emotion_negative` | Negative affect — sadness, anger, fear, anxiety | sad, angry, fear, hate, hurt |
| `body` | Body parts, bodily states, physical sensation | hand, face, hungry, sleep, tired |
| `motion` | Movement, motor actions, locomotion | run, walk, move, jump, throw |
| `perception` | Sensory experience — sight, sound, touch, taste, smell | see, hear, feel, smell, taste |
| `time` | Temporal references — moments, durations, sequence | yesterday, soon, hour, future, past |
| `space` | Spatial relations and locations | above, near, far, inside, beyond |
| `math` | Quantity, number, calculation, measurement | number, equal, count, half, total |
| `science` | Scientific domains and inquiry | experiment, theory, atom, biology, hypothesis |
| `work` | Labor, professions, productivity, achievement | job, employer, task, deadline, project |
| `family` | Kinship and family relationships (subset of `social`) | mother, father, child, sibling, parent |
| `communication` | Language acts and verbal interaction (subset of `social`) | speak, write, say, message, ask |
| `money` | Finance, commerce, economic concepts | pay, buy, cost, dollar, business |

**Notes on overlap:** `family` and `communication` are intentional subsets of `social`. A word like `parent` will typically match both `family` and `social`; a word like `say` will match both `communication` and `social`. This overlap is by design — analyses can use the broader `social` category alone, or use the finer-grained subset categories for higher resolution. The `Sem_Categories` roll-up column lists every match (e.g., `social,family`).

**LIWC vs Empath alignment:** The 15 SOP-default category names map onto:

- LIWC equivalents (LIWC-22 namespace): `Social`, `Cognition`, `posemo`/`tone_pos`, `negemo`/`tone_neg`, `body`, `motion`, `perception`/`Percept`, `time`, `space`, `quant`, (no direct `science` — derived from category subsets), `work`, `family`, `Social/Communication`, `money`. Exact LIWC-22 category mapping documented in `Feature_Extraction_Guide.md`.
- Empath equivalents: `social`, `cognitive_processes`, `positive_emotion`, `negative_emotion`, `body`, `movement`, `sensory`, `time`, `space`, `math`, `science`, `work`, `family`, `communication`, `money`.

When a SOP-default category has no exact equivalent in the active dictionary, the loader falls back to the closest available category and logs the substitution.
