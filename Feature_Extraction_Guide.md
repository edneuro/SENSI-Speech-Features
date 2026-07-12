# Feature Extraction Guide

## Overview

SensiExtractor produces word-level features from audio and text stimuli. Features fall into four categories:

| Category | Features | Source |
|----------|----------|--------|
| Structural | Document/sentence location, character offsets | spaCy |
| Lexical-linguistic | POS, lemma, morphology, syntax | spaCy |
| Statistical | Word frequency, contextual surprisal | wordfreq, GPT-2 |
| Acoustic | Word onset and offset times | Montreal Forced Aligner |

---

## 1. Structural features

These track the position of each token within the document. They are bookkeeping columns produced automatically by the spaCy pipeline.

| Column | Description | Example |
|--------|-------------|---------|
| `doc_id` | Document index — always 1 per file | 1 |
| `paragraph_id` | Paragraph number — usually 1 for short stimuli | 1 |
| `sentence_id` | Sentence number within the document | 3 |
| `sentence` | Full text of the sentence this token belongs to | "She left the room." |
| `start` | Character offset of the token's first character within `sentence` | 5 |
| `end` | Character offset of the token's last character within `sentence` | 8 |
| `token_id` | Global token counter within the file | 47 |
| `token_in_sentence_id` | Token index within the current sentence | 2 |

> ⚠️ `start`/`end` (lowercase) are character positions in the sentence string — not audio times. Do not confuse with `Onset`/`End` (uppercase), which are seconds in the audio file.

---

## 2. Lexical-linguistic features

Extracted using **spaCy** with the `en_core_web_sm` statistical model, trained on OntoNotes 5.0. These follow the standard CoNLL-U morphosyntactic annotation layers used in Universal Dependencies.

### Tokenization and lemmatization

| Column | Description | Example (token: "ran") |
|--------|-------------|------------------------|
| `token` | Word as it appears in the text | "ran" |
| `lemma` | Dictionary base form | "run" |

Lemmatization is rule-based and context-sensitive — it uses POS tag information to distinguish, for example, *saw* (past tense of *see*) from *saw* (the tool).

### Part-of-speech tags

Two tagging schemes are provided simultaneously:

**Universal POS (`upos`)** — coarse, cross-linguistically consistent tags from the Universal Dependencies framework.

| Tag | Meaning | Examples |
|-----|---------|---------|
| NOUN | Common noun | dog, city |
| PROPN | Proper noun | London, Mary |
| VERB | Verb | run, think |
| ADJ | Adjective | fast, happy |
| ADV | Adverb | quickly, never |
| ADP | Adposition (preposition/postposition) | in, on, under |
| DET | Determiner | the, a, this |
| PUNCT | Punctuation | . , ! |
| NUM | Numeral | 42, three |
| PART | Particle | not, 's |
| CCONJ | Coordinating conjunction | and, but, or |
| SCONJ | Subordinating conjunction | that, because, if |

**Penn Treebank POS (`xpos`)** — finer-grained tags. Examples: `NN` (singular noun), `NNS` (plural noun), `VBD` (past tense verb), `VBZ` (3rd-person singular present), `JJ` (adjective), `RB` (adverb).

### Morphological features (`feats`)

Encodes inflectional properties in UD key=value format:

```
Number=Sing|Person=3|Tense=Past|VerbForm=Fin
```

Returns `NA` for uninflected words (prepositions, determiners, etc.).

### Dependency parse

| Column | Description | Example |
|--------|-------------|---------|
| `dep_rel` | Grammatical function of this token relative to its head | "nsubj", "obj", "amod" |
| `head_token_id` | `token_id` (global, whole-document index) of the syntactic head; a root points to its own `token_id` | 3 |
| `deps` | Enhanced dependency representation (rarely populated for English) | "NA" |
| `misc` | Miscellaneous flags | "SpaceAfter=No" |

Common dependency relations:

| Label | Meaning | Example |
|-------|---------|---------|
| `root` | Main predicate of the sentence | *She **left** the room.* |
| `nsubj` | Nominal subject | ***She** left the room.* |
| `obj` | Direct object | *She left **the room**.* |
| `amod` | Adjectival modifier | *a **tall** building* |
| `advmod` | Adverbial modifier | *She **quickly** left.* |
| `det` | Determiner | ***The** cat sat.* |
| `prep` / `case` | Preposition/case marker | *She left **in** a hurry.* |

---

## 3. Statistical features

### Word frequency — `Zipf_Freq`

**What it measures:** How common a word is in everyday English, on a logarithmic scale.

**Source:** SUBTLEX-US (Brysbaert & New, 2009) — a 51-million-word corpus of American English film and TV subtitles, chosen because it better predicts human lexical processing times than traditional written-text corpora. Queried via the `wordfreq` library.

**The Zipf scale** is a log-transform of raw corpus frequency:

$$\text{Zipf} = \log_{10}\!\left(\frac{\text{occurrences}}{\text{corpus size}} \times 10^9\right) + 3$$

The +3 offset puts the scale in a convenient range. Rough landmarks:

| Zipf | Approximate frequency | Typical words |
|------|-----------------------|---------------|
| 7–8 | > 10,000 per million | *the*, *be*, *and* |
| 5–6 | ~100–1,000 per million | *run*, *house*, *think* |
| 3–4 | ~1–10 per million | *canopy*, *abolish* |
| 1–2 | < 0.1 per million | rare / technical terms |
| 0 | not in corpus | out-of-vocabulary |

Frequency is looked up per **lemma** so that *ran*, *runs*, and *running* all receive the frequency of *run*. Punctuation tokens receive NaN.

**Relevance to EEG:** Word frequency is one of the strongest predictors of the N400 component. Lower-frequency words produce larger (more negative) N400 amplitudes, reflecting greater difficulty during lexical access.

---

### Contextual surprisal — `Surprisal`

**What it measures:** How unpredictable a word is given everything that preceded it in the text — a direct operationalization of predictive processing.

**Mathematical definition:**

$$S(w_t) = -\log_2 P(w_t \mid w_1, w_2, \ldots, w_{t-1})$$

Surprisal is the self-information of a word under the conditional probability distribution given its full left context. The unit is **bits**:

- 0 bits = the word was perfectly predicted
- 1 bit = the word was one of two equally likely options
- Each additional bit halves the prior probability

**Model:** GPT-2 (Radford et al., 2019) — a 117M-parameter autoregressive transformer language model trained on WebText (~40 GB of curated web text). GPT-2 assigns a probability to each word given all preceding words, which is exactly the conditional distribution needed for surprisal. The model runs entirely on your local CPU; no internet connection is required after the initial download.

**Extraction procedure:**

1. The transcript is tokenized into GPT-2's subword vocabulary using byte-pair encoding (BPE).
2. A single forward pass computes the log-probability of each subword token given its left context.
3. For words that BPE splits into multiple pieces (e.g., *unbelievable* → *un·believ·able*), subword log-probabilities are summed to give the joint log-probability of the full word.
4. Surprisal in bits: $S(w) = -\sum_k \log_2 P(\text{subword}_k \mid \text{left context})$

**Context window:** GPT-2 processes up to 1,024 subword tokens (~750 words) at a time. For longer transcripts, the pipeline automatically applies a sliding window (512-token steps with 512-token context overlap) to avoid truncation artifacts at the cost of slightly reduced context for early words in each chunk.

**Typical values:**

| Surprisal (bits) | Interpretation | Example |
|-----------------|----------------|---------|
| 0–3 | Highly expected | *the* after a determiner slot |
| 4–8 | Normal content word | typical nouns and verbs in context |
| 9–15 | Unexpected | semantically anomalous words |
| > 15 | Very surprising | out-of-place words |

**Relationship to word frequency:** Frequency captures how common a word is in general; surprisal captures how predictable it is in this specific context. A rare word used in a context that strongly predicts it will have low surprisal despite low frequency. The two measures are correlated but dissociable in well-controlled stimuli.

**Relevance to EEG:** Surprisal is a primary predictor of N400 amplitude in predictive processing accounts of language comprehension (Kuperberg & Jaeger, 2016). It captures the top-down, context-driven component of N400 beyond what frequency alone explains.

---

## 4. Acoustic features — forced alignment

**What it measures:** The precise start and end time (in seconds) of each word within the audio file.

| Column | Description | Example |
|--------|-------------|---------|
| `Transcript` | Word form sent to the aligner — numbers spelled out | "twenty-four" |
| `Onset` | Word start time in the audio file (seconds) | 3.4210 |
| `End` | Word end time in the audio file (seconds) | 3.8870 |

**Tool:** Montreal Forced Aligner (MFA; McAuliffe et al., 2017) — a Kaldi-based forced alignment system that takes a known transcript and an audio file and recovers the precise acoustic boundaries of each word and phoneme.

**How forced alignment works:**

1. **Transcript preparation:** Numbers are spelled out (*24* → *twenty-four*) because the pronunciation dictionary contains English words, not digit strings.
2. **Pronunciation lookup:** A dictionary (`english_us_arpa`, based on the CMU Pronouncing Dictionary) maps each word to its canonical phoneme sequence using ARPABET notation. Example: *cat* → `K AE1 T`.
3. **Acoustic modeling:** A GMM-HMM acoustic model (`english_us_arpa`, trained on LibriSpeech) characterizes the spectral properties of each phoneme across time frames.
4. **Viterbi decoding:** Given the audio and the expected phoneme sequence, the Viterbi algorithm finds the most probable frame-by-frame alignment between the audio and the phoneme sequence.
5. **Word boundary extraction:** Phoneme boundaries are aggregated into word boundaries and written to a Praat TextGrid file, which the pipeline parses into `Onset` / `End` values.

**Why not use automatic speech recognition (ASR)?** ASR transcribes unknown audio and introduces transcription errors and timing heuristics. Forced alignment takes a *known* transcript and finds exactly where in the audio each word occurs, giving phoneme-level precision suitable for EEG time-locking.

**Prerequisites:**
- Audio must be a `.wav` file (mono or stereo).
- The transcript must match the audio content. Words present in the transcript but absent from the audio (or vice versa) will cause alignment failures or silently incorrect boundaries.
- MFA must be installed via conda (see README).

---

## Software stack summary

| Feature group | Library | Version | Reference |
|---------------|---------|---------|-----------|
| All linguistic annotation | spaCy (`en_core_web_sm`) | ≥ 3.7 | Honnibal et al. (2020) |
| Word frequency | wordfreq + SUBTLEX-US | ≥ 3.0 | Brysbaert & New (2009) |
| Surprisal | HuggingFace Transformers + GPT-2 | ≥ 4.30 | Radford et al. (2019) |
| Forced alignment | Montreal Forced Aligner | ≥ 2.0 | McAuliffe et al. (2017) |
| Acoustic model | `english_us_arpa` (LibriSpeech GMM-HMM) | — | MFA model zoo |
| Number normalization | num2words | ≥ 0.5 | — |
| TextGrid parsing | tgt | ≥ 1.4 | — |

---

## References

- Brysbaert, M., & New, B. (2009). Moving beyond Kučera and Francis: A critical evaluation of current word frequency norms and the introduction of a new and improved word frequency measure for American English. *Behavior Research Methods, 41*(4), 977–990.
- Honnibal, M., Montani, I., Van Landeghem, S., & Boyd, A. (2020). spaCy: Industrial-strength Natural Language Processing in Python. Zenodo. https://doi.org/10.5281/zenodo.1212303
- Kuperberg, G. R., & Jaeger, T. F. (2016). What do we mean by prediction in language comprehension? *Language, Cognition and Neuroscience, 31*(1), 32–59.
- McAuliffe, M., Socolof, M., Mihuc, S., Wagner, M., & Sonderegger, M. (2017). Montreal Forced Aligner: Trainable text-speech alignment using Kaldi. *Proceedings of Interspeech 2017*, 498–502.
- Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). Language models are unsupervised multitask learners. *OpenAI Blog, 1*(8).
- Shannon, C. E. (1948). A mathematical theory of communication. *Bell System Technical Journal, 27*(3), 379–423.

---

## Batch 1 features (added 2026-05-10)

This section documents the features and GUI changes introduced by Batch 1.
For the design rationale, see `docs/superpowers/specs/2026-05-10-feature-batch-1-design.md`.

### EWFG grade-band frequencies — `Edu_Zipf_*`

Five word-level columns derived from the *Educator's Word Frequency Guide* (Zeno
et al. 1995):

- `Edu_Zipf_K2`     — Zipf-scale frequency in K-grade-2 educational texts
- `Edu_Zipf_3to5`   — grades 3–5
- `Edu_Zipf_6to8`   — grades 6–8
- `Edu_Zipf_9to12`  — grades 9–12
- `Edu_Zipf_Overall` — aggregate across all grades

Lookup data is **not bundled** (license-bound). The pipeline emits NaN for these
columns until you place a parquet lookup at `dictionaries/edu_zipf.parquet`. See
the spec's Appendix A for acquisition steps. Override path: `--edu-freq-lookup PATH`.

### Semantic-domain indicators — `Sem_*`

One binary column per category for content words (NOUN, VERB, ADJ, ADV, PROPN);
function words and punctuation get NaN.

- Backend: **Empath** (open-source, pip-installed) — the default and only auto-selected backend. **LIWC is disabled**: the LIWC-22 EULA (<https://www.liwc.app/help/eula>) forbids using its dictionary outside LIWC's own software, so SensiExtractor neither ships a `.dic` nor auto-selects LIWC. A license holder can still opt in from the CLI with `--sem-dict path/to/their.dic` (the GUI radio is greyed out). To license LIWC: <https://liwcsoftware.onfastspring.com/>.
- Default category set (15 categories): `social, cognition, emotion_positive, emotion_negative, body, motion, perception, time, space, math, science, work, family, communication, money`.
- Each category's intended meaning and example words: see Appendix C of the spec doc.
- Roll-up column `Sem_Categories`: comma-separated list of categories the word matched.

Configure via the GUI's "Configure categories…" modal (saved to `config/preferences.json`)
or via CLI `--sem-categories cat1,cat2,...`.

### LLM-cloze — `LLM_Cloze`, `LLM_Surprisal`, `LLM_TopMatch`

Three columns from a single LLM forward pass per token:

| Column | Meaning |
|---|---|
| `LLM_Cloze` | P(word \| preceding context) — empirical-cloze proxy |
| `LLM_Surprisal` | -log2 P(word \| preceding context), in bits |
| `LLM_TopMatch` | 1 if model's top-1 next-token == target, else 0 |

The existing `Surprisal` column (GPT-2) is **kept unchanged** as a baseline.

#### Model selection (auto-detect)

| Selected when | Model | Approx. cloze r vs Provo | Cost per ~3000-word stimulus |
|---|---|---|---|
| `OPENAI_API_KEY` is set | `gpt-4o-mini` | ~0.83 | $0.10–0.30 (API) |
| GPU VRAM ≥ 48 GB | `meta-llama/Meta-Llama-3-70B` | ~0.84 | Free (electricity) |
| GPU VRAM ≥ 12 GB | `meta-llama/Meta-Llama-3-8B` | ~0.79 | Free |
| GPU VRAM ≥ 8 GB | `meta-llama/Llama-3.2-3B` | ~0.76 | Free |

Override with `--llm-model gpt-4o|gpt-4o-mini|llama-3-70b|llama-3-8b|llama-3.2-3b|<hf-id>`.
Run `python utils/validate_cloze.py` once after install to confirm the chosen model
correlates with human cloze norms in the expected range.

### Derived timing — `Word_Duration`, `Word_Gap`

Computed from MFA `Onset` / `End`:

- `Word_Duration` = `End - Onset`
- `Word_Gap` = `Onset[i] - End[i-1]` (NaN for the first word in a stimulus)

NaN for words MFA couldn't align.

### Document-level summary — `<stimulus>_summary.csv`

In batch mode, every successful pair emits a one-row sidecar CSV with:

- counts: `n_tokens, n_content_tokens, n_sentences, n_paragraphs`
- means: `mean_zipf_freq, mean_edu_zipf_overall, mean_surprisal, mean_llm_surprisal, mean_llm_cloze`
- per-category: `count_<cat>, prop_<cat>` for each active semantic category
- `dictionary_used`: `LIWC` or `Empath`

Disable with `--no-doc-summary`.

### GUI workflow

1. Launch: `python gui.py`
2. Pick a **Transcripts folder** (text files: `.txt, .tsv, .md`)
3. Pick an **Audio folder** (audio files: `.wav, .flac, .mp3`)
4. Pick an **Output folder** (where features CSVs and run logs go)
5. Enter your **Analyzer name** (required, persisted between runs)
6. Choose LLM model, dictionary backend, semantic categories, and feature toggles
7. Click **Run extraction**

The GUI scans only the **top level** of each folder (no recursion).
Files are paired by base filename (case-insensitive). Unmatched files are
listed in `run_<ts>_no-match.csv`. Errors on individual files are recorded in
the manifest and don't halt the batch.

### Persistence — `config/features.yaml` + `config/run.yaml`

The GUI persists the analyzer name, last folder paths, model choice, dictionary
backend, semantic categories, and feature toggle states between sessions, split
across two gitignored YAML files: `features.yaml` (global "what to extract" —
"Reset settings" rewrites exactly this file) and `run.yaml` (machine-specific
folders/analyzer/token paths). Override the directory holding both via the
`SENSI_CONFIG_DIR` env var.

### Run log artifacts (batch mode)

Three artifacts in the output folder, named with the run timestamp:

- `run_<ts>.log` — human-readable log
- `run_<ts>_manifest.json` — structured manifest with environment, config, and per-pair details
- `run_<ts>_no-match.csv` — files that didn't pair, with reason

The manifest captures the analyzer, system info, configuration snapshot, and
per-pair runtimes — sufficient to reproduce or debug a run later.

### Validation: `utils/validate_cloze.py`

Runs the configured LLM over Provo Corpus and reports correlation with human
cloze norms. Use after installation to confirm the LLM-cloze setup is working.

---

## Batch 1.5 — Tabbed GUI and document-level features (added 2026-05-10)

This batch restructures the GUI into independent Word-level and Document-level
tabs, adds new document-level feature groups (lexical diversity, readability,
LIWC summary scores), and migrates `preferences.json` to a nested schema.

### What's new structurally

- **Tabs**: The GUI is now a `ttk.Notebook` with `Word level` and `Document level` tabs.
- **Shared inputs at top**: Folder pickers, analyzer name, and dictionary backend are shared between tabs.
- **Independent feature toggles, categories, and run buttons per tab**.
- **Document-level output**: a single master CSV `documents_summary_<ts>.csv` with one row per stimulus, replacing the previous per-stim `<stim>_summary.csv` sidecar.

### New document-level features

| Group | Columns | When emitted |
|---|---|---|
| Counts | `n_tokens, n_content_tokens, n_sentences, n_paragraphs` | Counts toggle on |
| Aggregates | `mean_zipf_freq, mean_edu_zipf_<band>, mean_surprisal, mean_llm_cloze, mean_llm_surprisal, mean_llm_top_match_prop` | Each aggregate has its own toggle |
| Lexical diversity | `ttr`, `mtld` | TTR / MTLD toggles |
| Readability | `flesch_reading_ease`, `flesch_kincaid_grade` | Flesch-Ease / FK-Grade toggles |
| Category proportions | `count_<cat>`, `prop_<cat>` per active category | Category-proportions toggle |
| Composite psychological scores (Pennebaker-formula approximations) | `psych_authentic, psych_analytic, psych_tone, psych_clout` | any psych-* toggle on (no backend requirement) |

### Notes on composite psychological scores (Pennebaker-formula approximations)

These are approximations computed from publicly documented Pennebaker et al. (2014) formulas. As of 2026-06-10 they are produced **license-clean** (`extractors/psych_composite.py`): the grammatical inputs (articles, prepositions, pronouns by person, auxiliaries, …) come from spaCy's POS/morphology, and Tone comes from the VADER valence lexicon — **no LIWC dictionary is used**. They therefore no longer require (or depend on) the LIWC backend. For LIWC-22's official Authentic/Analytic/Tone/Clout numbers, use the LIWC desktop software (liwc.app). The `psych_` prefix is intentional: naming them `liwc_*` would overstate the authority of the output.

### Categories dialog change

Closing the categories dialog via the window X button now **saves** the current selection (matching most settings-panel UX conventions). The Cancel button still discards changes.

### Preferences migration

The `config/preferences.json` schema has changed from a flat layout (batch 1) to a three-section layout (`shared`, `word_level`, `document_level`). When SensiExtractor starts, an old flat-shape file is auto-migrated on first load; subsequent loads use the new shape.

### New CLI flags (batch mode)

- `--level word|document` — selects which level to run. Default `word` (backward compatible).
- `--no-doc-counts`, `--no-doc-mean-zipf`, `--no-doc-mean-edu-zipf`, `--no-doc-mean-surprisal`, `--no-doc-mean-llm-cloze`, `--no-doc-ttr`, `--no-doc-mtld`, `--no-doc-flesch-ease`, `--no-doc-fk-grade`, `--no-doc-category-proportions`, `--no-doc-psych-authentic`, `--no-doc-psych-analytic`, `--no-doc-psych-tone`, `--no-doc-psych-clout` — skip the corresponding doc-level feature. (The psych flags were renamed from `--no-doc-liwc-*` on 2026-06-10 when the scores became license-clean; output CSV column names already used the `psych_` prefix.)

### CLI example — document-level batch

```bash
python main.py \
  --level document \
  --transcripts-folder path/to/transcripts \
  --audio-folder       path/to/audio \
  --output-folder      path/to/output \
  --analyzer           "Your Name"
```

### What's removed in batch 1.5

- `<stim>_summary.csv` per-stim sidecar (replaced by document-level master CSV).
- `--no-doc-summary` CLI flag (replaced by the per-feature toggles above).

---

## Batch 2 — PySide6 GUI redesign (added 2026-05-11)

This batch replaces the tkinter tabbed interface with a modern PySide6-based sidebar/content architecture, adds continuous inline validation and in-session API-key handling, and introduces per-band educational word-frequency filtering.

### New `gui/` package structure

The legacy tkinter files (`gui.py`, `gui_word_tab.py`, `gui_document_tab.py`, etc.) have been removed. The new `gui/` package is organized around a sidebar navigation + content panes model:

- `main_window.py` — Main QMainWindow container, sidebar, and status strip
- `theme.py` — QSS stylesheets and design-system tokens
- `widgets.py` — Reusable components (SectionHeader, FolderRow, ToggleRow, AmberDotIndicator)
- `section_inputs.py` — Input paths, analyzer name, API-key field
- `section_features.py` — Feature toggles organized by group (Structural, Lex-ling, Frequency, Predictability, Semantic, Acoustic, Doc-level)
- `section_run.py` — Output checkboxes, readiness status, cost panel, Run button
- `section_logs.py` — Live log viewer and past-runs list
- `status_strip.py` — Persistent bottom bar with runtime and result indicators
- `category_picker.py` — Inline modal for semantic-category selection
- `preflight.py` — Continuous validation warnings (missing files, invalid API key, etc.)
- `cost_estimate.py` — OpenAI API cost estimation via tiktoken
- `run_dispatch.py` — Background-thread orchestrator for long-running extractions

### Column rename: `liwc_*` → `psych_*`

Document-level summary CSV now uses `psych_authentic`, `psych_analytic`, `psych_tone`, `psych_clout` instead of `liwc_*` prefixes. These are Pennebaker-formula approximations computed license-clean from spaCy POS/morphology + VADER valence (since 2026-06-10; previously from a LIWC-style dictionary) — **not** official LIWC-22 proprietary scores. Users requiring official Pennebaker scores should run the LIWC desktop software (liwc.app).

### Per-band EWFG output filtering

The Features page now exposes five independent toggles for Educator's Word Frequency Guide (EWFG) grade-band columns:

- `Edu_Zipf_K2` — K–2 grades (toggle: K-2)
- `Edu_Zipf_3to5` — grades 3–5 (toggle: 3-5)
- `Edu_Zipf_6to8` — grades 6–8 (toggle: 6-8)
- `Edu_Zipf_9to12` — grades 9–12 (toggle: 9-12)
- `Edu_Zipf_Overall` — aggregate (toggle: Overall)

**CLI flag:** `--edu-bands k2,3to5,overall` (comma-separated list; default = all bands). Example:

```bash
python main.py --transcripts-folder ... --edu-bands k2,overall
```

Outputs only `Edu_Zipf_K2` and `Edu_Zipf_Overall`; omits 3-5, 6-8, 9-12.

### Preferences schema v2 → v3 migration

The `config/preferences.json` schema has been further refined from v2 (nested `word_level`, `document_level`) to v3 (flattened `features`, `output`). Semantic category lists are now split by backend:

- `categories_liwc` — Categories selected when LIWC backend is active
- `categories_empath` — Categories selected when Empath backend is active

Auto-migration runs once on first launch; subsequent loads use v3. Old v2 files are backed up to `config/preferences.v2.backup.json`.

### Design and planning documents

- **Spec:** `docs/superpowers/specs/2026-05-11-gui-redesign-design.md`

---

## Batch 3 — Word/Doc Separation, Modal Categories, Per-Column Toggles (added 2026-05-11)

### GUI restructure

The sidebar now has **five items**:

1. **Inputs**
2. **Word-level features**
3. **Document-level features**
4. **Run**
5. **Logs**

Word-level and document-level configuration each live on their own page, with their own feature subsections and their own "Configure categories…" button.

### Per-column toggles

Four feature blocks expanded from single toggles into per-column checklists:

| Block | Columns | New per-column toggle keys |
|---|---|---|
| Structural (word-level) | doc_id, paragraph_id, sentence_id, sentence, start, end, token_id, token_in_sentence_id | `feat_struct_doc_id`, …, `feat_struct_token_in_sentence_id` |
| Acoustic (word-level) | Onset, End, Word_Duration, Word_Gap | `feat_acoustic_onset`, …, `feat_acoustic_word_gap` |
| Composite psych (doc-level) | psych_authentic, psych_analytic, psych_tone, psych_clout | `feat_psych_authentic`, …, `feat_psych_clout` |
| Counts (doc-level) | n_tokens, n_content_tokens, n_sentences, n_paragraphs | `out_count_tokens`, …, `out_count_paragraphs` |

Column dropping is applied in `main.run_word_level_batch` and `main.run_doc_level_batch` via `args.feature_toggles` and `args.output_toggles` dicts. CLI invocations without these attributes retain all columns (backward compat).

### Independent word vs document semantic categories

Word-level and document-level category selections are now separate. The Configure-categories button on each page opens a modal dialog and writes to its own prefs key:

- Word-level: `semantic.word_categories_liwc`, `semantic.word_categories_empath`
- Doc-level:  `semantic.doc_categories_liwc`, `semantic.doc_categories_empath`

On migration from v3, both levels start with the same selection (whatever was previously in the single shared list); they can diverge as the user edits one.

### Preferences schema v3 → v4 migration

Schema bump to v4:
- The single `features.structural`, `features.acoustic_boundaries`, `features.sem_composite_psych`, and `output.doc_counts` toggles are each expanded into per-column children. The parent toggle's value is copied to all children at migration time.
- `semantic.categories_{liwc,empath}` becomes four lists (`word_*` and `doc_*`).

The full v1 → v2 → v3 → v4 chain runs on load so files from any prior batch still work.

### Visual

- Each feature subsection is wrapped in a **Card** (white panel with a subtle border) for visual separation.
- Accent color is **SENSI cyan `#00B2D0`** (matches `assets/icon.svg`). The icon shows in the title bar and taskbar.
- Light theme only (dark theme removed in batch 3).

### Design and planning documents

- **Spec:** `docs/superpowers/specs/2026-05-11-batch-3-design.md`


---

## Acoustic Prosody (added 2026-05-12)

This batch adds **word-level (10 columns)** and **document-level (16 columns)** prosodic features extracted from the audio file via Praat (`praat-parselmouth`). MFA word boundaries (`Onset`/`End`) are used to slice the per-frame pitch and intensity contours into per-word windows.

### Word-level columns (in `<stimulus>_features.csv`)

| Column | Definition | Unit |
|---|---|---|
| `Pitch_Mean` | Mean F0 over voiced frames in the word | Hz or semitones (see config) |
| `Pitch_Min` / `Pitch_Max` | 5th / 95th percentile (robust to tracker errors) | same |
| `Pitch_Std` | Std of voiced frames | same |
| `Volume_Mean` / `Volume_Min` / `Volume_Max` / `Volume_Std` | Same aggregates over Praat intensity | dB |
| `Pitch_Delta` / `Volume_Delta` | Word-to-word change (`X[i] - X[i-1]`); NaN for first word | same |

**Pitch units** — `--pitch-units {hz, semitones}` or set in the GUI (Word features → Prosody card). Semitones are computed relative to the per-stimulus voiced-pitch median. The unit is recorded in the run manifest.

**Pitch search range** — `--pitch-floor` / `--pitch-ceiling` (defaults 75 / 500 Hz). Lower the floor for male bass voices, raise the ceiling for children/sopranos to avoid octave errors.

**Unvoiced words** → `Pitch_*` columns NaN; `Volume_*` still valid. **No alignment** → all 10 prosody columns NaN (prosody depends on MFA `Onset`/`End`).

### Document-level columns (in `documents_summary_<ts>.csv`)

| Column | Definition | Unit |
|---|---|---|
| `doc_pitch_mean` / `_min` / `_max` / `_std` | Pitch aggregates over the whole stimulus (voiced frames; p5/p95 trim for min/max) | Hz or st |
| `doc_volume_mean` / `_min` / `_max` / `_std` | Same for intensity | dB |
| `doc_pitch_delta_mean` / `doc_volume_delta_mean` | Mean of `|Pitch_Delta|` / `|Volume_Delta|` (absolute, so rises and falls do not cancel) | same |
| `doc_hnr` | Harmonics-to-Noise Ratio, mean over voiced frames (Boersma 1993) | dB |
| `doc_articulation_rate` | Total syllables (via `pyphen` on each word's surface form) / total spoken time, pauses excluded | syll/sec |
| `doc_pause_count` | Inter-word gaps `>` pause threshold | count |
| `doc_pause_mean_duration` / `_total_duration` | Mean / sum of those gap durations | sec |
| `doc_pause_pct_silence` | `doc_pause_total_duration / (End[last] - Onset[first])` | proportion |

**Pause threshold** — `--pause-threshold` (default 0.25 s) or set in the GUI (Doc features → Prosody card). Gaps strictly greater than the threshold count as pauses.

### CLI flags (group-level)

```
--no-prosody              Skip all 10 word-level prosody columns
--no-doc-prosody          Skip all 16 doc-level prosody + pause columns
--pitch-units {hz, semitones}
--pause-threshold SECONDS
--pitch-floor HZ
--pitch-ceiling HZ
```

Per-column granularity is GUI-only (via `prefs["features"]["feat_prosody_*"]` and `feat_doc_prosody_*`).

### Dependencies

- `praat-parselmouth>=0.4` — pitch (autocorrelation), intensity, HNR
- `pyphen>=0.14` — syllable counting for articulation rate

Both are in `requirements.txt`. The preflight panel warns if either is missing or if prosody is enabled without MFA alignment.

### Design and planning documents

- **Spec:** `docs/superpowers/specs/2026-05-12-prosody-design.md`
