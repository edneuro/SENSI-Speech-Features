# Feature Dependencies Map

**Purpose.** Bottom-up inventory of every feature SensiExtractor produces, grouped by **level** (word / sub-word / sentence / document) and by **cluster** (features that share libraries, inputs, and outputs). The top section lists **shared stages** — heavy infrastructure (spaCy parse, MFA, Praat-Parselmouth) that produces level-independent artifacts multiple feature clusters consume.

**Audience.** Anyone designing the dispatcher / planner that translates the user's output selection into a minimal pipeline run. The planner walks this graph backward: from "what columns/CSVs the user wants" → "which feature clusters" → "which shared stages must run".

**Format.** Each cluster declares:
- **Features** — the columns or sidecar rows it emits
- **Library** — Python package + any external binary
- **Inputs** — what artifacts it consumes (text, audio, output of stage X, etc.)
- **Outputs** — DataFrames / columns / files
- **Per-column toggles** — if any (so the planner knows which feature keys gate this cluster)

A cluster is **one entry per shared-library-and-input set**, per the user's clustering rule. Two extractors that use different libraries get separate clusters even if they conceptually belong to the same "group" (e.g. SUBTLEX vs EWFG frequencies).

---

## 0. Shared stages (level-independent)

These don't emit user-facing columns directly. They produce raw artifacts that one or more level-specific clusters consume. The planner should run each at most once per stimulus regardless of how many downstream consumers want it.

### 0.1 spaCy parse — `extractors/text_parser.py`

- **Library:** `spacy` + `en_core_web_sm` model
- **Inputs:** transcript text (str)
- **Outputs:** `word_df` (one row per token) with columns:
  - Identifiers: `doc_id, paragraph_id, sentence_id, sentence, start, end, term_id, token_id`
  - Linguistic: `token, lemma, upos, xpos, feats, dep_rel, head_token_id`
- **Consumers:** every word-level cluster except pure-acoustic; every sub-word cluster (needs `doc_id/sentence_id/token_id` for FK joins); every doc-level cluster that aggregates word features.
- **Notes:** Tokenization is the canonical "what is a word" definition for the pipeline. `start/end` are **character offsets** in the sentence string, not seconds — easy to confuse with `Onset/End` from MFA.

### 0.2 MFA forced alignment — `extractors/aligner.py`

- **Library:** `montreal-forced-aligner` (conda, not pip) subprocess; `tgt` for parsing the TextGrid output
- **External assets:** `english_us_arpa` acoustic model + dictionary (in `mfa_data/`)
- **Inputs:** audio path (.wav), prepared transcript (digits spelled out by `utils/transcript_prep.py` using `num2words`)
- **Outputs:** two DataFrames
  - `word_alignment_df` with columns `Transcript, Onset, End` (seconds, one row per spoken word)
  - `phones_df` with columns `phone, onset, offset` (seconds, one row per phoneme, ARPABET labels with stress digits 0/1/2 on vowels)
- **Consumers:** word-level acoustic + prosody; all sub-word clusters; doc-level prosody.
- **Notes:** Single subprocess call per stimulus; expensive (~10-60s on real audio). Words MFA fails to align come back missing — the per-word merge in `main.py:_merge_alignment` leaves `Onset/End` as NaN for those, and downstream consumers must tolerate NaN.

### 0.3 Praat-Parselmouth acoustic pass — `extractors/audio_features.py`

- **Library:** `praat-parselmouth`
- **Inputs:** audio path (.wav), pitch floor/ceiling (defaults 75/500 Hz)
- **Outputs:** `Contours` dataclass containing:
  - `pitch_hz: np.ndarray` (NaN at unvoiced frames)
  - `intensity_db: np.ndarray` (dB SPL)
  - `times: np.ndarray` (frame timestamps, seconds, default 10 ms step)
  - `hnr_db: float` (mean Harmonics-to-Noise Ratio over voiced frames)
  - `speaker_median_pitch_hz: float` (semitone reference)
- **Consumers:** word-level prosody; sub-word syllable prosody; doc-level prosody (pause stats, pitch/volume aggregates).
- **Notes:** Single Parselmouth pass per stimulus — the entire `Contours` object is computed once and sliced by every consumer that needs a sub-interval. Never re-call.

---

## 1. Word level

Emits `<stim>_features.csv` (one row per token).

### 1.1 Structural identifiers

- **Features:** `doc_id, paragraph_id, sentence_id, sentence, start, end, term_id, token_id`
- **Library:** (none — passthrough from §0.1)
- **Inputs:** §0.1 spaCy parse
- **Outputs:** column passthrough on `word_df`
- **Per-column toggles:** `feat_struct_doc_id, feat_struct_paragraph_id, feat_struct_sentence_id, feat_struct_sentence_text, feat_struct_char_start, feat_struct_char_end, feat_struct_term_id, feat_struct_token_id`
- **Notes:** These don't trigger any extractor — they're just whether each column survives the CSV-write drop step.

### 1.2 Lexical-linguistic annotation

- **Features:** `token, lemma, upos, xpos, feats, dep_rel, head_token_id`
- **Library:** (none — passthrough from §0.1)
- **Inputs:** §0.1 spaCy parse
- **Per-column toggles:** `lexling_tokens, lexling_pos, lexling_morph, lexling_dep`
- **Notes:** Same provenance as §1.1; split only because the GUI presents the two as separate cards.

### 1.3 SUBTLEX-style frequency — `extractors/word_freq.py`

- **Features:** `Zipf_Freq`
- **Library:** `wordfreq`
- **Inputs:** §0.1 spaCy parse (consumes `token` / `lemma` per row)
- **Per-column toggle:** `freq_subtlex`

### 1.4 Educator Word Frequency Guide — `extractors/educator_freq.py`

- **Features:** `Edu_Zipf_K2, Edu_Zipf_3to5, Edu_Zipf_6to8, Edu_Zipf_9to12, Edu_Zipf_Overall`
- **Library:** `pyarrow` (parquet I/O)
- **External asset:** `dictionaries/edu_zipf.parquet` (not redistributable; warn at preflight if missing)
- **Inputs:** §0.1 spaCy parse
- **Per-column toggles:** `freq_ewfg_k2, freq_ewfg_3to5, freq_ewfg_6to8, freq_ewfg_9to12, freq_ewfg_overall` (one per band; cluster fires if any are on)

### 1.5 GPT-2 surprisal — `extractors/surprisal.py`

- **Features:** `Surprisal`
- **Library:** `transformers` + `torch` (GPT-2 small)
- **External asset:** GPT-2 weights in `hf_cache/` (~500 MB; bundled with project)
- **Inputs:** §0.1 spaCy parse (token text); sliding window for transcripts >1024 subwords
- **Per-column toggle:** `pred_surprisal_gpt2`

### 1.6 LLM-derived predictability — `extractors/llm_cloze.py`

- **Features:** `LLM_Surprisal, LLM_Cloze, LLM_TopMatch`
- **Library:** `openai` (default: `gpt-4o-mini`, requires API key) OR local `transformers` model
- **Inputs:** §0.1 spaCy parse + selected LLM model id
- **Per-column toggles:** `pred_surprisal_llm, pred_cloze, pred_topmatch` (cluster fires if any on; OpenAI path also needs `OPENAI_API_KEY` in session env)
- **Notes:** Cost is non-trivial; the Run page shows a tiktoken-based estimate when an OpenAI model is selected.

### 1.7 Semantic-domain indicators — `extractors/semantic_domain.py`

- **Features:** `Sem_<Backend>_<Category>` columns (one per selected category) + a `Sem_Categories` summary column
- **Library:** `empath` (default, free) OR `liwc` (requires licensed `.dic`)
- **External asset:** `dictionaries/*.dic` if LIWC backend selected
- **Inputs:** §0.1 spaCy parse + user-selected categories (level-specific: `word_categories_<backend>` from prefs)
- **Per-column toggle:** `sem_indicators` (single master); the *categories* list selects which `Sem_*` columns get emitted.

### 1.8 Word acoustic alignment

- **Features:** `Onset, End, Transcript`
- **Library:** (none — passthrough from §0.2)
- **Inputs:** §0.2 MFA word tier merged into §0.1 spaCy df via `main.py:_merge_alignment`
- **Per-column toggles:** `feat_acoustic_onset, feat_acoustic_end`

### 1.9 Derived timing — `extractors/timing.py`

- **Features:** `Word_Duration, Word_Gap`
- **Library:** pure pandas (no external lib)
- **Inputs:** §1.8 (`Onset/End` on word df)
- **Per-column toggles:** `feat_acoustic_word_duration, feat_acoustic_word_gap`

### 1.10 Word prosody — `extractors/prosody.py`

- **Features:** `Pitch_Mean, Pitch_Min, Pitch_Max, Pitch_Std, Pitch_Delta, Volume_Mean, Volume_Min, Volume_Max, Volume_Std, Volume_Delta`
- **Library:** numpy (consumes pre-computed Contours from §0.3)
- **Inputs:** §0.1 spaCy parse (for `upos` to skip PUNCT) + §0.2 MFA word tier (`Onset/End`) + §0.3 Contours; pitch units (Hz vs semitones) from prefs
- **Per-column toggles:** `feat_prosody_pitch_*, feat_prosody_volume_*` (10 keys total; cluster fires if any are on)

---

## 2. Sub-word level

Emits `<stim>_syllables.csv` and `<stim>_phonemes.csv`. **Always emitted as sidecars** — they live alongside the word CSV, joined back via `doc_id/sentence_id/token_id`.

### 2.1 Phoneme intervals — `extractors/syllabify.py:syllabify_pairs` (phoneme half)

- **Features (rows in `<stim>_phonemes.csv`):** `doc_id, sentence_id, token_id, token, syllable_id, phoneme_id, Label, Onset, End, Duration`
- **Library:** pure pandas (consumes MFA phones tier directly)
- **Inputs:** §0.1 spaCy parse (for `doc_id/sentence_id/token_id/token` FK columns) + §0.2 MFA phones_df + §0.2 MFA word tier merged into word df (for `Onset/End` interval bounds)
- **Per-column toggles:** `feat_phoneme_label, feat_phoneme_onset, feat_phoneme_end, feat_phoneme_duration`
- **Master sidecar toggle:** `output.phonemes_csv`
- **Notes:** Label is ARPABET with stress digit. Phonemes whose `[onset, offset]` falls fully within a word's `[Onset, End]` are assigned to that word; phones outside any word are silently dropped.

### 2.2 Syllabification + orthographic labels — `extractors/syllabify.py:syllabify_pairs` (syllable half)

- **Features (rows in `<stim>_syllables.csv`):** `doc_id, sentence_id, token_id, token, syllable_id, Label, ARPABET, Onset, End, Duration`
- **Library:** `pyphen` (optional, for orthographic syllable labels in `Label` column); falls back to ARPABET-without-stress if pyphen is missing
- **Inputs:** §0.1 spaCy parse + §0.2 MFA word + phones tiers + per-word phone grouping (Maximum Onset Principle implemented in `syllabify_word`)
- **Per-column toggles:** `feat_syllable_onset, feat_syllable_end, feat_syllable_duration, feat_syllable_label, feat_syllable_arpabet`
- **Master sidecar toggle:** `output.syllables_csv`

### 2.3 Syllable prosody — `extractors/syllable_prosody.py`

- **Features:** `Pitch_Mean, Volume_Mean, Pitch_Delta, Volume_Delta` columns added to syllables_df
- **Library:** numpy (consumes Contours from §0.3, slices by syllable `[Onset, End]`)
- **Inputs:** §2.2 syllables_df + §0.3 Contours + pitch units from prefs
- **Per-column toggles:** `feat_syllable_pitch_mean, feat_syllable_volume_mean, feat_syllable_pitch_delta, feat_syllable_volume_delta`

### 2.4 Syllable P-center — `extractors/pcenter.py`

- **Features:** `P_Center` (seconds from audio start) — one column added to `<stim>_syllables.csv`
- **Library:** pure pandas/numpy (no external model)
- **Inputs:** §2.2 syllables_df + §2.1 phonemes_df (for per-phoneme onset/duration + nucleus detection)
- **Formula:** `P_Center = nucleus_onset − k × onset_consonant_duration`
  - `nucleus_onset` = onset time of the first ARPABET vowel label (ending in `0`/`1`/`2`)
  - `onset_consonant_duration` = sum of durations of phonemes before the nucleus
  - `k` = Scott 1993 coefficient, default 0.3, stored at `prefs["prosody"]["pcenter_k"]`
- **Edge cases:** vowel-initial → `onset_consonant_duration = 0`; no vowel nucleus → fallback to `syllable.Onset`; missing MFA alignment → `NaN`
- **Per-column toggle:** `feat_syllable_pcenter`
- **Master toggle:** `master_syllable_pcenter`
- **Reference:** Marcus (1981), Scott (1993)

---

## 3. Sentence level

**Not currently emitted as a dedicated CSV.** Sentence identifiers (`sentence_id, sentence`) appear as columns on the word CSV. Reserved here as a placeholder — if/when we add sentence-level aggregates (e.g. mean surprisal per sentence, sentence-level prosody), they'd consume §0.1 + the word-level features that already ran, and would emit `<stim>_sentences.csv` as a third sidecar.

Rough sketch for future planning:
- Inputs: word-level features (any cluster the user enabled) grouped by `sentence_id`
- Output: one row per sentence with mean / std / count aggregates of the word features
- Dependency: identical to whichever word-level clusters get aggregated

---

## 4. Document level

Emits a single master `documents_summary_<timestamp>.csv` with one row per stimulus across the whole batch.

### 4.1 Token / structural counts — `utils/doc_summary.py`

- **Features:** `n_tokens, n_content_tokens, n_sentences, n_paragraphs`
- **Library:** pure pandas
- **Inputs:** §0.1 spaCy parse (uses `upos` to identify content tokens)
- **Per-column toggles:** `out_count_tokens, out_count_content_tokens, out_count_sentences, out_count_paragraphs`

### 4.2 Word-feature aggregates — `utils/doc_summary.py`

- **Features:** `mean_zipf, mean_edu_zipf_*, mean_surprisal, mean_llm_surprisal, mean_llm_cloze, mean_llm_topmatch`
- **Library:** pure pandas
- **Inputs:** whichever word-level clusters from §1 ran — this is a **derived** cluster; it only emits a `mean_X` if `X` is present on the word df
- **Master toggle:** `doc_aggregates`
- **Notes:** Each mean has an implicit dependency on the matching word-level cluster. Per the bug history, the dispatcher must NOT request `mean_surprisal` while leaving `feat_pred_surprisal_gpt2` off — main.py will then force the extractor back on and silently re-pay its cost.

### 4.3 Lexical diversity — `extractors/lexical_diversity.py`

- **Features:** `ttr, mtld`
- **Library:** pure-python (own implementation)
- **Inputs:** §0.1 spaCy parse (consumes `token` per row)
- **Master toggle:** `doc_lex_diversity`

### 4.4 Readability — `extractors/readability.py`

- **Features:** `flesch_reading_ease, fk_grade`
- **Library:** `textstat`
- **Inputs:** raw transcript text (does its own tokenization internally)
- **Master toggle:** `doc_readability`

### 4.5 LIWC composite — `extractors/liwc_summary.py`

- **Features:** `psych_authentic, psych_analytic, psych_tone, psych_clout`
- **Library:** `liwc` (requires licensed `.dic`)
- **External asset:** `dictionaries/*.dic`
- **Inputs:** raw transcript text + LIWC dictionary
- **Per-column toggles:** `feat_psych_authentic, feat_psych_analytic, feat_psych_tone, feat_psych_clout` (cluster fires if any on; per-column drop happens after compute)

### 4.6 Category proportions — `utils/doc_summary.py`

- **Features:** `count_<cat>, prop_<cat>` (one pair per selected category)
- **Library:** pure pandas
- **Inputs:** §1.7 semantic indicators must have run on the word df (`Sem_*` columns + selected `doc_categories_<backend>` from prefs)
- **Master toggle:** `doc_category_proportions`

### 4.7 Document prosody — `utils/doc_summary.py` (`compute_doc_prosody`)

- **Features:** `doc_pitch_mean, doc_pitch_min, doc_pitch_max, doc_pitch_std, doc_volume_mean, doc_volume_min, doc_volume_max, doc_volume_std, doc_pitch_delta_mean, doc_volume_delta_mean, doc_hnr, doc_pause_count, doc_pause_mean_duration, doc_pause_total_duration, doc_pause_pct_silence`
- **Library:** numpy + pandas (consumes Contours)
- **Inputs:** §0.2 MFA word tier + §0.3 Contours
- **Per-column toggles:** `feat_doc_prosody_pitch_*, feat_doc_prosody_volume_*, feat_doc_prosody_pause_*, feat_doc_prosody_hnr` (14 keys, all in `feat_doc_prosody_*` namespace; cluster fires if any are on)

### 4.8 Articulation rate — `utils/doc_summary.py`

- **Features:** `doc_articulation_rate` (syllables per second of spoken time)
- **Library:** `pyphen` (counts syllables per word orthographically) + numpy
- **Inputs:** §0.1 spaCy parse + §0.2 MFA word tier (for spoken-time interval)
- **Per-column toggle:** `feat_doc_prosody_articulation_rate`
- **Notes:** Split from §4.7 because it's the only doc-prosody column that needs `pyphen`; preflight warns separately when pyphen is missing.

---

## 5. Plumbing utilities (not features themselves)

| Module | Library | Role |
|---|---|---|
| `utils/transcript_prep.py` | `num2words` | Spells out digits, normalizes whitespace, for MFA's text input. |
| `utils/textgrid_parser.py` | `tgt` | Parses MFA's TextGrid into word + phone DataFrames. |
| `utils/file_pairing.py` | stdlib | Matches `.txt` ↔ `.wav` by filename stem. |
| `utils/run_logger.py` | stdlib + `psutil` | Per-run manifest JSON, no-match CSV, log tee. |
| `utils/system_info.py` | `psutil` | OS / RAM / GPU info into the manifest. |

---

## 6. Implications for the planner refactor

Reading this map top-down, three structural facts pop out:

1. **Three shared stages dominate compute cost.** spaCy parse (~1-3 s), MFA (~10-60 s), Parselmouth Contours (~1-2 s). The planner's job is to run each *at most once per stimulus* and ensure every consumer that wants its output gets it. Right now `gui/run_dispatch.py` handles this via cascading `if needs_X: enable_Y` rules — easy to miss a case (which is exactly how we got both the sub-word-parse-forcing and the syllable-prosody-contours-forcing bugs).

2. **Sub-word and doc levels both consume MFA + Contours.** Today MFA emission lives inside `main.run()`'s "word pipeline" function — but its output is consumed by sub-word sidecars (§2) and doc prosody (§4.7), neither of which is conceptually a "word feature". The bug we just shipped (`df03cfa`) was a direct consequence: sidecars lived inside the word batch and a per-pair drop step erased columns the sub-word stage needed. A cleaner model is for **MFA to be a stage** (§0.2) that any level can request, not a step buried in the word pipeline.

3. **Per-column toggles vs cluster gates are conflated.** Look at §1.8 (acoustic alignment) — turning off `feat_acoustic_onset` is meant to say "don't include `Onset` in the *output CSV*", not "don't run MFA". But the current `_word_columns_to_drop` reads those same toggles to strip columns from the in-memory df, which broke the sub-word path. The planner should distinguish:
   - **Stage gates** (does extractor X run?) — derived from "is any consumer of X enabled?"
   - **Column gates** (is column Y written to CSV?) — applied only to the to-be-written df copy, never the in-memory one

A v1 planner could be a single function:

```python
def plan(user_selection: UserSelection) -> Plan:
    """user_selection: which CSVs + which columns the user wants.
       Plan: list of stages to run + their config, and post-stage column-drop sets."""
```

…that emits the same `no_*` flags `main.py` currently consumes. No structural change to `main.py` required for the first cut — just one new module that *generates* the dispatcher's argparse Namespace declaratively from a feature-stage graph derived from this map.

---

## 7. Open questions for the planner design

Things this map doesn't yet answer, worth nailing down before the brainstorm:

- **Should "stages" be a class hierarchy or a registry of pure functions?** The `Contours` dataclass already implies a shared object model — a stage produces one canonical output object that consumers slice.
- **Where does the `Sem_Categories` summary column live?** §1.7 emits both per-category columns and a summary string; if the user picks per-column toggles for each category, is the summary always on or also toggleable?
- **Are sentence-level aggregates worth modeling now even though no consumer exists?** §3 is a placeholder; if the planner is built without a sentence tier, adding it later is a small extension. If it's built with one, we pay for an empty rung today.
- **`Pitch_Delta` etc. — should they be a separate stage from `Pitch_Mean`, or one bundle?** Both consume Contours but `Delta` requires the per-word mean to be computed first. Currently bundled.

---

*Source of truth: the extractor source files in `extractors/` and `utils/doc_summary.py`. When code disagrees with this doc, the code is right and this doc needs updating.*
