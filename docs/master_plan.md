# SensiExtractor — Master Plan

**Last updated:** 2026-06-13

This document is the single entry point for project state. Read this first; deeper specs / plans / batch designs are linked from here.

---

## 1. Mission

SensiExtractor is the standardized SOP tool for the SENSI naturalistic-EEG pipeline (PI: Blair Kaneshiro). It takes audio (.wav) + transcript (.txt) and emits a publication-ready word-level feature CSV plus a document-level summary CSV. Lives inside the larger `../Data Curation/` effort.

## 2. Current state — what we're working on RIGHT NOW

**Acoustic level just shipped (2026-06-13)** — awaiting manual validation on a real stimulus (see table below). Older feature batches all await one consolidated manual-validation pass.

> ⚠️ **Bookkeeping gap:** prefs are now at **v13** but this plan only documented through v11. The v11→v12 **lexical-norms** batch (AoA/concreteness/MRC/VAD word norms) shipped to the codebase but its §3 row was never written, and follow-on data/doc work for it is still uncommitted in the working tree. Add the v12 row when that batch is finalized.

### Awaiting user review (just shipped)

**In-app feature documentation system (2026-05-24)** — first scaling pass complete; **408 passed, 2 skipped**.

- 41 per-feature `.md` files under `docs/features/` covering every GUI toggle across all 4 feature pages (~100 toggles, with shared-doc mappings collapsing related toggles like `pitch_*` to one file).
- Clickable ⓘ info-icon next to each toggle opens a Qt modal rendering the per-feature `.md` body via `QTextBrowser`.
- New modules: `gui/feature_docs.py` (loader + `_SHARED_DOC_MAP`), `gui/info_popup.py` (`InfoIcon` + `FeatureInfoDialog`). New asset: `assets/info.svg`. `ToggleRow` now auto-renders an icon when a matching `.md` exists.
- Feedback memories saved: [`feedback-feature-docs-authorship`](../memory/feedback-feature-docs-authorship.md), [`feedback-feature-docs-citations`](../memory/feedback-feature-docs-citations.md).
- **Open for user**: skim the docs (especially the `## Implementation in SensiExtractor` sections of the most-touched extractors) and grep `docs/features/` for `⚠️` to find every place I extrapolated rather than read directly from the code/spec — those are the lines that benefit most from a second pair of eyes. A separate citation-audit pass on **2026-05-27** swept the `## Papers` sections for wrong DOIs, wrong years, and fabricated references — see `docs/feature-docs-audit.md` for the diff and the standing checklist for future review.

### Awaiting end-to-end manual validation on a real ISC-Wu Tsai stimulus

The feature batches below all ship implemented + tests-green; what's outstanding is the same thing: someone runs `python gui.py` on a real stimulus and eyeballs the outputs. They're rolled up here so §2 doesn't repeat the same status for every batch.

| Batch | Date | What to look at |
|---|---|---|
| Acoustic level (audio-only TRF features) | 2026-06-13 | Run `python gui.py` (or `python main.py --level acoustic --audio-folder … --output-folder …`) on a real stimulus; confirm `acoustic/<stim>_acoustic.mat` loads in MATLAB/scipy with `envelope`, `onset_envelope`, `logmel`, `mel_center_freqs` (and `mfcc` if enabled) all sharing the same length `N = round(duration × analysis_rate)` and aligned to the `t`/`fs` vectors. Sanity-check the envelope tracks syllabic rhythm and `analysis_rate` matches the intended EEG rate. |
| Embeddings (Word2Vec + BERT) | 2026-05-22 | Download Google News Word2Vec model; verify `word/<stim>_word2vec.mat` and `word/<stim>_bert_word.mat` and `sentence/<stim>_bert_sentence.mat` shapes and content. |
| Sentence-level tier | 2026-05-22 | `sentence/<stim>_sentences.csv` appears with correct timing and HNR. |
| Output reorganization | 2026-05-22 | Output appears in `word/` / `syllable/` / `phoneme/` / `document/` subfolders. |
| P-center | 2026-05-20 | `P_Center` column appears in `syllable/<stim>_syllables.csv` with reasonable values. |
| Phoneme / Syllable batch | 2026-05-13 | Both sidecars appear with sane shapes; ≥ 1 syllable per word; no overlapping intervals; ARPABET phoneme labels present. |
| Acoustic prosody | 2026-05-12 | Per-column expected ranges below. |

**Prosody expected-range cheatsheet** (keep handy when validating):

| Column | Expected range | If way off, suspect… |
|---|---|---|
| `Pitch_Mean` (word CSV) | 60–300 Hz, or −12 to +12 semitones | Pitch floor/ceiling wrong for speaker. |
| `Volume_Mean` (word CSV) | 50–75 dB SPL (clear studio) | Audio level off; Praat assumes normalized input. |
| `Pitch_Delta` (word CSV) | Mix of +/-, NaN on first word | Always 0 → identical values; always NaN past first → bug. |
| `doc_hnr` (master CSV) | 10–25 dB for clear speech | Wildly negative → unvoiced-sentinel fix `460801b` didn't take. |
| `doc_articulation_rate` (master CSV) | 3–6 syll/sec | NaN → `pyphen` missing; 0 → no aligned words. |
| `doc_pause_count` (master CSV) | Reasonable integer | 0 on stimulus with obvious pauses → check `--pause-threshold` (default 0.25 s). |

Specs for each batch are linked from §3 (implementation plans were deleted as part of the 2026-05-24 docs curation — code + spec are now the canonical record).

## 3. Completed batches (newest first)

| Date | Batch | Summary |
|---|---|---|
| 2026-06-13 | **Self-describing embedding sidecars** | The three embedding `.mat` sidecars (`word/<stim>_word2vec.mat`, `word/<stim>_bert_word.mat`, `sentence/<stim>_bert_sentence.mat`) now carry their own labels instead of being bare matrices: each file is written as **flat top-level variables** — `embeddings` (unchanged shape, still top-level so existing consumers keep working) plus `tokens`/`word_id`/`sentence_id`/`is_valid` for the word level and `sentence_id`/`sentence`/`is_valid` for the sentence level. `tokens`/`sentence` are `dtype=object` so MATLAB sees a proper **cellstr** (unpadded), not a padded char matrix. **Unified punctuation handling:** `is_valid = (upos != PUNCT) AND embedded`; `embed_word2vec` now forces PUNCT → zero (it previously could emit a stray Google News vector for `,`/`.`), so a PUNCT row is a zero vector + `is_valid=False` in *both* word-level models. Extractor signatures: `embed_word2vec → (matrix, is_valid)`; `embed_bert_words_and_sentences → (word_emb, word_valid, sent_emb, sent_valid)`. Two helpers in `main.py` (`_word_embedding_vars` / `_sentence_embedding_vars`, with row/group-alignment guards) build the savemat dicts from the parsed `df`. No new files/prefs/planner/GUI — serialization change only. TDD on the extractor masks; savemat→loadmat roundtrip integration test. Three feature docs updated with `.mat` field tables. Spec: `docs/superpowers/specs/2026-06-13-embedding-sidecar-structs-design.md`; plan: `docs/superpowers/plans/2026-06-13-embedding-sidecar-structs.md`. |
| 2026-06-13 | **Acoustic level (audio-only TRF features)** | New audio-only `acoustic` level emitting one combined `acoustic/<stim>_acoustic.mat` per stimulus with four continuous acoustic time series on a single shared analysis grid (`N = round(duration × analysis_rate)`, with `t`/`fs` vectors) for TRF / cortical-tracking analyses: **broadband amplitude envelope** (default `rms` method faithfully replicating `mTRFenvelope.m` — square → boxcar-decimate → sqrt → power-law compress with `comp = log₁₀(2) ≈ 0.3`; `hilbert` analytic-signal method also available), **onset envelope** (half-wave-rectified envelope derivative), **log-mel spectrogram** (`n_mels` × time; speech/music/general band presets), and **MFCC** (off by default — log-mel preferred for TRF). New `extractors/acoustic_ts.py` (librosa + scipy, pure functions, TDD). `analysis_rate` configurable to match the user's EEG. Executes through its own audio-only batch runner `run_acoustic_level_batch` (main.py) that globs the audio folder directly (no transcript needed), registered in the planner registry (stage `acoustic_ts` + cluster; `acoustic_ts_mat` → `acoustic` in `_OUTPUT_CSV_MAP`). `plan.output_csvs` is now derived from active clusters (`live_csvs`), so an output key left on with its master off no longer writes empty files — a spec-aligned fix to shared planner behavior. Prefs **v12→v13** additive (`acoustic_ts` config section + 4 `feat_acoustic_ts_*` + `master_acoustic_ts` (opt-in, default off) + `acoustic_ts_mat`). New GUI "Acoustic" page + CLI `--level acoustic`. Four feature docs with live-verified citations (Crosse 2016, Stevens 1955 & 1937, Lalor & Foxe 2010, Ding & Simon 2012, Di Liberto 2015, Oganian & Chang 2019, Davis & Mermelstein 1980). Suite **481 passed, 2 skipped**. Spec: `docs/superpowers/specs/2026-06-13-acoustic-level-trf-features-design.md`; plan: `docs/superpowers/plans/2026-06-13-acoustic-level-trf-features.md`. Note: this batch's docs were authored in-session and are *not* part of the feature-doc author-review backlog. |
| 2026-06-10 | **LIWC removal + license-clean composite psych** | The LIWC-22 EULA (forbids using LIWC dictionary material outside LIWC's software) means LIWC can no longer ship as a backend. **(1)** Word-level semantic-indicator LIWC backend disabled: the GUI radio is greyed out with an EULA tooltip + note (Empath is the default), and `detect_backend()` always returns Empath — a license holder can still opt in via CLI `--sem-dict your.dic`. **(2)** Doc-level composite psych scores (Authentic/Analytic/Tone/Clout) — which were LIWC-`.dic`-based — **reimplemented license-clean** in new `extractors/psych_composite.py`: the Pennebaker (2014) formulas now read grammatical categories from the spaCy-parsed token DataFrame (POS/morphology) and Tone from VADER (`vaderSentiment`, lazy-loaded). Old `extractors/liwc_summary.py` retained but unwired. Planner stage `doc_liwc`→`doc_psych`, CLI flags `--no-doc-liwc-*`→`--no-doc-psych-*`, psych card no longer backend-gated. Prefs **v10→v11**: default `semantic.backend` → `empath`, migration coerces stored `liwc`→`empath`. Docs `doc_psych_composite.md` + `sem_indicators.md` rewritten with the EULA note and licensing pointer (<https://liwcsoftware.onfastspring.com/>). Spec: `docs/superpowers/specs/2026-06-10-liwc-removal-psych-reimpl-design.md`. 9 new extractor tests + v11 migration tests; suite green (441 passed, 2 skipped). |
| 2026-05-24 | **In-app feature documentation system** | 41 per-feature `.md` files under `docs/features/` covering ~100 GUI toggles across all 4 feature pages. Single source of truth: YAML frontmatter (key, levels, libraries, status, depth) + Markdown body (meaning, EEG rationale, math, **Implementation in SensiExtractor** sourced from the extractor + design spec with ⚠️-flagged interpolation, library snippet, citation-anchored Papers section, Datasets placeholder for the future per-dataset cross-ref). Clickable ⓘ info icon next to every documented toggle opens a Qt modal that renders the `.md` via `QTextBrowser`. New modules: `gui/feature_docs.py` (`has_feature_doc` / `load_feature_doc` + `_SHARED_DOC_MAP` routing related toggles to one file — e.g. all five `pitch_*` toggles → `prosody_pitch.md`), `gui/info_popup.py` (`InfoIcon` + `FeatureInfoDialog`). New asset `assets/info.svg`. `ToggleRow` now auto-renders the icon when a matching `.md` exists, so adding a new doc requires no GUI wiring. 14 new tests (`tests/gui/test_feature_docs.py`). Feedback memories: [`feedback-feature-docs-authorship`](../memory/feedback-feature-docs-authorship.md), [`feedback-feature-docs-citations`](../memory/feedback-feature-docs-citations.md). No spec/plan files — built iteratively in conversation. |
| 2026-05-22 | **Embeddings (Word2Vec + BERT)** | Three new `.mat` sidecars per stimulus: `word/<stim>_word2vec.mat` (n_words × 300 float32), `word/<stim>_bert_word.mat` (n_words × 12 × 768 float32), `sentence/<stim>_bert_sentence.mat` (n_sentences × 12 × 768 float32). New `extractors/word2vec_embed.py` (gensim, Google News 300d, OOV→zero, lowercase fallback) and `extractors/bert_embed.py` (bert-base-uncased, all 12 layers, 510-token sliding window, PUNCT excluded from input). Prefs v9→v10 additive (3 feature masters + 3 output keys); planner `word2vec_embed` + `bert_embed` stages + 3 master-toggle-only clusters; main.py writes .mat via scipy.io.savemat; GUI toggles in "Semantic similarity" cards on the Word-level page (Word2Vec + BERT word, below Semantic indicators) and the Sentence-features page (BERT sentence). Spec: `docs/superpowers/specs/2026-05-22-embeddings-design.md`. |
| 2026-05-22 | **Sentence-level tier** | New `sentence/<stim>_sentences.csv` sidecar per stimulus — sentence timing (min/max Onset/End over words) + 11 prosody columns (Pitch_{Mean,Min,Max,Std,Delta}, Volume_{Mean,Min,Max,Std,Delta}, HNR). New `extractors/sentence_prosody.py`; prefs v8→v9 additive (14 `feat_sentence_*` + 2 master toggles + `sentences_csv`); planner `sentence_prosody` stage + 2 clusters; CLI `--no-sentences`; GUI "Sentence features" page. Spec: `docs/superpowers/specs/2026-05-22-sentence-level-tier-design.md`. |
| 2026-05-22 | **Output reorganization** | Per-level subfolders: `word/`, `syllable/`, `phoneme/`, `document/` under the output root. Lazy mkdir (folder created only when output written). `session_units` CSVs moved to `logs/`. `_level_dir` helper in `main.py`; 6 path sites updated; 7 new integration tests. **Breaking change** to all output paths. Spec: `docs/superpowers/specs/2026-05-22-output-reorganization-design.md`. |
| 2026-05-20 | **P-center** | `P_Center` column added to `<stim>_syllables.csv`. Formula: `nucleus_onset − k × onset_consonant_duration` (Scott 1993, k=0.3 default). New `extractors/pcenter.py`; planner `pcenter` stage + `p-center` cluster; prefs v7 → v8 (adds `feat_syllable_pcenter`, `master_syllable_pcenter`, `prosody.pcenter_k`); P-center card in Sub-word features GUI page; CLI `--no-syl-pcenter` / `--pcenter-k`. Spec: `docs/superpowers/specs/2026-05-20-pcenter-design.md`. |
| 2026-05-19 | **GUI validation unification + cluster UX** | The 2026-05-14 planner became the single source of truth for which stages run; this batch makes it the single source for everything that *consumes* that truth: preflight, the Run-tab cost panel, the cluster master toggles, and unit metadata. Two new declarative fields on `FeatureCluster` (`master_toggle`, `available_when`) enable per-cluster master toggles in the GUI (grayed-out children when off) and the LIWC-only psych composite gating. `gui/preflight.py` was rewritten to consume `plan.stages_to_run`; the cost panel hides when no LLM features are active; the Syllables card split into Intervals + Labels + Prosody sub-clusters. Semantic categories unified into `semantic.categories_<backend>` (word-level page is canonical editor; doc-level shows count + hint). Per-run `<output>/<run_ts>_<level>_session_units.csv` records column → unit. Prefs v6 → v7 additive migration. Spec: `docs/superpowers/specs/2026-05-19-gui-unification-design.md`. |
| 2026-05-15 | **Planner module** | Replaced `gui/run_dispatch.py:build_args_word_level / build_args_doc_level` (~200 lines of hand-rolled cascading rules) with a declarative `gui/planner/` package: `Stage` + `FeatureCluster` dataclasses, a `STAGES` + `CLUSTERS` registry derived from `docs/feature_dependencies.md`, a pure `plan_from_prefs` function (DAG closure + topo sort), and a `to_namespace` adapter that emits the same `argparse.Namespace` `main.py` already consumes. No changes to `main.py` or any extractor. Adding a new feature is now a single edit in `gui/planner/registry.py`. A `_word_only_stages` filter in `emit.py` prevents doc-level stage requirements from bleeding into the word-level namespace; `doc_prosody → no_prosody=False` forces word-prosody computation when doc-prosody needs the deltas. Schema-drift guard test prevents the registry from falling out of sync with prefs. 313 tests green (37 new in `tests/planner/`). Spec: `docs/superpowers/specs/2026-05-14-planner-design.md`. |
| 2026-05-13 | **Phoneme / Syllable** | Two new sidecar CSVs per stimulus (`<stim>_syllables.csv`, `<stim>_phonemes.csv`) joined back to the word CSV via FK columns. Maximum Onset Principle syllabification on MFA's phones tier; syllable prosody (mean pitch/volume + deltas) reuses the existing Parselmouth `Contours` object. New "Sub-word features" GUI page with Syllables + Phonemes cards, Run-page master switches, preflight warnings for pyphen-missing and alignment-off. Prefs v5 → v6 additive (9 syllable + 4 phoneme keys + 2 sidecar output keys). Spec: `docs/superpowers/specs/2026-05-13-phoneme-syllable-design.md`. |
| 2026-05-12 | **GUI bug-fix sweep** | Nine bugs surfaced by real-stimulus testing: doc-level forced LLM/OpenAI calls regardless of toggles; CategoryPicker wrote to `document_categories_*` instead of `doc_categories_*`; `main.run()` read the obsolete v3 categories key; `no_parse` was only gated on structural toggles so sem-only word runs produced nothing; doc dispatcher hardcoded freq/edu_freq on; `sys.exit` on empty pipeline froze the GUI; Logs panel didn't tail per-step progress (`RunLogger` now tees stdout/stderr); EWFG warning leaked an internal doc path. Regression tests added in `tests/gui/test_run_dispatch.py` and `tests/gui/test_category_picker.py`. |
| 2026-05-12 | **Logs tidy-up** | Log artifacts in `<output>/logs/`; no-match.csv skipped when empty; live-tail in GUI Logs tab. Commit `98cc365`. |
| 2026-05-12 | **Acoustic prosody** | 10 word-level + 16 doc-level prosody columns via Praat-Parselmouth. Pitch units (Hz/semitones) and pause threshold configurable. Prefs v4→v5. Spec: `docs/superpowers/specs/2026-05-12-prosody-design.md`. |
| 2026-05-11 | **Batch 3 (GUI)** | Word vs document on separate sidebar pages, modal category dialogs per level, per-column toggles (structural × 8, acoustic × 4, psych × 4, doc counts × 4), prefs v3→v4, SENSI cyan theme. Spec: `superpowers/specs/2026-05-11-batch-3-design.md`. |
| 2026-05-11 | **PySide6 GUI redesign** | Switched from tkinter to PySide6. Sidebar+content architecture, continuous inline validation, in-session API-key paste. Spec: `superpowers/specs/2026-05-11-gui-redesign-design.md`. |
| 2026-05-10 | **Batch 1.5 (doc-level features)** | TTR, MTLD, Flesch reading ease, FK grade, LIWC summary scores (Authentic/Analytic/Tone/Clout); `documents_summary_<ts>.csv` master output; CLI `--level word\|document`. Spec: `superpowers/specs/2026-05-10-batch-1.5-tabbed-gui-design.md`. |
| 2026-05-10 | **Batch 1 (feature additions)** | EWFG grade-band frequencies (`Edu_Zipf_*`), semantic-domain indicators (LIWC/Empath), LLM-cloze (`LLM_*`), derived timing (`Word_Duration`, `Word_Gap`), folder-pair batch mode + run-log artifacts + analyzer field, GUI overhaul. Spec: `superpowers/specs/2026-05-10-feature-batch-1-design.md`. |
| Pre-history | **Original pipeline** | spaCy annotation, SUBTLEX Zipf via `wordfreq`, GPT-2 surprisal (sliding window for >1024 tokens), MFA forced alignment (replacing GENTLE). |

## 4. Roadmap — future batches

Ordered roughly by priority, not committed. Each item gets its own brainstorm → spec → plan → implement cycle.

### Documentation tracks (next focus)

| Item | Scope | Status | Notes |
|---|---|---|---|
| **Feature-doc author-review pass** | Walk through `docs/features/*.md` and resolve every ⚠️ marker (interpolated *why*). Citation pass landed 2026-05-27 (`docs/feature-docs-audit.md`); ⚠️ interpolation review still outstanding. | In progress | Self-contained — no code changes, just doc edits. Highest priority before external publication. |
| **External-doc deliverable (track #2)** | Convert the 41 per-feature `.md` files into a single browsable artifact for researchers outside the app — combined `docs/features.md` (TOC + grouped by level), or static HTML site via mkdocs / pandoc, or PDF. Same source files; just an additional render path. | Not started | User specifically said "maybe we convert into pdf later" — defer the format choice until external sharing is needed. |
| **Dataset registry (track #3)** | `docs/datasets/<dataset_slug>.md` per SENSI stimulus listing: compatible features, analyzed features (with DOI / paper), not-yet-analyzed features (with comments). Auto-generated `docs/datasets/index.md` matrix. Build script also back-fills the `## Datasets` section of every `docs/features/*.md` (currently a placeholder HTML comment in all 41 files). | Not started | User will provide the dataset list when ready. Markdown only at first — no interactive UI until N grows. |

### Feature roadmap

| Item | Scope | Status | Notes |
|---|---|---|---|
| **P-center** | Perceptual onset of syllables (Scott 1993 / Villing 2010). Syllable level. | **Shipped (2026-05-20)** | See §3 completed batches. |
| **BERT contextual embeddings** | 768-dim vectors per word in a sidecar .mat (all 12 layers). | **Shipped (2026-05-22)** | See §3 completed batches. Includes Word2Vec 300d. |
| **Human-rated cloze norms** | Not an extractor — data acquisition / collaboration with cloze-norming corpora. | Parked | |
| **Document-level enrichment expansion** | Anything beyond what Batch 1.5 / Batch 3 already cover. | Mostly already done | Revisit only if a specific gap is identified. |
| **Correctness scoring** | Either MFA alignment confidence (cheap) or ASR-vs-transcript agreement (medium). | Parked | Needs collaborator clarification on which they want. |
| **Brain Treebank parity** | `Index_in_Sentence`, `Word_Head`. | Already covered | `token_id` and `head_token_id` provide these. |

## 5. Naming conventions (read this to avoid confusion)

We have two parallel "batch" numbering systems that have collided:

- **GUI batches** (1, 1.5, PySide6 redesign, batch 3) — numbered iterations of the GUI's structure.
- **Feature roadmap batches** (originally numbered 1–7 in the batch-1 spec §11) — numbered iterations of the *extractor* feature set.

To prevent further collisions, **new batches are named by their feature** ("prosody batch", "phoneme/syllable batch") rather than a number.

Each batch has a design spec at:
- `docs/superpowers/specs/YYYY-MM-DD-<topic>-design.md`

Step-by-step implementation plans (`docs/superpowers/plans/`) are no longer kept after a batch ships — the code is the canonical execution record. The spec captures intent; the code captures realization.

## 6. Durable conventions

- **Prefs schema:** versioned in `utils/preferences.py`. Each batch may bump the version, **always additively**. Migration chain handles all prior versions → current. Current version: **v13** (v12 = lexical-norms batch, v13 = acoustic level).
- **Per-column drops:** GUI per-column toggles flow through `args.feature_toggles` (and `args.output_toggles` for output-section keys) → `_word_columns_to_drop` / `_doc_columns_to_drop` in `main.py`. CLI flags are group-level only.
- **Tests:** Unit tests for extractors (synthetic fixtures, no real audio where avoidable), integration tests in `tests/integration/`, GUI tests in `tests/gui/` via `pytest-qt`. Target: `python -m pytest` green on every commit.
- **Commits:** one per task during implementation; commit directly to `master` (per user preference). Feature plan referenced in commit message when applicable.
- **Per-feature docs:** every GUI toggle key has (or should have) a `docs/features/<key>.md` file rendered in-app via `gui/info_popup.py`. New batch → new toggle → add the `.md` (the ⓘ icon appears automatically; no GUI wiring required). Shared content across related toggles (e.g. all `pitch_*`) is routed via `gui/feature_docs.py::_SHARED_DOC_MAP`. Authoring rules in [`feedback-feature-docs-authorship`](../memory/feedback-feature-docs-authorship.md) and [`feedback-feature-docs-citations`](../memory/feedback-feature-docs-citations.md).

Environment activation lives in `CLAUDE.md`; durable user preferences live in `memory/MEMORY.md`.
