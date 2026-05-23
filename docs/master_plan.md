# SensiExtractor — Master Plan

**Last updated:** 2026-05-22

This document is the single entry point for project state. Read this first; deeper specs / plans / batch designs are linked from here.

---

## 1. Mission

SensiExtractor is the standardized SOP tool for the SENSI naturalistic-EEG pipeline (PI: Blair Kaneshiro). It takes audio (.wav) + transcript (.txt) and emits a publication-ready word-level feature CSV plus a document-level summary CSV. Lives inside the larger `../Data Curation/` effort.

## 2. Current state — what we're working on RIGHT NOW

**No active batch.** All planned features are implemented. Next step: manual end-to-end validation on a real ISC-Wu Tsai stimulus.

**Just completed:** Embeddings — Word2Vec + BERT (2026-05-22) — **implemented, full test suite green (393 passed, 2 skipped).**

- Spec: [`docs/superpowers/specs/2026-05-22-embeddings-design.md`](superpowers/specs/2026-05-22-embeddings-design.md)
- Plan: [`docs/superpowers/plans/2026-05-22-embeddings.md`](superpowers/plans/2026-05-22-embeddings.md)
- Status: All 9 plan tasks complete; `extractors/word2vec_embed.py` (gensim, Google News 300d, OOV→zero); `extractors/bert_embed.py` (bert-base-uncased, all 12 layers, sliding window, PUNCT excluded from input); `gensim>=4.3` added to requirements.txt; download_models.py updated; prefs v9→v10 additive (3 feature + 3 output keys); planner `word2vec_embed` + `bert_embed` stages + 3 clusters (master-toggle-only, empty toggle_keys); main.py writes `word/<stim>_word2vec.mat`, `word/<stim>_bert_word.mat`, `sentence/<stim>_bert_sentence.mat`; GUI toggles live in a "Semantic similarity" card on the Word-level page (Word2Vec + BERT word) and the Sentence-features page (BERT sentence) — no separate sidebar entry.
- Open: (1) Download Google News Word2Vec model and validate `.mat` output on a real stimulus. (2) Validate BERT `.mat` shapes and content on a real stimulus.

**Also completed:** Sentence-level tier (2026-05-22) — **implemented, full test suite green.**

- Spec: [`docs/superpowers/specs/2026-05-22-sentence-level-tier-design.md`](superpowers/specs/2026-05-22-sentence-level-tier-design.md)
- Plan: [`docs/superpowers/plans/2026-05-22-sentence-level-tier.md`](superpowers/plans/2026-05-22-sentence-level-tier.md)
- Status: All 8 plan tasks complete; new `sentence/<stim>_sentences.csv` sidecar; `extractors/sentence_prosody.py`; prefs v8→v9 additive (16 new keys); planner wired; CLI `--no-sentences`; GUI "Sentence features" page (timing + prosody ClusterCards).
- Open: Validate with a real ISC-Wu Tsai stimulus that `sentence/<stim>_sentences.csv` appears with correct timing and HNR values.

**Also completed:** Output reorganization (2026-05-22) — **implemented, full test suite green.**

- Spec: [`docs/superpowers/specs/2026-05-22-output-reorganization-design.md`](superpowers/specs/2026-05-22-output-reorganization-design.md)
- Plan: [`docs/superpowers/plans/2026-05-22-output-reorganization.md`](superpowers/plans/2026-05-22-output-reorganization.md)
- Status: All 5 plan tasks complete; `_level_dir` helper in `main.py`; 6 path sites updated; 7 new integration tests. Two NumPy 2.0 / pandas.NA compatibility fixes applied (`audio_features.py`, `prosody.py`).
- Open: Validate with a real ISC-Wu Tsai stimulus that output appears in `word/`, `syllable/`, `phoneme/`, `document/` subfolders as expected.

**Also awaiting manual validation:** P-center (2026-05-20) — **implemented, full test suite green.**

- Spec: [`docs/superpowers/specs/2026-05-20-pcenter-design.md`](superpowers/specs/2026-05-20-pcenter-design.md)
- Plan: [`docs/superpowers/plans/2026-05-20-pcenter.md`](superpowers/plans/2026-05-20-pcenter.md)
- Status: All 6 plan tasks complete; `P_Center` column added to `<stim>_syllables.csv` (now in `syllable/` subfolder); new `extractors/pcenter.py`; prefs v7 → v8 additive; P-center card in Sub-word features GUI; CLI `--no-syl-pcenter` / `--pcenter-k`.
- Open: on a real ISC-Wu Tsai stimulus with MFA alignment, confirm `P_Center` column appears in `syllable/<stim>_syllables.csv` with values in a reasonable range.

**Also awaiting manual validation:** Acoustic prosody (2026-05-12) — **implemented, full test suite green.**

- Spec: [`docs/superpowers/specs/2026-05-12-prosody-design.md`](superpowers/specs/2026-05-12-prosody-design.md)
- Plan: [`docs/superpowers/plans/2026-05-12-prosody.md`](superpowers/plans/2026-05-12-prosody.md)
- Status: All 13 plan tasks complete; **231 tests pass** (after the 2026-05-12 GUI bug-fix sweep added regression tests); 1 intentional skip (synthetic-WAV MFA alignment). Requires `pyarrow` for EWFG parquet loading and a working `libsndfile` for soundfile (transitively pulled by transformers) — both added to requirements.txt / repaired via `pip install --force-reinstall --no-deps soundfile`.
- Open: run one real ISC-Wu Tsai stimulus end-to-end (`python gui.py`, both word-level and document-level) and eyeball the prosody columns against these expected ranges:

| Column | Expected range | If way off, suspect… |
|---|---|---|
| `Pitch_Mean` (word CSV) | 60–300 Hz, or −12 to +12 if `--pitch-units semitones` | Pitch floor/ceiling wrong for the speaker (raise for children, lower for bass). |
| `Volume_Mean` (word CSV) | 50–75 dB SPL for clear studio recordings | Audio level too low/high; Praat intensity defaults assume normalized input. |
| `Pitch_Delta` (word CSV) | Mix of positive and negative, NaN on the first word | Always 0 → all values identical; always NaN past first → bug. |
| `doc_hnr` (master CSV) | 10–25 dB for clear speech (positive) | Wildly negative → the unvoiced-sentinel fix in `460801b` didn't take. |
| `doc_articulation_rate` (master CSV) | 3–6 syllables/sec | NaN → `pyphen` import / dict lang wrong; 0 → spoken_time was 0 (no aligned words). |
| `doc_pause_count` (master CSV) | Reasonable integer for the duration | 0 on a stimulus with obvious pauses → check `--pause-threshold` (default 0.25s). |

**Also awaiting manual validation:** Phoneme / Syllable batch (2026-05-13) — **implemented, full test suite green (264 passed, 2 skipped).**

- Spec: [`docs/superpowers/specs/2026-05-13-phoneme-syllable-design.md`](superpowers/specs/2026-05-13-phoneme-syllable-design.md)
- Plan: [`docs/superpowers/plans/2026-05-13-phoneme-syllable.md`](superpowers/plans/2026-05-13-phoneme-syllable.md)
- Status: All 13 plan tasks complete; two new sidecar CSVs (`<stim>_syllables.csv`, `<stim>_phonemes.csv`), new "Sub-word features" GUI page, prefs v5 → v6 additive.
- Open: on the same real-stimulus run used for prosody validation, confirm the two sidecars appear with sane shapes — ≥ 1 syllable per word, no overlapping `[Onset, End]` intervals within a word, syllable `Pitch_Mean`/`Volume_Mean` within the prosody table's expected ranges, ARPABET phoneme labels present (e.g. `AE1`, `T`).

**Recent follow-up batch:** Logs tidy-up (2026-05-12) — log artifacts moved to `<output>/logs/`, no-match.csv skipped when empty + deduped across word+doc batches, live tail of the active log in the Logs tab. No separate spec; committed as `98cc365`.

## 3. Completed batches (newest first)

| Date | Batch | Summary |
|---|---|---|
| 2026-05-22 | **Embeddings (Word2Vec + BERT)** | Three new `.mat` sidecars per stimulus: `word/<stim>_word2vec.mat` (n_words × 300 float32), `word/<stim>_bert_word.mat` (n_words × 12 × 768 float32), `sentence/<stim>_bert_sentence.mat` (n_sentences × 12 × 768 float32). New `extractors/word2vec_embed.py` (gensim, Google News 300d, OOV→zero, lowercase fallback) and `extractors/bert_embed.py` (bert-base-uncased, all 12 layers, 510-token sliding window, PUNCT excluded from input). Prefs v9→v10 additive (3 feature masters + 3 output keys); planner `word2vec_embed` + `bert_embed` stages + 3 master-toggle-only clusters; main.py writes .mat via scipy.io.savemat; GUI toggles in "Semantic similarity" cards on the Word-level page (Word2Vec + BERT word, below Semantic indicators) and the Sentence-features page (BERT sentence). Spec/plan: `docs/superpowers/specs/2026-05-22-embeddings-design.md`, `docs/superpowers/plans/2026-05-22-embeddings.md`. |
| 2026-05-22 | **Sentence-level tier** | New `sentence/<stim>_sentences.csv` sidecar per stimulus — sentence timing (min/max Onset/End over words) + 11 prosody columns (Pitch_{Mean,Min,Max,Std,Delta}, Volume_{Mean,Min,Max,Std,Delta}, HNR). New `extractors/sentence_prosody.py`; prefs v8→v9 additive (14 `feat_sentence_*` + 2 master toggles + `sentences_csv`); planner `sentence_prosody` stage + 2 clusters; CLI `--no-sentences`; GUI "Sentence features" page. Spec/plan: `docs/superpowers/specs/2026-05-22-sentence-level-tier-design.md`, `docs/superpowers/plans/2026-05-22-sentence-level-tier.md`. |
| 2026-05-22 | **Output reorganization** | Per-level subfolders: `word/`, `syllable/`, `phoneme/`, `document/` under the output root. Lazy mkdir (folder created only when output written). `session_units` CSVs moved to `logs/`. `_level_dir` helper in `main.py`; 6 path sites updated; 7 new integration tests. **Breaking change** to all output paths. Spec/plan: `docs/superpowers/specs/2026-05-22-output-reorganization-design.md`, `docs/superpowers/plans/2026-05-22-output-reorganization.md`. |
| 2026-05-20 | **P-center** | `P_Center` column added to `<stim>_syllables.csv`. Formula: `nucleus_onset − k × onset_consonant_duration` (Scott 1993, k=0.3 default). New `extractors/pcenter.py`; planner `pcenter` stage + `p-center` cluster; prefs v7 → v8 (adds `feat_syllable_pcenter`, `master_syllable_pcenter`, `prosody.pcenter_k`); P-center card in Sub-word features GUI page; CLI `--no-syl-pcenter` / `--pcenter-k`. Spec/plan: `docs/superpowers/specs/2026-05-20-pcenter-design.md`, `docs/superpowers/plans/2026-05-20-pcenter.md`. |
| 2026-05-19 | **GUI validation unification + cluster UX** | The 2026-05-14 planner became the single source of truth for which stages run; this batch makes it the single source for everything that *consumes* that truth: preflight, the Run-tab cost panel, the cluster master toggles, and unit metadata. Two new declarative fields on `FeatureCluster` (`master_toggle`, `available_when`) enable per-cluster master toggles in the GUI (grayed-out children when off) and the LIWC-only psych composite gating. `gui/preflight.py` was rewritten to consume `plan.stages_to_run`; the cost panel hides when no LLM features are active; the Syllables card split into Intervals + Labels + Prosody sub-clusters. Semantic categories unified into `semantic.categories_<backend>` (word-level page is canonical editor; doc-level shows count + hint). Per-run `<output>/<run_ts>_<level>_session_units.csv` records column → unit. Prefs v6 → v7 additive migration. Spec/plan: `docs/superpowers/specs/2026-05-19-gui-unification-design.md`, `docs/superpowers/plans/2026-05-19-gui-unification.md`. |
| 2026-05-15 | **Planner module** | Replaced `gui/run_dispatch.py:build_args_word_level / build_args_doc_level` (~200 lines of hand-rolled cascading rules) with a declarative `gui/planner/` package: `Stage` + `FeatureCluster` dataclasses, a `STAGES` + `CLUSTERS` registry derived from `docs/feature_dependencies.md`, a pure `plan_from_prefs` function (DAG closure + topo sort), and a `to_namespace` adapter that emits the same `argparse.Namespace` `main.py` already consumes. No changes to `main.py` or any extractor. Adding a new feature is now a single edit in `gui/planner/registry.py`. A `_word_only_stages` filter in `emit.py` prevents doc-level stage requirements from bleeding into the word-level namespace; `doc_prosody → no_prosody=False` forces word-prosody computation when doc-prosody needs the deltas. Schema-drift guard test prevents the registry from falling out of sync with prefs. 313 tests green (37 new in `tests/planner/`). Spec/plan: `docs/superpowers/specs/2026-05-14-planner-design.md`, `docs/superpowers/plans/2026-05-14-planner.md`. |
| 2026-05-13 | **Phoneme / Syllable** | Two new sidecar CSVs per stimulus (`<stim>_syllables.csv`, `<stim>_phonemes.csv`) joined back to the word CSV via FK columns. Maximum Onset Principle syllabification on MFA's phones tier; syllable prosody (mean pitch/volume + deltas) reuses the existing Parselmouth `Contours` object. New "Sub-word features" GUI page with Syllables + Phonemes cards, Run-page master switches, preflight warnings for pyphen-missing and alignment-off. Prefs v5 → v6 additive (9 syllable + 4 phoneme keys + 2 sidecar output keys). Spec/plan linked in §2. |
| 2026-05-12 | **GUI bug-fix sweep** | Nine bugs surfaced by real-stimulus testing: doc-level forced LLM/OpenAI calls regardless of toggles; CategoryPicker wrote to `document_categories_*` instead of `doc_categories_*`; `main.run()` read the obsolete v3 categories key; `no_parse` was only gated on structural toggles so sem-only word runs produced nothing; doc dispatcher hardcoded freq/edu_freq on; `sys.exit` on empty pipeline froze the GUI; Logs panel didn't tail per-step progress (`RunLogger` now tees stdout/stderr); EWFG warning leaked an internal doc path. Regression tests added in `tests/gui/test_run_dispatch.py` and `tests/gui/test_category_picker.py`. |
| 2026-05-12 | **Logs tidy-up** | Log artifacts in `<output>/logs/`; no-match.csv skipped when empty; live-tail in GUI Logs tab. Commit `98cc365`. |
| 2026-05-12 | **Acoustic prosody** | 10 word-level + 16 doc-level prosody columns via Praat-Parselmouth. Pitch units (Hz/semitones) and pause threshold configurable. Prefs v4→v5. Spec/plan linked above. |
| 2026-05-11 | **Batch 3 (GUI)** | Word vs document on separate sidebar pages, modal category dialogs per level, per-column toggles (structural × 8, acoustic × 4, psych × 4, doc counts × 4), prefs v3→v4, SENSI cyan theme. Spec: `superpowers/specs/2026-05-11-batch-3-design.md`. |
| 2026-05-11 | **PySide6 GUI redesign** | Switched from tkinter to PySide6. Sidebar+content architecture, continuous inline validation, in-session API-key paste. Spec: `superpowers/specs/2026-05-11-gui-redesign-design.md`. |
| 2026-05-10 | **Batch 1.5 (doc-level features)** | TTR, MTLD, Flesch reading ease, FK grade, LIWC summary scores (Authentic/Analytic/Tone/Clout); `documents_summary_<ts>.csv` master output; CLI `--level word\|document`. Spec: `superpowers/specs/2026-05-10-batch-1.5-tabbed-gui-design.md`. |
| 2026-05-10 | **Batch 1 (feature additions)** | EWFG grade-band frequencies (`Edu_Zipf_*`), semantic-domain indicators (LIWC/Empath), LLM-cloze (`LLM_*`), derived timing (`Word_Duration`, `Word_Gap`), folder-pair batch mode + run-log artifacts + analyzer field, GUI overhaul. Spec: `superpowers/specs/2026-05-10-feature-batch-1-design.md`. |
| Pre-history | **Original pipeline** | spaCy annotation, SUBTLEX Zipf via `wordfreq`, GPT-2 surprisal (sliding window for >1024 tokens), MFA forced alignment (replacing GENTLE). |

## 4. Roadmap — future batches

Ordered roughly by priority, not committed. Each item gets its own brainstorm → spec → plan → implement cycle.

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

Each batch lives at:
- `docs/superpowers/specs/YYYY-MM-DD-<topic>-design.md` (design)
- `docs/superpowers/plans/YYYY-MM-DD-<topic>.md` (implementation plan)

## 6. Durable conventions

- **Prefs schema:** versioned in `utils/preferences.py`. Each batch may bump the version, **always additively**. Migration chain handles all prior versions → current. Current version: **v10**.
- **Per-column drops:** GUI per-column toggles flow through `args.feature_toggles` (and `args.output_toggles` for output-section keys) → `_word_columns_to_drop` / `_doc_columns_to_drop` in `main.py`. CLI flags are group-level only.
- **Tests:** Unit tests for extractors (synthetic fixtures, no real audio where avoidable), integration tests in `tests/integration/`, GUI tests in `tests/gui/` via `pytest-qt`. Target: `python -m pytest` green on every commit.
- **Commits:** one per task during implementation; commit directly to `master` (per user preference). Feature plan referenced in commit message when applicable.

Environment activation lives in `CLAUDE.md`; durable user preferences live in `memory/MEMORY.md`.
