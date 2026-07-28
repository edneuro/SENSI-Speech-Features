# SensiExtractor — Master Plan

**Last updated:** 2026-07-27

This document is the single entry point for project state. Read this first; deeper specs / plans / batch designs are linked from here.

---

## 1. Mission

SensiExtractor is the standardized SOP tool for the SENSI naturalistic-EEG pipeline (PI: Blair Kaneshiro). It takes audio (.wav) + transcript (.txt) and emits a publication-ready word-level feature CSV plus a document-level summary CSV. Lives inside the larger `../Data Curation/` effort.

## 2. Current state — what we're working on RIGHT NOW

**No batch is mid-flight.** The most recent shipped work is the **phoneme label split + IPA + phoneme-coverage correctness** batch (2026-07-27) — implemented, tests-green, on branch `feat/phoneme-labels-ipa-oov`, with real-stimulus validation outstanding (see the table below; it is the one batch whose validation is *load-bearing*, since it changed row counts and fixed two silent data defects). Before that, the config/predictability cluster (all 2026-07-10 → 07-12): Config ergonomics (defaults templates + persistent HF token), Unified YAML config (Project C), per-model predictability context size (Project B), Predictability model unification (Project A), and the release-hardening correctness sweep. Full summaries are in §3 — this section no longer restates them.

**The one thing outstanding across all of that: end-to-end manual validation on a real ISC-Wu Tsai stimulus** (see table below). Everything ships implemented + tests-green, but this environment has only empty placeholder audio fixtures, so no real MFA-dependent run has been exercised.

**Also open (documentation):** the feature-doc **author-review pass** — grep `docs/features/` for `⚠️` and resolve each interpolated *why*. The citation-audit pass already landed (2026-05-27, `docs/feature-docs-audit.md`); the ⚠️ interpolation review is the remainder. See §4 "Documentation tracks."

Prefs schema is at **v20**; the version history and the two-file YAML storage move (Project C, 2026-07-12) are documented once in §6 — not repeated here.

> ⚠️ **Bookkeeping gap (still open):** the v11→v12 **lexical-norms** batch (AoA/concreteness/MRC/VAD word norms) and the v13→v14 (struct-id rename) / v14→v15 (opt-in defaults) batches shipped to the codebase but their §3 rows were never written. Add those rows when convenient.

**Fixed: headless `python run.py` no longer breaks on a pre-v20 `features.yaml`.** `run.py` still validates the raw config **before** `load_preferences()` runs the version-migration chain (`run.py:81-82`, `:108`) — that ordering is intentional, since raw validation exists to catch a malformed hand-edited config before an hour of work, not to duplicate migration. What changed: a fixed set of **retired keys** (`utils/config_yaml.RETIRED_FEATURE_KEYS` / `RETIRED_PREDICTABILITY_KEYS` — `feat_phoneme_label`, the v18 `pred_*` toggles, `feat_struct_term_id`, `predictability.llm_model`) is now tolerated by `validate_features_yaml` and reported as a non-fatal `[warn] retired toggle …` line (`retired_keys_present`) instead of an error, so a v19-shaped `features.yaml` reaches `load_preferences()`'s migration and `dispatch()` runs normally. A genuinely unknown/typo'd key is still a hard error. Also fixed the edge case this exposed: `load_preferences()` used to `setdefault` a **missing** `schema_version` to the *current* version rather than v19, which skipped every pending finalize block (including v20's `feat_phoneme_label` removal) and left the file never rewritten (`changed` stayed `False`); a **quoted** `schema_version` (e.g. `'19'`) hit the same bug because the int-keyed version-branch comparison silently fell through to the v1 flat-json migration path. Both now normalize to v19 before the migration chain runs. See `tests/test_run_py.py` and `tests/utils/test_preferences_yaml.py` for the regression coverage.

### Awaiting end-to-end manual validation on a real ISC-Wu Tsai stimulus

The feature batches below all ship implemented + tests-green; what's outstanding is the same thing: someone runs `python gui.py` on a real stimulus and eyeballs the outputs. They're rolled up here so §2 doesn't repeat the same status for every batch.

| Batch | Date | What to look at |
|---|---|---|
| **Phoneme label split + IPA + phoneme coverage** | 2026-07-27 | **The highest-priority validation in this table — this batch changed output values and row counts, and two of its fixes are for defects that failed silently.** Re-run all ten `validation/` stimuli and check, in order of how badly a regression would hurt: **(1) `P_Center` must be bit-identical** to the archived CSVs — the nucleus test moved from a `Label` string-suffix check to `Stress.notna()`, and if that migration were wrong every P-centre would silently collapse to the syllable onset with no exception. **(2) Row counts** drop by exactly the duplicate copies and by zero elsewhere: `3_M` phonemes 111 → **105** (−6) and syllables 41 → **39** (−2); `LJ003-0007` phonemes 124 → **119** (−5) and syllables 48 → **46** (−2); the other eight unchanged. (The removed count is *half* the rows flagged as sharing an interval — 12 flagged in `3_M` = 6 kept + 6 removed.) **(3) `Stress`** empty on every consonant row and populated on every vowel row — and read as `Int64`, not float. **(4) `IPA`** non-empty for every non-`spn` row; spot-check `AH0`→`ə` vs `AH1`→`ʌ` and `ER0`→`ɚ` vs `ER1`→`ɝ`. **(5)** `paragraph_id == 1` throughout, zero SPACE rows, and `head_token_id` resolving to a real `token_id` on the three hard-wrapped `LJ*` stimuli. **(6)** The `Zorblax` OOV corpus: with `alignment.g2p_oov` **off**, one `oov_unresolved` phoneme row with real timing; with it **on**, real phones marked `g2p`. **(7)** `phoneme_surprisal.mat` no longer double-counting at the duplicated contraction onsets. Also worth exercising the documented per-document staleness gap: with an existing `Montreal Forced Alignment/<stem>.TextGrid`, confirm the **batch** path re-aligns when `g2p_oov` flips (sidecar `mfa_g2p`), and that the per-document fallback does not — see [`g2p_oov`](features/g2p_oov.md). |
| Unified YAML config (Project C) | 2026-07-12 | Launch `python gui.py` once on a machine that still has a `config/preferences.json` from a prior session — confirm it becomes `preferences.json.bak` and `config/features.yaml` + `config/run.yaml` appear with the analyzer name and folder paths carried over intact. Click "Reset settings" and confirm the folder paths (and analyzer name) survive — only `features.yaml` should be rewritten. Then a headless smoke test: `python run.py` (or `--config-dir` pointed at a scratch config folder) against a real stimulus, confirming it produces the same outputs as an equivalent GUI run. |
| Predictability model unification (Project A) | 2026-07-10/11 | Select at least one gated Llama model (needs a HuggingFace token + accepted license — see `docs/features/llm_credentials.md`) alongside GPT-2 on the Predictability card, enable all three metrics, and run on a real stimulus. Confirm the word CSV has one `<slug>_<Metric>` column per model × metric (e.g. `gpt2_Surprisal`, `llama3_8b_Cloze`), first-word/PUNCT rows are `NaN`, and `gpt2_Surprisal` values are unchanged vs. the pre-unification `Surprisal` column on the same stimulus (GPT-2 is pinned full-precision CPU specifically so this should hold; a unit test pins the NaN/positive shape of the values but not byte-identical equality against a real archived run — worth spot-checking here). Confirm `doc_agg_predictability` produces a `mean_<slug>_<metric>` doc-summary column for each selected model × metric. Not yet run against a real gated Llama checkpoint in the implementation environment (no GPU / gated-model download there). |
| Persistent, batch-once MFA alignment | 2026-07-08 | Run a real batch twice on the same stimuli (`python gui.py`, or `python main.py --transcripts-folder … --audio-folder … --output-folder …`). **Run 1:** confirm `<output_folder>/Montreal Forced Alignment/<stem>.TextGrid` + `<stem>.json` appear and the log reads `[mfa] N stimuli: 0 cached, N aligned in 1 run`. **Run 2 (unchanged inputs):** log reads `[mfa] N stimuli: N cached, 0 aligned` — no MFA subprocess, noticeably faster startup — and outputs match run 1. **Staleness:** edit one transcript, re-run; only that stimulus re-aligns (`1 aligned`, rest cached) and its TextGrid/sidecar are rewritten. Confirm word-CSV `Onset`/`End` values are unchanged vs the pre-change baseline for the untouched stimuli. |
| Phoneme surprisal + cohort entropy | 2026-07-06 | Enable `feat_phoneme_surprisal` / `feat_phoneme_cohort_entropy` / `phoneme_surprisal_mat` (GUI Sub-word page + Run page, or CLI) and run on a real stimulus with MFA alignment on. Confirm `phoneme/<stim>_phonemes.csv` has `phoneme_surprisal` + `cohort_entropy` columns with sane values (first phoneme of a word > 0; entropy reaches exactly 0 on the final phoneme of a monosyllabic word once its cohort narrows to one candidate). Confirm `phoneme/<stim>_phoneme_surprisal.mat` loads with four length-`N` trains (`phoneme_surprisal`, `cohort_entropy`, `phoneme_onset`, `word_onset`) plus `t`/`fs`/`config`, `N` matching the acoustic `.mat`'s grid when the same `analysis_sr` is used. Not yet run against a real stimulus in the implementation environment — MFA is installed but only empty placeholder audio fixtures are present there. |
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

One line per batch. **Full detail (the exhaustive per-batch write-ups) lives in [`docs/changelog.md`](changelog.md)** — not loaded each session; open it only when you need the deep history of a specific batch. Design specs are under `docs/superpowers/specs/`.

| Date | Batch | One-line summary + spec |
|---|---|---|
| 2026-07-27 | **Phoneme label split + IPA + phoneme coverage** | Phoneme CSV's fused `Label` → `ARPABET` + `Stress` (nullable `Int64`) + `IPA` + `Pron_Source` — **BREAKING**; syllable CSV gains dash-joined `IPA`; two silent data defects fixed (OOV words vanished from all sub-word output; contractions duplicated every phoneme, also doubling the `.mat` surprisal trains); whitespace tokens no longer occupy word-CSV rows; opt-in MFA G2P (`alignment.g2p_oov`). Prefs v19→v20 **removal-migration**. `2026-07-27-phoneme-labels-ipa-oov-design.md` |
| 2026-07-12 | **Config ergonomics: defaults templates + persistent HF token** | Committed `config/*_defaults.yaml` templates for a GUI-free headless setup + persistent HF-token file (`config/hf_token.txt`) with a GUI Save button. No schema bump (v19). `2026-07-12-config-ergonomics-design.md` |
| 2026-07-12 | **Unified YAML config (Project C)** | Replaced `preferences.json` with a two-file YAML store (`features.yaml` + `run.yaml`) behind the unchanged `preferences.py` API; new headless `run.py`; Sherlock export machinery retired. Storage-only change (v19). `2026-07-12-unified-yaml-config-design.md` |
| 2026-07-11 | **Per-model predictability context size (Project B)** | Optional per-model context cap (clamped to each model's `native_ctx`); `predictability_models.effective_context`. Prefs v18→v19 additive (`predictability.context`). `2026-07-11-predictability-context-size-design.md` |
| 2026-07-10/11 | **Predictability model unification (Project A)** | Roster-driven namespaced `<slug>_<Metric>` columns (GPT-2 + 3 Llama-3) replacing `Surprisal`/`LLM_*` — **BREAKING**. `doc_agg_predictability` auto-discovers columns; OpenAI removed. Prefs v17→v18 removal-migration. `2026-07-10-predictability-model-unification-design.md` |
| 2026-07-10 | **Release-hardening correctness sweep** | Value-correctness audit: token↔alignment merge rewrite (`align_merge.py`), `env_comp`→`log10(2)`, OpenAI cloze removed, surprisal/Zipf/Word_Gap/MTLD fixes. Prefs v16→v17. No spec (directed audit). |
| 2026-07-08 | **Persistent, batch-once MFA alignment** | MFA output persists to `<output>/Montreal Forced Alignment/` and is reused across runs; one `mfa align` subprocess per dispatch (was per-doc). `2026-07-08-persistent-mfa-alignment-design.md` |
| 2026-07-06 | **Phoneme surprisal + cohort entropy** | Context-independent `phoneme_surprisal` + `cohort_entropy` columns + `.mat` sidecar on a shared analysis grid. Prefs v15→v16 additive. `2026-07-06-phoneme-surprisal-design.md` |
| 2026-06-19 | **Acoustic page relayout + lazy planner imports** | Acoustic page single-card relayout; planner extractor callables bound lazily (`import gui.planner` ~51 s→0.03 s). No prefs change. `2026-06-19-acoustic-page-redesign-and-lazy-imports-design.md` |
| 2026-06-13 | **Self-describing embedding sidecars** | Embedding `.mat` sidecars carry labels (`tokens`/`word_id`/`is_valid`…); unified PUNCT→zero. Serialization-only. `2026-06-13-embedding-sidecar-structs-design.md` |
| 2026-06-13 | **Acoustic level (audio-only TRF features)** | New audio-only `acoustic` level: envelope/onset-env/log-mel/MFCC on a shared grid → `acoustic/<stim>_acoustic.mat`. Prefs v12→v13 additive. `2026-06-13-acoustic-level-trf-features-design.md` |
| 2026-06-10 | **LIWC removal + license-clean composite psych** | LIWC dropped (EULA); doc psych (Authentic/Analytic/Tone/Clout) reimplemented from spaCy POS + VADER (`psych_composite.py`). Prefs v10→v11. `2026-06-10-liwc-removal-psych-reimpl-design.md` |
| 2026-05-24 | **In-app feature documentation system** | 41 per-feature `docs/features/*.md` rendered in-app via the ⓘ icon (`feature_docs.py` + `info_popup.py`). No spec. |
| 2026-05-22 | **Embeddings (Word2Vec + BERT)** | Three `.mat` sidecars: word2vec (300d), bert_word (12×768), bert_sentence. Prefs v9→v10 additive. `2026-05-22-embeddings-design.md` |
| 2026-05-22 | **Sentence-level tier** | `sentence/<stim>_sentences.csv` (timing + 11 prosody cols). Prefs v8→v9 additive. `2026-05-22-sentence-level-tier-design.md` |
| 2026-05-22 | **Output reorganization** | Per-level subfolders `word/` `syllable/` `phoneme/` `document/`. **Breaking** path change. `2026-05-22-output-reorganization-design.md` |
| 2026-05-20 | **P-center** | `P_Center` column on `syllables.csv` (Scott 1993, `pcenter.py`). Prefs v7→v8. `2026-05-20-pcenter-design.md` |
| 2026-05-19 | **GUI validation unification + cluster UX** | Planner became the single source for preflight/cost-panel/cluster masters; `FeatureCluster` gained `master_toggle`/`available_when`. Prefs v6→v7. `2026-05-19-gui-unification-design.md` |
| 2026-05-15 | **Planner module** | Replaced hand-rolled arg-building with the declarative `gui/planner/` package (Stage + FeatureCluster registry, `plan_from_prefs`, `to_namespace`). `2026-05-14-planner-design.md` |
| 2026-05-13 | **Phoneme / Syllable** | Two sidecar CSVs (syllables/phonemes) joined to the word CSV via FK; new Sub-word GUI page. Prefs v5→v6 additive. `2026-05-13-phoneme-syllable-design.md` |
| 2026-05-12 | **GUI bug-fix sweep** | Nine real-stimulus bugs fixed (forced LLM, category-key mismatch, empty-pipeline freeze, log tailing…). Regression tests added. No spec. |
| 2026-05-12 | **Logs tidy-up** | Log artifacts in `<output>/logs/`; live-tail in GUI. Commit `98cc365`. |
| 2026-05-12 | **Acoustic prosody** | 10 word + 16 doc prosody columns via Praat-Parselmouth. Prefs v4→v5. `2026-05-12-prosody-design.md` |
| 2026-05-11 | **Batch 3 (GUI)** | Word/document on separate pages, modal category dialogs, per-column toggles, SENSI cyan theme. Prefs v3→v4. `2026-05-11-batch-3-design.md` |
| 2026-05-11 | **PySide6 GUI redesign** | tkinter → PySide6; sidebar+content, inline validation. `2026-05-11-gui-redesign-design.md` |
| 2026-05-10 | **Batch 1.5 (doc-level features)** | TTR, MTLD, Flesch, FK, LIWC summary; `documents_summary_<ts>.csv`; CLI `--level`. `2026-05-10-batch-1.5-tabbed-gui-design.md` |
| 2026-05-10 | **Batch 1 (feature additions)** | EWFG grade-band freqs, semantic-domain indicators, LLM-cloze, derived timing, folder-pair batch mode + run-log. `2026-05-10-feature-batch-1-design.md` |
| Pre-history | **Original pipeline** | spaCy annotation, wordfreq Zipf, GPT-2 surprisal (sliding window), MFA forced alignment. |

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
| **Predictability context size (Project B)** | Per-model configurable context size (max context, clamped to each roster model's `native_ctx`) for the predictability extractor. Rides directly on the Project A roster (`extractors/predictability_models.py`'s `native_ctx` field already carries the data B needs). | **Shipped (2026-07-11)** | See §3 completed batches. |
| **Predictability shim cleanup (T8b)** | Delete `extractors/surprisal.py` / `extractors/llm_cloze.py` once nothing imports them (thin compatibility shims over `extractors/predictability.py`, kept during Project A for a smaller diff). | **Done (2026-07-11)** | Shims deleted at the end of the Project A branch (task T8b); `validate_cloze` migrated off them. |
| **Gating simplification (masters + output toggles + Run-tab summary)** | A removal-migration (next free schema version — v20 was taken by the 2026-07-27 phoneme batch) mapping each cluster's master toggle onto its children (masters currently double up with the per-column toggles they gate); planner/registry rework + preflight updates + a GUI card redesign to match; the Run tab becomes a **read-only** expected-output summary + readiness (no more independent output-toggle checkboxes to keep in sync with the feature pages). | Deferred until after the manual-validation pass | Decision from the 2026-07-12 discussion — large, cross-cutting change (prefs schema, planner, GUI, docs) deliberately parked behind the outstanding real-stimulus validation in §2 so it isn't built on top of unverified extractor output. |
| **P-center** | Perceptual onset of syllables (Scott 1993 / Villing 2010). Syllable level. | **Shipped (2026-05-20)** | See §3 completed batches. |
| **BERT contextual embeddings** | 768-dim vectors per word in a sidecar .mat (all 12 layers). | **Shipped (2026-05-22)** | See §3 completed batches. Includes Word2Vec 300d. |
| **Human-rated cloze norms** | Not an extractor — data acquisition / collaboration with cloze-norming corpora. | Parked | |
| **Document-level enrichment expansion** | Anything beyond what Batch 1.5 / Batch 3 already cover. | Mostly already done | Revisit only if a specific gap is identified. |
| **Correctness scoring** | Either MFA alignment confidence (cheap) or ASR-vs-transcript agreement (medium). | Parked | Needs collaborator clarification on which they want. |
| **Brain Treebank parity** | `Index_in_Sentence`, `Word_Head`. | Already covered — **as of 2026-07-27** | `token_in_sentence_id` gives `Index_in_Sentence`; `head_token_id` gives `Word_Head`. ⚠️ **Correction to the previous wording, which was false:** before 2026-07-27 `token_in_sentence_id` was *not* a true `Index_in_Sentence` for any transcript containing a hard line wrap. spaCy emits a literal `\n` as a `SPACE` token, and `text_parser.parse` iterated every token, so each newline consumed a `token_in_sentence_id` (and a `token_id`) and shifted every following word by one — affecting 3 of the 10 validation stimuli (the `LJ*` files; the single-line ISC stimuli were unaffected, which is why it went unnoticed). Whitespace tokens are now skipped, ids count emitted tokens only, and the parity claim holds. As of 2026-07-06 `head_token_id` names the head by **global `token_id`** (self-contained, unambiguous across sentences); the same batch renamed the id columns (`term_id`→`token_id`, `token_id`→`token_in_sentence_id`). If a consumer needs a sentence-local head index, derive it as `head_token_id - (token_id - token_in_sentence_id)` within the head's sentence. |

### Infrastructure / portability

| Item | Scope | Status | Notes |
|---|---|---|---|
| **Cross-platform CI matrix** | Run the suite on Windows + macOS + Linux (GitHub Actions) so portability is verified, not assumed. | Not started (next infra task) | Triggered by the 2026-06-14 MKL-on-AMD crash (commit `ec620e9`): the acoustic mel matmul crashed every real run while tests passed, because the crash workaround lived only in `tests/conftest.py`. CI is the only real guard for OS/CPU-specific breakage. See [`project-mkl-threading-entrypoints`](../memory/project-mkl-threading-entrypoints.md). |
| **Move one-time CLI tools to `scripts/`** | Relocate `utils/download_models.py`, `utils/download_lexical_norms.py`, `utils/validate_cloze.py` into `scripts/` — they're one-time setup/validation tools, not library code imported by the pipeline, so they belong alongside `relock.py` / `export_default_config.py` rather than in `utils/`. | Not started | Neatness only, no behavior change. Touches `setup.sh:29` / `setup.bat:40` (the invocation paths), `CLAUDE.md` / `README.md` (the tree + any documented commands), and any tests that import these modules by their current path. |
| **Acoustic-level portability audit** | One-off review of the acoustic code for OS/CPU assumptions. | **Done (2026-06-14)** | Clean: extractor + runner are pure numpy/librosa/scipy/`soundfile` (no MFA/Kaldi, no ffmpeg — libsndfile ≥1.1 reads WAV/FLAC/MP3 natively), all `pathlib`. MKL guard (`MKL_THREADING_LAYER=sequential` via `setdefault` in `main.py`/`gui.py`) is a no-op where numpy isn't an MKL build (OpenBLAS / macOS Accelerate), so it's safe on every platform. Alternative if MKL ever bites again: ship the env on conda-forge/OpenBLAS (`nomkl`). |

## 5. Naming conventions (read this to avoid confusion)

We have two parallel "batch" numbering systems that have collided:

- **GUI batches** (1, 1.5, PySide6 redesign, batch 3) — numbered iterations of the GUI's structure.
- **Feature roadmap batches** (originally numbered 1–7 in the batch-1 spec §11) — numbered iterations of the *extractor* feature set.

To prevent further collisions, **new batches are named by their feature** ("prosody batch", "phoneme/syllable batch") rather than a number.

Each batch has a design spec at:
- `docs/superpowers/specs/YYYY-MM-DD-<topic>-design.md`

Step-by-step implementation plans (`docs/superpowers/plans/`) are no longer kept after a batch ships — the code is the canonical execution record. The spec captures intent; the code captures realization.

## 6. Durable conventions

- **Prefs schema:** versioned in `utils/preferences.py`. Each batch may bump the version, **usually additively** — the rare exception is a removal-migration handled in the finalize block (precedent: v14's struct-id rename; also v18, below). Migration chain handles all prior versions → current. Current version: **v20** (v12 = lexical-norms batch, v13 = acoustic level, v14 = struct-id rename, v15 = opt-in defaults, v16 = phoneme surprisal + cohort entropy, v17 = release-hardening default corrections: `env_comp`→`log10(2)`, stored OpenAI `llm_model`→`""`, v18 = predictability model unification — **removal-migration**: the four `pred_surprisal_gpt2`/`pred_surprisal_llm`/`pred_cloze`/`pred_topmatch` toggles and `predictability.llm_model` are replaced by `predictability: {models: [...], metrics: {surprisal, cloze, topmatch}}`, v19 = per-model predictability context size — **additive**: adds `predictability.context` (slug→max-context map, empty by default), v20 = phoneme label split + IPA + G2P — **removal-migration**: `feat_phoneme_label` is **removed** and replaced by the four per-column toggles `feat_phoneme_arpabet` / `feat_phoneme_stress` / `feat_phoneme_ipa` / `feat_phoneme_pron_source`, each **inheriting the removed key's stored value** so a user who had the label column off keeps all four off; additive alongside it are `feat_syllable_ipa` (default `False`, matching `feat_syllable_arpabet`'s verbose-column precedent) and a **new top-level `alignment` section** holding `g2p_oov` (default `False`). The `alignment` section needs no `config_yaml` routing change — `_RUN_SECTIONS` is `("shared",)` and everything else goes to `features.yaml`, which is the semantically right home: it makes the GUI's "Reset settings" turn `g2p_oov` back off. Migration handled in the `_merged_from_user` finalize block alongside v17/v18, per the v14/v18 precedent). **Storage** (independent of schema version): since 2026-07-12 (Project C) the schema is persisted as a two-file YAML store (`config/features.yaml` + `config/run.yaml`) instead of a single `preferences.json`, behind the same load/save API — the migration chain above is unaffected.
- **Per-column drops:** GUI per-column toggles flow through `args.feature_toggles` (and `args.output_toggles` for output-section keys) → `_word_columns_to_drop` / `_doc_columns_to_drop` in `main.py`. CLI flags are group-level only.
- **Tests:** Unit tests for extractors (synthetic fixtures, no real audio where avoidable), integration tests in `tests/integration/`, GUI tests in `tests/gui/` via `pytest-qt`. Target: `python -m pytest` green on every commit.
- **Commits:** one per task during implementation; commit directly to `master` (per user preference). Feature plan referenced in commit message when applicable.
- **Per-feature docs:** every GUI toggle key has (or should have) a `docs/features/<key>.md` file rendered in-app via `gui/info_popup.py`. New batch → new toggle → add the `.md` (the ⓘ icon appears automatically; no GUI wiring required). Shared content across related toggles (e.g. all `pitch_*`) is routed via `gui/feature_docs.py::_SHARED_DOC_MAP`. Authoring rules in [`feedback-feature-docs-authorship`](../memory/feedback-feature-docs-authorship.md) and [`feedback-feature-docs-citations`](../memory/feedback-feature-docs-citations.md).

Environment activation lives in `CLAUDE.md`; durable user preferences live in `memory/MEMORY.md`.
