# SensiExtractor — Master Plan

**Last updated:** 2026-09-02

This document is the single entry point for project state. Read this first; deeper specs / plans / batch designs are linked from here.

---

## 1. Mission

SensiExtractor is the standardized SOP tool for the SENSI naturalistic-EEG pipeline (PI: Blair Kaneshiro). It takes audio (.wav) + transcript (.txt) and emits a publication-ready word-level feature CSV plus a document-level summary CSV. Lives inside the larger `../Data Curation/` effort.

## 2. Current state — what we're working on RIGHT NOW

**In flight: the audit fix branch `fix/mfa-reporting`** (2026-10-21, unmerged; see §2a), with the **output-switches branch `feat/output-switches`** (2026-10-26, unmerged, branched from it) stacked on top. The most recent work merged to `main` is the **phoneme label split + IPA + phoneme-coverage correctness** batch — **merged to `main` on 2026-07-28** (22 commits, `474a5f4..c0fda8e`; the `feat/phoneme-labels-ipa-oov` branch has been deleted) with its **real-stimulus validation PASSED**, not outstanding. See §2b for exactly what that run covered and the three things it did *not*. Before that, the config/predictability cluster (all 2026-07-10 → 07-12): Config ergonomics (defaults templates + persistent HF token), Unified YAML config (Project C), per-model predictability context size (Project B), Predictability model unification (Project A), and the release-hardening correctness sweep. Full summaries are in §3 — this section no longer restates them.

**Real-stimulus validation is no longer the blanket gap it was.** On 2026-07-28 a full `python run.py --config-dir validation/config` pass ran over the ten real stimuli in `validation/data/` with everything CPU-feasible enabled, and `validation/verify_phoneme_batch.py` exits 0. That single run exercised most of the backlog below — see §2b. The repo's own test fixtures are still empty placeholders, so the suite alone still proves nothing about MFA-dependent behaviour; the `validation/` corpus is what does.

**Also open (documentation):** the feature-doc **author-review pass** — grep `docs/features/` for `⚠️` and resolve each interpolated *why*. The citation-audit pass already landed (2026-05-27, `docs/feature-docs-audit.md`); the ⚠️ interpolation review is the remainder. See §4 "Documentation tracks."

Prefs schema is at **v20** on `main` and `fix/mfa-reporting`, **v21** on `feat/output-switches`; the version history and the two-file YAML storage move (Project C, 2026-07-12) are documented once in §6 — not repeated here.

> **Bookkeeping gap closed (2026-08-04).** The v11→v12 lexical-norms, v13→v14 struct-id-rename and v14→v15 opt-in-defaults batches now have §3 rows. They were missing because v14 and v15 both shipped inside the omnibus commit `7ae44d5` ("many updates and sherlock test") and v12 inside `12ad6a5`, so there was no per-batch commit to summarise. Every prefs version v1→v20 is now accounted for in §3.

**Fixed: headless `python run.py` no longer breaks on a pre-v20 `features.yaml`.** `run.py` still validates the raw config **before** `load_preferences()` runs the version-migration chain (`run.py:81-82`, `:108`) — that ordering is intentional, since raw validation exists to catch a malformed hand-edited config before an hour of work, not to duplicate migration. What changed: a fixed set of **retired keys** (`utils/config_yaml.RETIRED_FEATURE_KEYS` / `RETIRED_PREDICTABILITY_KEYS` — `feat_phoneme_label`, the v18 `pred_*` toggles, `feat_struct_term_id`, `predictability.llm_model`) is now tolerated by `validate_features_yaml` and reported as a non-fatal `[warn] retired toggle …` line (`retired_keys_present`) instead of an error, so a v19-shaped `features.yaml` reaches `load_preferences()`'s migration and `dispatch()` runs normally. A genuinely unknown/typo'd key is still a hard error. Also fixed the edge case this exposed: `load_preferences()` used to `setdefault` a **missing** `schema_version` to the *current* version rather than v19, which skipped every pending finalize block (including v20's `feat_phoneme_label` removal) and left the file never rewritten (`changed` stayed `False`); a **quoted** `schema_version` (e.g. `'19'`) hit the same bug because the int-keyed version-branch comparison silently fell through to the v1 flat-json migration path. Both now normalize to v19 before the migration chain runs. See `tests/test_run_py.py` and `tests/utils/test_preferences_yaml.py` for the regression coverage.

## 2a. NEXT SESSION STARTS HERE — audit fixes on `fix/mfa-reporting` (2026-10-21)

> **Stacked on top: `feat/output-switches` (2026-10-26, unmerged, not pushed).** Section
> switches on every feature card + one Run-tab switch per output level, prefs v21 (see the
> 2026-10-26 row in §3). It branches from `fix/mfa-reporting` (`0587fef`), so merge that
> first (or merge both together). Real headless run on `validation/data` gave outputs
> identical to `fix/mfa-reporting` (92 files); the owner's hand check in the GUI is still pending.

The 2026-08-04 full-project audit was **re-verified claim by claim on 2026-10-21** (read-only,
against the code). The old audit mis-framed roughly half its "wrong published numbers" (e.g.
the delta means and syllable deltas were code-correct, doc-wrong). Authority for what is
real: `audit/REVERIFY-2026-10-21.md`.
> ⚠️ `audit/` is still **git-excluded** (`.git/info/exclude`); confirm it exists before
> relying on it.

**Fixed on branch `fix/mfa-reporting` (not yet merged to `main`, not pushed)** — see the
2026-10-21 rows in §3: MFA at most once per doc after a batch failure; MFA problems in the
manifest; per-call MFA temp dir; one MFA speaker per stimulus (batch onsets = single-file
onsets); fail-fast input folders; the owner's value decisions (percent silence, standard
articulation rate, blanks for undefined means, stem `stimulus_id`, full-precision cloze,
syllable deltas skip unvoiced syllables); config value checks shared by `run.py` and the Run
tab; Cancel wired; Run-tab readiness; corrupt-TextGrid recovery; save retry; relative
`run.yaml` paths anchored to the config folder; log wording; CI import smoke covers `gui/` +
`run.py`; ~20 doc-only fixes + a complete units sidecar; dead per-column CLI flags removed.

**Before merging:** try the branch in the GUI (`python gui.py`) on real stimuli. Every
stimulus re-aligns once on its first run (sidecar v3), and several published values change
(percent silence ×100, articulation rate +1–10 %, `stimulus_id` loses `.txt`, onsets shift
by up to ~0.05–0.08 s for files previously aligned in a mixed batch).

**Still open:**
- **Sherlock:** MFA still writes `command_history.yaml` / `extracted_models/` under
  `MFA_ROOT_DIR` (inside the read-only `.sif`); unverified whether that fails there. Rebuild
  the image after merging and run a one-stimulus test job (Sherlock 2.0 nodes retired
  2026-10-06).
- **Install:** `setup.sh` should refuse an Intel (`osx-64`) conda and not install
  `conda-lock` into the base env (a collaborator's old Anaconda stalled for an hour).
- A stimulus MFA cannot align at all is still attempted once per pass.
- Leftover dead code: `Plan.column_drops` (computed and tested, never consumed) and the
  unreachable LIWC-no-.dic preflight warning.
- **Verification gap:** the `validation/` corpus has no out-of-dictionary word, so OOV/G2P
  is never exercised; no stimulus is long enough to exercise predictability chunking
  end-to-end.
- **Release blockers:** no `LICENSE`; the ⚠️ markers in `docs/features/*.md` (owner's domain
  review, §4); CI runs install + import smoke only (the test suite never runs in CI); no
  version/tag or `CITATION.cff`.

**Two surfaces came back clean in the audit — do not re-audit:** the `.mat` sidecars
(MATLAB R2024b) and the predictability math (2172/2172 pairs exact, native and chunked, via
`sherlock/validate_llama_predictability.py --model-slug gpt2`).

**Validated in real use since the audit:** the user's live `config/features.yaml`
migrated itself v19 → v20 on 2026-08-11 during ordinary use, correctly.

## 2b. Real-stimulus validation — what the 2026-07-28 run covered

One `python run.py --config-dir validation/config` pass over the ten stimuli in
`validation/data/` (5 ISC + 5 LJSpeech), everything CPU-feasible enabled, followed by
`validation/verify_phoneme_batch.py` (exit 0). Artifacts produced and inspected:

| Batch | Status after 2026-07-28 |
|---|---|
| **Phoneme label split + IPA + phoneme coverage** (2026-07-27) | **VALIDATED.** Row deltas exactly as predicted (`3_M` phonemes 111→105, syllables 41→39; `LJ003-0007` 124→119 and 48→46; other eight unchanged). Zero leftover duplicate `(Onset,End,ARPABET)` triples. `Stress` is `Int64`, empty on consonants. `IPA` non-empty on every non-`spn` row. `Pron_Source` = `dictionary` throughout (the check that would have caught the spaCy-token bug, which would have marked every number `g2p`). `Zorblax` OOV corpus correct both ways. `.mat` value trains no longer double-count. |
| Persistent, batch-once MFA alignment (2026-07-08) | **VALIDATED.** `[mfa] 10 stimuli: 0 cached, 10 aligned in 1 run` (the `SIDECAR_VERSION` 1→2 bump forcing exactly one re-align), then `10 cached, 0 aligned` on an immediate re-run with no subprocess. 10 TextGrids + 10 freshness sidecars on disk. |
| Phoneme surprisal + cohort entropy (2026-07-06) | **VALIDATED.** 10 `_phoneme_surprisal.mat` sidecars produced and diffed against the pre-change baseline. |
| Output reorganization (2026-05-22) | **VALIDATED.** `word/` 30, `syllable/` 10, `phoneme/` 20, `sentence/` 20, `document/` 5, `acoustic/` 11 files. |
| P-center (2026-05-20) | **VALIDATED.** Bit-identical across all 422 syllables when the alignment is held fixed — see the ⚠️ note below. |
| Phoneme / Syllable batch (2026-05-13) | **VALIDATED** via the above. |
| Sentence-level tier (2026-05-22) | **Artifacts confirmed** — 10 `_sentences.csv` + 10 `_bert_sentence.mat`. Column *values* not eyeballed. |
| Embeddings (Word2Vec + BERT) (2026-05-22) | **Artifacts confirmed** — 10 `_word2vec.mat`, 10 `_bert_word.mat`, 10 `_bert_sentence.mat`. Shapes/contents not opened in MATLAB. |
| Acoustic level (2026-06-13) | **Artifacts confirmed** — 10 `_acoustic.mat`. Grid alignment (`N = round(duration × analysis_rate)`) not re-verified in MATLAB. |
| Unified YAML config (Project C, 2026-07-12) | **VALIDATED, and better than asked** — the run used a `features.yaml` deliberately left at `schema_version: 19` carrying `feat_phoneme_label`, so it exercised the real v19→v20 removal-migration end to end. |
| Predictability model unification (Project A, 2026-07-10/11) | **PARTIAL.** `gpt2_Surprisal` / `gpt2_Cloze` / `gpt2_TopMatch` all present on real audio with correct NaN shape (0 PUNCT rows carry a value). **Not covered: any gated Llama model** — needs a HF token + accepted licence, and no GPU here. |
| Acoustic prosody (2026-05-12) | **VALIDATED against the cheatsheet below**, with one query — see `doc_hnr`. |

Measured on the 50 document-summary rows and 340 word rows of that run:

| Column | Measured | Expected | Verdict |
|---|---|---|---|
| `doc_articulation_rate` | 3.45 – 4.75 syll/s | 3–6 | OK |
| `doc_volume_mean` | 58.2 – 65.4 dB | 50–75 | OK |
| `doc_pause_count` | 1 – 4 | sane integer | OK |
| `doc_pitch_mean` | −1.06 – 0.78 st | relative, ~0 | OK |
| `doc_hnr` | **8.97 – 13.01 dB** (mean 11.2) | 10–25 | ⚠️ **15 of 50 rows below the 10 dB floor.** Not negative, so the unvoiced-sentinel fix is holding. Most likely the source audio: these are audiobook and classroom recordings, not studio. **Needs a human ear to confirm before publication** — if it is just the corpus, recalibrate the floor in the cheatsheet. |
| `Pitch_Mean` (word) | −13.9 – 10.5 st | −12 to +12 | 3 of 302 marginally outside |
| `Volume_Mean` (word) | 51.8 – 78.7 dB | 50–75 | 4 of 303 marginally outside |
| `gpt2_Surprisal` | 0.02 – 38.9 bits | >0, NaN on PUNCT | OK; 1 of 293 above 30 bits (a rare word) |

> ⚠️ **`P_Center` "bit-identical" needs its precondition stated, or the next person will chase a false failure.** It is bit-identical **only when the alignment is held fixed** — i.e. run the new code against the *archived* TextGrids. A full-pipeline re-run cannot match exactly, because any batch that bumps `SIDECAR_VERSION` forces a re-align and **MFA is not bit-reproducible at frame granularity** (measured: up to 0.02 s ≈ 2 frames on 7 of 422 syllables; `LJ005-0063`, which has no whitespace rows and so no renumbering, still moved 3 phoneme onsets). Also **never join old-vs-new on `token_in_sentence_id`** — this batch renumbers it, so an id join silently pairs *different words* (it paired new `often` against old `were`). Key on token text + occurrence, or compare positionally.

### Still genuinely outstanding

1. **A gated Llama predictability run** — needs a HF token, an accepted licence, and ideally a GPU.
2. **MATLAB-side loading** of the `.mat` sidecars (acoustic grid length, embedding shapes). Produced and readable by `scipy`, never opened in MATLAB.
3. **The `doc_hnr` judgement call** above.
4. **Sentence-CSV and embedding value inspection** — artifacts exist, values unexamined.


**Prosody expected-range cheatsheet** (keep handy when validating):

| Column | Expected range | If way off, suspect… |
|---|---|---|
| `Pitch_Mean` (word CSV) | 60–300 Hz, or −12 to +12 semitones | Pitch floor/ceiling wrong for speaker. |
| `Volume_Mean` (word CSV) | 50–75 dB SPL (clear studio) | Audio level off; Praat assumes normalized input. |
| `Pitch_Delta` (word CSV) | Mix of +/-, NaN on first word | Always 0 → identical values; always NaN past first → bug. |
| `doc_hnr` (master CSV) | 10–25 dB for clear studio speech; **the `validation/` corpus measures 9–13 dB** (see §2b) | Wildly negative → unvoiced-sentinel fix `460801b` didn't take. Merely *low* (9–13) is expected on audiobook/classroom source audio — do not treat it as a regression without comparing against a studio recording. |
| `doc_articulation_rate` (master CSV) | 3–6 syll/sec | NaN → `pyphen` missing; 0 → no aligned words. |
| `doc_pause_count` (master CSV) | Reasonable integer | 0 on stimulus with obvious pauses → check `--pause-threshold` (default 0.25 s). |

Specs for each batch are linked from §3 (implementation plans were deleted as part of the 2026-05-24 docs curation — code + spec are now the canonical record).

## 3. Completed batches (newest first)

One line per batch. **Full detail (the exhaustive per-batch write-ups) lives in [`docs/changelog.md`](changelog.md)** — not loaded each session; open it only when you need the deep history of a specific batch. Design specs are under `docs/superpowers/specs/`.

| Date | Batch | One-line summary + spec |
|---|---|---|
| 2026-10-26 | **Output switches (section switches + Run-tab levels)** | Every feature card's master is now an on/off **section switch** next to its title ("Included" / "Included — nothing selected" / "Left out of outputs"); off greys every control in the card and never changes ticks; "All · None" links per group; shared pitch settings stay editable. The Run tab's seven per-file output checkboxes become **one switch per level** (Word, Document, Sub-word, Sentence, Acoustic), each listing the files it will write (struck through when nothing ticked produces them); the Acoustic card and Acoustic level are one setting. New section keys (word semantic indicators/similarity, doc aggregates/lex diversity/readability/category proportions, sentence similarity) + `feat_phoneme_surprisal_mat`; planner `LEVEL_ARTIFACTS` + uniform section gating. One behaviour change: document predictability means are now computed when Word Predictability is left out but doc means are on. Prefs v20→v21 removal-migration (ten `output.*` file keys → `output.level_*` + section switches) keeps every config's files identical — checked on the owner's config over `validation/data` (92 files, 0 differences vs `fix/mfa-reporting`). Spec: [`2026-10-26-section-switch-design.md`](superpowers/specs/2026-10-26-section-switch-design.md). |
| 2026-10-21 | **Audit fix batch 2: owner value decisions, config checks, GUI/CLI robustness, doc fixes** | `doc_pause_pct_silence` is a percent of the speech span; `doc_articulation_rate` = syllables (MFA vowel phones) / (span − pauses above threshold), so the threshold matters and contractions count once (values +1–10 % on validation); `doc_pause_mean_duration`, `prop_<cat>`, `*_coverage` blank instead of 0 when undefined; `stimulus_id` = transcript stem; `*_Cloze` to 6 significant digits (never 0); syllable deltas ffill over unvoiced syllables. New `utils/config_checks.py` (ranges, allowed values, floor < ceiling, n_mfcc ≤ n_mels, model slugs, Empath names) used by `run.py` and the Run tab; Cancel wired; audio folder required only when used; corrupt TextGrid re-aligned + reported; save retry + visible failure; relative `run.yaml` folders anchored to the config folder; CI import smoke covers `gui/` + `run.py`; ~20 doc-only fixes + complete `session_units`; dead per-column CLI flags removed. No schema change. No spec (owner decisions recorded in `audit/REVERIFY-2026-10-21.md`). |
| 2026-10-21 | **MFA failure handling + input-folder checks (audit fix batch 1)** | After a failed batch pre-align, the per-doc fallback now persists its TextGrids and later passes skip already-aligned docs, so MFA runs at most once per doc; disk reuse requires a fresh sidecar; MFA batch failures and per-stimulus misses reach the run manifest; each MFA call gets its own `--temporary_directory`, so a back-to-back run can't hit a lingering lock on the shared `mfa_data/corpus/corpus.db` (MFA still writes `command_history.yaml` / `extracted_models/` under `MFA_ROOT_DIR`, so the read-only Sherlock `.sif` is **not** shown to work); each stimulus is staged as its own MFA speaker, so CMVN/fMLLR no longer pool audio across a batch — batch onsets now equal single-file onsets exactly (3 real stimuli; the old pooled layout differed by up to 0.05 s here, 0.08 s in the audit), sidecar v3 re-aligns old TextGrids once; missing/typo'd transcripts or audio folders and empty transcript folders now fail fast (`InputFolderError`, `run.py` exit 1), and audio-less transcripts during an aligned run are named in the manifest. No schema change. No spec (from the re-verified audit, `audit/REVERIFY-2026-10-21.md`, git-excluded). |
| 2026-07-27 | **Phoneme label split + IPA + phoneme coverage** | Phoneme CSV's fused `Label` → `ARPABET` + `Stress` (nullable `Int64`) + `IPA` + `Pron_Source` — **BREAKING**; syllable CSV gains dash-joined `IPA`; two silent data defects fixed (OOV words vanished from all sub-word output; contractions duplicated every phoneme, also doubling the `.mat` surprisal trains); whitespace tokens no longer occupy word-CSV rows; opt-in MFA G2P (`alignment.g2p_oov`). Prefs v19→v20 **removal-migration**. `2026-07-27-phoneme-labels-ipa-oov-design.md` |
| 2026-07-12 | **Config ergonomics: defaults templates + persistent HF token** | Committed `config/*_defaults.yaml` templates for a GUI-free headless setup + persistent HF-token file (`config/hf_token.txt`) with a GUI Save button. No schema bump (v19). `2026-07-12-config-ergonomics-design.md` |
| 2026-07-12 | **Unified YAML config (Project C)** | Replaced `preferences.json` with a two-file YAML store (`features.yaml` + `run.yaml`) behind the unchanged `preferences.py` API; new headless `run.py`; Sherlock export machinery retired. Storage-only change (v19). `2026-07-12-unified-yaml-config-design.md` |
| 2026-07-11 | **Per-model predictability context size (Project B)** | Optional per-model context cap (clamped to each model's `native_ctx`); `predictability_models.effective_context`. Prefs v18→v19 additive (`predictability.context`). `2026-07-11-predictability-context-size-design.md` |
| 2026-07-10/11 | **Predictability model unification (Project A)** | Roster-driven namespaced `<slug>_<Metric>` columns (GPT-2 + 3 Llama-3) replacing `Surprisal`/`LLM_*` — **BREAKING**. `doc_agg_predictability` auto-discovers columns; OpenAI removed. Prefs v17→v18 removal-migration. `2026-07-10-predictability-model-unification-design.md` |
| 2026-07-10 | **Release-hardening correctness sweep** | Value-correctness audit: token↔alignment merge rewrite (`align_merge.py`), `env_comp`→`log10(2)`, OpenAI cloze removed, surprisal/Zipf/Word_Gap/MTLD fixes. Prefs v16→v17. No spec (directed audit). |
| 2026-07-08 | **Persistent, batch-once MFA alignment** | MFA output persists to `<output>/Montreal Forced Alignment/` and is reused across runs; one `mfa align` subprocess per dispatch (was per-doc). `2026-07-08-persistent-mfa-alignment-design.md` |
| 2026-07-06 | **Opt-in defaults** | A fresh install / "Reset settings" now produces the minimum — only `word/<stim>_features.csv`. Cluster **master** toggles stay ON so the level cards are usable, but every non-structural feature is OFF and only the `word_level_csv` output is on; users opt into the expensive levels (MFA-based prosody, sub-word, embeddings). Prefs v14→v15. Shipped inside omnibus commit `7ae44d5`, which is why this row was missing. No spec. |
| 2026-07-06 | **Structural-id rename** | `term`/`token` were near-synonyms and easy to confuse, so both the prefs keys and the emitted CSV columns renamed in lockstep: `feat_struct_term_id` → `feat_struct_token_id` (global whole-document index) and the old `feat_struct_token_id` → `feat_struct_token_in_sentence_id` (position within the sentence); columns `term_id` → `token_id`, `token_id` → `token_in_sentence_id`. Prefs v13→v14 — **removal-migration**, the precedent later reused by v18 and v20. Shipped inside omnibus commit `7ae44d5`. No spec. |
| 2026-07-06 | **Phoneme surprisal + cohort entropy** | Context-independent `phoneme_surprisal` + `cohort_entropy` columns + `.mat` sidecar on a shared analysis grid. Prefs v15→v16 additive. `2026-07-06-phoneme-surprisal-design.md` |
| 2026-06-19 | **Acoustic page relayout + lazy planner imports** | Acoustic page single-card relayout; planner extractor callables bound lazily (`import gui.planner` ~51 s→0.03 s). No prefs change. `2026-06-19-acoustic-page-redesign-and-lazy-imports-design.md` |
| 2026-06-13 | **Self-describing embedding sidecars** | Embedding `.mat` sidecars carry labels (`tokens`/`word_id`/`is_valid`…); unified PUNCT→zero. Serialization-only. `2026-06-13-embedding-sidecar-structs-design.md` |
| 2026-06-10 | **Lexical norms (AoA / concreteness / MRC / VAD)** | Per-word age-of-acquisition, concreteness, imageability, familiarity and valence-arousal-dominance, looked up from `dictionaries/*.parquet` (`extractors/lexical_norms.py`). Opt-in: the master defaults OFF and no licensed data ships — the free research tables (Kuperman AoA, Brysbaert concreteness, MRC, Warriner VAD) are fetched by `utils/download_lexical_norms.py`. Per-column children default ON so flipping the master on yields every column. Prefs v11→v12 additive. Commit `12ad6a5`; this row was missing. No spec. |
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
| ~~**Gating simplification (masters + output toggles + Run-tab summary)**~~ — **shipped 2026-10-26** (`feat/output-switches`, see §3) | Shipped with the 2026-10-26 design's scope, not the full July sketch: masters were **kept** (presented as section switches, not mapped onto their children), and the Run tab keeps **one switch per level** instead of becoming read-only (owner decision 2026-10-26). The per-file output toggles are gone (v21). | Done | Spec: [`2026-10-26-section-switch-design.md`](superpowers/specs/2026-10-26-section-switch-design.md). The read-only Run tab and master→children mapping were dropped, not deferred. |
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

- **Prefs schema:** versioned in `utils/preferences.py`. Each batch may bump the version, **usually additively** — the rare exception is a removal-migration handled in the finalize block (precedent: v14's struct-id rename; also v18, below). Migration chain handles all prior versions → current. Current version: **v21** on `feat/output-switches`, **v20** on `main` (v12 = lexical-norms batch, v13 = acoustic level, v14 = struct-id rename, v15 = opt-in defaults, v16 = phoneme surprisal + cohort entropy, v17 = release-hardening default corrections: `env_comp`→`log10(2)`, stored OpenAI `llm_model`→`""`, v18 = predictability model unification — **removal-migration**: the four `pred_surprisal_gpt2`/`pred_surprisal_llm`/`pred_cloze`/`pred_topmatch` toggles and `predictability.llm_model` are replaced by `predictability: {models: [...], metrics: {surprisal, cloze, topmatch}}`, v19 = per-model predictability context size — **additive**: adds `predictability.context` (slug→max-context map, empty by default), v20 = phoneme label split + IPA + G2P — **removal-migration**: `feat_phoneme_label` is **removed** and replaced by the four per-column toggles `feat_phoneme_arpabet` / `feat_phoneme_stress` / `feat_phoneme_ipa` / `feat_phoneme_pron_source`, each **inheriting the removed key's stored value** so a user who had the label column off keeps all four off; additive alongside it are `feat_syllable_ipa` (default `False`, matching `feat_syllable_arpabet`'s verbose-column precedent) and a **new top-level `alignment` section** holding `g2p_oov` (default `False`). The `alignment` section needs no `config_yaml` routing change — `_RUN_SECTIONS` is `("shared",)` and everything else goes to `features.yaml`, which is the semantically right home: it makes the GUI's "Reset settings" turn `g2p_oov` back off. Migration handled in the `_merged_from_user` finalize block alongside v17/v18, per the v14/v18 precedent; v21 = output switches — **removal-migration**: the ten per-file `output.*` toggles (`word_level_csv` … `bert_sentence_mat`) become five `output.level_*` switches, anything finer than a level maps onto section switches (new section keys default on) and `feat_phoneme_surprisal_mat`, so a migrated config writes the same files — `_migrate_outputs_to_levels`). **Storage** (independent of schema version): since 2026-07-12 (Project C) the schema is persisted as a two-file YAML store (`config/features.yaml` + `config/run.yaml`) instead of a single `preferences.json`, behind the same load/save API — the migration chain above is unaffected.
- **Per-column drops:** GUI per-column toggles flow through `args.feature_toggles` (and `args.output_toggles` for output-section keys) → `_word_columns_to_drop` / `_doc_columns_to_drop` in `main.py`. CLI flags are group-level only.
- **Tests:** Unit tests for extractors (synthetic fixtures, no real audio where avoidable), integration tests in `tests/integration/`, GUI tests in `tests/gui/` via `pytest-qt`. Target: `python -m pytest` green on every commit.
- **Commits:** one per task during implementation; commit directly to `master` (per user preference). Feature plan referenced in commit message when applicable.
- **Per-feature docs:** every GUI toggle key has (or should have) a `docs/features/<key>.md` file rendered in-app via `gui/info_popup.py`. New batch → new toggle → add the `.md` (the ⓘ icon appears automatically; no GUI wiring required). Shared content across related toggles (e.g. all `pitch_*`) is routed via `gui/feature_docs.py::_SHARED_DOC_MAP`. Authoring rules in [`feedback-feature-docs-authorship`](../memory/feedback-feature-docs-authorship.md) and [`feedback-feature-docs-citations`](../memory/feedback-feature-docs-citations.md).

Environment activation lives in `CLAUDE.md`; durable user preferences live in `memory/MEMORY.md`.
