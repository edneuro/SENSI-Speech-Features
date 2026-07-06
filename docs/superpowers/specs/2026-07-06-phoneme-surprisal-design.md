# Phoneme surprisal + cohort entropy — design spec

**Date:** 2026-07-06
**Status:** design approved, pending spec review → plan → implement
**Level:** Sub-word (phoneme)
**Deps:** audio + MFA alignment (required); `wordfreq` + the MFA dictionary (both already in-repo — no new deps)

---

## 1. What we're adding

Context-independent **phoneme surprisal** and **cohort entropy** as word-onset-agnostic,
phoneme-onset-aligned regressors for TRF / cortical-tracking analyses, plus a unit-height
**phoneme onset** control. All three are emitted as impulse trains in a `.mat` sidecar; the
two scalar features are additionally written as per-phoneme columns on the existing phoneme CSV.

These are **acoustically/lexically driven, not LM-driven**: the cohort model uses only a
frequency-weighted unigram prior, keeping the features decorrelated from the GPT-2 word-level
`Surprisal` feature (which we do not touch).

## 2. The cohort model (verified against the hand-computable check)

A frequency-weighted cohort over a phoneme-prefix trie. Each trie node caches two accumulators
over every lexicon word whose (stress-stripped, primary) pronunciation passes through it:

```
Z   = Σ f(w)                 # total frequency mass in the cohort at this node
SFL = Σ f(w) · log2 f(w)     # "sum of f·log f", for the entropy closed form
```

For the aligned phoneme sequence `φ1..φk..φn` of a word, walking root → node-after-φ1 → …:

```
surprisal(φk) = log2(Z_before) − log2(Z_after)      # Z_before(φ1) = Z_root (empty context)
entropy(φk)   = log2(Z_after) − SFL_after / Z_after  # entropy of the cohort AFTER φk
```

`Z_before` is the node reached before consuming φk; `Z_after` is the node after consuming it.
`entropy` is evaluated at the **after** node (this is load-bearing — see §8 check). Both are
O(1) per phoneme because `Z`/`SFL` are precomputed at trie-build time.

**OOV path** (aligned sequence steps off the trie — a pronunciation variant not in the lexicon):
from the first missing phoneme onward, `surprisal = cap` and `entropy = 0`, where
`cap = −log2(f_min / Z_root)` (the surprisal of the single rarest lexicon word). Never `log(0)`.

## 3. Sources (reuse, no new deps)

- **Pronunciation lexicon:** `mfa_data/pretrained_models/dictionary/english_us_arpa.dict` — the
  exact dictionary MFA aligns against, so the trie's phoneme inventory matches the aligned
  `phones` tier. Format: `word\tPH PH PH` (one line per pronunciation variant).
- **Frequency f(w):** `wordfreq.word_frequency(w, "en")` (linear), the same SUBTLEX-US-backed
  source behind the existing `Zipf_Freq` column. Words with `f(w) == 0` (absent from wordfreq)
  are excluded from the cohort.
- **Aligned pronunciation:** MFA `phones` tier → `parse_phones` → the per-word phoneme sequence
  already grouped by `syllabify_pairs` (the `phonemes_df` rows for a word, in order). This is
  "the variant the aligner chose."

## 4. Extractor — `extractors/phoneme_surprisal.py` (pure functions, TDD)

- `load_mfa_lexicon(dict_path, *, strip_stress=True, primary_only=True) -> list[tuple[tuple[str,...], float]]`
  Parse the MFA `.dict`; `strip_stress` drops the trailing ARPABET stress digit (`EH1`→`EH`);
  `primary_only` keeps only the first-listed variant per word. Join each word to `f(w)`.
- `build_cohort_trie(entries) -> CohortTrie` — nodes cache `z`, `sfl`; also exposes `z_root`
  and `f_min` for the cap. Built once, cached per run (module-level cache keyed by
  `(dict_path, strip_stress, primary_only)`).
- `phoneme_scores(trie, phones, *, context_independent=True, first_phoneme_empty_context=True)`
  `-> (surprisals: list[float], entropies: list[float])`. `phones` is the stress-stripped
  (per `strip_stress`) aligned phoneme sequence for one word. The two keyword flags are
  **guard flags**: any non-default value raises `NotImplementedError` (LM-reweighted /
  non-empty-context modes are a future batch). They are **not** prefs/GUI keys — they document
  the modeling commitment at the API level without adding permanent prefs-schema surface (cf.
  the repo's regret over vestigial prefs keys like `master_acoustic_ts`).
- `rasterize(events, duration_s, analysis_sr) -> dict[str, np.ndarray]` — build the three
  length-`N` impulse trains (`N = round(duration_s * analysis_sr)`), impulses at
  `round(onset_s * analysis_sr)` using MATLAB-style rounding consistent with
  `acoustic_ts._mround`. `events` = list of `(onset_s, surprisal, entropy)`. Same-sample
  collisions: value trains sum; `phoneme_onset` stays unit height (1).

Config surface split:
- **Functional prefs/GUI flags** (change the numbers): `strip_stress`, `primary_only`, `analysis_sr`.
- **Extractor-API guard flags** (one implemented behavior): `context_independent`,
  `first_phoneme_empty_context`.

## 5. Outputs

**CSV** — extend `_PHON_COLUMNS` in `extractors/syllabify.py` with `phoneme_surprisal`,
`cohort_entropy`; the `phoneme_surprisal` stage appends these to `phonemes_df`. One row per
aligned phoneme, keyed as today (`doc_id`, `sentence_id`, `token_in_sentence_id`, `token`,
`syllable_id`, `phoneme_id`).

**`.mat`** — `phoneme/<stim>_phoneme_surprisal.mat`, self-describing like
`acoustic/<stim>_acoustic.mat`:
- `phoneme_surprisal`, `cohort_entropy`, `phoneme_onset` — three length-`N` impulse trains,
  each intended for its own banded-ridge band downstream.
- `t` (= `arange(N)/analysis_sr`), `fs` (= `analysis_sr`).
- `config` provenance: `analysis_sr`, `strip_stress`, `primary_only`, `context_independent`,
  `first_phoneme_empty_context`, `cap`, `n_phonemes`, `duration_s`, `lexicon` path.
- Grid `duration_s` comes from the **audio** (via librosa/soundfile, as the acoustic runner
  already loads it) so this `.mat` stacks sample-for-sample with the acoustic `.mat` when the
  same `analysis_sr` is used. Default `analysis_sr` = the acoustic level's default (`128`).

## 6. Wiring (mirrors GPT-2 surprisal + acoustic_ts)

- **Stage** `phoneme_surprisal` in `gui/planner/registry.py`: consumes `phonemes_df` (already
  grouped per word by `syllabify_pairs`); produces `phonemes_df.phoneme_surprisal`,
  `phonemes_df.cohort_entropy` — augmenting the artifact in place, mirroring how
  `syllable_prosody` consumes/produces `syllables_df`. Lazy-bound like the others.
- **FeatureCluster** `phoneme surprisal` — `output_csv="phonemes"`, toggles
  `feat_phoneme_surprisal` / `feat_phoneme_cohort_entropy`, `master_phoneme_surprisal`,
  `requires_stages=("spacy_parse","mfa_align","syllabify_pairs","phoneme_surprisal")`.
- **`.mat` gating:** new Run-tab output key `phoneme_surprisal_mat` (opt-in, default OFF, like
  `acoustic_ts_mat`). When on, `main.py` rasterizes the augmented `phonemes_df` + audio duration
  into the `.mat`. Mapped in `_OUTPUT_CSV_MAP` → `phoneme` subfolder.
- Requires audio + MFA — **no text-only mode** (onsets come from alignment).

## 7. Prefs — v13 → v14 (additive)

- `features`: `master_phoneme_surprisal` (default OFF, per v15 opt-in policy),
  `feat_phoneme_surprisal`, `feat_phoneme_cohort_entropy`.
- `output`: `phoneme_surprisal_mat` (default OFF).
- New `phoneme_surprisal` config section: `{ analysis_sr: 128, strip_stress: true, primary_only: true }`.
- Migration chain `v13 → v14` deep-merges the new keys; schema-drift guard updated.
- GUI: a card on the **Sub-word features** page (`gui/section_subword_features.py`) with the two
  feature toggles + `analysis_sr` / `strip_stress` / `primary_only` params; `phoneme_surprisal_mat`
  appears as a Run-tab output toggle.

## 8. Tests

- **Hand-computable check** (the spec's own acceptance test): lexicon `{cat:3, cab:1, dog:2}`
  → for `cat`, `surprisal ≈ [0.585, 0.0, 0.415]`, `entropy ≈ [0.811, 0.811, 0.0]`. This is the
  case I verified by hand; note that using the *before* node for entropy breaks position 3
  (0 vs a wrong nonzero), so the test pins the "after" convention.
- OOV-path cap: an aligned sequence that leaves the trie → capped surprisal + 0 entropy.
- Rasterization: impulse placement (`round(onset*sr)`), grid length `N`, unit-height onset
  control, same-sample collision behavior.
- Guard flags raise on non-default.
- v13→v14 migration + schema-drift guard.
- Docs: `docs/features/phoneme_surprisal.md` (+ `phoneme_cohort_entropy.md` or shared) with
  citations (Marslen-Wilson cohort model; Brodbeck et al. phoneme surprisal in TRF; Ettinger
  et al. — verify live before citing per the citations policy).

## 9. Non-goals

- No LM-reweighted / context-dependent cohort priors (guarded off; future batch).
- No change to the GPT-2 word `Surprisal` feature or `extractors/surprisal.py`.
- No new runtime dependency.
