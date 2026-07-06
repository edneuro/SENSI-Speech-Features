# Phoneme surprisal + cohort entropy — design spec

**Date:** 2026-07-06
**Status:** design approved, pending spec review → plan → implement
**Level:** Sub-word (phoneme)
**Deps:** audio + MFA alignment (required); `wordfreq` + the MFA dictionary (both already in-repo — no new deps)

---

## 1. What we're adding

Context-independent **phoneme surprisal** and **cohort entropy** as phoneme-onset-aligned
regressors for TRF / cortical-tracking analyses, plus unit-height **phoneme onset** and
**word onset** controls (see §2b — Brodbeck-style; the two controls together let a downstream
analyst reproduce the standard word-onset + non-word-initial-phoneme-onset control set).
All four are emitted as impulse trains in a `.mat` sidecar; the two scalar features are
additionally written as per-phoneme columns on the existing phoneme CSV.

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

**OOV path** (aligned sequence steps off the trie — a pronunciation **not in the MFA
dictionary at all**, e.g. a rare variant the aligner produced): from the first missing phoneme
onward, `surprisal = cap` and `entropy = 0`, where `cap = −log2(f_min / Z_root)` (the surprisal
of the single rarest lexicon word). Never `log(0)`. This is distinct from a merely low-frequency
word (§3) — those stay in the trie with a floored frequency.

## 2b. Onset controls

Following Brodbeck et al. (2018/2022), which include both a word-onset control and a
phoneme-onset control (the latter at non-word-initial phonemes only, to avoid collinearity):
the `.mat` emits **two** unit-height control trains — `phoneme_onset` (impulse at *every*
phoneme onset) and `word_onset` (impulse at each word-initial phoneme onset). Emitting both
(rather than a single all-phoneme onset) lets the analyst either use them directly or recover
Brodbeck's exact set (`word_onset` plus non-word-initial phoneme onset = `phoneme_onset −
word_onset`). Word boundaries are already known from the per-word grouping of `phonemes_df`.

## 3. Sources (reuse, no new deps)

- **Pronunciation lexicon:** `mfa_data/pretrained_models/dictionary/english_us_arpa.dict` — the
  exact dictionary MFA aligns against, so the trie's phoneme inventory matches the aligned
  `phones` tier. Format: `word\tPH PH PH` (one line per pronunciation variant).
- **Frequency f(w):** `wordfreq.word_frequency(w, "en")` (linear), the same SUBTLEX-US-backed
  source behind the existing `Zipf_Freq` column, giving each word a prior proportional to its
  frequency (Brodbeck/Gillis convention). Dictionary words absent from wordfreq (`f==0`) are
  **floored** to `f_min` (the minimum nonzero frequency observed across the lexicon), *not*
  excluded — so the full MFA lexicon stays in the trie and the OOV cap fires only for
  pronunciations genuinely absent from the dictionary (see §2). `f_min` doubles as the cap
  numerator, tying the two together.
- **Aligned pronunciation:** MFA `phones` tier → `parse_phones` → the per-word phoneme sequence
  already grouped by `syllabify_pairs` (the `phonemes_df` rows for a word, in order). This is
  "the variant the aligner chose."

## 4. Extractor — `extractors/phoneme_surprisal.py` (pure functions, TDD)

- `load_mfa_lexicon(dict_path, *, strip_stress=True, primary_only=True) -> list[tuple[tuple[str,...], float]]`
  Parse the MFA `.dict`; `strip_stress` drops the trailing ARPABET stress digit (`EH1`→`EH`);
  `primary_only` keeps only the first-listed variant per word. Join each word to `f(w)`.
- `build_cohort_trie(entries) -> CohortTrie` — computes `f_min` (min nonzero freq), floors any
  `f==0` word to `f_min`, then builds the trie; nodes cache `z`, `sfl`; also exposes `z_root`
  and `f_min` for the cap. Built once, cached per run (module-level cache keyed by
  `(dict_path, strip_stress, primary_only)`).
- `phoneme_scores(trie, phones, *, context_independent=True, first_phoneme_empty_context=True)`
  `-> (surprisals: list[float], entropies: list[float])`. `phones` is the stress-stripped
  (per `strip_stress`) aligned phoneme sequence for one word. The two keyword flags are
  **guard flags**: any non-default value raises `NotImplementedError` (LM-reweighted /
  non-empty-context modes are a future batch). They are **not** prefs/GUI keys — they document
  the modeling commitment at the API level without adding permanent prefs-schema surface (cf.
  the repo's regret over vestigial prefs keys like `master_acoustic_ts`).
- `rasterize(events, duration_s, analysis_sr) -> dict[str, np.ndarray]` — build the four
  length-`N` impulse trains (`N = round(duration_s * analysis_sr)`), impulses at
  `round(onset_s * analysis_sr)` using MATLAB-style rounding consistent with
  `acoustic_ts._mround`. `events` = list of `(onset_s, surprisal, entropy, is_word_initial)`.
  Same-sample collisions: value trains (`phoneme_surprisal`, `cohort_entropy`) sum;
  `phoneme_onset` / `word_onset` stay unit height (1).

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
- `phoneme_surprisal`, `cohort_entropy` — two value-modulated impulse trains, each intended
  for its own banded-ridge band downstream.
- `phoneme_onset`, `word_onset` — two unit-height control trains (§2b).
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
- Frequency floor: a dictionary word absent from wordfreq stays in the trie at `f_min` (its
  aligned path does *not* hit the OOV cap).
- Rasterization: impulse placement (`round(onset*sr)`), grid length `N`, unit-height
  `phoneme_onset` (all phonemes) and `word_onset` (word-initial only), same-sample collision
  behavior.
- Guard flags raise on non-default.
- v13→v14 migration + schema-drift guard.
- Docs: `docs/features/phoneme_surprisal.md` (+ `phoneme_cohort_entropy.md` or shared) with
  the §8b citation set (verify live at authoring time per the citations policy).

## 8b. Research fidelity — how this matches the literature

Verified against the primary sources during design (live-fetched 2026-07-06):

- **Surprisal formula.** Gillis et al. 2021 (*J Neurosci* 41(50):10316): `surprisal_i =
  −log2(freq(cohort_i)/freq(cohort_{i−1}))` — identical to our `log2(Z_before) − log2(Z_after)`.
- **Entropy convention.** Cohort entropy is `−Σ p·log2 p` over the cohort matching phonemes
  "from word onset **to** the current phoneme" (includes the current phoneme) — our **after-node**
  evaluation. (Using the *before* node breaks the hand-check at position 3, §8.)
- **Frequency-weighted prior.** Brodbeck et al. 2022 (*eLife* e72056): "the prior for each word
  is proportional to its frequency" (COCA); Gillis uses SUBTLEX-NL. We use SUBTLEX-US via
  `wordfreq` — same method, US norms.
- **First phoneme.** Brodbeck models it with the word-initial (empty-context) prior: "At word
  onset, the prior for each word is proportional to its frequency… With each subsequent phoneme,
  [inconsistent words] set to 0." We follow Brodbeck; Gillis instead *drops* the first phoneme.
  Our `first_phoneme_empty_context` guard flag records this choice.
- **Onset controls.** Brodbeck: "All models included control predictors for word onsets… and
  phoneme onsets (equal value impulse at the onset of each non-word-initial phoneme)." We emit
  `word_onset` + all-phoneme `phoneme_onset` so both this exact set and an all-phoneme control
  are recoverable (§2b).
- **Representation & rate.** Both papers place "impulses at phoneme onsets modulated by the
  value" and analyze at **128 Hz** — our impulse-train `.mat` and default `analysis_sr=128`.
- **Separate predictors.** Each feature enters as its own predictor with its own TRF — our
  one-band-per-regressor layout.

Citations to place in the feature doc(s), all live-verifiable (verify again at authoring time
per the citations policy): Marslen-Wilson & Welsh 1978 (cohort model); Brodbeck, Hong & Simon
2018 (*Current Biology*); Brodbeck et al. 2022 (*eLife* e72056); Gillis, Vanthornhout, Simon,
Francart & Di Liberto 2021 (*J Neurosci* 41(50):10316); Donhauser & Baillet 2020 (*Neuron*).

## 9. Non-goals

- No LM-reweighted / context-dependent cohort priors (guarded off; future batch).
- No change to the GPT-2 word `Surprisal` feature or `extractors/surprisal.py`.
- No new runtime dependency.
