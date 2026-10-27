# P-center Feature — Design Spec

**Date:** 2026-05-20
**Status:** Approved
**Depends on:** Phoneme/Syllable batch (2026-05-13) — `syllabify_pairs` + `phonemes_df` must be available.

---

## 1. What is P-center?

The **Perceptual Center** (P-center) of a syllable is the internal subjective "beat" — the moment a listener perceives the syllable as having occurred. It differs from the acoustic onset (start of the first phoneme) and is empirically closest to the vowel nucleus onset, shifted back by a fraction of the preceding consonant cluster duration.

Reference: Marcus (1981), Scott (1993). Used in speech rhythm research and as a timing landmark for EEG stimulus tagging.

---

## 2. Formula

```
P_Center = nucleus_onset − k × onset_consonant_duration
```

| Term | Definition |
|---|---|
| `nucleus_onset` | Onset time (seconds) of the first vowel phoneme in the syllable. In ARPABET, vowels are labels ending in a stress digit (`0`, `1`, `2`). |
| `onset_consonant_duration` | Sum of durations of all phonemes before the nucleus (the onset cluster). Zero for vowel-initial syllables. |
| `k` | Scott 1993 coefficient. Default `0.3`. User-configurable via `prefs["prosody"]["pcenter_k"]`. |

Result is rounded to 4 decimal places (matching `Duration` in `syllables_df`).

### Edge cases

| Situation | Handling |
|---|---|
| Vowel-initial syllable (no onset consonants) | `onset_consonant_duration = 0` → `P_Center = nucleus_onset` |
| No-vowel syllable (`nucleus = None`, e.g. "hmm") | `P_Center = syllable.Onset` (acoustic onset fallback) |
| Missing MFA alignment (`Onset` is NaN) | `P_Center = NaN` |

---

## 3. Output

`P_Center` is appended as a column in `<stim>_syllables.csv` — the existing sub-word sidecar. One value per syllable row. No new CSV is introduced.

Consumers wanting a word-level P-center can take the first syllable per `token_id` from the syllables CSV.

---

## 4. New file: `extractors/pcenter.py`

Single public function:

```python
def add_pcenter(
    syllables_df: pd.DataFrame,
    phonemes_df: pd.DataFrame,
    *,
    k: float = 0.3,
) -> pd.DataFrame:
    """Append P_Center column to syllables_df.

    For each syllable, P_Center = nucleus_onset - k * onset_consonant_duration.
    NaN when MFA alignment is missing. Falls back to syllable Onset when no
    vowel nucleus exists.
    """
```

**Algorithm:**
1. Group `phonemes_df` by `(doc_id, sentence_id, token_id, syllable_id)`.
2. For each group:
   a. Identify nucleus = first row whose `Label` ends in a digit.
   b. `onset_consonant_duration` = sum of `Duration` of rows before the nucleus.
   c. Apply formula; handle edge cases.
3. Build a lookup Series keyed by the four-column composite index.
4. Map onto `syllables_df` (left join) → `P_Center` column.
5. Return `syllables_df.copy()` with the new column.

No Parselmouth, no external model. Pure pandas/numpy.

`k` is passed by the planner's `emit.py:to_namespace` adapter as `args.pcenter_k`, following the same pattern as `args.pitch_units` for prosody. The Stage runner in `main.py` extracts it and forwards it as a keyword argument when calling `add_pcenter`.

---

## 5. Planner integration (`gui/planner/registry.py`)

### New Stage

```python
Stage(
    name="pcenter",
    consumes=("syllables_df", "phonemes_df"),
    produces=("syllables_df.P_Center",),
    run=add_pcenter,
)
```

### New FeatureCluster

```python
FeatureCluster(
    name="p-center",
    toggle_keys=("feat_syllable_pcenter",),
    output_csv="syllables",
    requires_stages=("spacy_parse", "mfa_align", "syllabify_pairs", "pcenter"),
    emits_columns=("P_Center",),
    master_toggle="master_syllable_pcenter",
)
```

---

## 6. Preferences (v7 → v8, additive)

New keys — all additive; migration is a no-op copy (v8 defaults fill the gaps):

| Section | Key | Default | Notes |
|---|---|---|---|
| `features` | `feat_syllable_pcenter` | `True` | Per-column toggle for P_Center in syllables CSV |
| `features` | `master_syllable_pcenter` | `True` | Master toggle for the P-center cluster card |
| `prosody` | `pcenter_k` | `0.3` | Scott 1993 coefficient; valid range 0–1 |

Migration: `_migrate_v7_into_v8` copies all existing sections and leaves new keys at their v8 defaults (same pattern as every prior additive migration).

---

## 7. GUI

- **`gui/section_word_features.py`** — Sub-word page, Syllables card: add one `ToggleRow` for `feat_syllable_pcenter` ("P-center"), adjacent to the syllable prosody toggles. Master toggle `master_syllable_pcenter` drives a new card or sub-section header.
- **`gui/section_run.py`** (prosody settings panel): add a `pcenter_k` spin-box alongside `pitch_floor_hz` / `pause_threshold_s`. Label: "P-center k coefficient (0–1)". Step 0.05, range 0.0–1.0.

No changes to `main.py`, `syllabify.py`, or any existing extractor.

---

## 8. Tests

**`tests/extractors/test_pcenter.py`** — synthetic DataFrames, no audio:

| Test | Fixture | Assertion |
|---|---|---|
| Standard onset cluster | `[B, AE1, T]`, `k=0.3` | `P_Center = AE1.onset − 0.3 × B.duration` |
| Vowel-initial syllable | `[AE1, T]` | `P_Center = AE1.onset` |
| No-vowel syllable | `[HH, M]` | `P_Center = syllable.Onset` |
| Multi-consonant onset | `[S, T, R, AY1]` | correction = sum of S+T+R durations × k |
| NaN propagation | `Onset = NaN` | `P_Center = NaN` |
| k=0 | any syllable | `P_Center = nucleus_onset` exactly |
| Two-word stimulus | 4 syllables across 2 tokens | correct FK grouping, no cross-word bleed |

**`tests/test_preferences.py`** — v7→v8 migration round-trip asserts `pcenter_k` defaults to `0.3`.

The existing planner schema-drift guard catches missing prefs keys automatically.

---

## 9. Touch-points summary

| File | Change |
|---|---|
| `extractors/pcenter.py` | **New** — `add_pcenter` function |
| `gui/planner/registry.py` | +1 Stage, +1 FeatureCluster |
| `utils/preferences.py` | v8 constants + `_migrate_v7_into_v8` |
| `gui/section_word_features.py` | +1 ToggleRow + master toggle in Syllables card |
| `gui/section_run.py` | +1 spin-box for `pcenter_k` |
| `tests/extractors/test_pcenter.py` | **New** — 7 unit tests |
| `tests/test_preferences.py` | +1 migration test |
| `docs/feature_dependencies.md` | Add §2.4 P-center entry |
