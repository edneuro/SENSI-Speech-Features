# Phoneme / Syllable Batch — Design Spec

**Date:** 2026-05-13
**Author / brainstorming partner:** Amilcar + Claude Code
**Active batch in master_plan §4:** Phoneme / Syllable level (top roadmap item; blocks p-center).

---

## 1. Context

SensiExtractor today emits two artifacts per stimulus:

- `<stim>_features.csv` — one row per word/token
- one row in the master `documents_summary_<ts>.csv`

The next roadmap step introduces a **multi-level data model**: per-syllable and per-phoneme rows. Until now everything has fit into the word-level DataFrame as additional columns. Phoneme/syllable data doesn't — each word has multiple syllables and many phonemes. This batch is therefore the first to break the "one row per token" output shape.

MFA's TextGrid already contains both tiers (`words` and `phones`), and `utils/textgrid_parser.py:parse_phones()` already extracts the phoneme tier into a `(phone, onset, offset)` DataFrame. The pipeline simply hasn't been calling it. So the phoneme side is mostly a wiring problem; the syllable side is the substantive new work (MFA does not emit a syllable tier — we group phonemes ourselves).

## 2. Goals

1. Emit two new sidecar CSVs per stimulus: `<stim>_syllables.csv` and `<stim>_phonemes.csv`, with foreign-key columns linking each row back to its parent word in `<stim>_features.csv`.
2. Syllable-level prosody (mean pitch, mean volume, deltas) reusing the existing Parselmouth `Contours` object — no second audio decode per stimulus.
3. Per-column toggles on a new "Sub-word features" sidebar page, consistent with the Word-level / Document-level conventions established in Batch 3.
4. Sidecar-level master switches on the Run page so an analyst who only wants `<stim>_features.csv` doesn't get the new files.

## 3. Non-goals (deferred)

- **P-center.** Its own future batch. Belongs as either a new column on the syllable CSV (`P_Center`) or a separate sidecar — decision deferred. P-center is acoustically computed (Villing 2010-style intensity-rise within the syllable interval); the syllabification algorithm chosen here does not bottleneck p-center accuracy because vowel positions are unambiguous regardless of where syllable boundaries land.
- **Phoneme class features** (voicing / manner / place). Easy follow-up using a static ARPABET → class lookup; deferred unless a specific analysis needs them.
- **Doc-level aggregates over syllables.** No new master-CSV columns in this batch. The existing `doc_articulation_rate` currently uses pyphen-derived syllable counts; *upgrading* it to use MFA-derived syllables is a behavior change worth its own decision and is out of scope here.
- **Sentence-level or paragraph-level CSVs.** Out of scope.
- **Alternative syllabification algorithms** (NIST tsylb2, sonority sequencing principle, stress-sensitive variants, ambisyllabicity). MOP only for this batch. Algorithm is encapsulated behind a function boundary so a future swap is contained.

## 4. Output shape

### 4.1 `<stim>_syllables.csv`

One row per syllable. Columns:

| Group | Column | Type | Notes |
|---|---|---|---|
| Context / FK | `doc_id` | str | matches the word CSV |
| | `sentence_id` | int | matches the word CSV |
| | `token_id` | int | identifies the parent word |
| | `token` | str | the word string itself, for analyst readability |
| Identity | `syllable_id` | int | 1-indexed, **per-word** (resets each word) |
| Label | `Label` | str | orthographic syllable (pyphen-derived). Falls back to ARPABET-without-stress when pyphen's syllable count for the word disagrees with the phoneme-derived count |
| | `ARPABET` | str | dash-joined ARPABET phonemes including stress digit on the vowel, e.g. `B-IH0`, `T-W-ER0-K` |
| Timing | `Onset` | float | seconds; first phoneme's onset |
| | `End` | float | seconds; last phoneme's end |
| | `Duration` | float | `End - Onset` |
| Prosody | `Pitch_Mean` | float | mean of Parselmouth pitch contour over `[Onset, End]` |
| | `Volume_Mean` | float | mean of intensity contour over `[Onset, End]` |
| | `Pitch_Delta` | float | `Pitch_Mean(i) - Pitch_Mean(i-1)`; NaN on first syllable of the stimulus |
| | `Volume_Delta` | float | same, for volume |

Joining back to the word CSV: `pd.merge(syllables_df, word_df, on=['doc_id', 'sentence_id', 'token_id'])`.

### 4.2 `<stim>_phonemes.csv`

One row per phoneme. Columns:

| Group | Column | Type | Notes |
|---|---|---|---|
| Context / FK | `doc_id` | str | |
| | `sentence_id` | int | |
| | `token_id` | int | parent word |
| | `token` | str | word string |
| | `syllable_id` | int | parent syllable (FK into syllables CSV) |
| Identity | `phoneme_id` | int | 1-indexed, **per-syllable** |
| Label | `Label` | str | ARPABET with stress digit, e.g. `EH1`, `T`, `OW0`. Same string MFA emits. |
| Timing | `Onset` | float | seconds; from MFA's `phones` tier |
| | `End` | float | seconds; from MFA |
| | `Duration` | float | `End - Onset` |

Joining: `pd.merge(phonemes_df, syllables_df, on=['doc_id', 'sentence_id', 'token_id', 'syllable_id'])`.

### 4.3 Context columns are mandatory

`doc_id`, `sentence_id`, `token_id`, `token`, `syllable_id`, `phoneme_id` are emitted unconditionally — without them the sidecar is unusable. Only the **Label**, **Timing**, and **Prosody** groups are per-column toggleable.

## 5. Syllabification algorithm — Maximum Onset Principle (MOP)

### 5.1 Rationale

There is no single dominant convention in EEG-language research. Three camps exist: phoneme-only (skip syllabification), acoustic-syllable (intensity-peak via Praat / de Jong & Wempe), and phonological-syllable (MOP / NIST tsylb2 / forced-alignment-based). The phonological camp is the only one that fits a sidecar-CSV data model with phoneme→syllable foreign-key linkage; MOP is its standard. The "wrong" cases (e.g., `/nɛ.twɜrk/` rather than `/nɛt.wɜrk/`) shift the syllable boundary by ~50 ms but do not move the vowel and therefore do not move the eventual p-center. Citable to Selkirk 1984 and Kahn 1976.

### 5.2 Algorithm

```python
def syllabify_word(phonemes: list[Phone]) -> list[Syllable]:
    if not phonemes:
        return []                                   # word with no aligned phones

    nucleus_indices = [i for i, p in enumerate(phonemes)
                       if p.label[-1].isdigit()]    # vowels have stress digit

    if not nucleus_indices:
        return [Syllable(phonemes=phonemes)]        # no vowel → single syllable

    # Max Onset: between nucleus_i and nucleus_{i+1}, all interludial
    # consonants belong to syllable i+1's onset. The cut lands JUST AFTER
    # nucleus_i.
    cut_after = nucleus_indices[:-1]
    starts = [0, *(c + 1 for c in cut_after)]
    ends   = [*(c + 1 for c in cut_after), len(phonemes)]
    return [Syllable(phonemes=phonemes[s:e]) for s, e in zip(starts, ends)]
```

### 5.3 Verified examples

| Word | Phonemes | Nucleus indices | Syllables |
|---|---|---|---|
| `below` | `[B, IH0, L, OW1]` | `[1, 3]` | `[B, IH0]`, `[L, OW1]` |
| `network` | `[N, EH1, T, W, ER0, K]` | `[1, 4]` | `[N, EH1]`, `[T, W, ER0, K]` |
| `string` | `[S, T, R, IH1, NG]` | `[3]` | `[S, T, R, IH1, NG]` (single syllable) |
| `lamps` | `[L, AE1, M, P, S]` | `[1]` | `[L, AE1, M, P, S]` (single syllable) |
| `idea` | `[AY1, D, IY0, AH0]` | `[0, 2, 3]` | `[AY1]`, `[D, IY0]`, `[AH0]` |
| `hmm` (no vowel) | `[HH, M]` | `[]` | `[HH, M]` (whole sequence, with warning) |

### 5.4 Label assignment

After syllabification produces N syllable rows for a word:

1. Run `pyphen.Pyphen(lang='en_US').inserted(word)` to get pyphen's orthographic split, split on the hyphen → M fragments.
2. **If N == M**: assign fragments 1:1 in order. `Label` = the fragment.
3. **If N != M**: fall back to ARPABET-without-stress per syllable (strip the digit from the vowel). `Label` = e.g. `B-IH`, `L-OW`. Increment a per-stimulus counter; log final count at finalize: `5 words had pyphen/phoneme syllable count mismatch — Label fell back to ARPABET-no-stress`.
4. **If pyphen is not importable**: use ARPABET-no-stress for everything. Single warning at run start: `pyphen not available; syllable Labels will be ARPABET-without-stress instead of orthographic`.

The `ARPABET` column is always populated regardless of pyphen status — it's derived directly from the syllable's phoneme rows.

## 6. Code organization

| File | Status | Purpose |
|---|---|---|
| `utils/textgrid_parser.py` | unchanged | `parse_phones()` already exists |
| `extractors/syllabify.py` | new (~80 LoC) | `syllabify_word()`, `assign_labels()`, `Phone` / `Syllable` dataclasses |
| `extractors/syllable_prosody.py` | new (~60 LoC) | `add_syllable_prosody(syll_df, contours)` — slices `Contours` by `[Onset, End]`; mirrors `extractors/prosody.py:add_word_prosody` signature |
| `main.py` | modified | new CLI flags `--no-syllables` / `--no-phonemes` and per-column drop flags; `run_word_level_batch` per-pair loop writes two new CSVs |
| `gui/section_subword_features.py` | new (~120 LoC) | `SubwordFeaturesSection`: two cards (Syllables, Phonemes), per-column toggles |
| `gui/main_window.py` | modified (~5 LoC) | new sidebar entry between "Document-level features" and "Run" |
| `gui/run_dispatch.py` | modified (~25 LoC) | propagate new toggles + new no_* flags through `build_args_word_level` |
| `gui/section_run.py` | modified (~10 LoC) | new "Syllables CSV" and "Phonemes CSV" output checkboxes |
| `gui/preflight.py` | modified (~15 LoC) | warning if pyphen missing AND label toggle on; warning if alignment off AND any sub-word feature on |
| `utils/preferences.py` | modified (~40 LoC) | schema v5 → v6, additive, new keys default to True |

Pipeline call site inside `run_word_level_batch` (sketch):

```python
# After MFA alignment, before/alongside writing the word CSV:
if not args.no_syllables or not args.no_phonemes:
    from extractors.syllabify import syllabify_pairs
    from extractors.syllable_prosody import add_syllable_prosody
    phonemes_df, syllables_df = syllabify_pairs(
        words_df=df, phone_df=align_df_phones,
    )
    if not args.no_syllables:
        syllables_df = add_syllable_prosody(syllables_df, args._contours,
                                            pitch_units=args.pitch_units)
        drop = _syllable_columns_to_drop(args.feature_toggles)
        syllables_df.drop(columns=[c for c in drop if c in syllables_df.columns]
                         ).to_csv(output_dir / f'{stim_id}_syllables.csv',
                                  index=False)
    if not args.no_phonemes:
        drop = _phoneme_columns_to_drop(args.feature_toggles)
        phonemes_df.drop(columns=[c for c in drop if c in phonemes_df.columns]
                        ).to_csv(output_dir / f'{stim_id}_phonemes.csv',
                                 index=False)
```

The new per-column-drop helpers (`_syllable_columns_to_drop`, `_phoneme_columns_to_drop`) follow the existing pattern from `_word_columns_to_drop`.

## 7. Prefs schema v5 → v6

Additive only. Migration is `_migrate_v5_into_v6` deep-merging defaults; new keys default to `True` for features and `True` for output toggles (sidecars default-on, consistent with how word-level CSV and doc summary default-on).

```python
_FEATURES_DEFAULT_V6 = {
    **_FEATURES_DEFAULT_V5,
    # Syllable per-column
    "feat_syllable_onset":        True,
    "feat_syllable_end":          True,
    "feat_syllable_duration":     True,
    "feat_syllable_label":        True,
    "feat_syllable_arpabet":      False,   # opt-in; verbose
    "feat_syllable_pitch_mean":   True,
    "feat_syllable_volume_mean":  True,
    "feat_syllable_pitch_delta":  True,
    "feat_syllable_volume_delta": True,
    # Phoneme per-column
    "feat_phoneme_label":         True,
    "feat_phoneme_onset":         True,
    "feat_phoneme_end":           True,
    "feat_phoneme_duration":      True,
}

_OUTPUT_DEFAULT_V6 = {
    **_OUTPUT_DEFAULT_V5,
    "syllables_csv": True,
    "phonemes_csv":  True,
}
```

Identical chain to existing migrations: v1→v2→v3→v4→v5→v6, with each intermediate also chainable to v6.

## 8. GUI surface

### 8.1 Sidebar

```
Inputs  •  Word-level features  •  Document-level features  •  Sub-word features ◄ NEW  •  Run  •  Logs
```

### 8.2 "Sub-word features" page

Two cards:

```
┌─ Syllables ─────────────────────────────────────────────────────────┐
│ One row per syllable. Requires MFA alignment.                       │
│                                                                     │
│ Timing            ☐ Onset    ☐ End      ☐ Duration                  │
│ Label             ☐ Label (orthographic, pyphen-derived)            │
│                   ☐ ARPABET (dash-joined, with stress digits)       │
│ Prosody           ☐ Pitch_Mean  ☐ Volume_Mean                       │
│                   ☐ Pitch_Delta ☐ Volume_Delta                      │
└─────────────────────────────────────────────────────────────────────┘

┌─ Phonemes ──────────────────────────────────────────────────────────┐
│ One row per phoneme. Requires MFA alignment.                        │
│                                                                     │
│ Label             ☐ Label (ARPABET incl. stress digit)              │
│ Timing            ☐ Onset  ☐ End  ☐ Duration                        │
└─────────────────────────────────────────────────────────────────────┘
```

### 8.3 Run page additions

Two new checkboxes alongside `Word-level CSV` and `Document-level summary`:

- **Syllables CSV** (default on) — gates `<stim>_syllables.csv` emission
- **Phonemes CSV** (default on) — gates `<stim>_phonemes.csv` emission

## 9. Dependencies and blocking rules

| Condition | Effect |
|---|---|
| `feat_syllable_onset` AND `feat_syllable_end` both off | `feat_syllable_duration` greyed out; prosody columns greyed out (no interval to slice). |
| `feat_syllable_pitch_mean` off | `feat_syllable_pitch_delta` greyed out. |
| `feat_syllable_volume_mean` off | `feat_syllable_volume_delta` greyed out. |
| `feat_phoneme_onset` AND `feat_phoneme_end` both off | `feat_phoneme_duration` greyed out. |
| MFA alignment off entirely (no acoustic_* toggle on) | Both sidecars skipped; preflight warns. |
| `parselmouth` not importable | Syllable prosody columns emit as NaN with a single warning per run. Timing columns still emit. |
| `pyphen` not importable | Syllable `Label` falls back to ARPABET-no-stress; preflight warns. |
| MFA could not align a word | No syllable / phoneme rows emitted for that word. Per-stimulus counter in the run log. |

## 10. Tests

Per the project's TDD-scope feedback (`memory/feedback-tdd-scope.md`): extractor math uses TDD upfront; GUI / dispatcher / prefs wiring tests are added after implementation.

**Upfront (TDD)** — `extractors/syllabify.py`, `extractors/syllable_prosody.py`:

- `test_syllabify.py`:
  - ~12 hand-checked phoneme sequences (CV, CVC, CCVCC, vowel-initial, no-vowel, single-phoneme, polysyllabic, stress variants)
  - Roundtrip: `syllabify(parse_phones(real_textgrid_fixture))` produces non-overlapping intervals covering all non-silence phonemes
  - Label assignment: pyphen-match path, pyphen-mismatch fallback path, pyphen-missing fallback path
  - Edge cases: empty input, single-phoneme word, MFA dropped vowels
- `test_syllable_prosody.py`:
  - Synthetic contour + synthetic syllable intervals → expected mean / delta values
  - NaN propagation when contour is missing
  - Delta NaN on first syllable

**Test-after** — main.py wiring, dispatcher, GUI section, preflight, prefs migration:

- `test_pipeline_batch.py`: assert both sidecar files appear with non-zero rows for the fixture stimulus
- `test_run_dispatch.py`: toggle propagation cases
- `test_preflight.py`: new warning cases
- `test_preferences_migration.py`: v5 → v6 additive migration

## 11. Out-of-scope / follow-ups

Listed in §3 above. Each gets its own brainstorm → spec → plan cycle:

1. **P-center batch** — depends on this one; introduces acoustic p-center detection inside syllable intervals.
2. **Phoneme class features** (voicing/manner/place) — additive columns on phonemes CSV.
3. **Stress-sensitive syllabification** or **NIST tsylb2 integration** — drop-in replacement for `syllabify_word()`.
4. **MFA-derived articulation_rate** — replace the current pyphen-based count in doc-level prosody; behavior change requiring its own decision.
5. **Doc-level aggregates over syllables** — mean syllable duration, syllable rate, etc.

## 12. Risks

- **Pyphen / phoneme syllable-count mismatch frequency** is empirically uncertain — could be 5%, could be 20% depending on the stimulus vocabulary. The fallback to ARPABET-no-stress handles it gracefully but may surface confusion for analysts seeing two label conventions in the same column. The run-log counter at finalize is the mitigation; revisit if real-stimulus testing shows the mismatch rate is high enough to warrant a different policy.
- **MFA alignment quality** in the underlying audio gates everything downstream. Words MFA couldn't align produce no syllable / phoneme rows. Documented but worth surfacing in the run log per stimulus.
- **Schema v5 → v6 migration** must be additive only (the project convention). Tested via the existing migration-chain test pattern.

---

*End of spec. Implementation plan to be drafted via the writing-plans skill.*
