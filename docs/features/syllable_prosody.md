---
key: syllable_prosody
name: Syllable-level prosody (pitch + volume + deltas)
levels: [word]
gui_toggles: [feat_syllable_pitch_mean, feat_syllable_volume_mean, feat_syllable_pitch_delta, feat_syllable_volume_delta]
output_columns: [Pitch_Mean, Volume_Mean, Pitch_Delta, Volume_Delta]
libraries: [parselmouth, praat]
status: implemented
depth: deep
---

## Meaning
Four per-syllable prosody columns in `<stim>_syllables.csv`, scaled-down
analogues of the word-level [pitch](prosody_pitch.md) and
[volume](prosody_volume.md) features:

| Column | Statistic | Notes |
|---|---|---|
| `Pitch_Mean`   | Mean F0 inside the syllable's [Onset, End) window | Same units (semitones rel. speaker median, or Hz) as the word-level pitch column. |
| `Volume_Mean`  | Mean intensity (dB SPL) inside the window | Same units as the word-level volume column. |
| `Pitch_Delta`  | `Pitch_Mean[i] − Pitch_Mean[i-1]` within the same word | NaN for the first syllable of each word. |
| `Volume_Delta` | `Volume_Mean[i] − Volume_Mean[i-1]` within the same word | NaN for the first syllable of each word. |

Only the means and per-word deltas are emitted — no min/max/std at the
syllable level, since syllable windows are short (often < 10 frames)
and percentile-based extrema would be noisy.

## EEG rationale
Syllable-level prosody is the **finest grain at which prominence
contrasts can be expressed acoustically**. For naturalistic-EEG:

- Stressed-vs-unstressed-syllable contrasts on the same word are the
  cleanest prominence-isolation paradigm; pitch accents on stressed
  syllables elicit a P200 ERP response whose amplitude is modulated by
  the onset latency of the F0 peak ([Heim 2006]).
- Syllable-onset-locked envelope tracking benefits from
  syllable-aligned pitch / intensity covariates rather than
  word-averaged ones.
- The within-word `Pitch_Delta` captures **intra-word pitch contour
  shape** — a useful regressor for studies of intonational phrase
  structure.

## Implementation in SensiExtractor
Source: `extractors/syllable_prosody.py`. Reuses the same `Contours`
object as the word- and document-level prosody extractors — one Praat
call per audio file, three different slicing windows.

**What the code does:**

1. For each syllable row, slice the pitch and intensity contours to
   `[syllable.Onset, syllable.End)`.
2. Compute the mean of each contour over valid frames.
3. Within each word group (rows sharing `doc_id, sentence_id,
   token_id`), compute `diff()` on the per-syllable means to populate
   `Pitch_Delta` / `Volume_Delta` (NaN for the first syllable of each
   word).
4. PUNCT / unaligned rows → NaN for all four columns.

The pitch-units configuration (semitones vs. Hz) is shared with the
word-level extractor — there is no separate per-extractor toggle.

## Papers
- See [`prosody_pitch`](prosody_pitch.md) and
  [`prosody_volume`](prosody_volume.md) for the foundational pitch /
  intensity references (including Boersma 2018, Steinhauer 1999,
  Heim 2006, Wagner 2010, Heilbron 2022, Zelger 2024, Cole 2019) —
  syllable-level summaries inherit the same rationale.
- <a name="heim-2006"></a>**[Heim 2006]** Heim & Alter. *Prosodic pitch accents in language comprehension and production: ERP data and acoustic analyses.* Acta Neurobiologiae Experimentalis 66(1), 55–68. [doi:10.55782/ane-2006-1587](https://doi.org/10.55782/ane-2006-1587) — P200 elicited by pitch accents on stressed syllables; amplitude modulated by F0-peak onset latency.

[Heim 2006]: #heim-2006

## Datasets
<!-- auto-populated from docs/datasets/*.md once the dataset registry exists -->
