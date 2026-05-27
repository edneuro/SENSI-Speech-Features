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
  cleanest prominence-isolation paradigm and produce reliable
  early-pitch-tracking responses ([Park 2020], [Kayser 2015]).
- Syllable-onset-locked envelope tracking benefits from
  syllable-aligned pitch / intensity covariates rather than
  word-averaged ones.
- The within-word `Pitch_Delta` captures **intra-word pitch contour
  shape** — a useful regressor for studies of intonational phrase
  structure ([Cole 2017]).

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
  intensity references — syllable-level summaries inherit the same
  rationale.
- <a name="park-2020"></a>**[Park 2020]** Park, Ince, Schyns, Thut, Gross. *Frontal top-down signals increase coupling of auditory low-frequency oscillations to continuous speech.* Current Biology. [doi:10.1016/j.cub.2014.11.075](https://doi.org/10.1016/j.cub.2014.11.075)
- <a name="kayser-2015"></a>**[Kayser 2015]** Kayser, Ince, Gross, Kayser. *Irregular speech rate dissociates auditory cortical entrainment, evoked responses, and frontal alpha.* J. Neuroscience. [doi:10.1523/JNEUROSCI.0890-15.2015](https://doi.org/10.1523/JNEUROSCI.0890-15.2015)
- <a name="cole-2017"></a>**[Cole 2017]** Cole, Hualde, Smith, Eager, Mahrt, Napoleão de Souza. *Sound, structure and meaning: The bases of prominence ratings.* J. Phonetics. [doi:10.1016/j.wocn.2018.05.002](https://doi.org/10.1016/j.wocn.2018.05.002)

[Park 2020]: #park-2020
[Kayser 2015]: #kayser-2015
[Cole 2017]: #cole-2017

## Datasets
<!-- auto-populated from docs/datasets/*.md once the dataset registry exists -->
