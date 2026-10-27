---
key: prosody_volume
name: Word-level volume (intensity) features
levels: [word]
gui_toggles: [feat_prosody_volume_mean, feat_prosody_volume_min, feat_prosody_volume_max, feat_prosody_volume_std, feat_prosody_volume_delta]
output_columns: [Volume_Mean, Volume_Min, Volume_Max, Volume_Std, Volume_Delta]
libraries: [parselmouth, praat]
status: implemented
depth: deep
---

## Meaning
Five per-word summary statistics of the **intensity contour** (loudness
in dB SPL), computed within each word's MFA-aligned acoustic window
([`Onset`, `End`](acoustic_boundaries.md)):

| Column | Statistic | Notes |
|---|---|---|
| `Volume_Mean`  | Arithmetic mean intensity (dB SPL) inside the word | Primary loudness summary. |
| `Volume_Min`   | 5th-percentile intensity | Robust min — avoids transient silence frames. |
| `Volume_Max`   | 95th-percentile intensity | Robust max — avoids transient clipping frames. |
| `Volume_Std`   | Standard deviation of intensity | Within-word loudness variability. |
| `Volume_Delta` | `Volume_Mean[i] − Volume_Mean[j]`, `j` = most recent earlier word with a `Volume_Mean` | Word-to-word loudness change; PUNCT/unaligned rows are skipped (like `Word_Gap`). `NaN` for the first aligned word. |

Units are **dB SPL** as Praat returns them. For clean studio
recordings, `Volume_Mean` typically lies in **50–75 dB SPL**. Values way outside this range usually indicate
the audio was not normalized to Praat's expected reference, not a bug
in the extractor.

Words with no aligned acoustic interval (failed alignment, PUNCT) get
`NaN` for all five columns.

## EEG rationale
Intensity is the second primary acoustic carrier of **prosodic
prominence** (along with [pitch](prosody_pitch.md)). For
naturalistic-EEG analyses:

- N100 amplitude scales linearly with sound pressure level in
  controlled auditory paradigms ([Zelger 2024]) — the loudness
  dependence of the early auditory evoked response. At the word level
  in continuous speech, louder words inherit this scaling as a direct
  acoustic-energy effect on the N100 (and the subsequent P200 in the
  N1–P2 complex).
- Intensity is one of the **acoustic cues to perceived prominence**,
  alongside F0 (pitch) and phone-rate, in cross-linguistic prominence
  ratings ([Cole 2019]).
- `Volume_Delta` captures *loudness contours across the discourse* —
  useful as a continuous regressor for amplitude-driven prediction
  models.

`Volume_Mean` is also a standard nuisance regressor: it is partly
confounded with both [pitch](prosody_pitch.md) and word duration
([acoustic_durations.md](acoustic_durations.md)), so analyses isolating
higher-order linguistic effects typically include all three.

## Math / implementation details
- Intensity is estimated by Praat as RMS amplitude smoothed over a
  short window, converted to dB SPL ([Boersma 2018]).
- Aggregates are computed over **non-NaN frames inside [Onset, End)**.
- `Volume_Min` / `Volume_Max` use the 5th and 95th percentiles for the
  same robustness reasons as the [pitch column](prosody_pitch.md).
- Single-frame words: `Volume_Std = 0`, min/max/mean all collapse to
  the same value.

## Implementation in SensiExtractor
Source: `sensi/extractors/prosody.py:add_word_prosody`. Shared
forward-pass implementation as [pitch](prosody_pitch.md) — the same
loop slices and aggregates both contours per word. `Volume_Delta` is
`Volume_Mean − Volume_Mean.ffill().shift(1)`, i.e. measured back to the
most recent word with a value, as for `Pitch_Delta`.

The two extractor columns are populated together in
`add_word_prosody`; enabling either set of toggles (or any single one)
triggers the same prosody stage. Per-column toggles control output
emission, not compute.

**Why:**

- *Robust percentiles for min/max* — same reasoning as
  [`Pitch_Min/Max`](prosody_pitch.md).
- *No reference-level normalization* — the extractor reports dB SPL
  directly from Praat without an audio-level normalization pass.
  The assumption is that input audio is already loudness-normalized
  upstream of SensiExtractor; for cross-stimulus volume comparisons,
  perform your own normalization before invoking the extractor.

## Library
```python
import parselmouth, numpy as np
snd = parselmouth.Sound("stimulus.wav")
intensity = snd.to_intensity(minimum_pitch=75)
db = intensity.values[0]                        # dB SPL per frame
times = intensity.xs()
# Per-word slice: db_in_word = db[(times >= onset) & (times < end)]
```

## Papers

**Intensity / loudness in speech processing:**
- <a name="boersma-2018"></a>**[Boersma 2018]** Boersma & Weenink. *Praat: doing phonetics by computer.* [praat.org](https://www.praat.org/) — Praat reference; intensity estimation (`To Intensity`).
- <a name="zelger-2024"></a>**[Zelger 2024]** Zelger, Seebacher, Graf, Rossi. *Is it too loud? Ask your brain!* NeuroImage 299, 120796. [doi:10.1016/j.neuroimage.2024.120796](https://doi.org/10.1016/j.neuroimage.2024.120796) — EEG study of 27 normal-hearing adults presented with 500 ms noise stimuli at 55–95 dB; N100 amplitude shows a linear dependency on sound level and on subjective loudness rating.
- <a name="cole-2019"></a>**[Cole 2019]** Cole, Hualde, Smith, Eager, Mahrt, Napoleão de Souza. *Sound, structure and meaning: The bases of prominence ratings in English, French and Spanish.* J. Phonetics 75, 113–147. [doi:10.1016/j.wocn.2019.05.002](https://doi.org/10.1016/j.wocn.2019.05.002) — listener prominence ratings in three languages are jointly driven by F0, intensity, and phone-rate; intensity is one of the converging acoustic cues to perceived prominence.

[Boersma 2018]: #boersma-2018
[Zelger 2024]: #zelger-2024
[Cole 2019]: #cole-2019

## Datasets
<!-- auto-populated from docs/datasets/*.md once the dataset registry exists -->
