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
| `Volume_Delta` | `Volume_Mean[i] − Volume_Mean[i-1]` | Word-to-word loudness change. `NaN` for first word. |

Units are **dB SPL** as Praat returns them. For clean studio
recordings, `Volume_Mean` typically lies in **50–75 dB SPL** (range
from `master_plan.md`). Values way outside this range usually indicate
the audio was not normalized to Praat's expected reference, not a bug
in the extractor.

Words with no aligned acoustic interval (failed alignment, PUNCT) get
`NaN` for all five columns.

## EEG rationale
Intensity is the second primary acoustic carrier of **prosodic
prominence** (along with [pitch](prosody_pitch.md)). For
naturalistic-EEG analyses:

- Louder words mark focused / emphasized constituents and elicit
  larger early auditory components (N100/P200) — direct effects of
  acoustic energy ([Sanders 2003], [Picton 2013]).
- Volume modulation contributes to prominence-driven attentional
  effects, even after pitch is regressed out ([Cole 2017]).
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
Source: `extractors/prosody.py:add_word_prosody`. Shared
forward-pass implementation as [pitch](prosody_pitch.md) — the same
loop slices and aggregates both contours per word.

The two extractor columns are populated together in
`add_word_prosody`; enabling either set of toggles (or any single one)
triggers the same prosody stage. Per-column toggles control output
emission, not compute.

**Why** *(⚠️ flags my interpolation):*

- *Robust percentiles for min/max* — same reasoning as
  [`Pitch_Min/Max`](prosody_pitch.md).
- *No reference-level normalization* — the extractor reports dB SPL
  directly from Praat without an audio-level normalization pass.
  ⚠️ The assumption is that input audio is already loudness-normalized
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
- <a name="sanders-2003"></a>**[Sanders 2003]** Sanders & Neville. *An ERP study of continuous speech processing II. Segmentation, semantics, and syntax in non-native speakers.* Cognitive Brain Research. [doi:10.1016/S0926-6410(03)00112-0](https://doi.org/10.1016/S0926-6410(03)00112-0)
- <a name="picton-2013"></a>**[Picton 2013]** Picton. *Hearing in time: Evoked potential studies of temporal processing.* Ear and Hearing. [doi:10.1097/AUD.0b013e31827ada02](https://doi.org/10.1097/AUD.0b013e31827ada02)
- <a name="cole-2017"></a>**[Cole 2017]** Cole, Hualde, Smith, Eager, Mahrt, Napoleão de Souza. *Sound, structure and meaning: The bases of prominence ratings in English, French and Spanish.* J. Phonetics. [doi:10.1016/j.wocn.2018.05.002](https://doi.org/10.1016/j.wocn.2018.05.002)

[Boersma 2018]: #boersma-2018
[Sanders 2003]: #sanders-2003
[Picton 2013]: #picton-2013
[Cole 2017]: #cole-2017

## Datasets
<!-- auto-populated from docs/datasets/*.md once the dataset registry exists -->
