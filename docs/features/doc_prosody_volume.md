---
key: doc_prosody_volume
name: Document-level volume (intensity) summaries
levels: [document]
gui_toggles: [feat_doc_prosody_volume_mean, feat_doc_prosody_volume_min, feat_doc_prosody_volume_max, feat_doc_prosody_volume_std, feat_doc_prosody_volume_delta_mean]
output_columns: [doc_volume_mean, doc_volume_min, doc_volume_max, doc_volume_std, doc_volume_delta_mean]
libraries: [parselmouth, praat]
status: implemented
depth: deep
---

## Meaning
Five document-level summaries of intensity (dB SPL) across the entire
stimulus, in the same units as the
[word-level volume](prosody_volume.md) columns:

| Column | What it summarizes |
|---|---|
| `doc_volume_mean`        | Mean intensity across the audio. |
| `doc_volume_min`         | 5th-percentile intensity. |
| `doc_volume_max`         | 95th-percentile intensity. |
| `doc_volume_std`         | Standard deviation of intensity. |
| `doc_volume_delta_mean`  | Mean of the **absolute** word-level [`Volume_Delta`](prosody_volume.md) values (rises and falls don't cancel). |

The first four operate on the **raw intensity contour** (all frames),
the last on word-to-word deltas (averaged as magnitudes).

## EEG rationale
- `doc_volume_mean` is a **gross-level acoustic confound** — sound
  level drives the amplitude of the N100 component of the auditory
  evoked response in a linear fashion ([Zelger 2024]), so loudness
  differences across stimuli will be reflected in early ERPs
  irrespective of any linguistic contrast. Either equalize loudness
  during stimulus preparation or include this column as a regression
  nuisance.
- `doc_volume_std` captures dynamic-range variability — useful as a
  matching covariate when comparing studio-quality vs.
  in-the-wild recordings.
- `doc_volume_delta_mean` is the document-level analogue of the
  word-level `Volume_Delta` average (unsigned) — captures how much loudness
  *changes* from word to word, a continuous proxy for prosodic
  expressiveness.

## Implementation in SensiExtractor
See [word-level volume features](prosody_volume.md) — the document-level
columns reuse the same Praat intensity contour, aggregated over the
full audio. `doc_volume_delta_mean` is the NaN-aware mean of the
**absolute** word-level `Volume_Delta` values
(`word_df["Volume_Delta"].abs().mean()`), so rises and falls don't
cancel.

## Papers
[Zelger 2024]: prosody_volume.md#zelger-2024

See [`prosody_volume`](prosody_volume.md) — document-level summaries
inherit the same foundational references (Boersma 2018, Zelger 2024,
Cole 2019).

## Datasets
<!-- auto-populated from docs/datasets/*.md once the dataset registry exists -->
