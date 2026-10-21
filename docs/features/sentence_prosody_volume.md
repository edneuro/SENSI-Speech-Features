---
key: sentence_prosody_volume
name: Sentence-level volume (intensity) features
levels: [sentence]
gui_toggles: [feat_sentence_volume_mean, feat_sentence_volume_min, feat_sentence_volume_max, feat_sentence_volume_std, feat_sentence_volume_delta]
output_columns: [Volume_Mean, Volume_Min, Volume_Max, Volume_Std, Volume_Delta]
libraries: [parselmouth, praat]
status: implemented
depth: deep
---

## Meaning
Five per-sentence summaries of intensity (dB SPL), in the same units
as the [word-level volume](prosody_volume.md) columns:

| Column | Statistic |
|---|---|
| `Volume_Mean`  | Mean intensity inside the sentence's `[Onset, End)`. |
| `Volume_Min`   | 5th-percentile intensity. |
| `Volume_Max`   | 95th-percentile intensity. |
| `Volume_Std`   | Standard deviation of intensity within the sentence. |
| `Volume_Delta` | Sentence-to-sentence change in `Volume_Mean`, measured back to the most recent sentence with a value (sentences with no aligned word are skipped). `NaN` for the first aligned sentence. |

## EEG rationale
Sentence-level volume summaries serve the same purposes as
[sentence-level pitch](sentence_prosody_pitch.md): stimulus matching
and prosody-aware regression covariates. They are partly confounded
with sentence-level `Pitch_Std` and with the sentence's word count —
so analyses should include both.

`Volume_Delta` captures the discourse-level "loudness arc" of a
narrative (rising during climaxes, falling during calm sections);
useful as a continuous regressor for engagement-modulated EEG/MEG
responses.

## Implementation in SensiExtractor
Source: `extractors/sentence_prosody.py`. Same Praat intensity
contour as the word- and document-level extractors; sliced to each
sentence's `[Onset, End)`. The aggregation function
(`_agg(slice) = (mean, p5, p95, std)`) is shared with the word-level
volume extractor.

## Papers
See [`prosody_volume`](prosody_volume.md) — sentence-level summaries
inherit the same foundational references (Boersma 2018, Zelger 2024,
Cole 2019).
