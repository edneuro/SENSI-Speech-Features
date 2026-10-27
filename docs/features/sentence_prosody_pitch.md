---
key: sentence_prosody_pitch
name: Sentence-level pitch features
levels: [sentence]
gui_toggles: [feat_sentence_pitch_mean, feat_sentence_pitch_min, feat_sentence_pitch_max, feat_sentence_pitch_std, feat_sentence_pitch_delta]
output_columns: [Pitch_Mean, Pitch_Min, Pitch_Max, Pitch_Std, Pitch_Delta]
libraries: [parselmouth, praat]
status: implemented
depth: deep
---

## Meaning
Five per-sentence summaries of F0 across each sentence's audio window
([`Onset, End`](sentence_timing.md)), in the same units as the
[word-level pitch](prosody_pitch.md) columns:

| Column | Statistic |
|---|---|
| `Pitch_Mean`  | Mean F0 across voiced frames in the sentence. |
| `Pitch_Min`   | 5th-percentile F0 in the sentence (robust min). |
| `Pitch_Max`   | 95th-percentile F0 in the sentence (robust max). |
| `Pitch_Std`   | Standard deviation of F0 in the sentence. |
| `Pitch_Delta` | `Pitch_Mean[i] − Pitch_Mean[j]`, `j` = most recent earlier sentence with a `Pitch_Mean` — sentence-to-sentence pitch change. `NaN` for the first aligned sentence. |

## EEG rationale
Sentence-level pitch summaries are the **prosodic counterpart to the
sentence-level [BERT](master_bert_sentence.md) embedding** — they
describe the prosodic shape of a sentence the way the embedding
describes its meaning.

Useful for:
- **Stimulus matching at the sentence grain** — when pairing sentences
  for cross-sentence ERP contrasts, equalizing `Pitch_Mean` and
  `Pitch_Std` controls for gross prosodic differences.
- **Discourse-level intonation analyses** — sentence-to-sentence
  pitch movement (`Pitch_Delta`) captures rising/falling sentence
  intonation patterns that mark interrogative vs. declarative, topic
  shifts, and information structure ([Wagner 2010]).
- **Speaker-normalization checks** — `Pitch_Std` near zero across all
  sentences usually indicates Praat's pitch tracker failed (e.g.
  whispered speech, unvoiced material); `Pitch_Std` proportional to
  speaker variability is the healthy signature.

See [`prosody_pitch`](prosody_pitch.md) for the full theoretical and
methodological rationale — the only difference is the aggregation
window (sentence vs. word).

## Implementation in SensiExtractor
Source: `sensi/extractors/sentence_prosody.py`. Uses the same `Contours`
object as the word- and document-level prosody extractors; slices to
`[Onset, End)` per sentence and computes the same `_agg(slice) =
(mean, p5, p95, std)` tuple as the word-level extractor. `Pitch_Delta`
is `Pitch_Mean − Pitch_Mean.ffill().shift(1)` across sentences — a
sentence where no word aligned is skipped rather than turning the next
sentence's delta into NaN.

`Pitch_Min` / `Pitch_Max` use 5th / 95th percentiles for the same
robustness reason described in [`prosody_pitch`](prosody_pitch.md) —
octave-jump errors from Praat's autocorrelation tracker shouldn't
dominate the per-sentence extrema.

## Papers
See [`prosody_pitch`](prosody_pitch.md) — sentence-level summaries
inherit the same foundational references (Boersma 2018, Steinhauer 1999,
Heim 2006, Wagner 2010, Heilbron 2022).
