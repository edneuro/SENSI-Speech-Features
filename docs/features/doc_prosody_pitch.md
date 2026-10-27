---
key: doc_prosody_pitch
name: Document-level pitch summaries
levels: [document]
gui_toggles: [feat_doc_prosody_pitch_mean, feat_doc_prosody_pitch_min, feat_doc_prosody_pitch_max, feat_doc_prosody_pitch_std, feat_doc_prosody_pitch_delta_mean]
output_columns: [doc_pitch_mean, doc_pitch_min, doc_pitch_max, doc_pitch_std, doc_pitch_delta_mean]
libraries: [parselmouth, praat]
status: implemented
depth: deep
---

## Meaning
Five document-level summaries of F0 across the entire stimulus, in the
same units as the [word-level pitch](prosody_pitch.md) columns
(semitones relative to speaker median by default; Hz if configured).

| Column | What it summarizes |
|---|---|
| `doc_pitch_mean`        | Mean F0 across all voiced frames in the audio. |
| `doc_pitch_min`         | 5th-percentile F0 over the audio. |
| `doc_pitch_max`         | 95th-percentile F0 over the audio. |
| `doc_pitch_std`         | Standard deviation of F0 across the audio. |
| `doc_pitch_delta_mean`  | Mean of the **absolute** word-level [`Pitch_Delta`](prosody_pitch.md) values (rises and falls don't cancel). |

The first four operate on the **raw F0 contour** (all voiced frames),
not on the per-word means — so `doc_pitch_mean` ≠ `mean(Pitch_Mean)`
when some words contain few voiced frames.

## EEG rationale
Document-level pitch statistics are **stimulus-matching covariates**:
two stimuli that differ in mean pitch (e.g. a male vs. a female
narrator) will produce systematically different early auditory ERPs
that confound any higher-order linguistic contrast. Including
`doc_pitch_*` lets you regress out gross speaker-level pitch
differences without ascribing them to lexical or syntactic effects.

`doc_pitch_std` is a single-number prosody-variability summary —
useful as a covariate when matching stimuli on expressiveness across
recording sessions or speakers.

`doc_pitch_delta_mean` differs from the other four: it's the **average
word-to-word movement magnitude** (unsigned) rather than a contour-frame statistic. Useful
when the question is about overall prosodic dynamism rather than range.

## Implementation in SensiExtractor
Source: `sensi/extractors/prosody.py:compute_doc_prosody` — see the
document-batch finishing step in `sensi/pipeline.py` which calls it.

`doc_pitch_*` (other than the delta) come from a single Praat call
over the full audio file via the same `Contours` object used by the
word-level extractor. Percentile-based min/max for the same
robustness reason as the word-level columns: octave-jump errors
otherwise dominate true extrema.

`doc_pitch_delta_mean` is computed in pandas as
`word_df["Pitch_Delta"].abs().mean()` (NaN-aware) — the mean absolute
delta, so a rise followed by an equal fall counts as movement rather
than cancelling to zero. It requires the
word-level prosody stage to have already run (the planner enforces
this).

## Library
See [word-level pitch features](prosody_pitch.md) — same Parselmouth
calls, just aggregated over the whole audio rather than per-word
windows.

## Papers
See [`prosody_pitch`](prosody_pitch.md) for the foundational
references; the document-level summaries inherit the same theoretical
rationale.

## Datasets
<!-- auto-populated from docs/datasets/*.md once the dataset registry exists -->
