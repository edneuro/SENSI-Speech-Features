---
key: acoustic_durations
name: Acoustic word duration and gap
levels: [word]
gui_toggles: [feat_acoustic_word_duration, feat_acoustic_word_gap]
output_columns: [Word_Duration, Word_Gap]
libraries: [montreal-forced-aligner]
status: implemented
depth: shallow
---

## Meaning
Two derived timing columns, both in **seconds**, computed from
[`Onset` / `End`](acoustic_boundaries.md):

| Column | Formula | Meaning |
|---|---|---|
| `Word_Duration` | `End − Onset` | How long the word took to articulate. |
| `Word_Gap`      | `Onset[i] − End[i-1]` | Silence between the end of the previous word and the start of this one. `NaN` for the first word of the audio. May be negative for slightly overlapping words. |

## EEG rationale
- **`Word_Duration`** is a covariate for any word-locked analysis that
  averages across words of different lengths — longer words have more
  acoustic content within the analysis window and can dominate
  averaged responses if not controlled. It is also an index of
  *speech rate at the word level*; conversational speaking rate is
  itself sensitive to turn length, topic, and demographics
  ([Yuan 2006]).
- **`Word_Gap`** captures inter-word silence — including pauses,
  hesitations, and disfluency gaps. Pauses chunk continuous speech
  into phrasal units, and longer pre-word gaps elicit larger
  closure-positive-shifts (CPS) at the EEG level ([Steinhauer 1999]).

For naturalistic-EEG analyses, both columns serve as **nuisance
regressors** when isolating higher-order linguistic effects from raw
acoustic-temporal structure.

## Implementation in SensiExtractor
These columns are added downstream of
[`Onset` / `End`](acoustic_boundaries.md) (from MFA). The
derivation lives in `main.py`'s word-batch finishing step — both are
simple arithmetic on the alignment output, so the columns are
guaranteed to be NaN-aligned with the underlying acoustic boundaries
(e.g. if a word didn't align, all four — `Onset`, `End`,
`Word_Duration`, `Word_Gap` — are absent or NaN together).

`Word_Gap` for the first word of the audio is `NaN` (no previous
word). Negative values arise occasionally when MFA assigns
slightly-overlapping boundaries to fast-spoken words — these are
faithful to the acoustic data, not bugs.

## Papers
- <a name="yuan-2006"></a>**[Yuan 2006]** Yuan, Liberman, Cieri. *Towards an integrated understanding of speaking rate in conversation.* Interspeech 2006, pp. 541–544. [ISCA Archive](https://www.isca-archive.org/interspeech_2006/yuan06_interspeech.html) — speaking rate / word duration in conversational speech.
- <a name="steinhauer-1999"></a>**[Steinhauer 1999]** Steinhauer, Alter, Friederici. *Brain potentials indicate immediate use of prosodic cues in natural speech processing.* Nature Neuroscience 2(2), 191–196. [doi:10.1038/5757](https://doi.org/10.1038/5757) — closure-positive-shift (CPS) at prosodic boundaries.

[Yuan 2006]: #yuan-2006
[Steinhauer 1999]: #steinhauer-1999

## Datasets
<!-- auto-populated from docs/datasets/*.md once the dataset registry exists -->
