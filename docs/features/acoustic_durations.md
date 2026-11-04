---
key: acoustic_durations
name: Acoustic word duration and gaps
levels: [word]
gui_toggles: [feat_acoustic_word_duration, feat_acoustic_word_gap, feat_acoustic_word_gap_after]
output_columns: [Word_Duration, Word_Gap, Word_Gap_After]
libraries: [montreal-forced-aligner]
status: implemented
depth: shallow
---

## Meaning
Three derived timing columns, all in **seconds**, computed from
[`Onset` / `End`](acoustic_boundaries.md):

| Column | Formula | Meaning |
|---|---|---|
| `Word_Duration` | `End − Onset` | How long the word took to articulate. |
| `Word_Gap`      | `Onset[i] − End[j]`, `j` = most recent earlier spoken word | Silence **before** the word: from the end of the previous aligned word to the start of this one. `NaN` for the first aligned word of the audio. |
| `Word_Gap_After` | `Onset[k] − End[i]`, `k` = next spoken word | Silence **after** the word: from its end to the start of the next aligned word. Equal to the next word's `Word_Gap`. `NaN` for the last aligned word of the audio. |

Punctuation tokens (which carry no acoustic interval) are skipped over, so
the gap across a comma is still measured and lands on the words on either
side of it. MFA's word intervals do not overlap, so neither gap is ever
negative.

**Contractions.** spaCy splits a contraction into parts (*He'd* → `He` +
`'d`, *didn't* → `did` + `n't`), but the speaker says one word and MFA
times it as one, so both rows carry the same `Onset` / `End`. The **first
part holds the word's gaps**; the later parts have `NaN` in both gap
columns, because there is no silence between them. (In the
[word events .mat](word_events_mat.md) the parts are merged into one
spoken word, which keeps the first part's gaps.)

## EEG rationale
- **`Word_Duration`** is a covariate for any word-locked analysis that
  averages across words of different lengths — longer words have more
  acoustic content within the analysis window and can dominate
  averaged responses if not controlled. It is also an index of
  *speech rate at the word level*; conversational speaking rate is
  itself sensitive to turn length, topic, and demographics
  ([Yuan 2006]).
- **`Word_Gap`** and **`Word_Gap_After`** capture inter-word silence —
  including pauses, hesitations, and disfluency gaps. Pauses chunk
  continuous speech into phrasal units, and prosodic boundaries
  elicit a closure positive shift (CPS) at the EEG level
  ([Steinhauer 1999]).
- ⚠️ The two columns answer different questions about the same pause.
  `Word_Gap` asks what the listener heard just *before* a word (a pause
  that may build anticipation or mark a boundary); `Word_Gap_After`
  asks whether a word *ended* a phrase (a pause that follows it, as at
  a phrase-final word). They hold the same numbers shifted by one word,
  so use the one that matches your question, or both, rather than
  treating them as independent measures.

For naturalistic-EEG analyses, both columns serve as **nuisance
regressors** when isolating higher-order linguistic effects from raw
acoustic-temporal structure.

## Implementation in SENSI SpeechFeatures
These columns are added downstream of
[`Onset` / `End`](acoustic_boundaries.md) (from MFA). The
derivation lives in `sensi/extractors/timing.py` — both are simple
arithmetic on the alignment output. `Word_Duration` is NaN-aligned
with the underlying boundaries (a word that didn't align has NaN
`Onset`, `End`, and `Word_Duration`).

`Word_Gap` measures back to the **most recent spoken word with a known
`End`**, and `Word_Gap_After` forward to the **next one with a known
`Onset`**: punctuation rows and (rarely) words MFA could not align are
skipped over rather than producing a NaN gap on the neighbouring word.
Consequence: if an unaligned word sits between two aligned ones, the
gaps on either side span the unaligned word — "time since the last
known word ended" / "time until the next known word starts", not
"duration of pure silence" in that rare case. A row is a contraction's
later part when it repeats the previous aligned row's exact `Onset`
and `End`.

## Papers
- <a name="yuan-2006"></a>**[Yuan 2006]** Yuan, Liberman, Cieri. *Towards an integrated understanding of speaking rate in conversation.* Interspeech 2006, pp. 541–544. [ISCA Archive](https://www.isca-archive.org/interspeech_2006/yuan06_interspeech.html) — speaking rate / word duration in conversational speech.
- <a name="steinhauer-1999"></a>**[Steinhauer 1999]** Steinhauer, Alter, Friederici. *Brain potentials indicate immediate use of prosodic cues in natural speech processing.* Nature Neuroscience 2(2), 191–196. [doi:10.1038/5757](https://doi.org/10.1038/5757) — closure-positive-shift (CPS) at prosodic boundaries.

[Yuan 2006]: #yuan-2006
[Steinhauer 1999]: #steinhauer-1999

## Datasets
<!-- auto-populated from docs/datasets/*.md once the dataset registry exists -->
