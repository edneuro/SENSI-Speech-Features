---
key: acoustic_boundaries
name: Acoustic word boundaries (MFA)
levels: [word]
gui_toggles: [feat_acoustic_onset, feat_acoustic_end]
output_columns: [Onset, End]
libraries: [montreal-forced-aligner, kaldi]
status: implemented
depth: deep
---

## Meaning
The actual acoustic start and end times of each word in the audio,
in **seconds from audio onset**, from forced alignment of the transcript
against the audio with [Montreal Forced Aligner (MFA)](https://montreal-forced-aligner.readthedocs.io/).

| Column | Meaning |
|---|---|
| `Onset` | Time the word begins, in seconds from the start of the audio. |
| `End`   | Time the word ends, in seconds from the start of the audio. |

Words that did not align (silence, OOV, hesitations not in the
transcript) are dropped — they don't appear as rows in the word-level
CSV at all. PUNCT rows are present (from spaCy parsing) but their
`Onset` / `End` are `NaN`.

> ⚠️ Capitalized `Onset` / `End` (seconds) are **not the same** as the
> lowercase [`start` / `end`](structural_offsets.md) columns (character
> offsets within the sentence). Easy to confuse — two coordinate
> systems, different units.

## EEG rationale
Accurate word-onset times in audio are the **anchor for every
word-locked EEG/MEG analysis** of naturalistic listening. The N400,
P200, and other word-locked components are all defined relative to
audio onset, and ms-level errors in `Onset` translate directly into
smeared / mis-aligned event-related responses.

MFA's onset estimates are derived from **phoneme-level acoustic models**
([McAuliffe 2017]), so they reflect real acoustic boundaries — not
heuristic guesses from text-to-time interpolation. Published
naturalistic-EEG / MEG analyses of continuous speech rely on this kind
of alignment precision to regress neural responses against per-word
linguistic features ([Broderick 2018], [Heilbron 2022]).

`End` is the symmetric per-word audio offset; word duration
([`Word_Duration`](acoustic_durations.md)) is `End − Onset`.

## Implementation in SensiExtractor
Source: `extractors/aligner.py:align_with_phones`. TextGrid parsing:
`utils/textgrid_parser.py:parse_words`.

**What the code does:**

1. Prepares the transcript for MFA via
   `utils.transcript_prep:prepare_for_aligner` — number-spelling
   (`"42" → "forty-two"`), whitespace normalization, and any other
   tokenizer-friendly fixes MFA expects.
2. Writes the audio and the prepared transcript to a temporary
   directory in the layout MFA expects (`.wav` + matching `.lab`
   file).
3. Invokes MFA as a subprocess against the
   `english_us_arpa` acoustic model and `english_us_arpa`
   dictionary (Kaldi-based, ships via conda).
4. Parses the resulting `.TextGrid` file with
   `utils.textgrid_parser:parse_words`, which extracts every
   non-silence word interval and returns one row per word with
   `Transcript / Onset / End`.
5. The temp directory is cleaned up after the run.

This is also the **upstream step** for the syllable and phoneme
sidecars and for the prosody extractors — they all consume the same
`.TextGrid` (via `parse_phones` for the sub-word tier).

**Why:**

- *MFA over Gentle / Aeneas / heuristic alignment* — MFA is the
  current state-of-the-art for acoustic forced alignment with active
  maintenance and good documentation
  ([McAuliffe 2017]). Heuristic time-warp alignment (e.g.
  scaling text to audio duration) introduces 50–200 ms errors that
  smear EEG responses unacceptably for word-level analysis.
- *English-US ARPA acoustic + dictionary models* — chosen for spoken English stimuli (audiobook, podcast, spoken narration, etc).
  Other locales would require swapping the model names and possibly
  the dictionary.
- *Silence intervals dropped* — they would otherwise appear as `<eps>`
  or empty-label rows in the TextGrid; downstream code is simpler if
  the word-level CSV contains only actual words.

## Library
```python
# Manual MFA invocation
mfa align <corpus_dir> english_us_arpa english_us_arpa <output_dir>

# Programmatic use in this project:
from extractors.aligner import align_with_phones
words_df, phones_df = align_with_phones("stimulus.wav", transcript_text)
# words_df columns: Transcript, Onset, End
```

## Papers
- <a name="mcauliffe-2017"></a>**[McAuliffe 2017]** McAuliffe, Socolof, Mihuc, Wagner, Sonderegger. *Montreal Forced Aligner: trainable text-speech alignment using Kaldi.* Interspeech 2017. [doi:10.21437/Interspeech.2017-1386](https://doi.org/10.21437/Interspeech.2017-1386) — the MFA tool itself.
- <a name="broderick-2018"></a>**[Broderick 2018]** Broderick, Anderson, Di Liberto, Crosse, Lalor. *Electrophysiological Correlates of Semantic Dissimilarity Reflect the Comprehension of Natural, Narrative Speech.* Current Biology 28(5), 803–809. [doi:10.1016/j.cub.2018.01.080](https://doi.org/10.1016/j.cub.2018.01.080) — regresses EEG against per-word semantic dissimilarity in continuous narrative speech.
- <a name="heilbron-2022"></a>**[Heilbron 2022]** Heilbron, Armeni, Schoffelen, Hagoort, De Lange. *A hierarchy of linguistic predictions during natural language comprehension.* PNAS 119(32), e2201968119. [doi:10.1073/pnas.2201968119](https://doi.org/10.1073/pnas.2201968119) — regresses EEG/MEG against GPT-2 surprisal in naturalistic audiobook listening.

[McAuliffe 2017]: #mcauliffe-2017
[Broderick 2018]: #broderick-2018
[Heilbron 2022]: #heilbron-2022

## Datasets
<!-- auto-populated from docs/datasets/*.md once the dataset registry exists -->
