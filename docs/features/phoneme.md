---
key: phoneme
name: Phoneme-level columns
levels: [word]
gui_toggles: [feat_phoneme_label, feat_phoneme_onset, feat_phoneme_end, feat_phoneme_duration]
output_columns: [Label, Onset, End, Duration]
libraries: [montreal-forced-aligner]
status: implemented
depth: deep
---

## Meaning
One row per phone (phoneme) in `<stim>_phonemes.csv`. Each row joins
back to its word and syllable via
`(doc_id, sentence_id, token_in_sentence_id, syllable_id, phoneme_id)`.

| Column | Meaning |
|---|---|
| `Label`    | ARPABET phoneme symbol. Vowels carry a trailing stress digit: `AE1` (primary), `AE2` (secondary), `AE0` (unstressed). Consonants are plain (`T`, `K`, `S`). |
| `Onset`    | Time the phone begins, in seconds from audio start. |
| `End`      | Time the phone ends, in seconds from audio start. |
| `Duration` | `End − Onset`, in seconds. |

## EEG rationale
Phoneme-level alignment is the most-precise speech-unit anchor
available; for naturalistic-EEG / MEG:

- **Phoneme-onset-locked early auditory responses** are the standard
  anchor for fine-grained acoustic-tracking analyses ([Khalighinejad 2017],
  [Brodbeck 2018]).
- **Phoneme-class regressors** (e.g. plosive vs. fricative vs.
  vowel) reveal feature-specific neural responses that word-level
  analyses average over ([Mesgarani 2014]).
- **Stress-marked vowels** (`Label` ending in `1`) identify
  primary-stressed nuclei, which are the anchor points for
  stress-tracking analyses.
- Phoneme `Duration` is the foundation for [`P_Center`](feat_syllable_pcenter.md)
  — which depends on the duration of onset consonants preceding the
  syllable's nucleus.

## Implementation in SensiExtractor
Source: `extractors/syllabify.py:syllabify_pairs` (downstream of MFA
alignment). Parses the phones tier of MFA's TextGrid via
`utils.textgrid_parser:parse_phones` and emits one row per non-silent
phone.

The columns are ARPABET as MFA produces them — no remapping,
no IPA conversion. PUNCT and unaligned words contribute no rows.
`phoneme_id` is a per-syllable 1-indexed counter (so `phoneme_id == 1`
is the first phone of the syllable, `2` the second, etc.).

## Library
ARPABET is the phoneme tagset shipped with MFA's `english_us_arpa`
dictionary; the alignment step writes them directly into the TextGrid
`phones` tier. No third-party library is needed beyond MFA itself.

## Papers
- <a name="khalighinejad-2017"></a>**[Khalighinejad 2017]** Khalighinejad, Cruzatto da Silva, Mesgarani. *Dynamic encoding of acoustic features in neural responses to continuous speech.* J. Neuroscience. [doi:10.1523/JNEUROSCI.2383-16.2017](https://doi.org/10.1523/JNEUROSCI.2383-16.2017)
- <a name="brodbeck-2018"></a>**[Brodbeck 2018]** Brodbeck, Hong, Simon. *Rapid transformation from auditory to linguistic representations of continuous speech.* Current Biology. [doi:10.1016/j.cub.2018.10.042](https://doi.org/10.1016/j.cub.2018.10.042)
- <a name="mesgarani-2014"></a>**[Mesgarani 2014]** Mesgarani, Cheung, Johnson, Chang. *Phonetic feature encoding in human superior temporal gyrus.* Science. [doi:10.1126/science.1245994](https://doi.org/10.1126/science.1245994)

[Khalighinejad 2017]: #khalighinejad-2017
[Brodbeck 2018]: #brodbeck-2018
[Mesgarani 2014]: #mesgarani-2014

## Datasets
<!-- auto-populated from docs/datasets/*.md once the dataset registry exists -->
