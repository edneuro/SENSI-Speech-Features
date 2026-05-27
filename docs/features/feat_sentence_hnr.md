---
key: feat_sentence_hnr
name: Sentence-level Harmonics-to-Noise Ratio (HNR)
levels: [sentence]
gui_toggle: feat_sentence_hnr
output_columns: [HNR]
libraries: [parselmouth, praat]
status: implemented
depth: shallow
---

## Meaning
Mean **Harmonics-to-Noise Ratio**, in dB, across all voiced frames
inside each sentence's audio window. See
[`feat_doc_prosody_hnr`](feat_doc_prosody_hnr.md) for the formula
and interpretation — `feat_sentence_hnr` is the same quantity
computed over a sentence-length window rather than the whole document.

Typical values for clean speech: **10–25 dB**. Lower values indicate
breathier or noisier phonation.

## EEG rationale
Per-sentence HNR captures **voice-quality variation across the
discourse** — a stimulus where some sentences are whispered and others
shouted will have wildly varying sentence-level HNR even with
constant `doc_hnr`. Useful for:

- Identifying sentences with degraded voice quality that may need to
  be excluded from analyses.
- Sentence-level voice-quality covariates for emotion / affect
  studies, where prosody and voice quality co-vary with content
  ([Banse 1996], [Patel 2018]).

## Implementation in SensiExtractor
Source: `extractors/sentence_prosody.py`. Same Praat harmonicity
contour as the document-level extractor (the same
`To Harmonicity (cc)` call) — sliced per sentence and averaged with
the unvoiced sentinel (−200 dB) masked out, identical to the
[document-level extractor](feat_doc_prosody_hnr.md).

## Papers
See [`feat_doc_prosody_hnr`](feat_doc_prosody_hnr.md) — the
sentence-level column inherits the same foundational references
(Boersma 1993, Banse 1996, Patel 2018).
