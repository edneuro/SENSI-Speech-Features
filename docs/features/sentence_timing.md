---
key: sentence_timing
name: Sentence timing
levels: [sentence]
gui_toggles: [feat_sentence_onset, feat_sentence_end, feat_sentence_text]
output_columns: [Onset, End, Sentence]
libraries: [montreal-forced-aligner, spaCy]
status: implemented
depth: stub
---

## Meaning
Per-sentence timing and text in the `<stim>_sentences.csv` sidecar:

| Column | Meaning |
|---|---|
| `Onset`    | First word's [`Onset`](acoustic_boundaries.md) in the sentence — start of the sentence in audio, in seconds. |
| `End`      | Last word's [`End`](acoustic_boundaries.md) in the sentence — end of the sentence in audio, in seconds. |
| `Sentence` | Verbatim sentence text from spaCy (matches the [`sentence`](feat_struct_sentence_text.md) column on the word-level CSV). |

Each row joins back to the word-level CSV via `sentence_id`. Sentences
where no word aligned (very rare) → `NaN` for `Onset` / `End`.

## EEG rationale
Sentence-locked analyses look at responses anchored to **sentence
boundaries** — useful for time-locking ERPs / time-frequency analyses
to the start (or end) of a syntactic unit. Common use cases:

- Sentence-onset-locked ERPs / time-frequency analyses, especially for
  Closure Positive Shift (CPS) and discourse-level effects.
- Computing sentence-level [pitch](sentence_prosody_pitch.md) /
  [volume](sentence_prosody_volume.md) / [HNR](feat_sentence_hnr.md)
  summaries — those columns need the sentence timing to slice their
  windows.
- Grouping word-level data by sentence for sentence-level
  aggregates downstream of the SensiExtractor outputs.

## Implementation in SensiExtractor
Source: `extractors/sentence_prosody.py` (the sentence-level
extractor that builds the sidecar; despite the name, also handles the
timing columns). Implementation derives sentence timing as the
**min `Onset` / max `End`** over the words of each sentence — so the
sentence interval covers all aligned word audio for that sentence.

`Sentence` is copied from the word-level
[`sentence`](feat_struct_sentence_text.md) column (all words in a
sentence share the same value, so picking any row works).
