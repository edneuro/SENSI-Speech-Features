---
key: feat_struct_sentence_text
name: Sentence text
levels: [word]
gui_toggle: feat_struct_sentence_text
output_columns: [sentence]
libraries: [spaCy]
status: implemented
depth: stub
---

## Meaning
The verbatim text of the sentence containing each word, including
punctuation and original whitespace. Every word row in the same
`sentence_id` carries the same `sentence` string — redundant by design,
so downstream consumers can join sentence-level context without a
separate lookup.

## Usage
- Display context for a word during manual inspection of the word-level
  CSV.
- Reconstruct sentences for sentence-level analysis without needing the
  sentence-level sidecar.
- Combine with [`start` / `end`](structural_offsets.md) to reproduce the
  surface token from its sentence: `sentence[start-1:end]`.

## Implementation in SensiExtractor
Source: `extractors/text_parser.py:parse` — emitted directly from
`spaCy`'s `sent.text` for each sentence span.
