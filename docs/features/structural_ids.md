---
key: structural_ids
name: Structural identifier columns
levels: [word]
gui_toggles: [feat_struct_doc_id, feat_struct_paragraph_id, feat_struct_sentence_id, feat_struct_term_id, feat_struct_token_id]
output_columns: [doc_id, paragraph_id, sentence_id, term_id, token_id]
libraries: [spaCy]
status: implemented
depth: stub
---

## Meaning
Bookkeeping columns that uniquely locate each word within the document
hierarchy. All are **1-indexed integers**.

| Column | What it indexes |
|---|---|
| `doc_id` | Document. Currently always `1` — single-document pipeline. Placeholder for future multi-stimulus runs. |
| `paragraph_id` | Paragraph within the document. Currently always `1` — paragraph segmentation is not yet implemented; reserved for future use. |
| `sentence_id` | Sentence within the document, in spaCy reading order. First sentence is `1`. |
| `term_id` | Global token index across the entire document. First token is `1`. |
| `token_id` | Token index **within the current sentence**. Resets to `1` at each sentence boundary. |

## Usage
- Group by `sentence_id` to compute sentence-level aggregates from the
  word-level CSV (matches the sentence sidecar's `sentence_id`).
- Use `term_id` for whole-document linear position.
- Use `(sentence_id, token_id)` together to refer to a word by its
  position inside its sentence — this is what the dependency parse
  (`head_token_id`) refers to.

## Library
Derived from spaCy's `Doc.sents` iteration over the `en_core_web_sm`
pipeline; `term_id` and `token_id` are assigned by the extractor as
1-based counters during the sentence walk.

## Implementation in SensiExtractor
Source: `extractors/text_parser.py:parse`.

`doc_id` and `paragraph_id` are hardcoded to `1` in the extractor —
multi-document and paragraph-aware processing aren't part of the current
pipeline. The other three counters are populated during the spaCy
sentence walk: `sentence_id` from `enumerate(doc.sents, start=1)`,
`token_id` from `enumerate(sent, start=1)`, and `term_id` as a global
counter that increments across sentences.
