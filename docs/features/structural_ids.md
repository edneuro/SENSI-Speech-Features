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
| `doc_id` | Document. `1` for a single-document run; in batch mode each document gets its sequential 1-indexed position in the batch (1, 2, 3, …). Set explicitly via the CLI `--doc-id N` flag for one-off runs. |
| `paragraph_id` | Paragraph within the document. Paragraphs are detected as runs of text separated by one or more blank lines (`\n\n`) in the transcript. A transcript with no blank lines is a single paragraph (`paragraph_id == 1` throughout). |
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

`doc_id` is a parameter on `parse(text, doc_id=1)` — the CLI/GUI batch
loops in `main.py` set it to each pair's 1-indexed position in the batch
(`args.doc_id = i + 1`), and one-off CLI runs can override the default
via `--doc-id N`.

`paragraph_id` is computed from blank-line boundaries in the raw
transcript: `_paragraph_spans` collects `(start_char, end_char)` for
each paragraph (separators matched by `\n[ \t]*\n+`), and each
sentence's `paragraph_id` is looked up by its `sent.start_char`. A
transcript with no blank lines yields a single span covering the whole
text, so `paragraph_id` stays at `1`. The aligner path (which
normalises whitespace before feeding MFA) is unaffected — parser and
aligner each see the raw text first.

The other three counters are populated during the spaCy sentence walk:
`sentence_id` from `enumerate(doc.sents, start=1)`, `token_id` from
`enumerate(sent, start=1)`, and `term_id` as a global counter that
increments across sentences.
