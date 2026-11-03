---
key: structural_ids
name: Structural identifier columns
levels: [word]
gui_toggles: [feat_struct_doc_id, feat_struct_paragraph_id, feat_struct_sentence_id, feat_struct_token_id, feat_struct_token_in_sentence_id]
output_columns: [doc_id, paragraph_id, sentence_id, token_id, token_in_sentence_id]
libraries: [spaCy]
status: implemented
depth: stub
---

## Meaning
Bookkeeping columns that uniquely locate each word within the document
hierarchy. All are **1-indexed integers**.

| Column | What it indexes |
|---|---|
| `doc_id` | Document. `1` for a single-document run; in batch mode each document gets its sequential 1-indexed position in the batch (1, 2, 3, …). |
| `paragraph_id` | Paragraph within the document. Paragraphs are detected as runs of text separated by one or more blank lines (`\n\n`) in the transcript. A transcript with no blank lines is a single paragraph (`paragraph_id == 1` throughout). |
| `sentence_id` | Sentence within the document, in spaCy reading order. First sentence is `1`. |
| `token_id` | Global token index across the entire document. First token is `1`. Counts **emitted tokens only** — see below. |
| `token_in_sentence_id` | Token index **within the current sentence**. Resets to `1` at each sentence boundary. A true sentence position — see below. |

### Whitespace tokens are not rows

spaCy emits a literal newline in the transcript as its own token
(`upos == 'SPACE'`, text `"\n"`). The parser skips these tokens
entirely, so a hard line wrap does not produce a word-CSV row and does
not consume a `token_id` or a `token_in_sentence_id`. Consequences:

- `token_id` and `token_in_sentence_id` count **emitted tokens only** and
  are contiguous with no gaps.
- `token_in_sentence_id` **is** a true position within its sentence —
  word *n* of the sentence, as printed.
- `paragraph_id` is computed from **blank lines**, not single newlines, so
  a hard wrap inside a sentence is not a paragraph break: both halves keep
  the same `sentence_id` and `paragraph_id`.
- `head_token_id` resolves correctly; see the note in Implementation.

## Usage
- Group by `sentence_id` to compute sentence-level aggregates from the
  word-level CSV (matches the sentence sidecar's `sentence_id`).
- Use `token_id` for whole-document linear position. This is also what
  the dependency parse (`head_token_id`) refers to: a word's head is
  named by the head's global `token_id`, so `token_id` is the anchor
  needed to resolve the dependency tree.
- Use `(sentence_id, token_in_sentence_id)` together to refer to a word by
  its position inside its sentence. This pair is the foreign key the
  sub-word (syllable / phoneme) sidecars join back on.

## Library
Derived from spaCy's `Doc.sents` iteration over the `en_core_web_sm`
pipeline; `token_id` and `token_in_sentence_id` are assigned by the
extractor as 1-based counters during the sentence walk.

## Implementation in SENSI SpeechFeatures
Source: `sensi/extractors/text_parser.py:parse`.

`doc_id` is a parameter on `parse(text, doc_id=1)` — the batch loops in
`sensi/pipeline.py` set it to each pair's 1-indexed position in the batch
(`args.doc_id = i + 1`).

`paragraph_id` is computed from blank-line boundaries in the raw
transcript: `_paragraph_spans` collects `(start_char, end_char)` for
each paragraph (separators matched by `\n[ \t]*\n+`), and each
sentence's `paragraph_id` is looked up by its `sent.start_char`. A
transcript with no blank lines yields a single span covering the whole
text, so `paragraph_id` stays at `1`. The aligner path (which
normalises whitespace before feeding MFA) is unaffected — parser and
aligner each see the raw text first.

The other three counters are populated during the spaCy sentence walk:
`sentence_id` from `enumerate(doc.sents, start=1)`, `token_in_sentence_id`
as a per-sentence counter over non-whitespace tokens, and `token_id` as a
global counter that increments across sentences. Tokens for which
`token.is_space` is true are skipped and never become rows.

⚠️ **The whitespace skip requires remapping heads.** spaCy's own
document index (`token.head.i + 1`) counts SPACE tokens, so using it
directly would leave every `head_token_id` past the first newline
pointing at the wrong word.

So `parse` runs two passes: the first builds
`id_by_doc_index: {token.i → token_id}` over surviving tokens only, and
the second resolves each head through that map. A root token maps to its
own `token_id`. Because heads are resolved through the map rather than
arithmetic, `head_token_id` always names a `token_id` that exists in the
CSV.
