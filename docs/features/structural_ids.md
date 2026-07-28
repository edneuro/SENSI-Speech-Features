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
| `doc_id` | Document. `1` for a single-document run; in batch mode each document gets its sequential 1-indexed position in the batch (1, 2, 3, …). Set explicitly via the CLI `--doc-id N` flag for one-off runs. |
| `paragraph_id` | Paragraph within the document. Paragraphs are detected as runs of text separated by one or more blank lines (`\n\n`) in the transcript. A transcript with no blank lines is a single paragraph (`paragraph_id == 1` throughout). |
| `sentence_id` | Sentence within the document, in spaCy reading order. First sentence is `1`. |
| `token_id` | Global token index across the entire document. First token is `1`. Counts **emitted tokens only** — see below. |
| `token_in_sentence_id` | Token index **within the current sentence**. Resets to `1` at each sentence boundary. A true sentence position — see below. |

### Whitespace tokens are not rows (fixed 2026-07-27)

spaCy emits a literal newline in the transcript as its own token
(`upos == 'SPACE'`, text `"\n"`). Until 2026-07-27 the parser iterated
every token, so each hard line wrap became an **all-NaN word-CSV row that
consumed both a `token_id` and a `token_in_sentence_id`**:

```
token_id=10  token_in_sentence_id=10  token=\n     ← consumed position 10
token_id=11  token_in_sentence_id=11  token=were
```

Every word after a line break was therefore off by one in both id columns,
and `token_in_sentence_id` was **not** a true position within its sentence
— it was a position within the sentence's token stream *including* line
wraps. Three of the ten SENSI validation transcripts were affected (the
hard-wrapped `LJ*` files); the single-line ISC stimuli have no SPACE
tokens, which is why it went unnoticed.

Whitespace tokens are now skipped entirely. Consequences:

- `token_id` and `token_in_sentence_id` count **emitted tokens only** and
  are contiguous with no gaps.
- `token_in_sentence_id` **is** now a true position within its sentence —
  word *n* of the sentence, as printed.
- Nothing is lost by dropping the rows. They carried no features (every
  feature column was NaN), and paragraph membership was never encoded by
  them: `paragraph_id` is computed from **blank lines**, not single
  newlines, and has been all along. A hard wrap inside a sentence is not a
  paragraph break — in one validation stimulus a `\n` falls mid-sentence,
  and spaCy correctly assigns both halves the same `sentence_id`.
- `head_token_id` still resolves correctly; see the note in
  Implementation, because that part was **not** free.

⚠️ If you are comparing against a word CSV generated before 2026-07-27
from a transcript containing hard line wraps, the id columns will differ
by the number of preceding newlines. Join on `token`/`Onset` rather than
on `token_id` when reconciling old and new outputs.

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
`sentence_id` from `enumerate(doc.sents, start=1)`, `token_in_sentence_id`
as a per-sentence counter over non-whitespace tokens, and `token_id` as a
global counter that increments across sentences. Tokens for which
`token.is_space` is true are skipped and never become rows.

⚠️ **The whitespace skip required a second, non-obvious change**, and
omitting it would have silently corrupted the dependency parse. `parse`
previously computed `head_token_id = token.head.i + 1`, using spaCy's own
document index — which counts SPACE tokens. That agreed with `token_id`
only *by accident*, because `token_id` counted them too. Skipping the rows
without remapping would have left every `head_token_id` past the first
newline pointing at the wrong word, with no error raised.

So `parse` now runs two passes: the first builds
`id_by_doc_index: {token.i → token_id}` over surviving tokens only, and
the second resolves each head through that map. A root token maps to its
own `token_id`. Because heads are resolved through the map rather than
arithmetic, `head_token_id` always names a `token_id` that exists in the
CSV.
