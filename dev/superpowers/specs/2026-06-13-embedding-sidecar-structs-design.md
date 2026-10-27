# Embedding Sidecar Structs — Design Spec

**Date:** 2026-06-13
**Status:** Approved, pending implementation plan.

---

## 1. Problem

The three embedding sidecars are written as bare matrices —
`scipy.io.savemat(path, {'embeddings': M})` — with **no labels**. Alignment to
the word/sentence CSV is purely positional (`row i` of the `.mat` ↔ `row i` of
the CSV). Two issues follow:

1. **Anonymous matrices with dead rows.** A downstream analyst loading the
   `.mat` alone cannot tell which row is which word without perfectly
   preserving CSV row order. The matrices also contain meaningless rows:
   standalone punctuation tokens (spaCy `upos='PUNCT'`) get a row each.
2. **Inconsistent punctuation handling across the two word-level models.**
   - **BERT** zeros PUNCT rows by design (excluded from the forward pass).
   - **Word2Vec** looks PUNCT up like any token — `","` may hit a real Google
     News vector or fall through to zero.

   So the same punctuation token can be a zero row in one file and a nonzero
   row in the other, and a naive cosine over the matrix hits undefined
   (zero-vector) rows.

## 2. Outputs (changed)

Each `.mat` goes from a single bare matrix to **multiple top-level variables**
in the same file — the matrix plus its labels. **Row order and matrix shapes
are unchanged, and `embeddings` stays a top-level variable, so existing
consumers that read `embeddings` keep working (additive, non-breaking).**

> **Mechanism note.** `scipy.io.savemat` writes each *top-level* dict key as a
> separate `.mat` variable; only a *nested* dict value becomes a MATLAB struct
> (cf. `out["config"]` in `acoustic_ts.py:176`). We deliberately use the flat
> top-level form, **not** a nested struct: a nested struct (`S.word2vec.embeddings`)
> would relocate `embeddings` and break any code doing `load(f,'embeddings')` /
> `loadmat(f)['embeddings']`. The flat form delivers the same self-describing
> bundle (one file: matrix + tokens + ids + mask) with zero compatibility cost.
> In MATLAB, `load('<stim>_word2vec.mat')` injects `embeddings`, `tokens`,
> `word_id`, `sentence_id`, `is_valid` into the workspace.

### Word-level (`word2vec`, `bert_word`)

| Field | Shape | Notes |
|-------|-------|-------|
| `embeddings`  | `(n_words, 300)` / `(n_words, 12, 768)` float32 | unchanged |
| `tokens`      | `{n_words}` cellstr | surface token, e.g. `'cat'`, `','`, `"n't"` |
| `word_id`     | `(n_words,)` int | = word CSV `term_id` (global token index) |
| `sentence_id` | `(n_words,)` int | spaCy sentence id |
| `is_valid`    | `(n_words,)` logical | `false` for PUNCT / OOV (zero rows) |

Positional alignment to the word CSV is preserved; `word_id` / `sentence_id`
make the join explicit and reorder-safe. Strip dead rows with
`E = embeddings(is_valid, :)`.

### Sentence-level (`bert_sentence`)

| Field | Shape | Notes |
|-------|-------|-------|
| `embeddings`  | `(n_sentences, 12, 768)` float32 | unchanged |
| `sentence_id` | `(n_sentences,)` int | |
| `sentence`    | `{n_sentences}` cellstr | the sentence text |
| `is_valid`    | `(n_sentences,)` logical | `false` for empty / all-PUNCT sentences |

`n_words` = rows in `<stim>_features.csv`; `n_sentences` = unique
`sentence_id`.

## 3. Unified punctuation handling

Definition: **`is_valid = (upos != PUNCT) AND successfully embedded`**, applied
identically to both word-level models. A PUNCT row is a zero vector with
`is_valid=false` in **both** files; OOV content words are also zero +
`is_valid=false`.

- **`extractors/word2vec_embed.py`** — `embed_word2vec(df)` additionally reads
  the `upos` column, forces `PUNCT → zero`, and returns `(matrix, is_valid)`
  where `is_valid = (upos != 'PUNCT') AND in-vocab`. (In-vocab covers the exact
  token or its lowercase fallback, matching current lookup.)
- **`extractors/bert_embed.py`** — already zeros PUNCT; extend the return to
  carry the word-level and sentence-level validity masks so the rule lives in
  the extractor (unit-testable) rather than `main.py`. Word-level
  `is_valid = (upos != 'PUNCT') AND row embedded` (a word in a skipped/empty
  window stays zero → invalid); sentence-level `is_valid = sentence has ≥1
  embedded content word`.

This keeps validity logic in `extractors/` (covered by the project's TDD
convention for extractor math).

## 4. main.py wiring

At `main.py:746–772`, build the savemat dict from the extractor outputs plus
the `df` columns (`token`, `term_id`, `sentence_id`, `sentence`), e.g.:

```python
scipy.io.savemat(str(mat_path), {
    'embeddings':  w2v_mat,                       # unchanged key/shape
    'tokens':      np.array(tokens, dtype=object),  # -> MATLAB cellstr
    'word_id':     np.asarray(word_ids,  dtype=np.int64),
    'sentence_id': np.asarray(sent_ids,  dtype=np.int64),
    'is_valid':    np.asarray(is_valid,  dtype=bool),  # -> MATLAB logical
})
```

`tokens` / `sentence` pass through `np.array(..., dtype=object)` so they render
as MATLAB cellstr rather than padded char matrices. No new files, paths, prefs
keys, planner stages, or GUI toggles — this is a serialization change to
existing sidecars only, keeping the `embeddings` variable backward-compatible.

## 5. Tests

- **TDD (extractor math):** validity masks — PUNCT → zero + invalid, OOV → zero
  + invalid, in-vocab hit → valid, all-PUNCT / empty sentence → invalid.
- **Test-after (integration):** each `.mat` loads as a struct exposing the
  expected fields; word-row count matches the features CSV and sentence count
  matches the sentence CSV; `word_id`/`sentence_id` align to the CSV.

## 6. Docs

Update `docs/features/master_word2vec.md`, `master_bert_word.md`, and
`master_bert_sentence.md` (`output_files` front-matter + Implementation
sections) to document the struct fields and the unified PUNCT handling.

## 7. Out of scope (YAGNI)

- No change to matrix shapes, dtypes, or row order.
- No new output files, prefs keys, planner stages, or GUI toggles.
- No change to the model lookup logic beyond zeroing PUNCT in Word2Vec.
