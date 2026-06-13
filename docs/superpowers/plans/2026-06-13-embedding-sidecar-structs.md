# Embedding Sidecar Structs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the three embedding `.mat` sidecars (`word2vec`, `bert_word`, `bert_sentence`) self-describing — bundling the matrix with `tokens`, `word_id`, `sentence_id`, and an `is_valid` mask — and unify punctuation handling so a PUNCT row is a zero vector + `is_valid=false` in both word-level models.

**Architecture:** Extractors return their embedding matrices **plus validity masks** (they alone know vocab/embedding success). `main.py` attaches the **labels** (tokens, ids, sentence text) from the parsed `df` and writes everything as **flat top-level variables** in the `.mat` (so the existing `embeddings` variable stays put — additive, non-breaking). No new files, prefs keys, planner stages, or GUI toggles.

**Tech Stack:** Python, numpy, pandas, scipy.io (savemat/loadmat), gensim (Word2Vec), transformers/torch (BERT), pytest.

**Spec:** `docs/superpowers/specs/2026-06-13-embedding-sidecar-structs-design.md`

---

## File Structure

- `extractors/word2vec_embed.py` — `embed_word2vec` returns `(matrix, is_valid)`; forces `PUNCT → zero`.
- `extractors/bert_embed.py` — `embed_bert_words_and_sentences` returns `(word_emb, word_valid, sent_emb, sent_valid)`.
- `main.py` — two small helpers build the flat savemat dicts from extractor outputs + `df`; wire them into the embedding write block (`~746–772`).
- `tests/extractors/test_word2vec_embed.py` — update unpacking; add PUNCT/validity tests.
- `tests/extractors/test_bert_embed.py` — update unpacking; add PUNCT/validity tests.
- `tests/integration/test_embedding_struct.py` — **new**; savemat→loadmat roundtrip of the helpers.
- `docs/features/master_word2vec.md`, `master_bert_word.md`, `master_bert_sentence.md` — document the new fields + unified PUNCT handling.

---

## Task 1: Word2Vec — PUNCT→zero + `is_valid` mask

**Files:**
- Modify: `extractors/word2vec_embed.py` (the `embed_word2vec` function, lines 57–77)
- Test: `tests/extractors/test_word2vec_embed.py`

- [ ] **Step 1: Update existing tests to the new 2-tuple return + add a upos-aware df helper**

Replace the `make_word_df` helper and every `mat = embed_word2vec(...)` call site. At the top of `tests/extractors/test_word2vec_embed.py`, replace the helper:

```python
def make_word_df(tokens, upos=None):
    data = {"token": tokens}
    if upos is not None:
        data["upos"] = upos
    return pd.DataFrame(data)
```

Then in each of `test_known_word_returns_its_vector`, `test_oov_word_returns_zero_vector`, `test_lowercase_fallback`, `test_output_shape_multi_word`, change the call from:

```python
mat = embed_word2vec(df, model_path="dummy.bin")
```

to:

```python
mat, is_valid = embed_word2vec(df, model_path="dummy.bin")
```

(The two `pytest.raises` tests at the bottom do not capture a return value — leave them unchanged.)

- [ ] **Step 2: Add the new validity tests**

Append to `tests/extractors/test_word2vec_embed.py`:

```python
@patch("extractors.word2vec_embed._load_model")
def test_punct_is_zeroed_and_invalid_even_if_in_vocab(mock_load):
    from extractors.word2vec_embed import embed_word2vec

    vec = np.ones(300, dtype=np.float32)
    # "," IS in vocab, but upos marks it PUNCT — it must be forced to zero.
    mock_load.return_value = _make_mock_kv({",": vec, "cat": vec})
    df = make_word_df(["cat", ","], upos=["NOUN", "PUNCT"])

    mat, is_valid = embed_word2vec(df, model_path="dummy.bin")
    np.testing.assert_array_equal(mat[0], vec)          # content word kept
    np.testing.assert_array_equal(mat[1], np.zeros(300))  # PUNCT zeroed
    assert is_valid.tolist() == [True, False]


@patch("extractors.word2vec_embed._load_model")
def test_oov_content_word_is_invalid(mock_load):
    from extractors.word2vec_embed import embed_word2vec

    vec = np.ones(300, dtype=np.float32)
    mock_load.return_value = _make_mock_kv({"the": vec})
    df = make_word_df(["the", "xyzzy"], upos=["DET", "NOUN"])

    mat, is_valid = embed_word2vec(df, model_path="dummy.bin")
    assert is_valid.tolist() == [True, False]
    np.testing.assert_array_equal(mat[1], np.zeros(300))


@patch("extractors.word2vec_embed._load_model")
def test_is_valid_dtype_and_shape(mock_load):
    from extractors.word2vec_embed import embed_word2vec

    mock_load.return_value = _make_mock_kv({"a": np.ones(300, dtype=np.float32)})
    df = make_word_df(["a", "b", "c"])  # no upos column -> all treated non-PUNCT
    mat, is_valid = embed_word2vec(df, model_path="dummy.bin")
    assert is_valid.shape == (3,)
    assert is_valid.dtype == bool
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `conda activate ./env && python -m pytest tests/extractors/test_word2vec_embed.py -v`
Expected: the new tests FAIL (e.g. `ValueError: too many values to unpack` / `embed_word2vec` returns an ndarray, not a tuple), and the edited existing tests FAIL on unpacking.

- [ ] **Step 4: Implement the new return + PUNCT zeroing**

In `extractors/word2vec_embed.py`, replace the body of `embed_word2vec` (from the docstring's return-type line down) so it reads `upos`, zeroes PUNCT, and returns `(mat, is_valid)`:

```python
def embed_word2vec(
    word_df: pd.DataFrame,
    *,
    model_path: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (matrix, is_valid).

    matrix:   (n_words, 300) float32, row-aligned with word_df.
    is_valid: (n_words,) bool. True iff the row is a non-PUNCT token that was
              found in vocab (exact or lowercase). PUNCT and OOV rows are zero
              vectors with is_valid False.
    """
    mp = Path(model_path) if model_path is not None else _DEFAULT_MODEL_PATH
    model = _load_model(mp)

    tokens = word_df["token"].tolist()
    if "upos" in word_df.columns:
        upos = word_df["upos"].tolist()
    else:
        upos = [None] * len(tokens)

    n = len(tokens)
    mat = np.zeros((n, 300), dtype=np.float32)
    is_valid = np.zeros(n, dtype=bool)
    for i, (tok, pos) in enumerate(zip(tokens, upos)):
        if pos == "PUNCT":
            continue  # forced zero, is_valid stays False (matches BERT)
        if tok in model:
            mat[i] = model[tok]
            is_valid[i] = True
        elif tok.lower() in model:
            mat[i] = model[tok.lower()]
            is_valid[i] = True
        # else OOV: row stays zero, is_valid stays False
    return mat, is_valid
```

Also update the module docstring's last line (currently "OOV words receive a zero vector...") to: `PUNCT and OOV rows receive a zero vector and is_valid=False.`

- [ ] **Step 5: Run the tests to verify they pass**

Run: `conda activate ./env && python -m pytest tests/extractors/test_word2vec_embed.py -v`
Expected: PASS (all tests).

- [ ] **Step 6: Commit**

```bash
git add extractors/word2vec_embed.py tests/extractors/test_word2vec_embed.py
git commit -m "feat(word2vec): return is_valid mask; force PUNCT to zero"
```

---

## Task 2: BERT — word + sentence `is_valid` masks

**Files:**
- Modify: `extractors/bert_embed.py` (`embed_bert_words_and_sentences`, lines 54–150)
- Test: `tests/extractors/test_bert_embed.py`

- [ ] **Step 1: Update existing tests to the new 4-tuple return**

In `tests/extractors/test_bert_embed.py`, update every call site:

- `test_word_embed_shape`: `word_emb, sent_emb = ...` → `word_emb, word_valid, sent_emb, sent_valid = ...`
- `test_word_embed_dtype_is_float32`: `word_emb, _ = ...` → `word_emb, _, _, _ = ...`
- `test_sentence_embed_different_from_each_other`: `_, sent_emb = ...` → `_, _, sent_emb, _ = ...`
- `test_sliding_window_does_not_exceed_512_tokens`: `word_emb, sent_emb = ...` → `word_emb, _, sent_emb, _ = ...`
- `test_punct_excluded_from_sentence_mean`: both calls `_, sent_with = ...` / `_, sent_without = ...` → `_, _, sent_with, _ = ...` / `_, _, sent_without, _ = ...`

- [ ] **Step 2: Add the new validity tests**

Append to `tests/extractors/test_bert_embed.py`:

```python
def test_punct_word_is_zero_and_invalid():
    from extractors.bert_embed import embed_bert_words_and_sentences

    tokenizer, model = _make_tiny_bert()
    df = pd.DataFrame([
        {"doc_id": "d", "sentence_id": 0, "sentence": "Hello world.",
         "token": "Hello", "upos": "NOUN"},
        {"doc_id": "d", "sentence_id": 0, "sentence": "Hello world.",
         "token": "world", "upos": "NOUN"},
        {"doc_id": "d", "sentence_id": 0, "sentence": "Hello world.",
         "token": ".", "upos": "PUNCT"},
    ])
    word_emb, word_valid, _, _ = embed_bert_words_and_sentences(
        df, tokenizer=tokenizer, model=model
    )
    assert word_valid.tolist() == [True, True, False]
    np.testing.assert_array_equal(word_emb[2], np.zeros((2, 64)))  # PUNCT row zero


def test_sentence_valid_mask():
    from extractors.bert_embed import embed_bert_words_and_sentences

    tokenizer, model = _make_tiny_bert()
    df = make_word_df([
        ("d", 0, "The cat sat.", "The"),
        ("d", 0, "The cat sat.", "cat"),
        ("d", 1, "Fish swim.",   "Fish"),
    ])
    _, _, _, sent_valid = embed_bert_words_and_sentences(
        df, tokenizer=tokenizer, model=model
    )
    assert sent_valid.tolist() == [True, True]


def test_all_punct_sentence_is_invalid():
    from extractors.bert_embed import embed_bert_words_and_sentences

    tokenizer, model = _make_tiny_bert()
    df = pd.DataFrame([
        {"doc_id": "d", "sentence_id": 0, "sentence": "...",
         "token": ".", "upos": "PUNCT"},
        {"doc_id": "d", "sentence_id": 0, "sentence": "...",
         "token": ".", "upos": "PUNCT"},
    ])
    _, word_valid, _, sent_valid = embed_bert_words_and_sentences(
        df, tokenizer=tokenizer, model=model
    )
    assert sent_valid.tolist() == [False]
    assert word_valid.tolist() == [False, False]
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `conda activate ./env && python -m pytest tests/extractors/test_bert_embed.py -v`
Expected: FAIL — unpacking errors (function still returns a 2-tuple).

- [ ] **Step 4: Implement the masks**

In `extractors/bert_embed.py`:

(a) Update the signature/docstring return type of `embed_bert_words_and_sentences` to a 4-tuple:

```python
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (word_embeddings, word_valid, sentence_embeddings, sentence_valid).

    word_embeddings:     (n_words, n_layers, hidden_size) float32
    word_valid:          (n_words,) bool — True iff a non-PUNCT word was embedded
    sentence_embeddings: (n_sentences, n_layers, hidden_size) float32
    sentence_valid:      (n_sentences,) bool — True iff the sentence had >=1 embedded content word
    """
```

(b) After the `sent_emb = np.zeros(...)` line (currently line 79), add the two masks:

```python
    word_valid = np.zeros(n_words, dtype=bool)
    sent_valid = np.zeros(n_sentences, dtype=bool)
```

(c) Inside the per-word loop, where the word vector is assigned (currently `word_emb[idx_to_pos[orig_idx]] = word_vec`), also mark it valid:

```python
            word_vec = stacked[positions].mean(axis=0)
            word_emb[idx_to_pos[orig_idx]] = word_vec
            word_valid[idx_to_pos[orig_idx]] = True
            content_positions.extend(positions)
```

(d) Where the sentence vector is assigned (currently the `if content_positions:` block), also mark the sentence valid:

```python
        if content_positions:
            sent_emb[s_pos] = stacked[content_positions].mean(axis=0)
            sent_valid[s_pos] = True
```

(e) Change the final return:

```python
    return word_emb, word_valid, sent_emb, sent_valid
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `conda activate ./env && python -m pytest tests/extractors/test_bert_embed.py -v`
Expected: PASS (all tests).

- [ ] **Step 6: Commit**

```bash
git add extractors/bert_embed.py tests/extractors/test_bert_embed.py
git commit -m "feat(bert): return word + sentence is_valid masks"
```

---

## Task 3: main.py serialization helpers + wiring

**Files:**
- Modify: `main.py` (add `import numpy as np` near line 43; add two helpers; rewrite the embedding write block at ~746–772)
- Test: `tests/integration/test_embedding_struct.py` (**new**)

- [ ] **Step 1: Write the failing roundtrip test**

Create `tests/integration/test_embedding_struct.py`:

```python
"""savemat -> loadmat roundtrip for the embedding-struct helpers in main.py."""
import numpy as np
import pandas as pd
import scipy.io

from main import _word_embedding_vars, _sentence_embedding_vars


def _word_df():
    return pd.DataFrame([
        {"doc_id": 1, "sentence_id": 1, "sentence": "The cat sat.",
         "term_id": 1, "token": "The"},
        {"doc_id": 1, "sentence_id": 1, "sentence": "The cat sat.",
         "term_id": 2, "token": "cat"},
        {"doc_id": 1, "sentence_id": 1, "sentence": "The cat sat.",
         "term_id": 3, "token": "."},
        {"doc_id": 1, "sentence_id": 2, "sentence": "It rained.",
         "term_id": 4, "token": "It"},
    ])


def test_word_vars_roundtrip(tmp_path):
    df = _word_df()
    emb = np.arange(4 * 5, dtype=np.float32).reshape(4, 5)
    is_valid = np.array([True, True, False, True])

    out = _word_embedding_vars(emb, df, is_valid)
    # `embeddings` is still a top-level key (backward compatible).
    assert set(out) == {"embeddings", "tokens", "word_id", "sentence_id", "is_valid"}

    p = tmp_path / "w.mat"
    scipy.io.savemat(str(p), out)
    loaded = scipy.io.loadmat(str(p))

    np.testing.assert_array_equal(loaded["embeddings"], emb)
    assert loaded["word_id"].ravel().tolist() == [1, 2, 3, 4]
    assert loaded["sentence_id"].ravel().tolist() == [1, 1, 1, 2]
    assert loaded["is_valid"].ravel().astype(bool).tolist() == [True, True, False, True]
    toks = [str(t).strip() for t in loaded["tokens"].ravel()]
    assert toks == ["The", "cat", ".", "It"]


def test_sentence_vars_roundtrip(tmp_path):
    df = _word_df()
    emb = np.arange(2 * 3, dtype=np.float32).reshape(2, 3)
    is_valid = np.array([True, True])

    out = _sentence_embedding_vars(emb, df, is_valid)
    assert set(out) == {"embeddings", "sentence_id", "sentence", "is_valid"}

    p = tmp_path / "s.mat"
    scipy.io.savemat(str(p), out)
    loaded = scipy.io.loadmat(str(p))

    # one row per unique (doc_id, sentence_id), in first-seen order
    assert loaded["sentence_id"].ravel().tolist() == [1, 2]
    sents = [str(s).strip() for s in loaded["sentence"].ravel()]
    assert sents == ["The cat sat.", "It rained."]
    np.testing.assert_array_equal(loaded["embeddings"], emb)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `conda activate ./env && python -m pytest tests/integration/test_embedding_struct.py -v`
Expected: FAIL with `ImportError: cannot import name '_word_embedding_vars' from 'main'`.

- [ ] **Step 3: Add `import numpy as np` and the two helpers to main.py**

At the top of `main.py`, add after `import pandas as pd` (line 43):

```python
import numpy as np
```

Add these two module-level helpers (place them just above `run_doc_level_batch` / the embedding write site — anywhere at module scope is fine):

```python
def _word_embedding_vars(emb: np.ndarray, df: pd.DataFrame,
                         is_valid: np.ndarray) -> dict:
    """Flat top-level .mat variables for a word-level embedding sidecar.

    `embeddings` stays a top-level key so existing consumers keep working.
    Labels are taken from df (row-aligned to emb).
    """
    return {
        "embeddings":  emb,
        "tokens":      np.array(df["token"].tolist(), dtype=object),
        "word_id":     np.asarray(df["term_id"].to_numpy(), dtype=np.int64),
        "sentence_id": np.asarray(df["sentence_id"].to_numpy(), dtype=np.int64),
        "is_valid":    np.asarray(is_valid, dtype=bool),
    }


def _sentence_embedding_vars(emb: np.ndarray, df: pd.DataFrame,
                             is_valid: np.ndarray) -> dict:
    """Flat top-level .mat variables for the sentence-level embedding sidecar.

    Sentence rows follow the same first-seen (doc_id, sentence_id) grouping the
    BERT extractor uses, so they stay aligned to `emb`.
    """
    groups = list(df.groupby(["doc_id", "sentence_id"], sort=False))
    sent_ids = [key[1] for key, _ in groups]
    sent_texts = [g["sentence"].iloc[0] for _, g in groups]
    return {
        "embeddings":  emb,
        "sentence_id": np.asarray(sent_ids, dtype=np.int64),
        "sentence":    np.array(sent_texts, dtype=object),
        "is_valid":    np.asarray(is_valid, dtype=bool),
    }
```

- [ ] **Step 4: Run the roundtrip test to verify it passes**

Run: `conda activate ./env && python -m pytest tests/integration/test_embedding_struct.py -v`
Expected: PASS.

- [ ] **Step 5: Wire the helpers into the embedding write block**

In `main.py`, replace the Word2Vec write (currently lines 742–754) so it captures the mask and writes the struct:

```python
            if not getattr(args, 'no_word2vec', True):
                try:
                    import scipy.io
                    from extractors.word2vec_embed import embed_word2vec, Word2VecUnavailable
                    w2v_mat, w2v_valid = embed_word2vec(df)
                    mat_path = _level_dir(output_dir, 'word') / f'{pair.stimulus_id}_word2vec.mat'
                    scipy.io.savemat(str(mat_path), _word_embedding_vars(w2v_mat, df, w2v_valid))
                except Word2VecUnavailable as exc:
                    print(f'[word2vec] skipped for {pair.stimulus_id}: {exc}',
                          file=sys.stderr)
                except Exception as exc:
                    print(f'[word2vec] failed for {pair.stimulus_id}: {exc}',
                          file=sys.stderr)
```

Then replace the BERT write (currently lines 758–775) to capture the four returns and write both structs:

```python
            run_bert_word = not getattr(args, 'no_bert_word', True)
            run_bert_sent = not getattr(args, 'no_bert_sentence', True)
            if run_bert_word or run_bert_sent:
                try:
                    import scipy.io
                    from extractors.bert_embed import embed_bert_words_and_sentences
                    word_emb, word_valid, sent_emb, sent_valid = \
                        embed_bert_words_and_sentences(df)
                    if run_bert_word:
                        mat_path = (_level_dir(output_dir, 'word')
                                    / f'{pair.stimulus_id}_bert_word.mat')
                        scipy.io.savemat(str(mat_path),
                                         _word_embedding_vars(word_emb, df, word_valid))
                    if run_bert_sent:
                        mat_path = (_level_dir(output_dir, 'sentence')
                                    / f'{pair.stimulus_id}_bert_sentence.mat')
                        scipy.io.savemat(str(mat_path),
                                         _sentence_embedding_vars(sent_emb, df, sent_valid))
                except Exception as exc:
                    print(f'[bert] failed for {pair.stimulus_id}: {exc}',
                          file=sys.stderr)
```

- [ ] **Step 6: Verify nothing else broke (import + full extractor suite)**

Run: `conda activate ./env && python -c "import main" && python -m pytest tests/extractors/test_word2vec_embed.py tests/extractors/test_bert_embed.py tests/integration/test_embedding_struct.py -v`
Expected: clean import, all PASS.

- [ ] **Step 7: Commit**

```bash
git add main.py tests/integration/test_embedding_struct.py
git commit -m "feat(embeddings): write self-describing .mat sidecars (tokens/ids/is_valid)"
```

---

## Task 4: Documentation

**Files:**
- Modify: `docs/features/master_word2vec.md`, `docs/features/master_bert_word.md`, `docs/features/master_bert_sentence.md`

No tests. Follow the existing doc voice (see the feature-doc authorship/citation conventions in memory — no fabricated claims; the `is_valid` rationale is implementation fact, not an empirical claim).

- [ ] **Step 1: Update `master_word2vec.md`**

In the front-matter, leave `output_files` as `["word/<stim>_word2vec.mat"]` (path unchanged). In the **Implementation in SensiExtractor** section, update item 5 of "What the code does" and add the output-fields description:

Replace the current item 5 ("Output is `float32`, shape `(n_words, 300)`, row-aligned…") with:

```markdown
5. PUNCT tokens are forced to a **zero vector** (regardless of vocab
   membership) so punctuation is handled identically to BERT.
6. The extractor returns `(matrix, is_valid)`, where `is_valid[i]` is True
   iff row `i` is a non-PUNCT token found in vocab (exact or lowercase
   fallback). PUNCT and OOV rows are zero with `is_valid=False`.

**Output `.mat` fields** (flat top-level variables, not a nested struct, so
`embeddings` stays backward-compatible):

| Field | Shape | Meaning |
|---|---|---|
| `embeddings`  | `(n_words, 300)` float32 | the vectors, row-aligned to the word CSV |
| `tokens`      | `{n_words}` cellstr | surface token per row |
| `word_id`     | `(n_words,)` int | the word CSV `term_id` (global token index) |
| `sentence_id` | `(n_words,)` int | spaCy sentence id |
| `is_valid`    | `(n_words,)` logical | False for PUNCT / OOV (zero rows) |
```

- [ ] **Step 2: Update `master_bert_word.md`**

In the **Implementation in SensiExtractor** section, after the existing item 6 ("All 12 transformer layers are emitted…"), add:

```markdown
8. The extractor returns word- and sentence-level `is_valid` masks alongside
   the matrices. For word-level, `is_valid[i]` is True iff row `i` is a
   non-PUNCT word that was embedded (PUNCT rows and any word in a skipped
   window stay zero + False).

**Output `.mat` fields** (flat top-level variables; `embeddings` unchanged):

| Field | Shape | Meaning |
|---|---|---|
| `embeddings`  | `(n_words, 12, 768)` float32 | per-layer vectors, row-aligned to the word CSV |
| `tokens`      | `{n_words}` cellstr | surface token per row |
| `word_id`     | `(n_words,)` int | the word CSV `term_id` |
| `sentence_id` | `(n_words,)` int | spaCy sentence id |
| `is_valid`    | `(n_words,)` logical | False for PUNCT (zero rows) |
```

- [ ] **Step 3: Update `master_bert_sentence.md`**

In the **Implementation in SensiExtractor** section (or the Meaning/output section, matching the file's structure), add the field table:

```markdown
**Output `.mat` fields** (flat top-level variables; `embeddings` unchanged):

| Field | Shape | Meaning |
|---|---|---|
| `embeddings`  | `(n_sentences, 12, 768)` float32 | per-layer sentence vectors |
| `sentence_id` | `(n_sentences,)` int | spaCy sentence id |
| `sentence`    | `{n_sentences}` cellstr | the sentence text |
| `is_valid`    | `(n_sentences,)` logical | False for empty / all-PUNCT sentences |
```

- [ ] **Step 4: Commit**

```bash
git add docs/features/master_word2vec.md docs/features/master_bert_word.md docs/features/master_bert_sentence.md
git commit -m "docs(embeddings): document self-describing .mat fields + unified PUNCT handling"
```

---

## Final verification

- [ ] Run the full embedding + integration test set:

Run: `conda activate ./env && python -m pytest tests/extractors/test_word2vec_embed.py tests/extractors/test_bert_embed.py tests/integration/test_embedding_struct.py -v`
Expected: all PASS.

- [ ] Confirm `python -c "import main"` succeeds (no syntax/import regressions).

- [ ] Update `docs/master_plan.md` completed-batches table with this change (one row, following the existing format) and commit.
