# Embeddings Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Emit Word2Vec (.mat, n_words×300) and BERT (.mat, n_words×12×768 and n_sentences×12×768) embedding matrices per stimulus into `word/` and `sentence/` subfolders.

**Architecture:** Two new extractors: `word2vec_embed.py` (gensim, OOV→zero) and `bert_embed.py` (HuggingFace bert-base-uncased, sliding window, shared forward pass for word+sentence). `.mat` files written via scipy.io.savemat. Wired into the word batch in `main.py`. Prefs v9→v10 (or v8→v10 if this ships before sentence-level). New "Embeddings" GUI page.

**Tech Stack:** gensim>=4.3, transformers, torch, scipy (all via conda env), PySide6

---

## File Map

| Action | Path |
|--------|------|
| **Create** | `extractors/word2vec_embed.py` |
| **Create** | `tests/extractors/test_word2vec_embed.py` |
| **Create** | `extractors/bert_embed.py` |
| **Create** | `tests/extractors/test_bert_embed.py` |
| **Modify** | `requirements.txt` — add gensim>=4.3 |
| **Modify** | `download_models.py` — add Word2Vec download instructions |
| **Modify** | `utils/preferences.py` — v9→v10 (3 output toggles + 3 master feature toggles) |
| **Modify** | `tests/utils/test_preferences_migration.py` — v9→v10 round-trip |
| **Modify** | `gui/planner/registry.py` — 2 new Stages + 3 new FeatureClusters |
| **Modify** | `gui/planner/emit.py` — `_WORD_LEVEL_CSVS` + `ns.no_word2vec` etc |
| **Modify** | `gui/planner/plan.py` — `_OUTPUT_CSV_MAP` + `_stage_configs` |
| **Modify** | `tests/planner/test_schema_drift.py` — update to v10 constants |
| **Modify** | `main.py` — CLI flags + .mat output in batch |
| **Create** | `gui/section_embeddings.py` |
| **Modify** | `gui/main_window.py` — new sidebar item + section |
| **Modify** | `tests/integration/test_output_paths.py` — .mat path tests |
| **Modify** | `docs/master_plan.md` — update current state |

---

## Task 1: Word2Vec extractor

**Files:**
- Create: `extractors/word2vec_embed.py`
- Create: `tests/extractors/test_word2vec_embed.py`

- [ ] **Step 1: Write failing tests (with mocked gensim)**

Create `tests/extractors/test_word2vec_embed.py`:

```python
"""Tests for extractors.word2vec_embed.

Uses a mock gensim KeyedVectors so no real model file is needed.
"""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch


def make_word_df(tokens):
    return pd.DataFrame({"token": tokens})


def _make_mock_kv(vocab: dict[str, np.ndarray]):
    """Return a mock KeyedVectors that behaves like the real one for __contains__ and __getitem__."""
    kv = MagicMock()
    kv.__contains__ = lambda self, tok: tok in vocab
    kv.__getitem__ = lambda self, tok: vocab[tok]
    return kv


@patch("extractors.word2vec_embed._load_model")
def test_known_word_returns_its_vector(mock_load):
    from extractors.word2vec_embed import embed_word2vec

    vec = np.ones(300, dtype=np.float32)
    mock_load.return_value = _make_mock_kv({"cat": vec})

    df = make_word_df(["cat"])
    mat = embed_word2vec(df, model_path="dummy.bin")
    assert mat.shape == (1, 300)
    np.testing.assert_array_equal(mat[0], vec)


@patch("extractors.word2vec_embed._load_model")
def test_oov_word_returns_zero_vector(mock_load):
    from extractors.word2vec_embed import embed_word2vec

    mock_load.return_value = _make_mock_kv({})  # empty vocab
    df = make_word_df(["xyzzy"])
    mat = embed_word2vec(df, model_path="dummy.bin")
    assert mat.shape == (1, 300)
    np.testing.assert_array_equal(mat[0], np.zeros(300))


@patch("extractors.word2vec_embed._load_model")
def test_lowercase_fallback(mock_load):
    from extractors.word2vec_embed import embed_word2vec

    vec = np.full(300, 2.0, dtype=np.float32)
    mock_load.return_value = _make_mock_kv({"dog": vec})  # lowercase only

    df = make_word_df(["Dog"])  # capitalized
    mat = embed_word2vec(df, model_path="dummy.bin")
    np.testing.assert_array_equal(mat[0], vec)


@patch("extractors.word2vec_embed._load_model")
def test_output_shape_multi_word(mock_load):
    from extractors.word2vec_embed import embed_word2vec

    vec = np.eye(300, dtype=np.float32)[0]
    mock_load.return_value = _make_mock_kv({"the": vec, "cat": vec})
    df = make_word_df(["the", "cat", "sat"])  # "sat" is OOV
    mat = embed_word2vec(df, model_path="dummy.bin")
    assert mat.shape == (3, 300)


def test_raises_word2vec_unavailable_when_model_missing(tmp_path):
    from extractors.word2vec_embed import embed_word2vec, Word2VecUnavailable

    df = make_word_df(["hello"])
    with pytest.raises(Word2VecUnavailable, match="not found"):
        embed_word2vec(df, model_path=str(tmp_path / "nonexistent.bin"))
```

- [ ] **Step 2: Run tests to confirm they fail**

```
conda run -p env python -m pytest tests/extractors/test_word2vec_embed.py -v
```
Expected: all FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement extractors/word2vec_embed.py**

Create `extractors/word2vec_embed.py`:

```python
"""Word2Vec embedding extractor (Google News 300d via gensim).

Expected model: hf_cache/word2vec/GoogleNews-vectors-negative300.bin
Download from: https://code.google.com/archive/p/word2vec/
               (GoogleNews-vectors-negative300.bin.gz, ~1.7 GB compressed)

OOV words receive a zero vector (300 float32 zeros).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any

_DEFAULT_MODEL_PATH = (
    Path(__file__).parent.parent / "hf_cache" / "word2vec"
    / "GoogleNews-vectors-negative300.bin"
)


class Word2VecUnavailable(RuntimeError):
    """Raised when gensim is missing or the model file is not found."""


def _load_model(model_path: str | Path) -> Any:
    """Load gensim KeyedVectors from a binary file. Raises Word2VecUnavailable."""
    try:
        from gensim.models import KeyedVectors  # type: ignore
    except ImportError:
        raise Word2VecUnavailable(
            "gensim is not installed. Run: pip install gensim"
        )
    p = Path(model_path)
    if not p.exists():
        raise Word2VecUnavailable(
            f"Word2Vec model not found at {p}. "
            "Download GoogleNews-vectors-negative300.bin from "
            "https://code.google.com/archive/p/word2vec/ and place it in "
            "hf_cache/word2vec/"
        )
    return KeyedVectors.load_word2vec_format(str(p), binary=True)


def embed_word2vec(
    word_df: pd.DataFrame,
    *,
    model_path: str | Path | None = None,
) -> np.ndarray:
    """Return (n_words, 300) float32 matrix.

    Row i corresponds to word_df.iloc[i]. OOV rows are zero vectors.
    """
    mp = Path(model_path) if model_path is not None else _DEFAULT_MODEL_PATH
    model = _load_model(mp)

    tokens = word_df["token"].tolist()
    n = len(tokens)
    mat = np.zeros((n, 300), dtype=np.float32)
    for i, tok in enumerate(tokens):
        if tok in model:
            mat[i] = model[tok]
        elif tok.lower() in model:
            mat[i] = model[tok.lower()]
    return mat
```

- [ ] **Step 4: Run tests to confirm they pass**

```
conda run -p env python -m pytest tests/extractors/test_word2vec_embed.py -v
```
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add extractors/word2vec_embed.py tests/extractors/test_word2vec_embed.py
git commit -m "feat(embed): add Word2Vec extractor (Google News 300d, OOV→zero)"
```

---

## Task 2: BERT extractor

**Files:**
- Create: `extractors/bert_embed.py`
- Create: `tests/extractors/test_bert_embed.py`

- [ ] **Step 1: Write failing tests (with tiny random-weight BERT config)**

Create `tests/extractors/test_bert_embed.py`:

```python
"""Tests for extractors.bert_embed.

Uses a tiny BertConfig (2 layers, 64 hidden, 2 heads) with random weights
so tests run in <5 s without GPU or real model downloads.
"""
import numpy as np
import pandas as pd
import pytest

transformers = pytest.importorskip("transformers")


def _make_tiny_bert():
    """Return (tokenizer, model) for a 2-layer, 64-dim BERT with random weights."""
    from transformers import BertConfig, BertModel, BertTokenizerFast
    import os, tempfile

    cfg = BertConfig(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=128,
        vocab_size=30522,   # same vocab as bert-base-uncased
        output_hidden_states=True,
    )
    model = BertModel(cfg)
    model.eval()

    # Use the real bert-base-uncased tokenizer vocab (already cached by transformers)
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    return tokenizer, model


def make_word_df(rows):
    """rows = list of (doc_id, sentence_id, sentence_text, token)"""
    return pd.DataFrame([
        {"doc_id": r[0], "sentence_id": r[1], "sentence": r[2],
         "token": r[3], "upos": "NOUN"}
        for r in rows
    ])


def test_word_embed_shape(tmp_path):
    from extractors.bert_embed import embed_bert_words_and_sentences

    tokenizer, model = _make_tiny_bert()
    df = make_word_df([
        ("doc1", 0, "The cat sat.", "The"),
        ("doc1", 0, "The cat sat.", "cat"),
        ("doc1", 0, "The cat sat.", "sat"),
        ("doc1", 1, "It rained.",   "It"),
        ("doc1", 1, "It rained.",   "rained"),
    ])
    word_emb, sent_emb = embed_bert_words_and_sentences(
        df, tokenizer=tokenizer, model=model
    )
    assert word_emb.shape == (5, 2, 64)    # n_words × n_layers × hidden
    assert sent_emb.shape == (2, 2, 64)    # n_sentences × n_layers × hidden


def test_word_embed_dtype_is_float32(tmp_path):
    from extractors.bert_embed import embed_bert_words_and_sentences

    tokenizer, model = _make_tiny_bert()
    df = make_word_df([("doc1", 0, "hello", "hello")])
    word_emb, _ = embed_bert_words_and_sentences(df, tokenizer=tokenizer, model=model)
    assert word_emb.dtype == np.float32


def test_sentence_embed_different_from_each_other(tmp_path):
    """Two sentences with different content should produce different embeddings."""
    from extractors.bert_embed import embed_bert_words_and_sentences

    tokenizer, model = _make_tiny_bert()
    df = make_word_df([
        ("doc1", 0, "The cat sat.",  "The"),
        ("doc1", 0, "The cat sat.",  "cat"),
        ("doc1", 1, "Fish swim fast.", "Fish"),
        ("doc1", 1, "Fish swim fast.", "swim"),
    ])
    _, sent_emb = embed_bert_words_and_sentences(df, tokenizer=tokenizer, model=model)
    assert not np.allclose(sent_emb[0], sent_emb[1])


def test_sliding_window_does_not_exceed_512_tokens():
    """Simulate a long document; assert no tokenization error."""
    from extractors.bert_embed import embed_bert_words_and_sentences

    tokenizer, model = _make_tiny_bert()
    # 40 sentences of 20 words each → well over 512 tokens total
    words_per_sent = 20
    rows = []
    for s in range(40):
        for w in range(words_per_sent):
            rows.append(("doc1", s, f"sentence{s}", f"word{w}"))
    df = make_word_df(rows)
    word_emb, sent_emb = embed_bert_words_and_sentences(
        df, tokenizer=tokenizer, model=model
    )
    assert word_emb.shape[0] == len(df)
    assert sent_emb.shape[0] == 40


def test_punct_excluded_from_sentence_mean():
    """PUNCT tokens should not contribute to sentence embeddings."""
    from extractors.bert_embed import embed_bert_words_and_sentences

    tokenizer, model = _make_tiny_bert()
    # Same sentence, one version has a PUNCT token
    df_with = pd.DataFrame([
        {"doc_id": "d", "sentence_id": 0, "sentence": "Hello.",
         "token": "Hello", "upos": "NOUN"},
        {"doc_id": "d", "sentence_id": 0, "sentence": "Hello.",
         "token": ".", "upos": "PUNCT"},
    ])
    df_without = pd.DataFrame([
        {"doc_id": "d", "sentence_id": 0, "sentence": "Hello",
         "token": "Hello", "upos": "NOUN"},
    ])
    _, sent_with    = embed_bert_words_and_sentences(df_with,    tokenizer=tokenizer, model=model)
    _, sent_without = embed_bert_words_and_sentences(df_without, tokenizer=tokenizer, model=model)
    # Sentence embeddings must be identical when PUNCT is excluded
    np.testing.assert_allclose(sent_with[0], sent_without[0], atol=1e-5)
```

- [ ] **Step 2: Run tests to confirm they fail**

```
conda run -p env python -m pytest tests/extractors/test_bert_embed.py -v
```
Expected: all FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement extractors/bert_embed.py**

Create `extractors/bert_embed.py`:

```python
"""BERT contextual embedding extractor (bert-base-uncased, all 12 layers).

Word-level:     (n_words, 12, 768) float32
                mean of subword token embeddings per word, per layer
Sentence-level: (n_sentences, 12, 768) float32
                mean of non-PUNCT word embeddings per sentence, per layer

Sliding window: left-context sentence-complete.
  - Each sentence gets as much preceding context as fits in 510 effective
    tokens (2 reserved for [CLS]/[SEP]).
  - Context is added by prepending complete preceding sentences (never
    cuts mid-sentence).

Requires: transformers, torch (already in requirements.txt).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from pathlib import Path

_DEFAULT_MODEL_NAME = "bert-base-uncased"
_MAX_EFFECTIVE_TOKENS = 510   # 512 - 2 for [CLS] and [SEP]


def _load_bert(model_name: str):
    from transformers import BertTokenizerFast, BertModel  # type: ignore
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name, output_hidden_states=True)
    model.eval()
    return tokenizer, model


def _sentence_groups(word_df: pd.DataFrame) -> list[dict]:
    """Return list of {doc_id, sent_id, tokens: list[str], indices: list[int], upos: list[str]}"""
    groups = []
    for (doc_id, sent_id), grp in word_df.groupby(
            ["doc_id", "sentence_id"], sort=False):
        groups.append({
            "doc_id":  doc_id,
            "sent_id": sent_id,
            "tokens":  grp["token"].tolist(),
            "indices": grp.index.tolist(),
            "upos":    grp["upos"].tolist() if "upos" in grp.columns else ["NOUN"] * len(grp),
        })
    return groups


def _count_tokens(tokenizer, words: list[str]) -> int:
    enc = tokenizer(words, is_split_into_words=True, add_special_tokens=False)
    return len(enc["input_ids"])


def embed_bert_words_and_sentences(
    word_df: pd.DataFrame,
    *,
    model_name: str = _DEFAULT_MODEL_NAME,
    tokenizer=None,   # injectable for tests
    model=None,       # injectable for tests
) -> tuple[np.ndarray, np.ndarray]:
    """Return (word_embeddings, sentence_embeddings).

    word_embeddings:     (n_words, n_layers, hidden_size) float32
    sentence_embeddings: (n_sentences, n_layers, hidden_size) float32

    n_layers = 12 for bert-base-uncased (2 for the test fixture).
    """
    if tokenizer is None or model is None:
        tokenizer, model = _load_bert(model_name)

    # Infer architecture from the model config
    n_layers = model.config.num_hidden_layers
    hidden   = model.config.hidden_size

    groups = _sentence_groups(word_df)
    n_words     = len(word_df)
    n_sentences = len(groups)

    word_emb = np.zeros((n_words, n_layers, hidden), dtype=np.float32)
    sent_emb = np.zeros((n_sentences, n_layers, hidden), dtype=np.float32)

    # Map original word_df index → position in word_emb
    idx_to_pos = {idx: i for i, idx in enumerate(word_df.index.tolist())}

    for s_pos, s in enumerate(groups):
        current_tokens = s["tokens"]
        n_current      = _count_tokens(tokenizer, current_tokens)

        # Build left-context window: prepend preceding sentences newest-first
        context_token_lists: list[list[str]] = []
        remaining = _MAX_EFFECTIVE_TOKENS - n_current
        for prev_s in reversed(groups[:s_pos]):
            n_prev = _count_tokens(tokenizer, prev_s["tokens"])
            if remaining - n_prev < 0:
                break
            context_token_lists.insert(0, prev_s["tokens"])
            remaining -= n_prev

        # Flatten context
        context_words = [tok for tlist in context_token_lists for tok in tlist]
        n_context_words = len(context_words)
        window_words = context_words + current_tokens

        # Tokenize the window
        enc = tokenizer(
            window_words,
            is_split_into_words=True,
            return_tensors="pt",
            max_length=512,
            truncation=True,
        )
        word_ids = enc.word_ids()  # list[int | None], length = n_input_tokens

        # BERT forward pass
        with torch.no_grad():
            out = model(**enc)

        # hidden_states: tuple of (n_layers+1) tensors, each (1, seq_len, hidden)
        # [0] is embedding layer; [1:] are the n_layers transformer layers
        stacked = torch.stack(out.hidden_states[1:], dim=0)   # (n_layers, 1, seq_len, hidden)
        stacked = stacked.squeeze(1).permute(1, 0, 2).numpy()  # (seq_len, n_layers, hidden)

        # Aggregate subword tokens per word for current sentence
        # word_ids[p] == n_context_words + w_local  ↔  word w_local of current sentence
        content_positions: list[int] = []  # positions of non-PUNCT words in hidden states
        for w_local, (orig_idx, upos) in enumerate(zip(s["indices"], s["upos"])):
            w_global = n_context_words + w_local
            positions = [p for p, wid in enumerate(word_ids) if wid == w_global]
            if not positions:
                continue
            word_vec = stacked[positions].mean(axis=0)  # (n_layers, hidden)
            word_emb[idx_to_pos[orig_idx]] = word_vec
            if upos != "PUNCT":
                content_positions.extend(positions)

        # Sentence embedding: mean of non-PUNCT word positions
        if content_positions:
            sent_emb[s_pos] = stacked[content_positions].mean(axis=0)

    return word_emb, sent_emb
```

- [ ] **Step 4: Run tests to confirm they pass**

```
conda run -p env python -m pytest tests/extractors/test_bert_embed.py -v
```
Expected: all pass (may take ~30 s on CPU due to BERT forward passes; the tiny config keeps it manageable)

- [ ] **Step 5: Commit**

```bash
git add extractors/bert_embed.py tests/extractors/test_bert_embed.py
git commit -m "feat(embed): add BERT extractor (word+sentence, sliding window, all 12 layers)"
```

---

## Task 3: Requirements + download_models.py

**Files:**
- Modify: `requirements.txt`
- Modify: `download_models.py`

- [ ] **Step 1: Add gensim to requirements.txt**

In `requirements.txt`, add:
```
gensim>=4.3
```

- [ ] **Step 2: Add Word2Vec guidance to download_models.py**

At the end of `download_models.py`, after the existing MFA block, add:

```python
print('\n--- Word2Vec (manual download required) ---')
print(
    'Google News 300d is ~1.7 GB compressed (~3.6 GB on disk).\n'
    'Automated download is not available. To use Word2Vec embeddings:\n'
    '  1. Download: https://code.google.com/archive/p/word2vec/\n'
    '     File: GoogleNews-vectors-negative300.bin.gz\n'
    '  2. Decompress: gunzip GoogleNews-vectors-negative300.bin.gz\n'
    '  3. Place at: hf_cache/word2vec/GoogleNews-vectors-negative300.bin\n'
    '  (BERT embeddings work without this file.)'
)
```

- [ ] **Step 3: Commit**

```bash
git add requirements.txt download_models.py
git commit -m "feat(deps): add gensim>=4.3; document Word2Vec manual download in download_models.py"
```

---

## Task 4: Prefs v9 → v10

**Files:**
- Modify: `utils/preferences.py`
- Modify: `tests/utils/test_preferences_migration.py`

> **Note:** If the sentence-level tier batch (plan `2026-05-22-sentence-level-tier.md`) has NOT been merged yet, bump v8 → v10 instead by composing both v9 and v10 changes into a single version. If it HAS been merged (current SCHEMA_VERSION == 9), apply the v9→v10 migration as described below.

- [ ] **Step 1: Write a failing migration test**

Add to `tests/utils/test_preferences_migration.py`:

```python
def test_v9_migrates_to_v10_with_embedding_defaults():
    """v9 prefs gain embedding output toggles and master feature toggles."""
    from utils.preferences import (
        DEFAULT_PREFS_V9, SCHEMA_VERSION_V10,
    )
    import copy, json, os
    from pathlib import Path
    from utils.preferences import load_preferences

    tmp_path = Path(os.environ.get("TMPDIR", "/tmp")) / "prefs_v9_embed_test.json"
    tmp_path.write_text(json.dumps({**DEFAULT_PREFS_V9, "schema_version": 9}))
    orig_env = os.environ.get("SENSI_PREFS_PATH")
    os.environ["SENSI_PREFS_PATH"] = str(tmp_path)
    try:
        prefs = load_preferences()
    finally:
        if orig_env is None:
            del os.environ["SENSI_PREFS_PATH"]
        else:
            os.environ["SENSI_PREFS_PATH"] = orig_env
        tmp_path.unlink(missing_ok=True)

    assert prefs["schema_version"] == SCHEMA_VERSION_V10
    assert prefs["output"]["word2vec_mat"] is True
    assert prefs["output"]["bert_word_mat"] is True
    assert prefs["output"]["bert_sentence_mat"] is True
    assert prefs["features"]["master_word2vec"] is True
    assert prefs["features"]["master_bert_word"] is True
    assert prefs["features"]["master_bert_sentence"] is True
```

- [ ] **Step 2: Run test to confirm it fails**

```
conda run -p env python -m pytest tests/utils/test_preferences_migration.py::test_v9_migrates_to_v10_with_embedding_defaults -v
```
Expected: FAIL — `ImportError: cannot import name 'SCHEMA_VERSION_V10'`

- [ ] **Step 3: Add v10 constants and migration to utils/preferences.py**

In `utils/preferences.py`, after the v9 block, add:

```python
# ---------------------------------------------------------------------------
# v10 constants (active schema)
#
# Embeddings batch (2026-05-22) changes vs v9:
#   - 3 new output toggles: word2vec_mat, bert_word_mat, bert_sentence_mat
#   - 3 new feature master toggles: master_word2vec, master_bert_word,
#     master_bert_sentence
# ---------------------------------------------------------------------------

SCHEMA_VERSION_V10 = 10

_FEATURES_DEFAULT_V10 = {
    **_FEATURES_DEFAULT_V9,
    "master_word2vec":       True,
    "master_bert_word":      True,
    "master_bert_sentence":  True,
}

_OUTPUT_DEFAULT_V10 = {
    **_OUTPUT_DEFAULT_V9,
    "word2vec_mat":      True,
    "bert_word_mat":     True,
    "bert_sentence_mat": True,
}

DEFAULT_PREFS_V10: dict[str, Any] = {
    **DEFAULT_PREFS_V9,
    "schema_version": SCHEMA_VERSION_V10,
    "features": dict(_FEATURES_DEFAULT_V10),
    "output":   dict(_OUTPUT_DEFAULT_V10),
}
```

Update the `ACTIVE SCHEMA ALIASES` block:
```python
SCHEMA_VERSION = SCHEMA_VERSION_V10  # active version
DEFAULT_PREFS: dict[str, Any] = DEFAULT_PREFS_V10  # active defaults
```

Add the migration function:
```python
def _migrate_v9_into_v10(merged: dict, v9: dict) -> None:
    """v9 → v10: additive only. v10 defaults supply all embedding keys."""
    _deep_merge_current(merged, v9)
    merged["schema_version"] = SCHEMA_VERSION_V10
```

Update `load_preferences` to add `elif version == SCHEMA_VERSION_V9:` and extend all existing arms to chain through v10. The pattern is identical to every previous version bump: each arm applies `_migrate_v9_into_v10(merged, v9_intermediate)` as the final step before `_write`.

The full updated chain for `if version == SCHEMA_VERSION_V10:` through the `else:` (v1 path) follows the same pattern as the existing v8 chain in the file — just add one more `_migrate_v9_into_v10` call at the end of each arm and a new `elif version == SCHEMA_VERSION_V9:` arm.

- [ ] **Step 4: Run migration test**

```
conda run -p env python -m pytest tests/utils/test_preferences_migration.py -v
```
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add utils/preferences.py tests/utils/test_preferences_migration.py
git commit -m "feat(prefs): v9 → v10 additive — embedding output toggles and master feature keys"
```

---

## Task 5: Planner registry + emit updates

**Files:**
- Modify: `gui/planner/registry.py`
- Modify: `gui/planner/emit.py`
- Modify: `gui/planner/plan.py`
- Modify: `tests/planner/test_schema_drift.py`

- [ ] **Step 1: Update schema drift test to reference v10 constants**

In `tests/planner/test_schema_drift.py`, update imports:
```python
from utils.preferences import _FEATURES_DEFAULT_V10, _OUTPUT_DEFAULT_V10
```

Update `_INTENTIONALLY_UNROUTED_FEATURES` to add the new master toggles:
```python
_INTENTIONALLY_UNROUTED_FEATURES: set[str] = {
    # ... existing set (unchanged) ...
    "master_word2vec",       # routes via master_toggle field
    "master_bert_word",      # routes via master_toggle field
    "master_bert_sentence",  # routes via master_toggle field
}
```

Update `_OUTPUT_KEYS_VIA_CSV_MAP`:
```python
_OUTPUT_KEYS_VIA_CSV_MAP: set[str] = {
    "word_level_csv", "syllables_csv", "phonemes_csv", "document_summary",
    "sentences_csv",      # sentence-level tier (v9)
    "word2vec_mat",       # embedding output (v10) — routed via _OUTPUT_CSV_MAP
    "bert_word_mat",      # embedding output (v10)
    "bert_sentence_mat",  # embedding output (v10)
    "master_doc_prosody",
    "master_doc_articulation_rate",
    "master_doc_counts",
}
```

Update all four test functions to use `_FEATURES_DEFAULT_V10` / `_OUTPUT_DEFAULT_V10`.

- [ ] **Step 2: Confirm schema drift test fails as expected**

```
conda run -p env python -m pytest tests/planner/test_schema_drift.py -v
```
Expected: `test_every_feature_pref_key_is_in_some_cluster` fails — 3 new master keys unrouted.

- [ ] **Step 3: Add stages + clusters to registry.py**

In `gui/planner/registry.py`, add imports:
```python
from extractors.word2vec_embed import embed_word2vec as _embed_word2vec
from extractors.bert_embed     import embed_bert_words_and_sentences as _embed_bert
```

Add new Stages (after pcenter stage / sentence_prosody stage):
```python
    # --- §5 Embeddings ---
    Stage(name="word2vec_embed",
          consumes=("word_df",),
          produces=("word2vec_mat",),
          run=_embed_word2vec),

    Stage(name="bert_embed",
          consumes=("word_df",),
          produces=("bert_word_mat", "bert_sentence_mat"),
          run=_embed_bert),
```

Add new FeatureClusters (at the end of CLUSTERS, before the final `]`):
```python
    # --- Embeddings (§5) ---

    # §5.1 Word2Vec
    FeatureCluster(
        name="word2vec",
        toggle_keys=(),           # no per-column toggles; master is the gate
        output_csv="word2vec",
        requires_stages=("spacy_parse", "word2vec_embed"),
        emits_columns=(),
        master_toggle="master_word2vec",
    ),

    # §5.2 BERT word-level
    FeatureCluster(
        name="bert word embeddings",
        toggle_keys=(),
        output_csv="bert_word",
        requires_stages=("spacy_parse", "bert_embed"),
        emits_columns=(),
        master_toggle="master_bert_word",
    ),

    # §5.3 BERT sentence-level
    FeatureCluster(
        name="bert sentence embeddings",
        toggle_keys=(),
        output_csv="bert_sentence",
        requires_stages=("spacy_parse", "bert_embed"),
        emits_columns=(),
        master_toggle="master_bert_sentence",
    ),
```

> **Note on empty toggle_keys:** A cluster with empty `toggle_keys` is active iff its output_csv is wanted AND its master_toggle is True. The `_active_clusters` function's step-4 guard `if any(_toggle_value(prefs, k) for k in c.toggle_keys)` will fail for empty `toggle_keys` (any([]) is False). Patch `_active_clusters` in `plan.py` to treat empty toggle_keys as "always active when output_csv is wanted and master is on":

In `gui/planner/plan.py`, update the final cluster-activation condition:
```python
        # 4. any child toggle on (empty toggle_keys = always active)
        if c.toggle_keys and not any(_toggle_value(prefs, k) for k in c.toggle_keys):
            continue
        actives.append(c)
```

- [ ] **Step 4: Update _OUTPUT_CSV_MAP in plan.py**

In `gui/planner/plan.py`, add to `_OUTPUT_CSV_MAP`:
```python
_OUTPUT_CSV_MAP = {
    "word_level_csv":   "features",
    "syllables_csv":    "syllables",
    "phonemes_csv":     "phonemes",
    "sentences_csv":    "sentences",
    "word2vec_mat":     "word2vec",        # new
    "bert_word_mat":    "bert_word",       # new
    "bert_sentence_mat":"bert_sentence",   # new
    "document_summary": "documents_summary",
}
```

- [ ] **Step 5: Update emit.py**

In `gui/planner/emit.py`:

1. Add `"word2vec"`, `"bert_word"`, `"bert_sentence"` to `_WORD_LEVEL_CSVS`:
   ```python
   _WORD_LEVEL_CSVS = frozenset({
       "features", "syllables", "phonemes", "sentences",
       "word2vec", "bert_word", "bert_sentence",
   })
   ```

2. In `to_namespace`, add after the `ns.no_sentences` line:
   ```python
   ns.no_word2vec      = "word2vec"      not in plan.output_csvs
   ns.no_bert_word     = "bert_word"     not in plan.output_csvs
   ns.no_bert_sentence = "bert_sentence" not in plan.output_csvs
   ```

- [ ] **Step 6: Run schema drift tests**

```
conda run -p env python -m pytest tests/planner/ -v
```
Expected: all pass

- [ ] **Step 7: Commit**

```bash
git add gui/planner/registry.py gui/planner/emit.py gui/planner/plan.py tests/planner/test_schema_drift.py
git commit -m "feat(planner): register word2vec_embed + bert_embed stages and clusters for v10"
```

---

## Task 6: main.py — CLI flags + .mat output

**Files:**
- Modify: `main.py`

- [ ] **Step 1: Add embedding CLI flags**

In `build_parser()`, after the sentence-level flags (or after `--no-phonemes` if sentence-level hasn't shipped), add:

```python
    # Embedding sidecar master toggles
    p.add_argument('--no-word2vec',      action='store_true',
                   help='Skip <stim>_word2vec.mat emission.')
    p.add_argument('--no-bert-word',     action='store_true',
                   help='Skip <stim>_bert_word.mat emission.')
    p.add_argument('--no-bert-sentence', action='store_true',
                   help='Skip <stim>_bert_sentence.mat emission.')
```

- [ ] **Step 2: Add .mat emission to run_word_level_batch**

In `run_word_level_batch`, after the sentence sidecar block (or after the phonemes block), add:

```python
            # Word2Vec embedding
            if not getattr(args, 'no_word2vec', False):
                try:
                    import scipy.io
                    from extractors.word2vec_embed import embed_word2vec, Word2VecUnavailable
                    w2v_mat = embed_word2vec(df)
                    mat_path = (_level_dir(output_dir, 'word')
                                / f'{pair.stimulus_id}_word2vec.mat')
                    scipy.io.savemat(str(mat_path), {'embeddings': w2v_mat})
                except Word2VecUnavailable as exc:
                    print(f'[word2vec] skipped ({exc})', file=sys.stderr)
                except Exception as exc:
                    print(f'[word2vec] failed for {pair.stimulus_id}: {exc}',
                          file=sys.stderr)

            # BERT embeddings (shared forward pass for word + sentence)
            run_bert_word = not getattr(args, 'no_bert_word', False)
            run_bert_sent = not getattr(args, 'no_bert_sentence', False)
            if run_bert_word or run_bert_sent:
                try:
                    import scipy.io
                    from extractors.bert_embed import embed_bert_words_and_sentences
                    word_emb, sent_emb = embed_bert_words_and_sentences(df)
                    if run_bert_word:
                        mat_path = (_level_dir(output_dir, 'word')
                                    / f'{pair.stimulus_id}_bert_word.mat')
                        scipy.io.savemat(str(mat_path), {'embeddings': word_emb})
                    if run_bert_sent:
                        mat_path = (_level_dir(output_dir, 'sentence')
                                    / f'{pair.stimulus_id}_bert_sentence.mat')
                        scipy.io.savemat(str(mat_path), {'embeddings': sent_emb})
                except Exception as exc:
                    print(f'[bert] failed for {pair.stimulus_id}: {exc}',
                          file=sys.stderr)
```

- [ ] **Step 3: Run existing tests to confirm no regressions**

```
conda run -p env python -m pytest tests/test_main_cli.py tests/integration/ -v
```
Expected: all pass (embedding blocks are gated by `--no-word2vec` and `--no-bert-*` which are False by default but the test fixtures don't hit them because they pass no such flags AND the Word2Vec model file is absent → Word2VecUnavailable exception is caught silently + BERT load attempt would fail gracefully too).

Actually, the default flags are `--no-word2vec=False`, meaning embeddings WOULD try to run by default. This could break existing tests. Fix: add `--no-word2vec --no-bert-word --no-bert-sentence` to the existing integration test helpers, OR set `getattr(args, 'no_word2vec', True)` (default True so existing args without the flag are skipped). Choose the second approach to avoid updating all test helper functions.

**Change the emission guards to default-skip:**
```python
            if not getattr(args, 'no_word2vec', True):   # default True = skip if flag absent
```
```python
            run_bert_word = not getattr(args, 'no_bert_word', True)
            run_bert_sent = not getattr(args, 'no_bert_sentence', True)
```

The GUI's emit.py sets these flags explicitly from prefs; the CLI and tests that don't pass the flags will silently skip embeddings. This is the right behavior since embedding runs require the model files to be present.

- [ ] **Step 4: Run tests again**

```
conda run -p env python -m pytest tests/test_main_cli.py tests/integration/ -v
```
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add main.py
git commit -m "feat(main): add Word2Vec + BERT .mat emission to word batch (--no-word2vec etc; default off)"
```

---

## Task 7: GUI Embeddings page

**Files:**
- Create: `gui/section_embeddings.py`
- Modify: `gui/main_window.py`

- [ ] **Step 1: Create gui/section_embeddings.py**

```python
"""Embeddings page — Word2Vec and BERT semantic similarity outputs."""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QSizePolicy
)

from gui.widgets import Card, ToggleRow
from utils.preferences import load_preferences, save_preferences


class EmbeddingsSection(QWidget):
    settings_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        prefs = load_preferences()
        f = prefs["features"]

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 16, 24, 16)
        layout.setSpacing(12)

        header = QLabel("Semantic Similarity")
        header.setObjectName("SectionHeader")
        layout.addWidget(header)

        note = QLabel(
            "Outputs are written as .mat files (MATLAB/scipy format) into the "
            "word/ and sentence/ output subfolders. "
            "Word2Vec requires GoogleNews-vectors-negative300.bin (see download_models.py)."
        )
        note.setWordWrap(True)
        note.setObjectName("Subtitle")
        layout.addWidget(note)

        # --- Word2Vec card ---
        w2v_card = Card("Word2Vec (Google News 300d)")
        w2v_toggle = ToggleRow(
            "word/<stim>_word2vec.mat  —  (n_words, 300) float32",
            f.get("master_word2vec", True),
        )
        def _on_w2v(checked):
            save_preferences({"features": {"master_word2vec": checked},
                               "output":   {"word2vec_mat": checked}})
            self.settings_changed.emit()
        w2v_toggle.toggled.connect(_on_w2v)
        w2v_card.layout().addWidget(w2v_toggle)
        layout.addWidget(w2v_card)

        # --- BERT card ---
        bert_card = Card("BERT (bert-base-uncased, all 12 layers)")
        bert_word_toggle = ToggleRow(
            "word/<stim>_bert_word.mat  —  (n_words, 12, 768) float32",
            f.get("master_bert_word", True),
        )
        bert_sent_toggle = ToggleRow(
            "sentence/<stim>_bert_sentence.mat  —  (n_sentences, 12, 768) float32",
            f.get("master_bert_sentence", True),
        )
        def _on_bert_word(checked):
            save_preferences({"features": {"master_bert_word": checked},
                               "output":   {"bert_word_mat": checked}})
            self.settings_changed.emit()
        def _on_bert_sent(checked):
            save_preferences({"features": {"master_bert_sentence": checked},
                               "output":   {"bert_sentence_mat": checked}})
            self.settings_changed.emit()
        bert_word_toggle.toggled.connect(_on_bert_word)
        bert_sent_toggle.toggled.connect(_on_bert_sent)
        bert_card.layout().addWidget(bert_word_toggle)
        bert_card.layout().addWidget(bert_sent_toggle)
        layout.addWidget(bert_card)

        layout.addStretch()
```

- [ ] **Step 2: Update gui/main_window.py**

1. Add import:
   ```python
   from gui.section_embeddings import EmbeddingsSection
   ```

2. Update `SECTIONS` to add "Embeddings":
   ```python
   SECTIONS = ["Inputs", "Word-level features", "Document-level features",
               "Sub-word features", "Sentence features", "Embeddings", "Run", "Logs"]
   ```
   (If sentence-level has not shipped yet, omit "Sentence features" from this list.)

3. In `__init__`, add after `self.sentence_section`:
   ```python
   self.embeddings_section = EmbeddingsSection()
   ```

4. Connect signal:
   ```python
   self.embeddings_section.settings_changed.connect(self.run_section.refresh)
   ```

5. Add to the `sections` list (between `sentence_section` and `run_section`):
   ```python
   self.embeddings_section,
   ```

- [ ] **Step 3: Run GUI smoke test**

```
conda run -p env python -m pytest tests/gui/test_main_window.py -v
```
Expected: pass

- [ ] **Step 4: Commit**

```bash
git add gui/section_embeddings.py gui/main_window.py
git commit -m "feat(gui): add Embeddings page (Word2Vec + BERT toggles)"
```

---

## Task 8: Integration tests for .mat paths

**Files:**
- Modify: `tests/integration/test_output_paths.py`

- [ ] **Step 1: Add .mat path tests**

In `tests/integration/test_output_paths.py`, add:

```python
def test_no_mat_files_by_default(tmp_path):
    """No Word2Vec or BERT model files present → no .mat files emitted, no crash."""
    # Default args don't set no_word2vec/no_bert_* flags, which in main.py
    # getattr-defaults to True (skip). So no .mat files should appear.
    run_word_level_batch(_word_args(tmp_path))
    out = tmp_path / 'out'
    mat_files = list(out.rglob("*.mat"))
    assert mat_files == [], f"Expected no .mat files, found: {mat_files}"


def test_bert_sentence_mat_lands_in_sentence_subfolder(tmp_path, monkeypatch):
    """When BERT runs, bert_sentence.mat lands in sentence/ not word/."""
    import numpy as np
    import scipy.io
    # Monkeypatch bert_embed to avoid real model load
    import extractors.bert_embed as bert_module

    def _fake_bert(word_df, **kwargs):
        n_words = len(word_df)
        n_sents = word_df["sentence_id"].nunique()
        return (
            np.zeros((n_words, 2, 64), dtype=np.float32),
            np.zeros((n_sents, 2, 64), dtype=np.float32),
        )
    monkeypatch.setattr(bert_module, "embed_bert_words_and_sentences", _fake_bert)

    args = _word_args(tmp_path)
    args.no_bert_word     = False
    args.no_bert_sentence = False
    args.no_word2vec      = True
    run_word_level_batch(args)
    out = tmp_path / 'out'
    assert (out / 'word'     / 'story1_bert_word.mat').exists()
    assert (out / 'sentence' / 'story1_bert_sentence.mat').exists()
```

- [ ] **Step 2: Run integration tests**

```
conda run -p env python -m pytest tests/integration/test_output_paths.py -v
```
Expected: all pass including new .mat tests

- [ ] **Step 3: Run full test suite**

```
conda run -p env python -m pytest -x -q
```
Expected: all pass (or same 1 intentional skip)

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_output_paths.py
git commit -m "test(integration): verify .mat embedding output paths and default-off behavior"
```

---

## Task 9: Update master_plan.md

- [ ] **Step 1: Update docs/master_plan.md**

Update §2 (Current state) and §3 (Completed batches) to record the embeddings batch. Update the prefs version comment to v10. Remove "BERT contextual embeddings" from the §4 Roadmap table.

- [ ] **Step 2: Commit**

```bash
git add docs/master_plan.md
git commit -m "docs: record embeddings batch in master_plan; remove from roadmap"
```
