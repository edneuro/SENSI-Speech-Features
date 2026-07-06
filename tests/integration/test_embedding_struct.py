"""savemat -> loadmat roundtrip for the embedding-struct helpers in main.py."""
import numpy as np
import pandas as pd
import scipy.io

from main import _word_embedding_vars, _sentence_embedding_vars


def _word_df():
    return pd.DataFrame([
        {"doc_id": 1, "sentence_id": 1, "sentence": "The cat sat.",
         "token_id": 1, "token": "The"},
        {"doc_id": 1, "sentence_id": 1, "sentence": "The cat sat.",
         "token_id": 2, "token": "cat"},
        {"doc_id": 1, "sentence_id": 1, "sentence": "The cat sat.",
         "token_id": 3, "token": "."},
        {"doc_id": 1, "sentence_id": 2, "sentence": "It rained.",
         "token_id": 4, "token": "It"},
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
    # object array -> MATLAB cellstr; each cell loads as a 1-element str array.
    toks = [str(t.item()) for t in loaded["tokens"].ravel()]
    assert toks == ["The", "cat", ".", "It"]


def test_sentence_vars_roundtrip(tmp_path):
    df = _word_df()
    emb = np.arange(2 * 3, dtype=np.float32).reshape(2, 3)
    is_valid = np.array([True, False])  # second sentence invalid (e.g. all-PUNCT)

    out = _sentence_embedding_vars(emb, df, is_valid)
    assert set(out) == {"embeddings", "sentence_id", "sentence", "is_valid"}

    p = tmp_path / "s.mat"
    scipy.io.savemat(str(p), out)
    loaded = scipy.io.loadmat(str(p))

    # one row per unique (doc_id, sentence_id), in first-seen order
    assert loaded["sentence_id"].ravel().tolist() == [1, 2]
    assert loaded["is_valid"].ravel().astype(bool).tolist() == [True, False]
    sents = [str(s.item()) for s in loaded["sentence"].ravel()]
    assert sents == ["The cat sat.", "It rained."]
    np.testing.assert_array_equal(loaded["embeddings"], emb)
