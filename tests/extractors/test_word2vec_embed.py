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
