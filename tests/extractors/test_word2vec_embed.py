"""Tests for extractors.word2vec_embed.

Uses a mock gensim KeyedVectors so no real model file is needed.
"""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch


def make_word_df(tokens, upos=None):
    data = {"token": tokens}
    if upos is not None:
        data["upos"] = upos
    return pd.DataFrame(data)


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
    mat, is_valid = embed_word2vec(df, model_path="dummy.bin")
    assert mat.shape == (1, 300)
    np.testing.assert_array_equal(mat[0], vec)


@patch("extractors.word2vec_embed._load_model")
def test_oov_word_returns_zero_vector(mock_load):
    from extractors.word2vec_embed import embed_word2vec

    mock_load.return_value = _make_mock_kv({})  # empty vocab
    df = make_word_df(["xyzzy"])
    mat, is_valid = embed_word2vec(df, model_path="dummy.bin")
    assert mat.shape == (1, 300)
    np.testing.assert_array_equal(mat[0], np.zeros(300))


@patch("extractors.word2vec_embed._load_model")
def test_lowercase_fallback(mock_load):
    from extractors.word2vec_embed import embed_word2vec

    vec = np.full(300, 2.0, dtype=np.float32)
    mock_load.return_value = _make_mock_kv({"dog": vec})  # lowercase only

    df = make_word_df(["Dog"])  # capitalized
    mat, is_valid = embed_word2vec(df, model_path="dummy.bin")
    np.testing.assert_array_equal(mat[0], vec)


@patch("extractors.word2vec_embed._load_model")
def test_output_shape_multi_word(mock_load):
    from extractors.word2vec_embed import embed_word2vec

    vec = np.eye(300, dtype=np.float32)[0]
    mock_load.return_value = _make_mock_kv({"the": vec, "cat": vec})
    df = make_word_df(["the", "cat", "sat"])  # "sat" is OOV
    mat, is_valid = embed_word2vec(df, model_path="dummy.bin")
    assert mat.shape == (3, 300)


def test_raises_word2vec_unavailable_when_gensim_missing(tmp_path):
    from extractors.word2vec_embed import embed_word2vec, Word2VecUnavailable
    import sys

    df = make_word_df(["hello"])
    # Block gensim entirely — sys.modules[key]=None makes import raise ImportError.
    with patch.dict(sys.modules, {
        "gensim": None, "gensim.models": None, "gensim.downloader": None
    }):
        with pytest.raises(Word2VecUnavailable, match="not installed"):
            embed_word2vec(df, model_path=str(tmp_path / "nonexistent.bin"))


def test_raises_word2vec_unavailable_when_download_fails(tmp_path):
    from extractors.word2vec_embed import embed_word2vec, Word2VecUnavailable
    import sys

    fake_kv = MagicMock()
    fake_models = MagicMock()
    fake_models.KeyedVectors = fake_kv
    fake_downloader = MagicMock()
    fake_downloader.load.side_effect = Exception("network error")
    fake_gensim = MagicMock()
    fake_gensim.models = fake_models
    fake_gensim.downloader = fake_downloader

    df = make_word_df(["hello"])
    with patch.dict(sys.modules, {
        "gensim": fake_gensim,
        "gensim.models": fake_models,
        "gensim.downloader": fake_downloader,
    }):
        with pytest.raises(Word2VecUnavailable, match="auto-download failed"):
            embed_word2vec(df, model_path=str(tmp_path / "nonexistent.bin"))


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
