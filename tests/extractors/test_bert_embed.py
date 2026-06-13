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

    cfg = BertConfig(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=128,
        vocab_size=30522,
        output_hidden_states=True,
    )
    model = BertModel(cfg)
    model.eval()

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
    word_emb, word_valid, sent_emb, sent_valid = embed_bert_words_and_sentences(
        df, tokenizer=tokenizer, model=model
    )
    assert word_emb.shape == (5, 2, 64)
    assert sent_emb.shape == (2, 2, 64)


def test_word_embed_dtype_is_float32(tmp_path):
    from extractors.bert_embed import embed_bert_words_and_sentences

    tokenizer, model = _make_tiny_bert()
    df = make_word_df([("doc1", 0, "hello", "hello")])
    word_emb, _, _, _ = embed_bert_words_and_sentences(df, tokenizer=tokenizer, model=model)
    assert word_emb.dtype == np.float32


def test_sentence_embed_different_from_each_other(tmp_path):
    from extractors.bert_embed import embed_bert_words_and_sentences

    tokenizer, model = _make_tiny_bert()
    df = make_word_df([
        ("doc1", 0, "The cat sat.",  "The"),
        ("doc1", 0, "The cat sat.",  "cat"),
        ("doc1", 1, "Fish swim fast.", "Fish"),
        ("doc1", 1, "Fish swim fast.", "swim"),
    ])
    _, _, sent_emb, _ = embed_bert_words_and_sentences(df, tokenizer=tokenizer, model=model)
    assert not np.allclose(sent_emb[0], sent_emb[1])


def test_sliding_window_does_not_exceed_512_tokens():
    from extractors.bert_embed import embed_bert_words_and_sentences

    tokenizer, model = _make_tiny_bert()
    words_per_sent = 20
    rows = []
    for s in range(40):
        for w in range(words_per_sent):
            rows.append(("doc1", s, f"sentence{s}", f"word{w}"))
    df = make_word_df(rows)
    word_emb, _, sent_emb, _ = embed_bert_words_and_sentences(
        df, tokenizer=tokenizer, model=model
    )
    assert word_emb.shape[0] == len(df)
    assert sent_emb.shape[0] == 40


def test_punct_excluded_from_sentence_mean():
    from extractors.bert_embed import embed_bert_words_and_sentences

    tokenizer, model = _make_tiny_bert()
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
    _, _, sent_with, _    = embed_bert_words_and_sentences(df_with,    tokenizer=tokenizer, model=model)
    _, _, sent_without, _ = embed_bert_words_and_sentences(df_without, tokenizer=tokenizer, model=model)
    np.testing.assert_allclose(sent_with[0], sent_without[0], atol=1e-5)


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


def test_sentence_valid_mixed():
    from extractors.bert_embed import embed_bert_words_and_sentences

    tokenizer, model = _make_tiny_bert()
    # Sentence 0 has a content word (valid); sentence 1 is all PUNCT (invalid).
    df = pd.DataFrame([
        {"doc_id": "d", "sentence_id": 0, "sentence": "Hello",
         "token": "Hello", "upos": "NOUN"},
        {"doc_id": "d", "sentence_id": 1, "sentence": "...",
         "token": ".", "upos": "PUNCT"},
    ])
    _, _, _, sent_valid = embed_bert_words_and_sentences(
        df, tokenizer=tokenizer, model=model
    )
    assert sent_valid.tolist() == [True, False]


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
