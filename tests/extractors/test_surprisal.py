"""Tests for the pure (model-free) parts of extractors/surprisal.py.

The GPT-2 forward pass itself is not exercised here (500 MB download);
these tests pin the subword→token bookkeeping, the no-context rule for
the document's first word, and the cross-pass cache.
"""
import math

import pandas as pd
import pytest

import extractors.surprisal as m


# --- map_tokens_to_subwords ---------------------------------------------------

def test_simple_two_token_mapping():
    text = "the cat"
    #                 "the"   " cat"  (GPT-2 offsets include the leading space)
    offsets = [(0, 3), (3, 7)]
    spans = m.map_tokens_to_subwords(text, ["the", "cat"], offsets)
    assert spans == [[0], [1]]


def test_straddling_subword_counted_once():
    # spaCy splits "don't" into "do" + "n't"; a GPT-2 subword "don" (0,3)
    # overlaps both pieces. Its log-prob must be charged to exactly one
    # token (the first), never summed into both.
    text = "don't go"
    offsets = [(0, 3), (3, 5), (5, 8)]   # "don", "'t", " go"
    spans = m.map_tokens_to_subwords(text, ["do", "n't", "go"], offsets)
    assert spans == [[0], [1], [2]]


def test_multi_subword_token():
    text = "antidisestablishment now"
    offsets = [(0, 4), (4, 12), (12, 20), (20, 24)]
    spans = m.map_tokens_to_subwords(
        text, ["antidisestablishment", "now"], offsets)
    assert spans == [[0, 1, 2], [3]]


def test_token_not_in_text_gets_empty_span():
    spans = m.map_tokens_to_subwords("abc", ["zzz"], [(0, 3)])
    assert spans == [[]]


# --- surprisal aggregation ----------------------------------------------------

def test_token_surprisal_sums_subword_logprobs_in_bits():
    # Token owns subwords 1 and 2 with natural-log probs -1 and -2:
    # surprisal = 3 / ln(2) bits.
    log_probs = [None, -1.0, -2.0]
    out = m._token_surprisals([[1, 2]], log_probs)
    assert out[0] == pytest.approx(3.0 / math.log(2), abs=1e-4)


def test_first_word_has_no_context_and_is_none():
    # Subword 0 has no preceding context: GPT-2 cannot assign it a
    # probability. The old code scored it 0 bits ("perfectly predictable"),
    # an artifact a researcher could mistake for data. It must be None.
    log_probs = [None, -1.0]
    out = m._token_surprisals([[0], [1]], log_probs)
    assert out[0] is None
    assert out[1] is not None


def test_unmatched_token_is_none():
    assert m._token_surprisals([[]], [None]) == [None]


# --- cross-pass cache -----------------------------------------------------------

def _df(tokens):
    return pd.DataFrame({"token": tokens, "upos": ["NOUN"] * len(tokens)})


def test_add_surprisal_computes_once_per_text(monkeypatch):
    calls = []

    def fake_compute(text, tokens):
        calls.append(text)
        return [1.0] * len(tokens)

    monkeypatch.setattr(m, "_compute_surprisals", fake_compute)
    monkeypatch.setattr(m, "_load", lambda: None)
    m._SURPRISAL_CACHE.clear()

    df = _df(["a", "b"])
    out1 = m.add_surprisal(df, "a b")          # word-level pass
    out2 = m.add_surprisal(df, "a b")          # doc-level pass, same doc
    assert calls == ["a b"]                    # computed once
    assert out1["Surprisal"].tolist() == out2["Surprisal"].tolist()

    m.add_surprisal(_df(["c"]), "c")           # different doc → recompute
    assert calls == ["a b", "c"]


def test_cache_hit_skips_model_load(monkeypatch):
    # The doc-level pass must not pay the GPT-2 load when the word-level
    # pass already scored this exact document.
    def boom():
        raise AssertionError("model loaded despite cache hit")

    monkeypatch.setattr(m, "_compute_surprisals",
                        lambda text, tokens: [1.0] * len(tokens))
    monkeypatch.setattr(m, "_load", lambda: None)
    m._SURPRISAL_CACHE.clear()
    m.add_surprisal(_df(["a"]), "a")

    monkeypatch.setattr(m, "_load", boom)
    m.add_surprisal(_df(["a"]), "a")           # served from cache
