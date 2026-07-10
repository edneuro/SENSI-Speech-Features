"""Tests for extractors/align_merge.py — token ↔ MFA-word merge.

The merge must be MONOTONIC: token i can never take timing from an
alignment word that belongs before an already-consumed word or arbitrarily
far ahead. The old implementation scanned the whole alignment for the first
unused same-form word, so one tokenization mismatch (a contraction, a digit)
let a token steal timing from a same-form word minutes later — silently.
"""
import numpy as np
import pandas as pd
import pytest

from extractors.align_merge import merge_alignment


def _tok_df(tokens, upos=None):
    upos = upos if upos is not None else ["NOUN"] * len(tokens)
    return pd.DataFrame({"token": tokens, "upos": upos})


def _align_df(words, t0=0.0, dur=0.4, gap=0.1):
    onsets, ends = [], []
    t = t0
    for _ in words:
        onsets.append(round(t, 4))
        ends.append(round(t + dur, 4))
        t += dur + gap
    return pd.DataFrame({"Transcript": words, "Onset": onsets, "End": ends})


# --- basic contract ----------------------------------------------------------

def test_one_to_one_in_order():
    tok = _tok_df(["Hello", "world"])
    ali = _align_df(["hello", "world"])
    out = merge_alignment(tok, ali)
    assert out["Onset"].tolist() == [0.0, 0.5]
    assert out["End"].tolist() == [0.4, 0.9]
    assert out["Transcript"].tolist() == ["hello", "world"]


def test_punctuation_gets_nan_and_consumes_nothing():
    tok = _tok_df(["Hello", ",", "world"], upos=["INTJ", "PUNCT", "NOUN"])
    ali = _align_df(["hello", "world"])
    out = merge_alignment(tok, ali)
    assert pd.isna(out["Onset"].iloc[1])
    assert out["Onset"].iloc[2] == 0.5


def test_does_not_mutate_inputs():
    tok = _tok_df(["a"])
    ali = _align_df(["a"])
    merge_alignment(tok, ali)
    assert "Onset" not in tok.columns
    assert "_norm" not in ali.columns


def test_empty_alignment_yields_nan_columns():
    tok = _tok_df(["hello", "world"])
    out = merge_alignment(tok, pd.DataFrame(columns=["Transcript", "Onset", "End"]))
    assert out["Onset"].isna().all()
    assert list(out.columns[-3:]) == ["Transcript", "Onset", "End"]


def test_repeated_words_assigned_in_order():
    tok = _tok_df(["the", "cat", "the", "dog"])
    ali = _align_df(["the", "cat", "the", "dog"])
    out = merge_alignment(tok, ali)
    assert out["Onset"].tolist() == [0.0, 0.5, 1.0, 1.5]


# --- the theft bug -----------------------------------------------------------

def test_contraction_does_not_steal_later_same_form_word():
    # spaCy splits "don't" -> "do" + "n't"; MFA aligns "don't" as one word.
    # The old merge matched the "do" piece to the LATER standalone "do",
    # assigning it timing from the wrong clause and stealing that row.
    tok = _tok_df(["I", "do", "n't", "know", "what", "to", "do", "now"])
    ali = _align_df(["i", "don't", "know", "what", "to", "do", "now"])
    out = merge_alignment(tok, ali)
    # "do" and "n't" both map onto the acoustic word "don't" (index 1).
    assert out["Onset"].iloc[1] == out["Onset"].iloc[2] == 0.5
    assert out["End"].iloc[1] == out["End"].iloc[2] == 0.9
    assert out["Transcript"].iloc[1] == out["Transcript"].iloc[2] == "don't"
    # The later standalone "do" keeps its own (later) timing.
    assert out["Onset"].iloc[6] == pytest.approx(2.5)
    assert out["Onset"].iloc[7] == pytest.approx(3.0)


def test_unmatched_token_resyncs_instead_of_stealing():
    # "kumquat" was never aligned; "the" appears again later. The unmatched
    # token must become NaN and the later "the" must keep its own timing.
    tok = _tok_df(["the", "kumquat", "sat", "on", "the", "mat"])
    ali = _align_df(["the", "sat", "on", "the", "mat"])
    out = merge_alignment(tok, ali)
    assert pd.isna(out["Onset"].iloc[1])
    assert out["Onset"].iloc[0] == 0.0
    assert out["Onset"].iloc[4] == pytest.approx(1.5)   # 4th aligned word
    assert out["Onset"].iloc[5] == pytest.approx(2.0)


# --- number repair -----------------------------------------------------------

def test_digit_token_matches_single_spelled_word():
    tok = _tok_df(["Bella", "has", "24", "apples"])
    ali = _align_df(["bella", "has", "twenty-four", "apples"])
    out = merge_alignment(tok, ali)
    assert out["Onset"].iloc[2] == pytest.approx(1.0)
    assert out["End"].iloc[2] == pytest.approx(1.4)


def test_digit_token_spans_multiword_spelling():
    # "1,500" -> "one thousand, five hundred": the digit token gets the
    # full acoustic span of the spelled sequence.
    tok = _tok_df(["over", "1,500", "points"])
    ali = _align_df(["over", "one", "thousand", "five", "hundred", "points"])
    out = merge_alignment(tok, ali)
    assert out["Onset"].iloc[1] == pytest.approx(0.5)    # "one" onset
    assert out["End"].iloc[1] == pytest.approx(2.4)      # "hundred" end
    assert out["Onset"].iloc[2] == pytest.approx(2.5)    # "points" unharmed


def test_hyphenated_spelling_split_by_mfa():
    # MFA may tokenize "twenty-four" into two words; the digit token must
    # still span both.
    tok = _tok_df(["has", "24", "apples"])
    ali = _align_df(["has", "twenty", "four", "apples"])
    out = merge_alignment(tok, ali)
    assert out["Onset"].iloc[1] == pytest.approx(0.5)
    assert out["End"].iloc[1] == pytest.approx(1.4)
    assert out["Onset"].iloc[2] == pytest.approx(1.5)


def test_hyphenated_word_token_split_by_mfa():
    tok = _tok_df(["her", "mother-in-law", "arrived"])
    ali = _align_df(["her", "mother", "in", "law", "arrived"])
    out = merge_alignment(tok, ali)
    assert out["Onset"].iloc[1] == pytest.approx(0.5)
    assert out["End"].iloc[1] == pytest.approx(1.9)
    assert out["Onset"].iloc[2] == pytest.approx(2.0)


# --- same-length substitution fallback ---------------------------------------

def test_equal_length_replacement_pairs_positionally():
    # "&" (SYM, not PUNCT) is spoken as "and" — same position, one word.
    tok = _tok_df(["salt", "&", "pepper"], upos=["NOUN", "SYM", "NOUN"])
    ali = _align_df(["salt", "and", "pepper"])
    out = merge_alignment(tok, ali)
    assert out["Onset"].iloc[1] == pytest.approx(0.5)
    assert out["Transcript"].iloc[1] == "and"


# --- case / normalization ----------------------------------------------------

def test_case_insensitive_match():
    tok = _tok_df(["HELLO", "World"])
    ali = _align_df(["hello", "world"])
    out = merge_alignment(tok, ali)
    assert out["Onset"].notna().all()


def test_symbol_only_token_gets_nan():
    tok = _tok_df(["wait", "—", "what"], upos=["VERB", "SYM", "PRON"])
    ali = _align_df(["wait", "what"])
    out = merge_alignment(tok, ali)
    assert pd.isna(out["Onset"].iloc[1])
    assert out["Onset"].iloc[2] == pytest.approx(0.5)


# --- reporting ---------------------------------------------------------------

def test_match_summary_printed(capsys):
    tok = _tok_df(["hello", "world"])
    ali = _align_df(["hello", "world"])
    merge_alignment(tok, ali)
    assert "[merge] aligned 2/2" in capsys.readouterr().out
