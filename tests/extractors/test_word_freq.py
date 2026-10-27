"""Tests for sensi/extractors/word_freq.py — Zipf frequency lookup."""
import math

import pandas as pd

from sensi.extractors.word_freq import add_word_freq


def _df(lemmas, upos=None):
    upos = upos if upos is not None else ["NOUN"] * len(lemmas)
    return pd.DataFrame({"lemma": lemmas, "upos": upos})


def test_common_word_has_high_zipf():
    out = add_word_freq(_df(["the"]))
    assert out["Zipf_Freq"].iloc[0] > 6.0


def test_punctuation_is_nan():
    out = add_word_freq(_df(["."], upos=["PUNCT"]))
    assert math.isnan(out["Zipf_Freq"].iloc[0])


def test_oov_is_nan_not_zero():
    # wordfreq returns 0.0 as an out-of-vocabulary sentinel, not a real
    # Zipf value. Emitting 0.0 silently dragged down doc-level mean_zipf
    # and looked like an (impossibly) rare-but-attested word. NaN matches
    # the pipeline's "no value" convention (cf. lexical_norms: never imputed).
    out = add_word_freq(_df(["zxqvwjkzzz"]))
    assert math.isnan(out["Zipf_Freq"].iloc[0])


def test_lookup_is_lemma_based():
    # Documented behavior: frequency of the LEMMA column, not the surface
    # token. ("ran" rows carry lemma "run" from spaCy.)
    df = pd.DataFrame({"lemma": ["run"], "token": ["ran"], "upos": ["VERB"]})
    out = add_word_freq(df)
    from wordfreq import zipf_frequency
    assert out["Zipf_Freq"].iloc[0] == zipf_frequency("run", "en")
