"""Tests for the license-clean composite psychological scores.

These mirror the four Pennebaker (2014) summary indices but source their
grammatical category proportions from a spaCy-parsed token DataFrame
(no LIWC dictionary) and valence from an injectable lexicon (VADER in
production). A fake valence lexicon is injected so the suite never needs
the `vaderSentiment` package installed.
"""

import math

import pandas as pd
import pytest

from extractors.psych_composite import compute_psych_composite, EXPECTED_KEYS


# A tiny valence lexicon for deterministic Tone tests.
_FAKE_VALENCE = {
    'happy': 2.0, 'love': 3.0, 'joy': 2.5, 'wonderful': 2.8,
    'sad': -2.0, 'angry': -2.5, 'hate': -3.0, 'terrible': -2.8,
}


def _df(tokens, upos, *, lemma=None, feats=None, dep=None):
    n = len(tokens)
    return pd.DataFrame({
        'token': tokens,
        'upos': upos,
        'lemma': lemma if lemma is not None else [t.lower() for t in tokens],
        'feats': feats if feats is not None else ['NA'] * n,
        'dep_rel': dep if dep is not None else ['dep'] * n,
    })


def test_returns_expected_keys_as_floats():
    df = _df(['The', 'cat', 'ran'], ['DET', 'NOUN', 'VERB'])
    out = compute_psych_composite(df, valence_lexicon=_FAKE_VALENCE)
    assert set(out.keys()) == EXPECTED_KEYS
    for v in out.values():
        assert isinstance(v, float)


def test_empty_df_returns_all_nan():
    df = _df([], [])
    out = compute_psych_composite(df, valence_lexicon=_FAKE_VALENCE)
    for k in EXPECTED_KEYS:
        assert math.isnan(out[k])


def test_punct_only_df_returns_all_nan():
    df = _df(['.', ','], ['PUNCT', 'PUNCT'])
    out = compute_psych_composite(df, valence_lexicon=_FAKE_VALENCE)
    for k in EXPECTED_KEYS:
        assert math.isnan(out[k])


def test_tone_higher_when_positive_words_dominate():
    pos = _df(['happy', 'love', 'joy', 'sad'],
              ['ADJ', 'NOUN', 'NOUN', 'ADJ'])
    neg = _df(['happy', 'sad', 'angry', 'hate'],
              ['ADJ', 'ADJ', 'ADJ', 'NOUN'])
    pos_out = compute_psych_composite(pos, valence_lexicon=_FAKE_VALENCE)
    neg_out = compute_psych_composite(neg, valence_lexicon=_FAKE_VALENCE)
    assert pos_out['tone'] > neg_out['tone']


def test_analytic_higher_for_article_prep_heavy_text():
    # Articles + prepositions raise Analytic; pronouns + auxiliaries lower it.
    formal = _df(['the', 'house', 'on', 'the', 'hill', 'in', 'the', 'valley'],
                 ['DET', 'NOUN', 'ADP', 'DET', 'NOUN', 'ADP', 'DET', 'NOUN'])
    narrative = _df(['i', 'was', 'we', 'were', 'they', 'are'],
                    ['PRON', 'AUX', 'PRON', 'AUX', 'PRON', 'AUX'])
    formal_out = compute_psych_composite(formal, valence_lexicon=_FAKE_VALENCE)
    narr_out = compute_psych_composite(narrative, valence_lexicon=_FAKE_VALENCE)
    assert formal_out['analytic'] > narr_out['analytic']


def test_clout_higher_with_second_third_person_than_first_singular():
    high = _df(['you', 'they', 'we', 'your'],
               ['PRON', 'PRON', 'PRON', 'PRON'])
    low = _df(['i', 'me', 'my', 'mine'],
              ['PRON', 'PRON', 'PRON', 'PRON'])
    high_out = compute_psych_composite(high, valence_lexicon=_FAKE_VALENCE)
    low_out = compute_psych_composite(low, valence_lexicon=_FAKE_VALENCE)
    assert high_out['clout'] > low_out['clout']


def test_robust_to_token_and_upos_only_df():
    """A df lacking lemma/feats/dep_rel must still produce four floats."""
    df = pd.DataFrame({
        'token': ['I', 'think', 'therefore', 'I', 'am'],
        'upos': ['PRON', 'VERB', 'ADV', 'PRON', 'AUX'],
    })
    out = compute_psych_composite(df, valence_lexicon=_FAKE_VALENCE)
    assert set(out.keys()) == EXPECTED_KEYS
    for v in out.values():
        assert not math.isnan(v)  # has words → real scores


def test_scores_clamped_to_0_100():
    # Pathological all-article text would push Analytic high; must clamp.
    df = _df(['the'] * 20, ['DET'] * 20)
    out = compute_psych_composite(df, valence_lexicon=_FAKE_VALENCE)
    for k in EXPECTED_KEYS:
        assert 0.0 <= out[k] <= 100.0


def test_default_valence_lexicon_loads_lazily_or_degrades():
    """Without an injected lexicon the call must not crash.

    If vaderSentiment is installed it loads the real lexicon; if not, the
    extractor still returns all four keys (Tone becomes NaN — see the
    dedicated test below).
    """
    df = _df(['the', 'cat', 'ran'], ['DET', 'NOUN', 'VERB'])
    out = compute_psych_composite(df)  # no lexicon injected
    assert set(out.keys()) == EXPECTED_KEYS


def test_missing_valence_lexicon_yields_nan_tone_not_fake_neutral():
    """No usable valence lexicon → Tone is NaN ("not computed"), never a
    misleading neutral 50. The other three scores still compute."""
    df = _df(['I', 'should', 'go', 'now'], ['PRON', 'AUX', 'VERB', 'ADV'])
    out = compute_psych_composite(df, valence_lexicon={})
    assert math.isnan(out['tone'])
    for k in ('authentic', 'analytic', 'clout'):
        assert not math.isnan(out[k])
