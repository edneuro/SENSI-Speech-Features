import math
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from extractors.liwc_summary import (
    compute_liwc_summary,
    _category_counts_from_text,
    EXPECTED_KEYS,
)


class _FakeParser:
    """Mock LIWC parser. Returns categories per token from a small lookup."""

    def __init__(self, lookup: dict[str, list[str]]):
        self._lookup = lookup

    def __call__(self, token: str) -> list[str]:
        return self._lookup.get(token.lower(), [])


def test_returns_expected_keys():
    parser = _FakeParser({})
    with patch('extractors.liwc_summary._load_parser', return_value=parser):
        out = compute_liwc_summary("hello world", Path('fake.dic'))
    assert set(out.keys()) == EXPECTED_KEYS


def test_tone_positive_higher_when_positive_words_dominate():
    pos = _FakeParser({
        'happy': ['posemo'], 'love': ['posemo'], 'joy': ['posemo'],
        'sad': ['negemo'],
    })
    neg = _FakeParser({
        'happy': ['posemo'],
        'sad': ['negemo'], 'angry': ['negemo'], 'hate': ['negemo'],
    })
    with patch('extractors.liwc_summary._load_parser', return_value=pos):
        pos_out = compute_liwc_summary("happy love joy sad", Path('x'))
    with patch('extractors.liwc_summary._load_parser', return_value=neg):
        neg_out = compute_liwc_summary("happy sad angry hate", Path('x'))
    assert pos_out['tone'] > neg_out['tone']


def test_empty_text_returns_nans():
    parser = _FakeParser({})
    with patch('extractors.liwc_summary._load_parser', return_value=parser):
        out = compute_liwc_summary("", Path('x'))
    for k in EXPECTED_KEYS:
        assert math.isnan(out[k])


def test_category_counts_from_text():
    parser = _FakeParser({
        'the': ['article'], 'cat': [],
        'i': ['i', 'pronoun', 'ppron'],
    })
    counts, n_words = _category_counts_from_text("The cat I i", parser)
    assert n_words == 4
    assert counts.get('article', 0) == 1
    assert counts.get('i', 0) == 2
    assert counts.get('pronoun', 0) == 2
    assert counts.get('ppron', 0) == 2


def test_old_liwc_column_names_are_gone():
    """build_doc_summary must emit psych_* columns, never liwc_* ones."""
    from utils.doc_summary import build_doc_summary, DOC_FEATURE_KEYS

    df = pd.DataFrame({
        'token': ['I', 'think', 'therefore', 'I', 'am'],
        'upos': ['PRON', 'VERB', 'ADV', 'PRON', 'AUX'],
        'sentence_id': [1, 1, 1, 1, 1],
        'paragraph_id': [1, 1, 1, 1, 1],
    })

    parser = _FakeParser({
        'i': ['i', 'pronoun', 'ppron'],
        'think': ['cogproc'],
    })

    toggles = {k: True for k in DOC_FEATURE_KEYS}

    with patch('extractors.liwc_summary._load_parser', return_value=parser):
        out = build_doc_summary(
            df,
            stimulus_id='test',
            dictionary_used='LIWC',
            text='I think therefore I am',
            toggles=toggles,
            liwc_dictionary_path=Path('fake.dic'),
        )

    for old in ('liwc_authentic', 'liwc_analytic', 'liwc_tone', 'liwc_clout'):
        assert old not in out, f"old column {old!r} still present in output"
    for new in ('psych_authentic', 'psych_analytic', 'psych_tone', 'psych_clout'):
        assert new in out, f"new column {new!r} missing from output"
