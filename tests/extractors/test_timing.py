import math

import pandas as pd
import pytest

from sensi.extractors.timing import add_timing_columns


def test_word_duration_basic():
    df = pd.DataFrame({
        'token': ['the', 'cat', 'sat'],
        'upos': ['DET', 'NOUN', 'VERB'],
        'Onset': [0.0, 0.5, 1.2],
        'End': [0.4, 1.0, 1.6],
    })
    out = add_timing_columns(df)
    assert list(out['Word_Duration']) == pytest.approx([0.4, 0.5, 0.4])


def test_word_gap_first_word_is_nan():
    df = pd.DataFrame({
        'token': ['the', 'cat'],
        'upos': ['DET', 'NOUN'],
        'Onset': [0.0, 0.5],
        'End': [0.4, 1.0],
    })
    out = add_timing_columns(df)
    assert math.isnan(out['Word_Gap'].iloc[0])
    assert out['Word_Gap'].iloc[1] == pytest.approx(0.1)  # 0.5 - 0.4


def test_nan_in_inputs_propagates():
    df = pd.DataFrame({
        'token': ['the', 'cat', '.'],
        'upos': ['DET', 'NOUN', 'PUNCT'],
        'Onset': [0.0, 0.5, None],
        'End': [0.4, 1.0, None],
    })
    out = add_timing_columns(df)
    assert math.isnan(out['Word_Duration'].iloc[2])
    assert math.isnan(out['Word_Gap'].iloc[2])


def test_word_gap_skips_punctuation_rows():
    # Punctuation rows have no acoustic interval, but the silence between
    # the words on either side of a comma is still a real, measurable gap.
    df = pd.DataFrame({
        'token': ['well', ',', 'yes'],
        'upos': ['INTJ', 'PUNCT', 'INTJ'],
        'Onset': [0.0, None, 0.9],
        'End': [0.4, None, 1.3],
    })
    out = add_timing_columns(df)
    assert out['Word_Gap'].iloc[2] == pytest.approx(0.5)  # 0.9 - 0.4
    assert math.isnan(out['Word_Gap'].iloc[1])


def test_word_gap_spans_unaligned_word():
    # A word MFA couldn't align sits between two aligned words: the gap is
    # measured back to the most recent aligned End (documented behavior).
    df = pd.DataFrame({
        'token': ['one', 'kumquat', 'three'],
        'upos': ['NUM', 'NOUN', 'NUM'],
        'Onset': [0.0, None, 2.0],
        'End': [0.5, None, 2.4],
    })
    out = add_timing_columns(df)
    assert out['Word_Gap'].iloc[2] == pytest.approx(1.5)  # 2.0 - 0.5


def test_no_alignment_columns_emits_nan():
    df = pd.DataFrame({'token': ['the'], 'upos': ['DET']})
    out = add_timing_columns(df)
    assert 'Word_Duration' in out.columns
    assert 'Word_Gap' in out.columns
    assert math.isnan(out['Word_Duration'].iloc[0])
    assert math.isnan(out['Word_Gap'].iloc[0])
