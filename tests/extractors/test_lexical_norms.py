"""Tests for the lexical-norms lookup extractor.

Norm files are user-supplied (never shipped); these tests write small temp
parquets and pass their paths, exercising the real lemma-keyed lookup,
token fallback, MRC 0→NaN handling, and missing-file degradation.
"""

import math

import numpy as np
import pandas as pd
import pytest

from extractors.lexical_norms import (
    add_lexical_norms, ALL_OUTPUT_COLUMNS, _lookup_cache,
)


@pytest.fixture(autouse=True)
def _clear_cache():
    _lookup_cache.clear()
    yield
    _lookup_cache.clear()


def _df(tokens, upos=None, lemma=None):
    n = len(tokens)
    return pd.DataFrame({
        'token': tokens,
        'upos': upos if upos is not None else ['NOUN'] * n,
        'lemma': lemma if lemma is not None else [t.lower() for t in tokens],
    })


def _make_parquets(tmp_path):
    pd.DataFrame({'word': ['dog', 'truth'],
                  'aoa_mean': [4.0, 9.5],
                  'aoa_sd': [1.2, 2.0]}).to_parquet(tmp_path / 'aoa.parquet')
    pd.DataFrame({'word': ['dog', 'truth'],
                  'concreteness': [4.85, 1.95]}).to_parquet(tmp_path / 'conc.parquet')
    # MRC: 'truth' imageability stored as 0 == missing
    pd.DataFrame({'word': ['dog', 'truth'],
                  'imageability': [600, 0],
                  'familiarity': [500, 450]}).to_parquet(tmp_path / 'mrc.parquet')
    pd.DataFrame({'word': ['dog', 'truth'],
                  'valence': [6.5, 5.0],
                  'arousal': [4.0, 3.5],
                  'dominance': [5.5, 5.0]}).to_parquet(tmp_path / 'warr.parquet')
    return {
        'aoa':          tmp_path / 'aoa.parquet',
        'concreteness': tmp_path / 'conc.parquet',
        'mrc':          tmp_path / 'mrc.parquet',
        'warriner':     tmp_path / 'warr.parquet',
    }


def test_adds_all_columns_with_values(tmp_path):
    paths = _make_parquets(tmp_path)
    out = add_lexical_norms(_df(['dog']), paths=paths)
    for col in ALL_OUTPUT_COLUMNS:
        assert col in out.columns
    row = out.iloc[0]
    assert row['AoA_Mean'] == 4.0
    assert row['AoA_SD'] == 1.2
    assert row['Concreteness'] == 4.85
    assert row['Imageability'] == 600
    assert row['Familiarity'] == 500
    assert row['Valence'] == 6.5
    assert row['Arousal'] == 4.0
    assert row['Dominance'] == 5.5


def test_oov_is_nan_never_imputed(tmp_path):
    paths = _make_parquets(tmp_path)
    out = add_lexical_norms(_df(['xyzzy']), paths=paths)
    for col in ALL_OUTPUT_COLUMNS:
        assert math.isnan(out.iloc[0][col]), col


def test_punct_rows_are_nan(tmp_path):
    paths = _make_parquets(tmp_path)
    out = add_lexical_norms(_df(['dog', '.'], upos=['NOUN', 'PUNCT']), paths=paths)
    assert out.iloc[0]['Concreteness'] == 4.85
    assert math.isnan(out.iloc[1]['Concreteness'])


def test_mrc_zero_treated_as_missing(tmp_path):
    paths = _make_parquets(tmp_path)
    out = add_lexical_norms(_df(['truth']), paths=paths)
    # imageability stored as 0 → NaN, but familiarity 450 is a real value
    assert math.isnan(out.iloc[0]['Imageability'])
    assert out.iloc[0]['Familiarity'] == 450


def test_token_fallback_when_lemma_misses(tmp_path):
    paths = _make_parquets(tmp_path)
    # lemma 'nonword' isn't in the table; surface token 'dog' is
    out = add_lexical_norms(_df(['dog'], lemma=['nonword']), paths=paths)
    assert out.iloc[0]['Concreteness'] == 4.85


def test_missing_file_yields_nan_for_that_source_only(tmp_path):
    paths = _make_parquets(tmp_path)
    paths['aoa'] = tmp_path / 'does_not_exist.parquet'
    out = add_lexical_norms(_df(['dog']), paths=paths)
    assert math.isnan(out.iloc[0]['AoA_Mean'])
    assert math.isnan(out.iloc[0]['AoA_SD'])
    # other sources still resolve
    assert out.iloc[0]['Concreteness'] == 4.85


def test_concrete_vs_abstract_known_values(tmp_path):
    paths = _make_parquets(tmp_path)
    out = add_lexical_norms(_df(['dog', 'truth']), paths=paths)
    assert out.iloc[0]['Concreteness'] > out.iloc[1]['Concreteness']
