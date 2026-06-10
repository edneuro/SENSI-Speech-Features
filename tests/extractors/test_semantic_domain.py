import math

import pandas as pd
import pytest

from extractors.semantic_domain import (
    add_semantic_domain,
    detect_backend,
    BackendNotAvailable,
)


def test_detect_backend_returns_empath_when_no_dic(tmp_path, monkeypatch):
    # Point dictionary path at empty tmp_path
    backend, dict_path = detect_backend(dictionaries_dir=tmp_path)
    assert backend == 'empath'
    assert dict_path is None


def test_detect_backend_never_auto_selects_liwc_even_with_dic(tmp_path):
    # LIWC is disabled under its EULA — a stray .dic must NOT activate it.
    (tmp_path / 'liwc-2022.dic').write_text('% header %\n')
    backend, dict_path = detect_backend(dictionaries_dir=tmp_path)
    assert backend == 'empath'
    assert dict_path is None


def test_empath_basic_categories(tmp_path):
    df = pd.DataFrame({
        'token': ['mother', 'run', 'the'],
        'lemma': ['mother', 'run', 'the'],
        'upos': ['NOUN', 'VERB', 'DET'],
    })
    out = add_semantic_domain(
        df,
        backend='empath',
        categories=['family', 'motion'],
    )
    assert 'Sem_Empath_family' in out.columns
    assert 'Sem_Empath_motion' in out.columns
    assert 'Sem_Categories' in out.columns
    # Function word gets NaN
    assert math.isnan(out['Sem_Empath_family'].iloc[2])


def test_function_words_get_nan(tmp_path):
    df = pd.DataFrame({
        'token': ['the', 'and'],
        'lemma': ['the', 'and'],
        'upos': ['DET', 'CCONJ'],
    })
    out = add_semantic_domain(df, backend='empath', categories=['social'])
    assert math.isnan(out['Sem_Empath_social'].iloc[0])
    assert math.isnan(out['Sem_Empath_social'].iloc[1])


def test_sem_categories_rollup_format(tmp_path):
    df = pd.DataFrame({
        'token': ['mother'],
        'lemma': ['mother'],
        'upos': ['NOUN'],
    })
    out = add_semantic_domain(df, backend='empath', categories=['family', 'social'])
    rollup = out['Sem_Categories'].iloc[0]
    # mother is in both family and social per Empath
    parts = rollup.split(',') if rollup else []
    assert 'family' in parts or 'social' in parts


def test_unavailable_backend_raises(tmp_path):
    df = pd.DataFrame({'token': ['x'], 'lemma': ['x'], 'upos': ['NOUN']})
    with pytest.raises(BackendNotAvailable):
        add_semantic_domain(df, backend='liwc', categories=['social'],
                            dictionary_path=tmp_path / 'nonexistent.dic')
