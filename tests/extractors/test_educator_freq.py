import math

import pandas as pd
import pytest

from extractors.educator_freq import add_edu_freq, EXPECTED_COLUMNS


def _write_lookup(tmp_path):
    """Create a small parquet lookup with a few words."""
    import pandas as pd
    lookup = pd.DataFrame([
        {'word': 'cat',  'edu_zipf_k2': 5.5, 'edu_zipf_3to5': 5.2, 'edu_zipf_6to8': 4.9, 'edu_zipf_9to12': 4.5, 'edu_zipf_overall': 5.0},
        {'word': 'photosynthesis', 'edu_zipf_k2': 0.0, 'edu_zipf_3to5': 1.0, 'edu_zipf_6to8': 3.5, 'edu_zipf_9to12': 4.2, 'edu_zipf_overall': 2.9},
    ])
    path = tmp_path / 'edu.parquet'
    lookup.to_parquet(path)
    return path


def test_columns_added(tmp_path):
    df = pd.DataFrame({'token': ['cat'], 'lemma': ['cat'], 'upos': ['NOUN']})
    out = add_edu_freq(df, lookup_path=_write_lookup(tmp_path))
    for col in EXPECTED_COLUMNS:
        assert col in out.columns


def test_known_word_values(tmp_path):
    df = pd.DataFrame({'token': ['cat'], 'lemma': ['cat'], 'upos': ['NOUN']})
    out = add_edu_freq(df, lookup_path=_write_lookup(tmp_path))
    assert out['Edu_Zipf_K2'].iloc[0] == pytest.approx(5.5)
    assert out['Edu_Zipf_Overall'].iloc[0] == pytest.approx(5.0)


def test_oov_returns_nan(tmp_path):
    df = pd.DataFrame({'token': ['xyzzy'], 'lemma': ['xyzzy'], 'upos': ['NOUN']})
    out = add_edu_freq(df, lookup_path=_write_lookup(tmp_path))
    assert math.isnan(out['Edu_Zipf_K2'].iloc[0])


def test_punct_gets_nan(tmp_path):
    df = pd.DataFrame({'token': ['.'], 'lemma': ['.'], 'upos': ['PUNCT']})
    out = add_edu_freq(df, lookup_path=_write_lookup(tmp_path))
    assert math.isnan(out['Edu_Zipf_Overall'].iloc[0])


def test_missing_lookup_emits_nan_columns_with_warning(tmp_path, caplog):
    df = pd.DataFrame({'token': ['cat'], 'lemma': ['cat'], 'upos': ['NOUN']})
    out = add_edu_freq(df, lookup_path=tmp_path / 'nonexistent.parquet')
    for col in EXPECTED_COLUMNS:
        assert col in out.columns
    assert math.isnan(out['Edu_Zipf_K2'].iloc[0])


def test_band_filter_emits_only_selected_columns(tmp_path):
    """When only k2 and overall are requested, other band columns must be absent."""
    df = pd.DataFrame({'token': ['cat', 'feline', 'the'], 'lemma': ['cat', 'feline', 'the'], 'upos': ['NOUN', 'NOUN', 'DET']})
    out = add_edu_freq(df, lookup_path=_write_lookup(tmp_path), bands=['k2', 'overall'])
    assert 'Edu_Zipf_K2' in out.columns
    assert 'Edu_Zipf_Overall' in out.columns
    assert 'Edu_Zipf_3to5' not in out.columns
    assert 'Edu_Zipf_6to8' not in out.columns
    assert 'Edu_Zipf_9to12' not in out.columns


def test_band_filter_all_bands_matches_previous_behavior(tmp_path):
    df = pd.DataFrame({'token': ['cat'], 'lemma': ['cat'], 'upos': ['NOUN']})
    out = add_edu_freq(df, lookup_path=_write_lookup(tmp_path), bands=['k2', '3to5', '6to8', '9to12', 'overall'])
    for col in ('Edu_Zipf_K2', 'Edu_Zipf_3to5', 'Edu_Zipf_6to8',
                'Edu_Zipf_9to12', 'Edu_Zipf_Overall'):
        assert col in out.columns
