import math

import pytest

from extractors.lexical_diversity import compute_ttr, compute_mtld


def test_ttr_basic():
    # 6 tokens, 5 unique
    tokens = ['the', 'cat', 'sat', 'on', 'the', 'mat']
    assert compute_ttr(tokens) == pytest.approx(5 / 6)


def test_ttr_all_unique():
    tokens = ['alpha', 'beta', 'gamma']
    assert compute_ttr(tokens) == 1.0


def test_ttr_empty_returns_nan():
    assert math.isnan(compute_ttr([]))


def test_ttr_case_insensitive():
    tokens = ['The', 'cat', 'THE', 'CAT']
    assert compute_ttr(tokens) == pytest.approx(2 / 4)


def test_mtld_empty_returns_nan():
    assert math.isnan(compute_mtld([]))


def test_mtld_single_token():
    # Single token: TTR is 1.0, never crosses threshold, partial factor case
    result = compute_mtld(['hello'])
    # With threshold 0.72: partial factor = (1 - 1.0) / (1 - 0.72) = 0
    # Factor count = 0 + 0 = 0 in one direction; both passes averaged
    # Edge case: when no factor was reached, fall back to NaN (ill-defined)
    assert math.isnan(result)


def test_mtld_uniform_returns_finite():
    # 100 unique tokens => never drops below threshold => 1 partial factor
    tokens = [f'word{i}' for i in range(100)]
    result = compute_mtld(tokens)
    assert not math.isnan(result)
    assert result > 0


def test_mtld_repetitive_text_lower_than_diverse():
    repetitive = ['the', 'cat', 'the', 'cat'] * 25  # very repetitive
    diverse = [f'word{i}' for i in range(100)]
    assert compute_mtld(diverse) > compute_mtld(repetitive)
