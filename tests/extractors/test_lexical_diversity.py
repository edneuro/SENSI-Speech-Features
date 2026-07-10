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


def test_mtld_all_unique_is_nan():
    # A text whose running TTR never reaches the threshold completes zero
    # factors and zero partial factor ((1-1.0)/(1-0.72) = 0): MTLD is
    # undefined (McCarthy & Jarvis 2010). The old code fabricated a
    # "1 full factor" here, which made MTLD equal the token count — a
    # value linear in text length, exactly the artifact MTLD exists to
    # remove. Degenerate for real English (function words repeat).
    tokens = [f'word{i}' for i in range(100)]
    assert math.isnan(compute_mtld(tokens))


def test_mtld_hand_computed_value():
    # tokens: a b a c d  (threshold 0.72)
    # forward: TTRs 1, 1, 2/3<=0.72 -> factor, reset; then c, d -> TTR 1.0
    #          -> partial (1-1.0)/0.28 = 0  -> factors=1 -> 5/1 = 5
    # reverse (d c a b a): TTRs 1,1,1,1, 4/5=0.8 -> no factor;
    #          partial (1-0.8)/0.28 = 0.714286 -> 5/0.714286 = 7
    # MTLD = mean(5, 7) = 6
    assert compute_mtld(['a', 'b', 'a', 'c', 'd']) == pytest.approx(6.0)


def test_mtld_partial_factor_is_proportional_not_binary():
    # tokens: a b a  ->  forward: factor completes exactly at token 3
    # (TTR 2/3 <= 0.72), nothing left over -> 3/1 = 3.
    # reverse (a b a) identical -> MTLD = 3.
    assert compute_mtld(['a', 'b', 'a']) == pytest.approx(3.0)


def test_mtld_repetitive_text_lower_than_diverse():
    repetitive = ['the', 'cat', 'the', 'cat'] * 25          # very repetitive
    diverse = [f'word{i % 50}' for i in range(200)]         # 50-type cycle
    assert compute_mtld(diverse) > compute_mtld(repetitive)
