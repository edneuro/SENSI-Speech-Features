import math
from unittest.mock import patch

import pandas as pd
import pytest

from extractors.llm_cloze import (
    add_llm_cloze,
    select_model,
    EXPECTED_COLUMNS,
    _check_context_limit,
    _chunk_slices,
    _OPENAI_CONTEXT_LIMITS,
)


class _FakeBackend:
    """Stand-in for an LLM backend used in unit tests."""

    def score(self, text: str, tokens: list[str]) -> list[dict]:
        # Return a fixed pattern: first token gets P=0.5; rest get 0.1
        out = []
        for i, tok in enumerate(tokens):
            p = 0.5 if i == 0 else 0.1
            out.append({
                'p': p,
                'surprisal': -math.log2(p),
                'top_match': 1 if i == 0 else 0,
            })
        return out


def test_columns_added():
    df = pd.DataFrame({'token': ['the', 'cat'], 'upos': ['DET', 'NOUN']})
    out = add_llm_cloze(df, text='the cat', backend=_FakeBackend())
    for col in EXPECTED_COLUMNS:
        assert col in out.columns


def test_punct_gets_nan():
    df = pd.DataFrame({'token': ['the', '.'], 'upos': ['DET', 'PUNCT']})
    out = add_llm_cloze(df, text='the .', backend=_FakeBackend())
    assert math.isnan(out['LLM_Cloze'].iloc[1])
    assert math.isnan(out['LLM_Surprisal'].iloc[1])
    assert math.isnan(out['LLM_TopMatch'].iloc[1])


def test_metric_values():
    df = pd.DataFrame({'token': ['hi', 'there'], 'upos': ['INTJ', 'ADV']})
    out = add_llm_cloze(df, text='hi there', backend=_FakeBackend())
    assert out['LLM_Cloze'].iloc[0] == pytest.approx(0.5)
    assert out['LLM_Cloze'].iloc[1] == pytest.approx(0.1)
    assert out['LLM_TopMatch'].iloc[0] == 1
    assert out['LLM_TopMatch'].iloc[1] == 0


def test_select_model_uses_openai_key(monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'sk-test')
    assert select_model() == 'gpt-4o-mini'


def test_select_model_picks_local_when_no_api_key(monkeypatch):
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    with patch('extractors.llm_cloze._available_vram_gb', return_value=8.0):
        assert select_model() == 'meta-llama/Llama-3.2-3B'
    with patch('extractors.llm_cloze._available_vram_gb', return_value=14.0):
        assert select_model() == 'meta-llama/Llama-3-8B'
    with patch('extractors.llm_cloze._available_vram_gb', return_value=50.0):
        assert select_model() == 'meta-llama/Llama-3-70B'


def test_context_limit_passes_when_under():
    _check_context_limit(n_subwords=1000, limit=1024, model_id='m')
    _check_context_limit(n_subwords=1024, limit=1024, model_id='m')


def test_context_limit_passes_when_limit_none():
    _check_context_limit(n_subwords=10**9, limit=None, model_id='unknown-model')


def test_context_limit_raises_with_helpful_message():
    # OpenAI is the only backend that can't chunk; the error fires there.
    with pytest.raises(ValueError) as excinfo:
        _check_context_limit(n_subwords=200_000, limit=128_000, model_id='gpt-4o-mini')
    msg = str(excinfo.value)
    assert '200000' in msg
    assert '128000' in msg
    assert 'gpt-4o-mini' in msg
    assert 'Split' in msg or 'split' in msg


def test_openai_context_limit_table_covers_listed_models():
    # Every OpenAI model in LLM_OPTIONS should have a known context limit so
    # _OpenAIBackend.score can validate before incurring API cost.
    from extractors.llm_cloze import LLM_OPTIONS
    for model_id, _label in LLM_OPTIONS:
        if model_id.startswith('gpt-'):
            assert model_id in _OPENAI_CONTEXT_LIMITS, (
                f"{model_id!r} listed in LLM_OPTIONS but missing from "
                f"_OPENAI_CONTEXT_LIMITS — score() will skip the overflow check."
            )


# ---- chunk-slice iteration for long-text handling in _HuggingFaceBackend ----


def test_chunk_slices_handles_short_input():
    # Caller branches to a single forward pass when n <= window; this helper
    # always chunks at window//2. Verify it still produces correct coverage
    # if invoked on a short input.
    slices = _chunk_slices(n=3, window=4)
    # chunk=2: pass 1 keeps positions 0..1, pass 2 keeps position 2
    assert slices == [(0, 2, 0), (0, 3, 2)]


def test_chunk_slices_exact_window_multiples():
    # n=8, window=4 (stride=2). New positions per pass = 2.
    # Expected coverage: 0..1, 2..3, 4..5, 6..7
    slices = _chunk_slices(n=8, window=4)
    assert slices == [
        (0, 2, 0),   # seg ids[0:2], keep all 2 → positions 0,1
        (0, 4, 2),   # seg ids[0:4], keep last 2 → positions 2,3
        (2, 6, 2),   # seg ids[2:6], keep last 2 → positions 4,5
        (4, 8, 2),   # seg ids[4:8], keep last 2 → positions 6,7
    ]


def test_chunk_slices_uneven_tail():
    # n=10, window=4 (stride=2). Last chunk should land short.
    slices = _chunk_slices(n=10, window=4)
    assert slices == [
        (0, 2, 0),
        (0, 4, 2),
        (2, 6, 2),
        (4, 8, 2),
        (6, 10, 2),
    ]
    # Last segment is ids[6:10] (length 4), keep_off=2 → 2 new positions (8,9).


def test_chunk_slices_coverage_no_overlap_no_gap():
    # The union of segment[keep_off:] ranges (in original-index space) must
    # cover 0..n-1 exactly once.
    for n, window in [(10, 4), (17, 6), (100, 8), (1, 4), (5, 4)]:
        covered = []
        for seg_start, seg_end, keep_off in _chunk_slices(n, window):
            covered.extend(range(seg_start + keep_off, seg_end))
        assert covered == list(range(n)), f"failed for n={n}, window={window}"


def test_chunk_slices_segment_never_exceeds_window():
    for n, window in [(10, 4), (50, 8), (1000, 128)]:
        for seg_start, seg_end, _keep in _chunk_slices(n, window):
            assert seg_end - seg_start <= window


def test_chunk_slices_rejects_tiny_window():
    with pytest.raises(ValueError):
        _chunk_slices(n=100, window=1)
