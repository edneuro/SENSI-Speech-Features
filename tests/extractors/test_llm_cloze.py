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
        assert select_model() == 'meta-llama/Meta-Llama-3-8B'
    with patch('extractors.llm_cloze._available_vram_gb', return_value=50.0):
        assert select_model() == 'meta-llama/Meta-Llama-3-70B'


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


def test_is_gated_local_model():
    from extractors.llm_cloze import is_gated_local_model
    assert is_gated_local_model('meta-llama/Meta-Llama-3-8B') is True
    assert is_gated_local_model('meta-llama/Llama-3.2-3B') is True
    assert is_gated_local_model('gpt-4o-mini') is False


def test_has_hf_token_true_when_get_token_returns(monkeypatch):
    import extractors.llm_cloze as m
    monkeypatch.setattr('huggingface_hub.get_token', lambda: 'hf_abc', raising=False)
    assert m.has_hf_token() is True


def test_has_hf_token_false_when_no_token(monkeypatch):
    import extractors.llm_cloze as m
    monkeypatch.setattr('huggingface_hub.get_token', lambda: None, raising=False)
    assert m.has_hf_token() is False


import types
import pytest


def test_backend_raises_actionable_error_on_gated(monkeypatch):
    import extractors.llm_cloze as m

    def raise_gated(*a, **k):
        raise OSError("401 Client Error: gated repo")

    monkeypatch.setattr('transformers.AutoTokenizer.from_pretrained', raise_gated)
    with pytest.raises(RuntimeError) as ei:
        m._HuggingFaceBackend('meta-llama/Meta-Llama-3-8B')
    msg = str(ei.value)
    assert 'license' in msg and 'token' in msg
    assert 'meta-llama/Meta-Llama-3-8B' in msg


# ---- device-tier cascade for the local HuggingFace backend ----


class _DummyParam:
    def __init__(self, device):
        self.device = device


class _DummyModel:
    def __init__(self, device='cpu'):
        self.config = types.SimpleNamespace(max_position_embeddings=1024)
        self._device = device

    def eval(self):
        return self

    def parameters(self):
        yield _DummyParam(self._device)


def _patch_transformers(monkeypatch, fake_model):
    monkeypatch.setattr('transformers.AutoTokenizer.from_pretrained',
                        lambda *a, **k: object())
    monkeypatch.setattr('transformers.AutoModelForCausalLM.from_pretrained', fake_model)
    monkeypatch.setattr('transformers.BitsAndBytesConfig', lambda **k: object())


def test_load_tiers_prefers_gpu_only_when_cuda():
    from extractors.llm_cloze import _load_tiers
    assert _load_tiers(True) == ['4-bit GPU', 'fp16 GPU', 'CPU']
    assert _load_tiers(False) == ['CPU']


def test_backend_loads_4bit_on_gpu(monkeypatch):
    import extractors.llm_cloze as m
    monkeypatch.setattr('torch.cuda.is_available', lambda: True)
    _patch_transformers(monkeypatch, lambda model_id, **kw: _DummyModel('cuda:0'))
    b = m._HuggingFaceBackend('meta-llama/Meta-Llama-3-8B')
    assert b._load_tier == '4-bit GPU'


def test_backend_falls_back_to_fp16_gpu_on_bnb_error(monkeypatch):
    """bitsandbytes broken (typical on Windows) but a GPU is present → fp16 GPU,
    NOT a silent drop to CPU."""
    import extractors.llm_cloze as m
    monkeypatch.setattr('torch.cuda.is_available', lambda: True)
    calls = []

    def fake_model(model_id, **kw):
        calls.append(kw)
        if 'quantization_config' in kw:
            raise RuntimeError("bitsandbytes not available on this platform")
        return _DummyModel('cuda:0')

    _patch_transformers(monkeypatch, fake_model)
    b = m._HuggingFaceBackend('meta-llama/Meta-Llama-3-8B')
    assert b._load_tier == 'fp16 GPU'
    assert any('quantization_config' in c for c in calls)          # tried 4-bit
    assert any(c.get('torch_dtype') is not None for c in calls)     # then fp16
    assert b._context_limit == 1024


def test_backend_falls_back_to_cpu_and_warns(monkeypatch, caplog):
    """Both GPU loads fail while CUDA is present → CPU, with a loud warning so
    the CPU landing is never silent."""
    import logging
    import extractors.llm_cloze as m
    monkeypatch.setattr('torch.cuda.is_available', lambda: True)

    def fake_model(model_id, **kw):
        if kw.get('device_map') == 'auto':      # both GPU tiers use device_map
            raise RuntimeError("gpu load failed")
        return _DummyModel('cpu')

    _patch_transformers(monkeypatch, fake_model)
    with caplog.at_level(logging.WARNING):
        b = m._HuggingFaceBackend('meta-llama/Meta-Llama-3-8B')
    assert b._load_tier == 'CPU'
    assert 'CPU' in caplog.text and 'slower' in caplog.text


def test_backend_cpu_only_when_no_cuda(monkeypatch, caplog):
    """No GPU at all → only the CPU tier is attempted (no bnb/CUDA load), and
    no false 'GPU idle' warning."""
    import logging
    import extractors.llm_cloze as m
    monkeypatch.setattr('torch.cuda.is_available', lambda: False)
    calls = []

    def fake_model(model_id, **kw):
        calls.append(kw)
        return _DummyModel('cpu')

    _patch_transformers(monkeypatch, fake_model)
    with caplog.at_level(logging.WARNING):
        b = m._HuggingFaceBackend('meta-llama/Llama-3.2-3B')
    assert b._load_tier == 'CPU'
    assert calls == [{}]                                    # no GPU load attempted
    assert not any(r.levelno >= logging.WARNING for r in caplog.records)
