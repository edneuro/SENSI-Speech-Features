import math
import types

import numpy as np
import pandas as pd
import pytest
from sensi.extractors.predictability import (
    add_predictability,
    map_tokens_to_subwords,
    _token_surprisals,
    _scores_from_subwords,
    _chunk_slices,
    _check_context_limit,
    _OPENAI_CONTEXT_LIMITS,
    _make_default_backend,
    _load_tiers,
    is_gated_local_model,
    has_hf_token,
    select_model,
    LLM_OPTIONS,
)

class _Fake:
    """Backend returning fixed per-token scores; first token is context-less."""
    def score(self, text, tokens):
        out = []
        for i, _ in enumerate(tokens):
            if i == 0:
                out.append({"p": float("nan"), "surprisal": float("nan"),
                            "top_match": None})
            else:
                out.append({"p": 0.5, "surprisal": 1.0, "top_match": 1})
        return out

def _df():
    return pd.DataFrame({"token": ["The", "cat", "."],
                         "upos": ["DET", "NOUN", "PUNCT"]})

def test_tiny_cloze_keeps_precision_instead_of_flooring_to_zero():
    """Cloze is stored to 6 significant digits, so a very unlikely word (38.9
    bits) keeps a positive probability consistent with its surprisal."""
    from sensi.extractors.predictability import _write_metric_columns
    p = 2.0 ** -38.9
    out = {}
    _write_metric_columns(out, "gpt2", ["surprisal", "cloze"], ["NOUN"],
                          [{"p": p, "surprisal": 38.9, "top_match": 0}])
    cloze = out["gpt2_Cloze"][0]
    assert cloze > 0
    assert -math.log2(cloze) == pytest.approx(38.9, abs=1e-4)


def test_namespaced_columns_per_model_and_metric():
    out = add_predictability(_df(), "The cat.", models=["gpt2", "llama3_8b"],
                             metrics=["surprisal", "cloze"],
                             backends={"gpt2": _Fake(), "llama3_8b": _Fake()})
    for slug in ("gpt2", "llama3_8b"):
        assert f"{slug}_Surprisal" in out.columns
        assert f"{slug}_Cloze" in out.columns
        assert f"{slug}_TopMatch" not in out.columns   # topmatch not requested

def test_first_word_and_punct_are_nan():
    out = add_predictability(_df(), "The cat.", models=["gpt2"],
                             metrics=["surprisal", "cloze", "topmatch"],
                             backends={"gpt2": _Fake()})
    assert math.isnan(out["gpt2_Surprisal"].iloc[0])   # context-less first word
    assert math.isnan(out["gpt2_Surprisal"].iloc[2])   # PUNCT
    assert out["gpt2_Surprisal"].iloc[1] == 1.0
    assert math.isnan(out["gpt2_TopMatch"].iloc[0])     # top_match None → NaN
    assert out["gpt2_TopMatch"].iloc[1] == 1

def test_no_models_or_no_metrics_writes_no_columns():
    base = _df()
    a = add_predictability(base, "x", models=[], metrics=["surprisal"],
                           backends={})
    b = add_predictability(base, "x", models=["gpt2"], metrics=[],
                           backends={"gpt2": _Fake()})
    assert list(a.columns) == list(base.columns)
    assert list(b.columns) == list(base.columns)

def test_unknown_slug_rejected():
    with pytest.raises(KeyError):
        add_predictability(_df(), "x", models=["nope"], metrics=["surprisal"],
                           backends={"nope": _Fake()})


def test_gpt2_via_auto_matches_legacy_surprisal():
    # AutoModelForCausalLM("gpt2") is GPT2LMHeadModel — identical logits.
    # Pin: the document-first token is NaN (no context); the first content
    # word's surprisal is finite/positive. Guards against load-path drift.
    df = pd.DataFrame({"token": ["The", "cat", "sat"],
                       "upos": ["DET", "NOUN", "VERB"]})
    out = add_predictability(df, "The cat sat", models=["gpt2"],
                             metrics=["surprisal"])
    assert math.isnan(out["gpt2_Surprisal"].iloc[0])
    assert out["gpt2_Surprisal"].iloc[1] > 0


# ---- GPT-2 must stay fp32/CPU even when CUDA is available (legacy parity) ----


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


def test_gpt2_backend_stays_cpu_even_when_cuda_available(monkeypatch):
    # Legacy sensi/extractors/surprisal.py hardcoded USE_GPU=False, always loading
    # GPT-2 in full fp32 precision on CPU. The shared _HuggingFaceBackend
    # must preserve that — quantized/fp16 GPT-2 would emit different
    # surprisal values than previously-computed CSVs, even though the GPU
    # tiers are correct for the Llama models.
    from sensi.extractors.predictability import _HuggingFaceBackend
    monkeypatch.setattr('torch.cuda.is_available', lambda: True)
    _patch_transformers(monkeypatch, lambda model_id, **kw: _DummyModel('cuda:0'))
    b = _HuggingFaceBackend('gpt2')
    assert b._load_tier == 'CPU'


# ---------------------------------------------------------------------------
# Ported from tests/extractors/test_surprisal.py (deleted — sensi/extractors/
# surprisal.py is now a retired shim). These pin the pure subword<->token
# bookkeeping shared by every local-HF backend.
# ---------------------------------------------------------------------------


def test_simple_two_token_mapping():
    text = "the cat"
    #                 "the"   " cat"  (GPT-2 offsets include the leading space)
    offsets = [(0, 3), (3, 7)]
    spans = map_tokens_to_subwords(text, ["the", "cat"], offsets)
    assert spans == [[0], [1]]


def test_straddling_subword_counted_once():
    # spaCy splits "don't" into "do" + "n't"; a GPT-2 subword "don" (0,3)
    # overlaps both pieces. Its log-prob must be charged to exactly one
    # token (the first), never summed into both.
    text = "don't go"
    offsets = [(0, 3), (3, 5), (5, 8)]   # "don", "'t", " go"
    spans = map_tokens_to_subwords(text, ["do", "n't", "go"], offsets)
    assert spans == [[0], [1], [2]]


def test_multi_subword_token():
    text = "antidisestablishment now"
    offsets = [(0, 4), (4, 12), (12, 20), (20, 24)]
    spans = map_tokens_to_subwords(
        text, ["antidisestablishment", "now"], offsets)
    assert spans == [[0, 1, 2], [3]]


def test_token_not_in_text_gets_empty_span():
    spans = map_tokens_to_subwords("abc", ["zzz"], [(0, 3)])
    assert spans == [[]]


def test_token_surprisal_sums_subword_logprobs_in_bits():
    # Token owns subwords 1 and 2 with natural-log probs -1 and -2:
    # surprisal = 3 / ln(2) bits.
    log_probs = [None, -1.0, -2.0]
    out = _token_surprisals([[1, 2]], log_probs)
    assert out[0] == pytest.approx(3.0 / math.log(2), abs=1e-4)


def test_first_word_has_no_context_and_is_none():
    # Subword 0 has no preceding context: the model cannot assign it a
    # probability. Scoring it 0 bits ("perfectly predictable") would be an
    # artifact a researcher could mistake for data. It must be None.
    log_probs = [None, -1.0]
    out = _token_surprisals([[0], [1]], log_probs)
    assert out[0] is None
    assert out[1] is not None


def test_unmatched_token_is_none():
    assert _token_surprisals([[]], [None]) == [None]


# ---------------------------------------------------------------------------
# Ported from tests/extractors/test_llm_cloze.py (deleted — sensi/extractors/
# llm_cloze.py is now a retired shim). Covers pieces of predictability.py
# not already exercised above: pure scoring math, model selection/gating,
# chunked long-text inference, and the device-tier load cascade.
# ---------------------------------------------------------------------------


def test_scores_from_subwords_first_token_has_no_context():
    # "hi there": subword 0 is context-less → its word must be NaN/NaN/None,
    # not p=1.0 / surprisal=0 ("perfectly predictable").
    text = "hi there"
    offsets = [(0, 2), (2, 8)]
    sub_log_p = [None, -1.0]
    sub_top1 = [None, 7]
    ids = [5, 7]
    out = _scores_from_subwords(text, ["hi", "there"], offsets,
                                sub_log_p, sub_top1, ids)
    assert math.isnan(out[0]['p']) and math.isnan(out[0]['surprisal'])
    assert out[0]['top_match'] is None
    assert out[1]['p'] == pytest.approx(math.exp(-1.0))
    assert out[1]['top_match'] == 1          # argmax 7 == actual id 7


def test_scores_from_subwords_straddling_subword_counted_once():
    # Subword "don" (0,3) straddles "do"+"n't": charged to "do" only.
    text = "don't go"
    offsets = [(0, 3), (3, 5), (5, 8)]
    sub_log_p = [None, -1.0, -2.0]
    sub_top1 = [None, 1, 2]
    ids = [0, 1, 9]
    out = _scores_from_subwords(text, ["do", "n't", "go"], offsets,
                                sub_log_p, sub_top1, ids)
    # "n't" owns exactly subword 1 (lp -1.0), "go" owns subword 2 (lp -2.0).
    assert out[1]['surprisal'] == pytest.approx(1.0 / math.log(2))
    assert out[2]['surprisal'] == pytest.approx(2.0 / math.log(2))


# ---- add_predictability's own cross-pass score cache (_SCORE_CACHE) -------


def test_add_predictability_caches_scores_across_passes(monkeypatch):
    import sensi.extractors.predictability as m

    calls = []

    class _CountingBackend:
        def __init__(self, tag):
            self.tag = tag

        def score(self, text, tokens):
            calls.append((self.tag, text))
            return [{"p": 0.5, "surprisal": 1.0, "top_match": 1} for _ in tokens]

    monkeypatch.setattr(m, "_get_backend",
                        lambda slug, max_context=None: _CountingBackend(slug))
    m._SCORE_CACHE.clear()
    df = pd.DataFrame({"token": ["a", "b"], "upos": ["DET", "NOUN"]})
    m.add_predictability(df, "a b", models=["gpt2"], metrics=["cloze"])          # word-level pass
    m.add_predictability(df, "a b", models=["gpt2"], metrics=["cloze"])          # doc-level pass, same doc+model
    assert calls == [("gpt2", "a b")]                                           # scored once
    m.add_predictability(df, "a b", models=["llama32_3b"], metrics=["cloze"])   # other model → rescore
    assert calls == [("gpt2", "a b"), ("llama32_3b", "a b")]


# ---- model selection / gating ---------------------------------------------


def test_select_model_never_returns_openai_even_with_key(monkeypatch):
    # OpenAI has no per-token scoring endpoint anymore (the legacy
    # /v1/completions echo+logprobs path is gone and chat models were never
    # served there). select_model must not route to a backend that can only
    # fail at run time, even if a key is present.
    monkeypatch.setenv('OPENAI_API_KEY', 'sk-test')
    monkeypatch.setattr('sensi.extractors.predictability._available_vram_gb', lambda: 0.0)
    assert select_model() == 'meta-llama/Llama-3.2-3B'


def test_select_model_picks_local_by_vram(monkeypatch):
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    monkeypatch.setattr('sensi.extractors.predictability._available_vram_gb', lambda: 8.0)
    assert select_model() == 'meta-llama/Llama-3.2-3B'
    monkeypatch.setattr('sensi.extractors.predictability._available_vram_gb', lambda: 14.0)
    assert select_model() == 'meta-llama/Meta-Llama-3-8B'
    monkeypatch.setattr('sensi.extractors.predictability._available_vram_gb', lambda: 50.0)
    assert select_model() == 'meta-llama/Meta-Llama-3-70B'


def test_llm_options_has_no_openai_entries():
    assert LLM_OPTIONS, "options list must not be empty"
    assert not any(mid.startswith('gpt-') for mid, _ in LLM_OPTIONS)


def test_make_default_backend_rejects_openai_with_actionable_error():
    with pytest.raises(RuntimeError) as ei:
        _make_default_backend('gpt-4o-mini')
    msg = str(ei.value).lower()
    assert 'openai' in msg
    assert 'local' in msg          # points the user at the working path
    assert 'meta-llama' in str(ei.value) or 'llama' in msg


def test_is_gated_local_model():
    assert is_gated_local_model('meta-llama/Meta-Llama-3-8B') is True
    assert is_gated_local_model('meta-llama/Llama-3.2-3B') is True
    assert is_gated_local_model('gpt-4o-mini') is False


def test_has_hf_token_true_when_get_token_returns(monkeypatch):
    monkeypatch.setattr('huggingface_hub.get_token', lambda: 'hf_abc', raising=False)
    assert has_hf_token() is True


def test_has_hf_token_false_when_no_token(monkeypatch):
    monkeypatch.setattr('huggingface_hub.get_token', lambda: None, raising=False)
    assert has_hf_token() is False


def test_backend_raises_actionable_error_on_gated(monkeypatch):
    from sensi.extractors.predictability import _HuggingFaceBackend

    def raise_gated(*a, **k):
        raise OSError("401 Client Error: gated repo")

    monkeypatch.setattr('transformers.AutoTokenizer.from_pretrained', raise_gated)
    with pytest.raises(RuntimeError) as ei:
        _HuggingFaceBackend('meta-llama/Meta-Llama-3-8B')
    msg = str(ei.value)
    assert 'license' in msg and 'token' in msg
    assert 'meta-llama/Meta-Llama-3-8B' in msg


# ---- context-limit guard (used by a future OpenAI backend) ----------------


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
    # a future OpenAI backend's score() could validate before incurring API
    # cost. Currently vacuous (LLM_OPTIONS is local-only) but guards against
    # a silent gap if an OpenAI entry is ever added back.
    for model_id, _label in LLM_OPTIONS:
        if model_id.startswith('gpt-'):
            assert model_id in _OPENAI_CONTEXT_LIMITS, (
                f"{model_id!r} listed in LLM_OPTIONS but missing from "
                f"_OPENAI_CONTEXT_LIMITS — score() will skip the overflow check."
            )


# ---- chunk-slice iteration for long-text handling in _HuggingFaceBackend --


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


# ---- device-tier cascade for the local HuggingFace backend -----------------
# (reuses _DummyParam / _DummyModel / _patch_transformers defined above)


def test_load_tiers_prefers_gpu_only_when_cuda():
    assert _load_tiers(True) == ['4-bit GPU', 'fp16 GPU', 'CPU']
    assert _load_tiers(False) == ['CPU']


def test_backend_loads_4bit_on_gpu(monkeypatch):
    from sensi.extractors.predictability import _HuggingFaceBackend
    monkeypatch.setattr('torch.cuda.is_available', lambda: True)
    _patch_transformers(monkeypatch, lambda model_id, **kw: _DummyModel('cuda:0'))
    b = _HuggingFaceBackend('meta-llama/Meta-Llama-3-8B')
    assert b._load_tier == '4-bit GPU'


def test_backend_falls_back_to_fp16_gpu_on_bnb_error(monkeypatch):
    """bitsandbytes broken (typical on Windows) but a GPU is present → fp16 GPU,
    NOT a silent drop to CPU."""
    from sensi.extractors.predictability import _HuggingFaceBackend
    monkeypatch.setattr('torch.cuda.is_available', lambda: True)
    calls = []

    def fake_model(model_id, **kw):
        calls.append(kw)
        if 'quantization_config' in kw:
            raise RuntimeError("bitsandbytes not available on this platform")
        return _DummyModel('cuda:0')

    _patch_transformers(monkeypatch, fake_model)
    b = _HuggingFaceBackend('meta-llama/Meta-Llama-3-8B')
    assert b._load_tier == 'fp16 GPU'
    assert any('quantization_config' in c for c in calls)          # tried 4-bit
    assert any(c.get('torch_dtype') is not None for c in calls)     # then fp16
    assert b._context_limit == 1024


def test_backend_falls_back_to_cpu_and_warns(monkeypatch, caplog):
    """Both GPU loads fail while CUDA is present → CPU, with a loud warning so
    the CPU landing is never silent."""
    import logging
    from sensi.extractors.predictability import _HuggingFaceBackend
    monkeypatch.setattr('torch.cuda.is_available', lambda: True)

    def fake_model(model_id, **kw):
        if kw.get('device_map') == 'auto':      # both GPU tiers use device_map
            raise RuntimeError("gpu load failed")
        return _DummyModel('cpu')

    _patch_transformers(monkeypatch, fake_model)
    with caplog.at_level(logging.WARNING):
        b = _HuggingFaceBackend('meta-llama/Meta-Llama-3-8B')
    assert b._load_tier == 'CPU'
    assert 'CPU' in caplog.text and 'slower' in caplog.text


def test_backend_cpu_only_when_no_cuda(monkeypatch, caplog):
    """No GPU at all → only the CPU tier is attempted (no bnb/CUDA load), and
    no false 'GPU idle' warning."""
    import logging
    from sensi.extractors.predictability import _HuggingFaceBackend
    monkeypatch.setattr('torch.cuda.is_available', lambda: False)
    calls = []

    def fake_model(model_id, **kw):
        calls.append(kw)
        return _DummyModel('cpu')

    _patch_transformers(monkeypatch, fake_model)
    with caplog.at_level(logging.WARNING):
        b = _HuggingFaceBackend('meta-llama/Llama-3.2-3B')
    assert b._load_tier == 'CPU'
    assert calls == [{}]                                    # no GPU load attempted
    assert not any(r.levelno >= logging.WARNING for r in caplog.records)


# --- Project B: per-model context cap threading ------------------------------

def test_backend_caps_context_limit(monkeypatch):
    import sensi.extractors.predictability as m
    monkeypatch.setattr('torch.cuda.is_available', lambda: False)
    _patch_transformers(monkeypatch, lambda model_id, **kw: _DummyModel('cpu'))
    # _DummyModel.config.max_position_embeddings == 1024; a lower cap wins.
    b = m._HuggingFaceBackend('gpt2', max_context=256)
    assert b._context_limit == 256


def test_backend_context_none_uses_native(monkeypatch):
    import sensi.extractors.predictability as m
    monkeypatch.setattr('torch.cuda.is_available', lambda: False)
    _patch_transformers(monkeypatch, lambda model_id, **kw: _DummyModel('cpu'))
    b = m._HuggingFaceBackend('gpt2')
    assert b._context_limit == 1024


def test_backend_cap_above_native_clamps(monkeypatch):
    import sensi.extractors.predictability as m
    monkeypatch.setattr('torch.cuda.is_available', lambda: False)
    _patch_transformers(monkeypatch, lambda model_id, **kw: _DummyModel('cpu'))
    b = m._HuggingFaceBackend('gpt2', max_context=5000)
    assert b._context_limit == 1024   # min(native 1024, 5000)


def test_get_backend_reuses_same_cap_but_evicts_other_caps(monkeypatch):
    # Within a run the same cap is reused (cached); a DIFFERENT cap for the
    # same model evicts the old instance so at most one loaded model per
    # model_id is retained (each holds GBs of weights).
    import sensi.extractors.predictability as m
    m._BACKEND_CACHE.clear()
    monkeypatch.setattr('torch.cuda.is_available', lambda: False)
    _patch_transformers(monkeypatch, lambda model_id, **kw: _DummyModel('cpu'))
    b1 = m._get_backend('gpt2', max_context=256)
    assert m._get_backend('gpt2', max_context=256) is b1   # reused
    b2 = m._get_backend('gpt2', max_context=512)            # re-cap → new
    assert b2 is not b1
    # Only the newest cap for gpt2 is retained; the 256 entry is gone.
    assert list(m._BACKEND_CACHE.keys()) == [('gpt2', 512)]


def test_get_backend_keeps_distinct_models(monkeypatch):
    import sensi.extractors.predictability as m
    m._BACKEND_CACHE.clear()
    monkeypatch.setattr('torch.cuda.is_available', lambda: False)
    _patch_transformers(monkeypatch, lambda model_id, **kw: _DummyModel('cpu'))
    m._get_backend('gpt2', max_context=256)
    m._get_backend('llama3_8b', max_context=256)
    # Different models coexist; eviction is per-model_id only.
    assert {k[0] for k in m._BACKEND_CACHE} == {'gpt2', 'meta-llama/Meta-Llama-3-8B'}


def test_add_predictability_threads_effective_context(monkeypatch):
    import sensi.extractors.predictability as m

    calls = []

    class _FakeB:
        def score(self, text, tokens):
            return [{"p": 0.5, "surprisal": 1.0, "top_match": 1} for _ in tokens]

    def fake_get_backend(slug, max_context=None):
        calls.append((slug, max_context))
        return _FakeB()

    monkeypatch.setattr(m, "_get_backend", fake_get_backend)
    m._SCORE_CACHE.clear()
    df = pd.DataFrame({"token": ["a", "b"], "upos": ["NOUN", "NOUN"]})
    # llama3_8b native is 8192; cap 2048 is honored (above floor).
    m.add_predictability(df, "a b", models=["llama3_8b"], metrics=["surprisal"],
                         context={"llama3_8b": 2048})
    assert calls == [("llama3_8b", 2048)]
