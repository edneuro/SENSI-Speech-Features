import math
import types

import numpy as np
import pandas as pd
import pytest
from extractors.predictability import add_predictability

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
    # Legacy extractors/surprisal.py hardcoded USE_GPU=False, always loading
    # GPT-2 in full fp32 precision on CPU. The shared _HuggingFaceBackend
    # must preserve that — quantized/fp16 GPT-2 would emit different
    # surprisal values than previously-computed CSVs, even though the GPU
    # tiers are correct for the Llama models.
    from extractors.predictability import _HuggingFaceBackend
    monkeypatch.setattr('torch.cuda.is_available', lambda: True)
    _patch_transformers(monkeypatch, lambda model_id, **kw: _DummyModel('cuda:0'))
    b = _HuggingFaceBackend('gpt2')
    assert b._load_tier == 'CPU'
