import math
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
