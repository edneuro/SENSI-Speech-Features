import math
from unittest.mock import patch

import pandas as pd
import pytest

from extractors.llm_cloze import (
    add_llm_cloze,
    select_model,
    EXPECTED_COLUMNS,
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
