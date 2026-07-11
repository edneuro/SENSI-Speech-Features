"""LLM-derived cloze metrics.

Compatibility shim: the single backend implementation (_HuggingFaceBackend,
loaded via AutoModelForCausalLM) and its shared helpers now live in
extractors/predictability.py, alongside extractors/surprisal.py's GPT-2 path.
This module keeps add_llm_cloze's public signature, LLM_OPTIONS, and the
legacy LLM_Cloze/LLM_Surprisal/LLM_TopMatch columns for existing callers
(main.py, gui/*, utils/validate_cloze.py) and tests; deletion of this shim
is deferred to a later cleanup task once those consumers migrate to
extractors.predictability directly.

Produces three columns from a single forward pass per token:
    LLM_Cloze     = P(target | preceding context)
    LLM_Surprisal = -log2 P(target | preceding context)   (bits)
    LLM_TopMatch  = 1 if argmax-next-token == target else 0

Backends are LOCAL Hugging Face causal LMs only. OpenAI is intentionally
not offered: scoring an existing string for per-token log-probabilities
requires the legacy /v1/completions `echo=True, logprobs` primitive,
which OpenAI has retired and never served for the gpt-4o family. A gpt-*
model id therefore raises an actionable error instead of failing mid-run
against the API (see _make_default_backend). If OpenAI ever ships a
scoring endpoint again, restore an OpenAI backend and re-add its entries
to LLM_OPTIONS.

Model selection (auto):
    GPU VRAM ≥ 48 GB → meta-llama/Meta-Llama-3-70B
    GPU VRAM ≥ 12 GB → meta-llama/Meta-Llama-3-8B
    otherwise        → meta-llama/Llama-3.2-3B (GPU if ≥8 GB, else CPU)

Override with `--llm-model <id>` (a local HF causal-LM id). The backend
interface is pluggable so tests can inject mock backends (see
_FakeBackend in the unit tests).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from extractors.predictability import (          # noqa: F401
    _Backend,
    _HuggingFaceBackend,
    _make_default_backend,
    _cached_backend,
    _BACKEND_CACHE,
    _available_vram_gb,
    is_gated_local_model,
    has_hf_token,
    _scores_from_subwords,
    _chunk_slices,
    _check_context_limit,
    _load_tiers,
    _model_load_hint,
    LLM_OPTIONS,
    _OPENAI_CONTEXT_LIMITS,
    _SCORE_CACHE,
    _SCORE_CACHE_MAX,
    _text_tokens_digest,
)

EXPECTED_COLUMNS = ['LLM_Cloze', 'LLM_Surprisal', 'LLM_TopMatch']


def select_model() -> str:
    """Pick the best LOCAL LLM model id given the current environment.

    Thin wrapper kept here (rather than a bare re-export) so that unit tests
    patching extractors.llm_cloze._available_vram_gb take effect — this
    function does a plain-name lookup within this module's own namespace.
    The underlying VRAM probe and OpenAI-exclusion rationale live in
    extractors.predictability (single implementation, imported above).
    """
    vram = _available_vram_gb()
    if vram >= 48:
        return 'meta-llama/Meta-Llama-3-70B'
    if vram >= 12:
        return 'meta-llama/Meta-Llama-3-8B'
    return 'meta-llama/Llama-3.2-3B'


def add_llm_cloze(
    token_df: pd.DataFrame,
    text: str,
    backend: _Backend | None = None,
    model_id: str | None = None,
) -> pd.DataFrame:
    """Append LLM_Cloze, LLM_Surprisal, LLM_TopMatch columns to token_df.

    Punctuation rows are NaN across all three columns. The document's
    first word is NaN/NaN/NaN too — it has no preceding context, so the
    model cannot score it (the backend reports top_match=None there).

    Kept here (rather than moved wholesale into predictability.py) so that
    unit tests patching extractors.llm_cloze._cached_backend take effect —
    this function does a plain-name lookup of _cached_backend within this
    module's own namespace. The score cache (_SCORE_CACHE) is the same
    object used by extractors.predictability.add_predictability, so a
    roster-driven pass and a legacy add_llm_cloze pass over the same model
    id + stimulus share one set of scores.
    """
    df = token_df.copy()

    tokens = [str(t) for t in df['token'].tolist()]
    if backend is None:
        mid = model_id or select_model()
        key = (mid, _text_tokens_digest(text, tokens))
        scores = _SCORE_CACHE.get(key)
        if scores is None:
            scores = _cached_backend(mid).score(text, tokens)
            _SCORE_CACHE[key] = scores
            while len(_SCORE_CACHE) > _SCORE_CACHE_MAX:
                _SCORE_CACHE.popitem(last=False)
    else:
        scores = backend.score(text, tokens)

    p_col, surp_col, top_col = [], [], []
    for upos, score in zip(df['upos'].tolist(), scores):
        if upos == 'PUNCT':
            p_col.append(np.nan)
            surp_col.append(np.nan)
            top_col.append(np.nan)
            continue
        p_col.append(round(score['p'], 6))
        surp_col.append(round(score['surprisal'], 4))
        top_col.append(np.nan if score['top_match'] is None
                       else int(score['top_match']))

    df['LLM_Cloze'] = p_col
    df['LLM_Surprisal'] = surp_col
    df['LLM_TopMatch'] = top_col
    return df
