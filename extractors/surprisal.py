"""
Surprisal extractor — GPT-2 contextual word predictability.

Compatibility shim: the single implementation (the shared local-HuggingFace
backend, loaded via AutoModelForCausalLM) now lives in
extractors/predictability.py, alongside extractors/llm_cloze.py's backend —
GPT-2 is loaded through the exact same `_HuggingFaceBackend` class as the
gated Llama models (AutoModelForCausalLM("gpt2") resolves to GPT2LMHeadModel,
so logits/weights are identical to the old direct-load path). This module
keeps add_surprisal's public signature and legacy 'Surprisal' column for
existing callers/tests; deletion of this shim is deferred to a later
cleanup task once main.py/gui/* migrate to extractors.predictability
directly.

Input:  raw transcript text (str) + token-level DataFrame with 'token'
        and 'upos' columns (output of text_parser.parse())
Output: same DataFrame with a new 'Surprisal' column appended

Unit: bits (log base 2). Higher = less predictable in context.

Punctuation tokens (upos == 'PUNCT') receive NaN. The document's FIRST
word also receives NaN: its first subword has no preceding context, so
the model assigns it no probability — emitting 0 bits there would fake a
"perfectly predictable" first word.

Results are memoized per (text, token sequence): when the word-level and
document-level batches process the same stimulus in one run, GPT-2 runs
once and the second pass is served from cache (without even loading the
model).
"""
from __future__ import annotations

from collections import OrderedDict

import pandas as pd

from extractors.predictability import (          # noqa: F401
    map_tokens_to_subwords, _token_surprisals, _get_backend,
    _text_tokens_digest,
)

# (text, tokens) → per-token surprisal list. Sized to comfortably hold one
# GUI dispatch (word pass + doc pass over the same stimulus); FIFO-evicted.
# Distinct from predictability._SCORE_CACHE: values here are already-reduced
# per-token surprisal floats (legacy shape), not raw backend score dicts.
_SURPRISAL_CACHE: OrderedDict[str, list] = OrderedDict()
_CACHE_MAX_ENTRIES = 256


def _load():
    """Preload/cache the GPT-2 backend (AutoModelForCausalLM; shared cache
    with extractors.predictability / extractors.llm_cloze)."""
    _get_backend("gpt2")


def _compute_surprisals(text: str, tokens: list[str]) -> list:
    """Return one surprisal value per token (None for context-less/unmatched)."""
    scores = _get_backend("gpt2").score(text, tokens)
    return [
        None if s["surprisal"] != s["surprisal"] else round(s["surprisal"], 4)
        for s in scores
    ]


def add_surprisal(token_df: pd.DataFrame, text: str) -> pd.DataFrame:
    """Append a Surprisal column to a token-level DataFrame.

    Args:
        token_df: DataFrame from text_parser.parse(). Must have 'token'
                  and 'upos' columns. Rows must appear in document order.
        text:     The same raw text that was passed to text_parser.parse().
                  Used to run GPT-2 over the full document context.

    Returns:
        A copy of token_df with 'Surprisal' column (float, bits).
        Punctuation rows and the document's first word have NaN.
    """
    tokens = [str(t) for t in token_df['token'].tolist()]
    key = _text_tokens_digest(text, tokens)
    surprisals = _SURPRISAL_CACHE.get(key)
    if surprisals is None:
        _load()
        surprisals = _compute_surprisals(text, tokens)
        _SURPRISAL_CACHE[key] = surprisals
        while len(_SURPRISAL_CACHE) > _CACHE_MAX_ENTRIES:
            _SURPRISAL_CACHE.popitem(last=False)

    df = token_df.copy()
    df['Surprisal'] = [
        None if upos == 'PUNCT' else surprisals[i]
        for i, upos in enumerate(token_df['upos'].tolist())
    ]
    return df
