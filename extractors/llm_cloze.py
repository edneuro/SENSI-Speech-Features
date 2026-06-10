"""LLM-derived cloze metrics.

Produces three columns from a single forward pass per token:
    LLM_Cloze     = P(target | preceding context)
    LLM_Surprisal = -log2 P(target | preceding context)   (bits)
    LLM_TopMatch  = 1 if argmax-next-token == target else 0

Model selection (auto):
    OPENAI_API_KEY set            → gpt-4o-mini
    GPU VRAM ≥ 48 GB              → meta-llama/Llama-3-70B
    GPU VRAM ≥ 12 GB              → meta-llama/Llama-3-8B
    GPU VRAM ≥  8 GB              → meta-llama/Llama-3.2-3B
    otherwise                     → meta-llama/Llama-3.2-3B (CPU)

Override with `--llm-model <id>`. The backend interface is pluggable so
tests can inject mock backends (see _FakeBackend in the unit tests).
"""

import math
import os
from typing import Protocol

import numpy as np
import pandas as pd

EXPECTED_COLUMNS = ['LLM_Cloze', 'LLM_Surprisal', 'LLM_TopMatch']

# Shared by GUI dropdown (gui.py) and the run dispatch (gui_run.py).
# Kept here to avoid circular imports between gui.py and gui_run.py.
LLM_OPTIONS = [
    ('gpt-4o-mini',                 'gpt-4o-mini   ($0.10–0.30/stim, r≈0.83, requires OPENAI_API_KEY)'),
    ('gpt-4o',                      'gpt-4o        ($1–3/stim, r≈0.85, requires OPENAI_API_KEY)'),
    ('meta-llama/Llama-3-70B',      'Llama-3-70B   (free, r≈0.84, needs ≥48GB VRAM)'),
    ('meta-llama/Llama-3-8B',       'Llama-3-8B    (free, r≈0.79, needs ≥12GB VRAM)'),
    ('meta-llama/Llama-3.2-3B',     'Llama-3.2-3B  (free, r≈0.76, needs ≥8GB VRAM)'),
]

# Maximum input subwords each OpenAI model accepts. HuggingFace models are
# checked against `model.config.max_position_embeddings` directly.
_OPENAI_CONTEXT_LIMITS = {
    'gpt-4o-mini': 128_000,
    'gpt-4o':      128_000,
}


def _check_context_limit(n_subwords: int, limit: int | None, model_id: str) -> None:
    """Raise ValueError if the tokenised stimulus exceeds the model's context window.

    Used by the OpenAI backend, which can't chunk arbitrary text via the API.
    Local HuggingFace backends fall back to chunked inference instead.
    """
    if limit is None or n_subwords <= limit:
        return
    raise ValueError(
        f"Stimulus has {n_subwords} subwords but model {model_id!r} accepts at "
        f"most {limit}. Split the transcript into shorter segments, or switch "
        f"to a local Llama backend (which chunks long text automatically)."
    )


def _chunk_slices(n: int, window: int) -> list[tuple[int, int, int]]:
    """Iteration pattern for sliding-window inference over ``n`` subwords.

    Yields ``(segment_start, segment_end, keep_offset)`` per forward pass.
    Stride is ``window // 2`` — each chunk past the initial window sees at
    least ``window // 2`` preceding subwords inside the same pass. The union
    of ``segment[keep_offset:]`` covers indices ``0..n-1`` exactly once.
    """
    chunk = window // 2
    if chunk <= 0:
        raise ValueError(f"window must be >= 2 to chunk (got {window})")
    out = []
    for start in range(0, n, chunk):
        context_start = max(0, start - chunk)
        segment_end = min(start + chunk, n)
        keep_offset = start - context_start
        out.append((context_start, segment_end, keep_offset))
    return out


class _Backend(Protocol):
    def score(self, text: str, tokens: list[str]) -> list[dict]:
        """Return one dict per token with keys: p, surprisal, top_match."""


def select_model() -> str:
    """Pick the best LLM model id given the current environment."""
    if os.environ.get('OPENAI_API_KEY'):
        return 'gpt-4o-mini'
    vram = _available_vram_gb()
    if vram >= 48:
        return 'meta-llama/Llama-3-70B'
    if vram >= 12:
        return 'meta-llama/Llama-3-8B'
    return 'meta-llama/Llama-3.2-3B'


def _available_vram_gb() -> float:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    except Exception:
        pass
    return 0.0


def add_llm_cloze(
    token_df: pd.DataFrame,
    text: str,
    backend: _Backend | None = None,
    model_id: str | None = None,
) -> pd.DataFrame:
    """Append LLM_Cloze, LLM_Surprisal, LLM_TopMatch columns to token_df."""
    df = token_df.copy()

    if backend is None:
        backend = _make_default_backend(model_id or select_model())

    tokens = list(df['token'])
    scores = backend.score(text, tokens)

    p_col, surp_col, top_col = [], [], []
    for (i, row), score in zip(df.iterrows(), scores):
        if row['upos'] == 'PUNCT':
            p_col.append(np.nan)
            surp_col.append(np.nan)
            top_col.append(np.nan)
            continue
        p_col.append(round(score['p'], 6))
        surp_col.append(round(score['surprisal'], 4))
        top_col.append(int(score['top_match']))

    df['LLM_Cloze'] = p_col
    df['LLM_Surprisal'] = surp_col
    df['LLM_TopMatch'] = top_col
    return df


# --------------------- backends ---------------------

def _make_default_backend(model_id: str) -> _Backend:
    if model_id.startswith('gpt-'):
        return _OpenAIBackend(model_id)
    return _HuggingFaceBackend(model_id)


class _HuggingFaceBackend:
    """Local Hugging Face causal-LM backend with 4-bit quantization when possible."""

    def __init__(self, model_id: str):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

        self._torch = torch
        self._model_id = model_id
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        try:
            quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
            self._model = AutoModelForCausalLM.from_pretrained(
                model_id, quantization_config=quant, device_map='auto',
            )
        except Exception:
            # Fall back to fp16/cpu if bitsandbytes isn't supported
            self._model = AutoModelForCausalLM.from_pretrained(model_id)
        self._model.eval()
        self._context_limit = getattr(self._model.config, 'max_position_embeddings', None)

    def score(self, text: str, tokens: list[str]) -> list[dict]:
        torch = self._torch
        enc = self._tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        ids = enc['input_ids']
        offsets = enc['offset_mapping']
        device = next(self._model.parameters()).device

        limit = self._context_limit
        if limit is None or len(ids) <= limit:
            ids_t = torch.tensor([ids], device=device)
            with torch.no_grad():
                logits = self._model(ids_t).logits[0]
        else:
            # Stimulus exceeds the model's context window — slide a chunked
            # window through it (same strategy as extractors/surprisal.py for
            # GPT-2). Stride = window // 2 so each post-window position sees
            # at least window // 2 preceding subwords inside its forward pass.
            chunks = []
            for seg_start, seg_end, keep_off in _chunk_slices(len(ids), limit):
                seg_ids = ids[seg_start:seg_end]
                ids_t = torch.tensor([seg_ids], device=device)
                with torch.no_grad():
                    seg_logits = self._model(ids_t).logits[0]
                chunks.append(seg_logits[keep_off:])
            logits = torch.cat(chunks, dim=0)

        log_probs = torch.log_softmax(logits, dim=-1)
        # Per-subword log-prob and top-1 prediction
        sub_log_p = [0.0]
        sub_top1 = [None]
        for i in range(1, len(ids)):
            sub_log_p.append(log_probs[i - 1, ids[i]].item())
            sub_top1.append(int(log_probs[i - 1].argmax().item()))

        # Map subwords → spaCy tokens via character offsets
        results = []
        char_pos = 0
        for tok in tokens:
            idx = text.find(tok, char_pos)
            if idx == -1:
                results.append({'p': float('nan'), 'surprisal': float('nan'), 'top_match': 0})
                continue
            tok_start, tok_end = idx, idx + len(tok)
            char_pos = tok_end
            total_lp = 0.0
            first_sub = None
            found = False
            for sub_idx, (s, e) in enumerate(offsets):
                if e <= tok_start:
                    continue
                if s >= tok_end:
                    break
                total_lp += sub_log_p[sub_idx]
                if first_sub is None:
                    first_sub = sub_idx
                found = True
            if not found:
                results.append({'p': float('nan'), 'surprisal': float('nan'), 'top_match': 0})
                continue
            p = math.exp(total_lp)
            surprisal = -total_lp / math.log(2)
            top_match = 1 if (first_sub is not None and sub_top1[first_sub] == ids[first_sub]) else 0
            results.append({'p': p, 'surprisal': surprisal, 'top_match': top_match})
        return results


class _OpenAIBackend:
    """OpenAI Chat Completions logprobs-based scorer."""

    def __init__(self, model_id: str):
        from openai import OpenAI
        self._client = OpenAI()
        self._model_id = model_id
        self._context_limit = _OPENAI_CONTEXT_LIMITS.get(model_id)

    def score(self, text: str, tokens: list[str]) -> list[dict]:
        if self._context_limit is not None:
            import tiktoken
            try:
                enc = tiktoken.encoding_for_model(self._model_id)
            except KeyError:
                enc = tiktoken.get_encoding('o200k_base')
            _check_context_limit(len(enc.encode(text)), self._context_limit, self._model_id)
        # NOTE: For per-token logprobs over a fixed continuation, OpenAI's
        # Completions endpoint with `logprobs` is the right primitive.
        # Newer Chat Completions support `logprobs` and `top_logprobs` for
        # generated text but not for arbitrary scoring of an existing string.
        # We use the legacy /completions endpoint via the Responses API:
        responses = self._client.completions.create(
            model=self._model_id,
            prompt=text,
            max_tokens=0,
            logprobs=5,
            echo=True,
        )
        choice = responses.choices[0]
        token_logprobs = choice.logprobs.token_logprobs
        offsets = choice.logprobs.text_offset
        sub_tokens = choice.logprobs.tokens
        top_choices = choice.logprobs.top_logprobs

        results = []
        char_pos = 0
        for tok in tokens:
            idx = text.find(tok, char_pos)
            if idx == -1:
                results.append({'p': float('nan'), 'surprisal': float('nan'), 'top_match': 0})
                continue
            tok_start, tok_end = idx, idx + len(tok)
            char_pos = tok_end
            total_lp = 0.0
            first_sub = None
            found = False
            for sub_idx, (off, sub) in enumerate(zip(offsets, sub_tokens)):
                if off + len(sub) <= tok_start:
                    continue
                if off >= tok_end:
                    break
                lp = token_logprobs[sub_idx] or 0.0
                total_lp += lp
                if first_sub is None:
                    first_sub = sub_idx
                found = True
            if not found:
                results.append({'p': float('nan'), 'surprisal': float('nan'), 'top_match': 0})
                continue
            p = math.exp(total_lp)
            surprisal = -total_lp / math.log(2)
            top_match = 0
            if first_sub is not None and top_choices[first_sub]:
                argmax_token = max(top_choices[first_sub], key=top_choices[first_sub].get)
                if argmax_token.strip() == sub_tokens[first_sub].strip():
                    top_match = 1
            results.append({'p': p, 'surprisal': surprisal, 'top_match': top_match})
        return results
