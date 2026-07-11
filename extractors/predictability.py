"""Unified multi-model predictability extractor (surprisal / cloze / top-match).

Replaces extractors/surprisal.py and the old extractors/llm_cloze.py. Each
selected roster model is scored once; only the requested metrics are written,
prefixed by the model slug. See the design spec §4.

This module is also the single home for the shared local-HuggingFace backend
(_HuggingFaceBackend, loaded via AutoModelForCausalLM) and its supporting
helpers. extractors/surprisal.py and extractors/llm_cloze.py are now thin
compatibility shims over this module — see their docstrings — so there is
exactly one backend implementation, used by add_predictability (this module),
add_surprisal (surprisal shim) and add_llm_cloze (llm_cloze shim) alike.
"""
from __future__ import annotations

import hashlib
import logging
import math
from collections import OrderedDict
from typing import Protocol

import numpy as np
import pandas as pd

from extractors.predictability_models import spec_for

logger = logging.getLogger(__name__)

_METRIC_COLS = {"surprisal": "Surprisal", "cloze": "Cloze", "topmatch": "TopMatch"}


# ---------------------------------------------------------------------------
# add_predictability: roster-driven, namespaced columns
# ---------------------------------------------------------------------------

def add_predictability(df, text, *, models, metrics, backends=None):
    metrics = [m for m in metrics if m in _METRIC_COLS]
    if not models or not metrics:
        return df.copy()
    for slug in models:
        spec_for(slug)          # validate; KeyError on unknown
    out = df.copy()
    upos = out["upos"].tolist()
    tokens = [str(t) for t in out["token"].tolist()]
    for slug in models:
        backend = (backends or {}).get(slug)
        if backend is not None:
            scores = backend.score(text, tokens)
        else:
            scores = _scored_by_slug(slug, text, tokens)
        _write_metric_columns(out, slug, metrics, upos, scores)
    return out


def _scored_by_slug(slug: str, text: str, tokens: list[str]) -> list[dict]:
    """Score via the cached backend, memoized per (hf_id, text/tokens digest)
    so the word-level and document-level passes over the same stimulus never
    re-run inference (mirrors the legacy add_llm_cloze caching, and shares
    its cache — see _SCORE_CACHE below)."""
    hf_id = spec_for(slug).hf_id
    key = (hf_id, _text_tokens_digest(text, tokens))
    scores = _SCORE_CACHE.get(key)
    if scores is None:
        scores = _get_backend(slug).score(text, tokens)
        _SCORE_CACHE[key] = scores
        while len(_SCORE_CACHE) > _SCORE_CACHE_MAX:
            _SCORE_CACHE.popitem(last=False)
    return scores


def _write_metric_columns(out, slug, metrics, upos, scores):
    cols = {m: [] for m in metrics}
    for pos_tag, score in zip(upos, scores):
        for m in metrics:
            if pos_tag == "PUNCT":
                cols[m].append(np.nan); continue
            if m == "surprisal":
                cols[m].append(round(score["surprisal"], 4)
                               if score["surprisal"] == score["surprisal"] else np.nan)
            elif m == "cloze":
                cols[m].append(round(score["p"], 6)
                               if score["p"] == score["p"] else np.nan)
            else:  # topmatch
                tm = score["top_match"]
                cols[m].append(np.nan if tm is None else int(tm))
    for m in metrics:
        out[f"{slug}_{_METRIC_COLS[m]}"] = cols[m]


def _get_backend(slug: str) -> "_HuggingFaceBackend":
    return _cached_backend(spec_for(slug).hf_id)


# ---------------------------------------------------------------------------
# subword <-> token bookkeeping (moved from extractors/surprisal.py; shared
# by the HuggingFace backend's score() below)
# ---------------------------------------------------------------------------

def map_tokens_to_subwords(
    text: str,
    spacy_tokens: list[str],
    offsets: list[tuple[int, int]],
) -> list[list[int]]:
    """Assign each subword to exactly one spaCy token.

    Tokens are located in `text` left to right; a subword belongs to the
    FIRST token whose character span it overlaps, and is then consumed —
    a subword straddling a token boundary (e.g. GPT-2 "don" over spaCy
    "do" + "n't") is never charged twice. Returns one (possibly empty)
    list of subword indices per token. Shared by every local-HF backend
    (GPT-2 and the gated Llama models alike).
    """
    spans: list[list[int]] = []
    char_pos = 0
    sub_idx = 0
    n_subs = len(offsets)

    for token in spacy_tokens:
        idx = text.find(token, char_pos)
        if idx == -1:
            spans.append([])
            continue
        tok_start, tok_end = idx, idx + len(token)
        char_pos = tok_end

        # Skip subwords that end before this token starts.
        while sub_idx < n_subs and offsets[sub_idx][1] <= tok_start:
            sub_idx += 1
        own: list[int] = []
        while sub_idx < n_subs and offsets[sub_idx][0] < tok_end:
            own.append(sub_idx)
            sub_idx += 1
        spans.append(own)

    return spans


def _token_surprisals(
    spans: list[list[int]],
    log_probs: list,
) -> list:
    """Per-token surprisal (bits) from subword spans and log-probs.

    A token is None when it has no subwords, or when any of its subwords
    has no log-prob (log_probs[i] is None — the context-less position 0).
    """
    results = []
    for own in spans:
        if not own or any(log_probs[i] is None for i in own):
            results.append(None)
            continue
        total_lp = sum(log_probs[i] for i in own)
        results.append(round(-total_lp / math.log(2), 4))
    return results


def _scores_from_subwords(
    text: str,
    tokens: list[str],
    offsets: list[tuple[int, int]],
    sub_log_p: list,
    sub_top1: list,
    ids: list[int],
) -> list[dict]:
    """Per-token {p, surprisal, top_match} from per-subword log-probs.

    A token is unscoreable — {nan, nan, None} — when it has no subwords or
    contains the context-less first subword (sub_log_p[i] is None).
    """
    spans = map_tokens_to_subwords(text, tokens, offsets)
    results = []
    for own in spans:
        if not own or any(sub_log_p[i] is None for i in own):
            results.append({'p': float('nan'), 'surprisal': float('nan'),
                            'top_match': None})
            continue
        total_lp = sum(sub_log_p[i] for i in own)
        first_sub = own[0]
        results.append({
            'p': math.exp(total_lp),
            'surprisal': -total_lp / math.log(2),
            'top_match': 1 if sub_top1[first_sub] == ids[first_sub] else 0,
        })
    return results


# ---------------------------------------------------------------------------
# model selection / gating (moved from extractors/llm_cloze.py)
# ---------------------------------------------------------------------------

# Shared by the GUI dropdown and the run dispatch. LOCAL models only — OpenAI
# is intentionally not offered: scoring an existing string for per-token
# log-probabilities requires the legacy /v1/completions `echo=True, logprobs`
# primitive, which OpenAI has retired and never served for the gpt-4o family.
LLM_OPTIONS = [
    ('meta-llama/Meta-Llama-3-70B', 'Llama-3-70B   (free, r≈0.84, needs ≥48GB VRAM)'),
    ('meta-llama/Meta-Llama-3-8B',  'Llama-3-8B    (free, r≈0.79, needs ≥12GB VRAM)'),
    ('meta-llama/Llama-3.2-3B',     'Llama-3.2-3B  (free, r≈0.76, needs ≥8GB VRAM, CPU-capable)'),
]

# Retained so a future OpenAI scoring backend (and the generic
# _check_context_limit guard) has a home; no OpenAI model is currently
# selectable. HuggingFace models are checked against
# `model.config.max_position_embeddings` directly.
_OPENAI_CONTEXT_LIMITS: dict[str, int] = {}


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
    """Pick the best LOCAL LLM model id given the current environment.

    OpenAI is never selected — even with OPENAI_API_KEY set — because there
    is no working OpenAI per-token scoring endpoint (see LLM_OPTIONS above).
    Selection is by available GPU VRAM.
    """
    vram = _available_vram_gb()
    if vram >= 48:
        return 'meta-llama/Meta-Llama-3-70B'
    if vram >= 12:
        return 'meta-llama/Meta-Llama-3-8B'
    return 'meta-llama/Llama-3.2-3B'


def is_gated_local_model(model_id: str) -> bool:
    """True for gated local HF models (the meta-llama/* family) that need a token."""
    return model_id.startswith("meta-llama/")


def has_hf_token() -> bool:
    """True if a HuggingFace token is already available — HF_TOKEN /
    HUGGING_FACE_HUB_TOKEN env, or a token saved by a prior `hf auth login`.
    Lazy import so importing this module (and starting the GUI) stays cheap."""
    try:
        from huggingface_hub import get_token
    except Exception:
        return False
    return bool(get_token())


def _available_vram_gb() -> float:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    except Exception:
        pass
    return 0.0


def _load_tiers(cuda: bool) -> list[str]:
    """Ordered model-load strategies, most-VRAM-efficient first.

    The GPU tiers ('4-bit GPU', 'fp16 GPU') are only attempted when CUDA is
    actually present; on a CPU-only machine we go straight to 'CPU' rather
    than fail a bitsandbytes/CUDA load first. The 4-bit tier depends on a
    working bitsandbytes build (frequently broken on Windows), so 'fp16 GPU'
    exists specifically so a GPU still gets used when quantization is
    unavailable — otherwise we'd silently drop to CPU with the GPU idle.
    """
    return ['4-bit GPU', 'fp16 GPU', 'CPU'] if cuda else ['CPU']


def _model_load_hint(model_id: str, exc: Exception) -> str:
    return (
        f"Could not load local LLM {model_id!r}: {type(exc).__name__}: {exc}\n"
        f"To fix: (1) accept the license at https://huggingface.co/{model_id} ; "
        f"(2) provide a HuggingFace token (GUI token box, HF_TOKEN env var, or "
        f"`hf auth login`); (3) on an offline machine, pre-download the weights "
        f"first. Or select an OpenAI model (set OPENAI_API_KEY) — no gating or download."
    )


class _HuggingFaceBackend:
    """Local Hugging Face causal-LM backend with 4-bit quantization when possible.

    Loads via AutoModelForCausalLM/AutoTokenizer for every model — including
    GPT-2, where AutoModelForCausalLM("gpt2") resolves to the same
    GPT2LMHeadModel class (identical weights/logits) that the legacy
    extractors/surprisal.py loaded directly.
    """

    def __init__(self, model_id: str):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

        self._torch = torch
        self._model_id = model_id
        cuda = bool(torch.cuda.is_available())

        def _load(tier: str):
            if tier == '4-bit GPU':
                quant = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
                return AutoModelForCausalLM.from_pretrained(
                    model_id, quantization_config=quant, device_map='auto')
            if tier == 'fp16 GPU':
                return AutoModelForCausalLM.from_pretrained(
                    model_id, torch_dtype=torch.float16, device_map='auto')
            return AutoModelForCausalLM.from_pretrained(model_id)  # CPU, default dtype

        try:
            # A gated / missing / offline repo fails on the tokenizer first, so
            # the actionable hint below fires before we try any device tier.
            self._tokenizer = AutoTokenizer.from_pretrained(model_id)
            last_exc: Exception | None = None
            self._load_tier = None
            for tier in _load_tiers(cuda):
                try:
                    self._model = _load(tier)
                    self._load_tier = tier
                    break
                except Exception as exc:   # noqa: BLE001 — many unrelated load failures
                    last_exc = exc
                    logger.info(
                        "Local LLM load tier %r failed for %s: %s", tier, model_id, exc)
            if self._load_tier is None:
                raise last_exc if last_exc is not None else RuntimeError("no load tier")
        except Exception as exc:
            raise RuntimeError(_model_load_hint(model_id, exc)) from exc

        self._model.eval()
        self._context_limit = getattr(self._model.config, 'max_position_embeddings', None)
        try:
            device = next(self._model.parameters()).device
        except Exception:
            device = 'unknown'
        logger.info(
            "Loaded local LLM %s via %s (device=%s)", model_id, self._load_tier, device)
        if cuda and self._load_tier == 'CPU':
            logger.warning(
                "GPU detected but %s loaded on CPU — the 4-bit and fp16 GPU loads "
                "both failed (check bitsandbytes / accelerate). Inference will be "
                "much slower; fix the GPU deps or pick a smaller model.", model_id)

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
            # window through it. Stride = window // 2 so each post-window
            # position sees at least window // 2 preceding subwords inside
            # its forward pass.
            chunks = []
            for seg_start, seg_end, keep_off in _chunk_slices(len(ids), limit):
                seg_ids = ids[seg_start:seg_end]
                ids_t = torch.tensor([seg_ids], device=device)
                with torch.no_grad():
                    seg_logits = self._model(ids_t).logits[0]
                chunks.append(seg_logits[keep_off:])
            logits = torch.cat(chunks, dim=0)

        log_probs = torch.log_softmax(logits, dim=-1)
        # Per-subword log-prob and top-1 prediction. Position 0 has no
        # preceding context — None marks it unscoreable (not "0 bits").
        sub_log_p: list = [None]
        sub_top1: list = [None]
        for i in range(1, len(ids)):
            sub_log_p.append(log_probs[i - 1, ids[i]].item())
            sub_top1.append(int(log_probs[i - 1].argmax().item()))

        return _scores_from_subwords(text, tokens, offsets,
                                     sub_log_p, sub_top1, ids)


def _make_default_backend(model_id: str) -> _Backend:
    if model_id.startswith('gpt-'):
        raise RuntimeError(
            f"OpenAI model {model_id!r} is not supported for cloze scoring. "
            "Per-token scoring of an existing string needs the legacy "
            "/v1/completions echo+logprobs primitive, which OpenAI has "
            "retired and never served for the gpt-4o family — a run would "
            "only fail at the API. Use a local model instead "
            "(e.g. --llm-model meta-llama/Llama-3.2-3B, which runs on CPU), "
            "or omit --llm-model to auto-select one by GPU VRAM."
        )
    return _HuggingFaceBackend(model_id)


# Process-wide backend cache, keyed by literal HuggingFace repo id (the same
# string space as roster hf_ids, e.g. "gpt2", and legacy free-form model_ids,
# e.g. "meta-llama/Meta-Llama-3-8B") — so a roster-driven load and a legacy
# add_llm_cloze load of the same model share one instance. Loading a local 8B
# model costs minutes; we load once per id and reuse it across every document
# (and the word/doc passes) in a run — the same "load once" intent that
# _align_cache gives MFA. Callers that pass an explicit `backend=` bypass
# this entirely.
_BACKEND_CACHE: dict[str, _Backend] = {}


def _cached_backend(model_id: str) -> _Backend:
    backend = _BACKEND_CACHE.get(model_id)
    if backend is None:
        backend = _make_default_backend(model_id)
        _BACKEND_CACHE[model_id] = backend
    return backend


# ---------------------------------------------------------------------------
# score cache: shared by add_predictability, add_llm_cloze (llm_cloze shim)
# ---------------------------------------------------------------------------

# (hf_id/model_id, text/tokens digest) → scores. Sized to hold one GUI
# dispatch (word pass + doc pass over the same stimuli); FIFO-evicted.
_SCORE_CACHE: OrderedDict = OrderedDict()
_SCORE_CACHE_MAX = 256


def _text_tokens_digest(text: str, tokens: list[str]) -> str:
    h = hashlib.sha256(text.encode('utf-8'))
    h.update(b'\x00')
    h.update('\x00'.join(tokens).encode('utf-8'))
    return h.hexdigest()
