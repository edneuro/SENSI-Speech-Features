"""Unified multi-model predictability extractor (surprisal / cloze / top-match).

Replaces the retired extractors/surprisal.py and extractors/llm_cloze.py
(both deleted once every consumer moved onto this module). Each selected
roster model is scored once; only the requested metrics are written, prefixed
by the model slug. See the design spec §4.

This module is the single home for the shared local-HuggingFace backend
(_HuggingFaceBackend, loaded via AutoModelForCausalLM) and its supporting
helpers, so there is exactly one backend implementation behind
add_predictability.
"""
from __future__ import annotations

import hashlib
import logging
import math
from collections import OrderedDict
from typing import Protocol

import numpy as np
import pandas as pd

from sensi.extractors.predictability_models import (
    spec_for, effective_context, effective_precision, load_plan, weights_gb,
    slug_for_hf_id)
from sensi.utils.transcript_prep import normalize_whitespace

logger = logging.getLogger(__name__)

_METRIC_COLS = {"surprisal": "Surprisal", "cloze": "Cloze", "topmatch": "TopMatch"}


# ---------------------------------------------------------------------------
# add_predictability: roster-driven, namespaced columns
# ---------------------------------------------------------------------------

def add_predictability(df, text, *, models, metrics, context=None, precision=None,
                       backends=None):
    metrics = [m for m in metrics if m in _METRIC_COLS]
    if not models or not metrics:
        return df.copy()
    for slug in models:
        spec_for(slug)          # validate; KeyError on unknown
    ctx_map = context or {}
    # Line breaks in a transcript are layout, not speech: a raw "\n" makes the
    # next word a space-less line start (LJ002-0321 "were": 22 bits vs 2.8).
    text = normalize_whitespace(text)
    out = df.copy()
    upos = out["upos"].tolist()
    tokens = [str(t) for t in out["token"].tolist()]
    for slug in models:
        backend = (backends or {}).get(slug)
        if backend is not None:
            scores = backend.score(text, tokens)
            _record_used(slug, precision)
        else:
            eff = effective_context(slug, ctx_map)   # per-model cap (Project B)
            scores = _scored_by_slug(slug, text, tokens, eff, precision)
        _write_metric_columns(out, slug, metrics, upos, scores)
    return out


def _scored_by_slug(slug: str, text: str, tokens: list[str],
                    max_context: int | None = None,
                    precision_map: dict | None = None) -> list[dict]:
    """Score via the cached backend, memoized per (hf_id, max_context,
    precision run, text/tokens digest) so the word-level and document-level
    passes over the same stimulus never re-run inference. The key includes
    max_context and the precision actually run, so re-capping a model or
    changing its precision re-scores rather than returning stale values
    (see _SCORE_CACHE below)."""
    hf_id = spec_for(slug).hf_id
    precision_run, _device = load_plan(slug, precision_map, _cuda_available())
    _record_used(slug, precision_map)
    key = (hf_id, max_context, precision_run, _text_tokens_digest(text, tokens))
    scores = _SCORE_CACHE.get(key)
    if scores is None:
        scores = _get_backend(slug, max_context, precision_map).score(text, tokens)
        _SCORE_CACHE[key] = scores
        while len(_SCORE_CACHE) > SCORE_CACHE_MAX:
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
                # 6 significant digits, not 6 decimals: a very unlikely word
                # keeps a tiny positive probability (e.g. 2.1e-12) instead of
                # flooring to 0, so Cloze = 2^(-Surprisal) and log(Cloze) hold.
                cols[m].append(float(f'{score["p"]:.6g}')
                               if score["p"] == score["p"] else np.nan)
            else:  # topmatch
                tm = score["top_match"]
                cols[m].append(np.nan if tm is None else int(tm))
    for m in metrics:
        out[f"{slug}_{_METRIC_COLS[m]}"] = cols[m]


def _cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:   # noqa: BLE001
        return False


def _get_backend(slug: str, max_context: int | None = None,
                 precision_map: dict | None = None) -> "_Backend":
    """The cached backend for `slug` at the precision/device load_plan picks.
    No GPU -> fp32 on CPU for every model, said once when that overrides the
    user's choice."""
    hf_id = spec_for(slug).hf_id
    precision_run, device = load_plan(slug, precision_map, _cuda_available())
    if ((hf_id, max_context, precision_run, device) not in _BACKEND_CACHE
            and device == "cpu"
            and effective_precision(slug, precision_map) != "fp32"):
        print(f"[predict] {slug}: no GPU — running fp32 on CPU "
              f"(~{weights_gb(slug, precision_run):.1f} GB RAM)", flush=True)
    return _cached_backend(hf_id, max_context, precision_run, device, slug=slug)


# ---------------------------------------------------------------------------
# subword <-> token bookkeeping (moved from sensi/extractors/surprisal.py; shared
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
# model selection / gating (moved from sensi/extractors/llm_cloze.py)
# ---------------------------------------------------------------------------

# Legacy model list — the GUI now reads the roster
# (sensi/extractors/predictability_models.all_specs) directly, so this constant is no
# longer consulted by production code. Retained (with tests asserting it holds
# no OpenAI entries) as the documented home for the local-model roster labels
# and a guardrail against an OpenAI backend silently reappearing. LOCAL models
# only — OpenAI can't score an existing string for per-token log-probabilities
# (the legacy /v1/completions echo+logprobs primitive is retired).
LLM_OPTIONS = [
    ('meta-llama/Meta-Llama-3-70B', 'Llama-3-70B   (free, gated)'),
    ('meta-llama/Meta-Llama-3-8B',  'Llama-3-8B    (free, gated)'),
    ('meta-llama/Llama-3.2-3B',     'Llama-3.2-3B  (free, gated)'),
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
        ...


def _model_id_for_vram(vram: float) -> str:
    """Pure VRAM-threshold decision, shared by every select_model caller.

    Kept side-effect-free (no VRAM probing here) so it needs no monkeypatch
    target of its own — select_model() below calls this after probing VRAM
    itself, via the module-level _available_vram_gb.
    """
    if vram >= 48:
        return 'meta-llama/Meta-Llama-3-70B'
    if vram >= 12:
        return 'meta-llama/Meta-Llama-3-8B'
    return 'meta-llama/Llama-3.2-3B'


def select_model() -> str:
    """Pick the best LOCAL LLM model id (a HuggingFace repo id, not a roster
    slug) given the current environment.

    Legacy entry point kept for callers that still want a free-form model id
    (e.g. scripts/validate_cloze.py) rather than driving add_predictability via
    the roster directly. A plain-name lookup within this module's own
    namespace, so unit tests can monkeypatch
    sensi.extractors.predictability._available_vram_gb to exercise each VRAM tier.
    """
    return _model_id_for_vram(_available_vram_gb())


def is_gated_local_model(model_id: str) -> bool:
    """True for gated local HF models (the meta-llama/* family) that need a token."""
    return model_id.startswith("meta-llama/")


def has_hf_token() -> bool:
    """True if a HuggingFace token is already available — HF_TOKEN /
    HUGGING_FACE_HUB_TOKEN env, a token saved by a prior `hf auth login`,
    or the token file named by prefs shared.hf_token_file (default
    config/hf_token.txt). Lazy imports keep GUI startup cheap."""
    try:
        from huggingface_hub import get_token
    except Exception:
        return False
    if bool(get_token()):
        return True
    try:
        from sensi.utils.credentials import read_hf_token
        from sensi.utils.preferences import load_preferences
        return bool(read_hf_token(load_preferences().get("shared", {})))
    except Exception:
        return False


def _available_vram_gb() -> float:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    except Exception:
        pass
    return 0.0


def _model_load_hint(model_id: str, exc: Exception) -> str:
    return (
        f"Could not load local LLM {model_id!r}: {type(exc).__name__}: {exc}\n"
        f"To fix: (1) accept the license at https://huggingface.co/{model_id} ; "
        f"(2) provide a HuggingFace token (GUI token box, HF_TOKEN env var, or "
        f"`hf auth login`); (3) on an offline machine, pre-download the weights "
        f"first. Or select an OpenAI model (set OPENAI_API_KEY) — no gating or download."
    )


def _quantized_load_hint(name: str, precision_run: str, exc: Exception) -> str:
    return (
        f"Could not load {name} at {precision_run}: {type(exc).__name__}: {exc}\n"
        f"8-bit / 4-bit loading needs a working bitsandbytes build on a CUDA GPU. "
        f"To fix: choose fp16 for this model on the Predictability card."
    )


def _is_cached_locally(model_id: str) -> bool:
    """True when the model's config.json is already in the local HF cache
    (no network: a local lookup only)."""
    try:
        from huggingface_hub import try_to_load_from_cache
        return isinstance(try_to_load_from_cache(model_id, "config.json"), str)
    except Exception:   # noqa: BLE001 — treat a failed lookup as "not cached"
        return False


class _HuggingFaceBackend:
    """Local Hugging Face causal-LM backend at one explicit precision.

    Loads via AutoModelForCausalLM/AutoTokenizer for every model — including
    GPT-2, where AutoModelForCausalLM("gpt2") resolves to the same
    GPT2LMHeadModel class (identical weights/logits) that the legacy
    sensi/extractors/surprisal.py loaded directly. `precision_run` / `device`
    come from predictability_models.load_plan; a failed load raises, it never
    falls back to another precision (that would change the values silently).
    """

    def __init__(self, model_id: str, max_context: int | None = None,
                 precision_run: str = "fp32", device: str = "cpu",
                 slug: str | None = None):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

        self._torch = torch
        self._model_id = model_id
        self._max_context = max_context
        self.precision_run = precision_run
        self.device = device
        slug = slug or slug_for_hf_id(model_id)
        name = slug or model_id

        if slug is not None and not _is_cached_locally(model_id):
            print(f"[predict] {slug}: downloading "
                  f"{spec_for(slug).download_gb:.2f} GB on first use…", flush=True)

        try:
            # A gated / missing / offline repo fails on the tokenizer first, so
            # the actionable hint below fires before any weights load.
            self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        except Exception as exc:
            raise RuntimeError(_model_load_hint(model_id, exc)) from exc
        try:
            if precision_run == "fp32":
                model = AutoModelForCausalLM.from_pretrained(
                    model_id, dtype=torch.float32)
                if device == "cuda":
                    model = model.to("cuda")
            elif precision_run == "fp16":
                model = AutoModelForCausalLM.from_pretrained(
                    model_id, dtype=torch.float16, device_map="auto")
            elif precision_run == "8bit":
                model = AutoModelForCausalLM.from_pretrained(
                    model_id, quantization_config=BitsAndBytesConfig(load_in_8bit=True),
                    device_map="auto")
            elif precision_run == "4bit":
                model = AutoModelForCausalLM.from_pretrained(
                    model_id, quantization_config=BitsAndBytesConfig(
                        load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16),
                    device_map="auto")
            else:
                raise ValueError(f"unknown precision {precision_run!r}")
        except Exception as exc:
            if precision_run in ("8bit", "4bit"):
                raise RuntimeError(
                    _quantized_load_hint(name, precision_run, exc)) from exc
            raise RuntimeError(_model_load_hint(model_id, exc)) from exc
        self._model = model

        self._model.eval()
        # Native context window, optionally capped by the user's per-model
        # max_context (Project B). A lower cap just makes score()'s chunking
        # kick in sooner; it never exceeds the model's real ceiling.
        native = getattr(self._model.config, 'max_position_embeddings', None)
        if max_context is None:
            self._context_limit = native
        elif native is None:
            self._context_limit = max_context
        else:
            self._context_limit = min(native, max_context)
        print(f"[predict] {name}: {precision_run} on "
              f"{'GPU' if device == 'cuda' else 'CPU'}", flush=True)

    def score(self, text: str, tokens: list[str]) -> list[dict]:
        torch = self._torch
        enc = self._tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        ids = enc['input_ids']
        offsets = enc['offset_mapping']
        n = len(ids)
        sub_log_p: list = [None] * n
        sub_top1: list = [None] * n
        if n == 0:
            return _scores_from_subwords(text, tokens, offsets, sub_log_p, sub_top1, ids)
        device = next(self._model.parameters()).device

        # Every forward pass is [BOS] + ids[a:b]: BOS is never scored and maps
        # to no word, and gives ids[0] a context, so the first word is scored
        # too. Without a BOS token (lead == 0) ids[0] stays unscored (None).
        bos = getattr(self._tokenizer, 'bos_token_id', None)
        lead = [] if bos is None else [bos]
        limit = self._context_limit
        if limit is None or n + len(lead) <= limit:
            segments = [(0, n, 0)]
        else:
            # Stimulus exceeds the model's context window — slide a chunked
            # window through it. Stride = window // 2 so each post-window
            # position sees at least window // 2 preceding subwords inside
            # its forward pass; BOS takes one slot of every window.
            segments = _chunk_slices(n, limit - len(lead))

        for a, b, keep in segments:
            ids_t = torch.tensor([lead + ids[a:b]], device=device)
            with torch.no_grad():
                log_probs = torch.log_softmax(self._model(ids_t).logits[0], dim=-1)
            # Logits at window position j predict ids[a + j + 1 - len(lead)];
            # keep positions j >= keep, so each subword is scored exactly once.
            for j in range(keep, b - a):
                t = a + j + 1 - len(lead)
                if t >= n:
                    continue
                sub_log_p[t] = log_probs[j, ids[t]].item()
                sub_top1[t] = int(log_probs[j].argmax().item())

        return _scores_from_subwords(text, tokens, offsets,
                                     sub_log_p, sub_top1, ids)


def _make_default_backend(model_id: str, max_context: int | None = None,
                          precision_run: str = "fp32", device: str = "cpu",
                          slug: str | None = None) -> _Backend:
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
    return _HuggingFaceBackend(model_id, max_context, precision_run, device, slug=slug)


# Process-wide backend cache, keyed by literal HuggingFace repo id (the same
# string space as roster hf_ids, e.g. "gpt2", and free-form model_ids, e.g.
# "meta-llama/Meta-Llama-3-8B" as select_model() returns) — so loads of the
# same model from different call sites share one instance. Loading a local 8B
# model costs minutes; we load once per id and reuse it across every document
# (and the word/doc passes) in a run — the same "load once" intent that
# _align_cache gives MFA. Keyed by (model_id, max_context, precision_run,
# device) so two context caps or precisions of the same model don't share one
# instance. Callers that pass an explicit `backend=` bypass this entirely.
_BACKEND_CACHE: dict[tuple, _Backend] = {}

# One {"slug", "hf_id", "precision", "device"} per backend built in this
# process, in load order (read through loaded_models()).
_LOADED: list[dict] = []


def loaded_models() -> list[dict]:
    return [dict(r) for r in _LOADED]


# Models a RUN scored with (cache hits included), in first-use order; unlike
# _LOADED this is per run: reset_used_models() is called when a run starts.
_USED: list[dict] = []


def reset_used_models() -> None:
    _USED.clear()


def used_models() -> list[dict]:
    return [dict(r) for r in _USED]


def _record_used(slug: str, precision_map: dict | None) -> None:
    precision_run, device = load_plan(slug, precision_map, _cuda_available())
    rec = {"slug": slug, "hf_id": spec_for(slug).hf_id,
           "precision": precision_run, "device": device}
    if rec not in _USED:
        _USED.append(rec)


def _cached_backend(model_id: str, max_context: int | None = None,
                    precision_run: str = "fp32", device: str = "cpu",
                    slug: str | None = None) -> _Backend:
    key = (model_id, max_context, precision_run, device)
    backend = _BACKEND_CACHE.get(key)
    if backend is None:
        # Keep at most one loaded instance per model_id: a run uses a single
        # context cap and precision per model, so another instance of the same
        # model is dead weight (each holds GBs of weights). Evict them before
        # loading — this keeps the cache bounded to the roster size even when
        # the user tries several caps/precisions across successive runs.
        for stale in [k for k in _BACKEND_CACHE if k[0] == model_id]:
            del _BACKEND_CACHE[stale]
        backend = _make_default_backend(model_id, max_context, precision_run,
                                        device, slug=slug)
        _BACKEND_CACHE[key] = backend
        _LOADED.append({"slug": slug or slug_for_hf_id(model_id), "hf_id": model_id,
                        "precision": precision_run, "device": device})
    return backend


# ---------------------------------------------------------------------------
# score cache: (model, text/tokens digest) → scores, reused across passes
# ---------------------------------------------------------------------------

# (hf_id/model_id, max_context, precision run, text/tokens digest) → scores.
# Sized to hold one GUI dispatch (word pass + doc pass over the same stimuli);
# FIFO-evicted.
_SCORE_CACHE: OrderedDict = OrderedDict()
SCORE_CACHE_MAX = 256


def _text_tokens_digest(text: str, tokens: list[str]) -> str:
    h = hashlib.sha256(text.encode('utf-8'))
    h.update(b'\x00')
    h.update('\x00'.join(tokens).encode('utf-8'))
    return h.hexdigest()
