"""Roster of local language models for the predictability feature.

Single source of truth for the GUI model list, column/aggregate prefixes,
gating, and (for the future context-size project) each model's native
context ceiling. `slug` is the only identifier used downstream; it is
column- and filename-safe.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    slug: str          # column/filename-safe id; the sole downstream identifier
    label: str         # GUI display text
    hf_id: str         # HuggingFace repo id passed to from_pretrained
    gated: bool        # needs a HF token + accepted license
    native_ctx: int    # max_position_embeddings ceiling (data only here)
    note: str          # short GUI hint
    params: float      # parameter count
    download_gb: float # size of the first-use download
    precisions: tuple[str, ...]  # offered load precisions; first = default


_LLAMA_PRECISIONS = ("fp16", "8bit", "4bit")

ALL_SPECS: list[ModelSpec] = [
    ModelSpec("gpt2", "GPT-2", "gpt2", False, 1024,
              "free, no token, 0.55 GB download on first use, ~0.5 GB RAM",
              124e6, 0.55, ("fp32",)),
    ModelSpec("gpt2_medium", "GPT-2 medium", "openai-community/gpt2-medium",
              False, 1024, "free, no token, 1.52 GB download on first use, ~1.4 GB RAM",
              355e6, 1.52, ("fp32",)),
    ModelSpec("gpt2_large", "GPT-2 large", "openai-community/gpt2-large",
              False, 1024, "free, no token, 3.25 GB download on first use, ~3.1 GB RAM",
              774e6, 3.25, ("fp32",)),
    ModelSpec("gpt2_xl", "GPT-2 XL", "openai-community/gpt2-xl",
              False, 1024, "free, no token, 6.43 GB download on first use, ~6.0 GB RAM",
              1.5e9, 6.43, ("fp32",)),
    ModelSpec("llama32_3b", "Llama-3.2-3B", "meta-llama/Llama-3.2-3B",
              True, 131072,
              "gated, free; 6.4 GB download on first use; ~6.4 GB in fp16 (needs a GPU)",
              3.21e9, 6.42, _LLAMA_PRECISIONS),
    ModelSpec("llama3_8b", "Llama-3-8B", "meta-llama/Meta-Llama-3-8B",
              True, 8192,
              "gated, free; 16 GB download on first use; ~16 GB in fp16 (needs a GPU)",
              8e9, 16.0, _LLAMA_PRECISIONS),
    ModelSpec("llama3_70b", "Llama-3-70B", "meta-llama/Meta-Llama-3-70B",
              True, 8192,
              "gated, free; 142 GB download on first use; ~142 GB in fp16 (needs a GPU)",
              71e9, 142.0, _LLAMA_PRECISIONS),
]

_BY_SLUG = {s.slug: s for s in ALL_SPECS}
_BY_HF_ID = {s.hf_id: s.slug for s in ALL_SPECS}


def all_specs() -> list[ModelSpec]:
    return list(ALL_SPECS)


def spec_for(slug: str) -> ModelSpec:
    return _BY_SLUG[slug]


def is_gated(slug: str) -> bool:
    return _BY_SLUG[slug].gated


def slug_for_hf_id(hf_id: str) -> str | None:
    return _BY_HF_ID.get(hf_id)


# Minimum sensible context window (subwords). Below this a chunked window gives
# each word almost no preceding context, so a user cap is clamped up to it.
FLOOR_CONTEXT = 128


def effective_context(slug: str, context_map: dict) -> int:
    """Resolve a model's effective max context (subwords) from a user cap.

    The single source of truth for context clamping — the GUI, the extractor,
    and any validation call this so they never disagree. A slug ABSENT from
    `context_map` is uncapped (the model's native ceiling). A present value is
    clamped to [FLOOR_CONTEXT, native_ctx].
    """
    native = _BY_SLUG[slug].native_ctx
    raw = context_map.get(slug)
    if raw is None:
        return native
    return max(FLOOR_CONTEXT, min(int(raw), native))


BYTES_PER_PARAM: dict[str, float] = {
    "fp32": 4.0, "fp16": 2.0, "8bit": 1.0, "4bit": 0.5,
}


def default_precision(slug: str) -> str:
    return _BY_SLUG[slug].precisions[0]


def effective_precision(slug: str, precision_map: dict | None) -> str:
    """The user's chosen precision if the model offers it, else its default."""
    chosen = (precision_map or {}).get(slug)
    if chosen in _BY_SLUG[slug].precisions:
        return chosen
    return default_precision(slug)


def load_plan(slug: str, precision_map: dict | None,
              cuda: bool) -> tuple[str, str]:
    """(precision actually run, device). No GPU -> fp32 on CPU for every model;
    with a GPU the chosen precision runs on it."""
    if not cuda:
        return "fp32", "cpu"
    return effective_precision(slug, precision_map), "cuda"


def weights_gb(slug: str, precision_run: str) -> float:
    """Weight memory (GB) of a model loaded at `precision_run`."""
    return _BY_SLUG[slug].params * BYTES_PER_PARAM[precision_run] / 1e9
