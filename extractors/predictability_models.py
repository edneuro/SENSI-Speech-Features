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


ALL_SPECS: list[ModelSpec] = [
    ModelSpec("gpt2", "GPT-2", "gpt2", False, 1024,
              "free, no token, ~500 MB, CPU-capable baseline"),
    ModelSpec("llama32_3b", "Llama-3.2-3B", "meta-llama/Llama-3.2-3B",
              True, 131072, "gated, free; ≥8 GB VRAM or CPU"),
    ModelSpec("llama3_8b", "Llama-3-8B", "meta-llama/Meta-Llama-3-8B",
              True, 8192, "gated, free; ≥12 GB VRAM"),
    ModelSpec("llama3_70b", "Llama-3-70B", "meta-llama/Meta-Llama-3-70B",
              True, 8192, "gated, free; ≥48 GB VRAM"),
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
