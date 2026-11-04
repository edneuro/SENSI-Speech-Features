import re
import pytest
from sensi.extractors.predictability_models import (
    all_specs, spec_for, is_gated, slug_for_hf_id, ALL_SPECS,
    default_precision, effective_precision, load_plan, weights_gb,
)

_SAFE = re.compile(r"^[a-z0-9_]+$")

def test_roster_has_gpt2_family_and_three_llamas():
    slugs = [s.slug for s in all_specs()]
    assert slugs == ["gpt2", "gpt2_medium", "gpt2_large", "gpt2_xl",
                     "llama32_3b", "llama3_8b", "llama3_70b"]

def test_slugs_are_column_and_filename_safe():
    for s in ALL_SPECS:
        assert _SAFE.match(s.slug), s.slug

def test_gpt2_is_free_and_1024():
    g = spec_for("gpt2")
    assert g.gated is False and g.hf_id == "gpt2" and g.native_ctx == 1024

def test_llamas_are_gated_with_ceilings():
    assert is_gated("llama3_8b") and spec_for("llama3_8b").native_ctx == 8192
    assert spec_for("llama32_3b").native_ctx == 131072
    assert spec_for("llama3_70b").native_ctx == 8192

def test_slug_for_hf_id_maps_stored_model():
    assert slug_for_hf_id("meta-llama/Meta-Llama-3-8B") == "llama3_8b"
    assert slug_for_hf_id("gpt2") == "gpt2"
    assert slug_for_hf_id("something/unknown") is None

def test_spec_for_unknown_raises():
    with pytest.raises(KeyError):
        spec_for("nope")


# --- effective_context: per-model cap clamped to [FLOOR, native_ctx] ---------

def test_effective_context_absent_slug_is_native():
    from sensi.extractors.predictability_models import effective_context, spec_for
    # No entry for the slug → uncapped → the model's native ceiling.
    assert effective_context("gpt2", {}) == spec_for("gpt2").native_ctx
    assert effective_context("llama3_8b", {}) == 8192


def test_effective_context_cap_below_native_is_honored():
    from sensi.extractors.predictability_models import effective_context
    assert effective_context("llama3_8b", {"llama3_8b": 2048}) == 2048


def test_effective_context_above_native_clamps_down():
    from sensi.extractors.predictability_models import effective_context
    # gpt2 native is 1024; asking for more clamps to native.
    assert effective_context("gpt2", {"gpt2": 999999}) == 1024


def test_effective_context_below_floor_clamps_to_floor():
    from sensi.extractors.predictability_models import effective_context, FLOOR_CONTEXT
    assert FLOOR_CONTEXT == 128
    assert effective_context("llama3_8b", {"llama3_8b": 1}) == 128


def test_effective_context_exactly_native():
    from sensi.extractors.predictability_models import effective_context
    assert effective_context("gpt2", {"gpt2": 1024}) == 1024


def test_gpt2_family_in_roster():
    for slug, hf, gb in [("gpt2", "gpt2", 0.55), ("gpt2_medium", "openai-community/gpt2-medium", 1.52),
                         ("gpt2_large", "openai-community/gpt2-large", 3.25),
                         ("gpt2_xl", "openai-community/gpt2-xl", 6.43)]:
        s = spec_for(slug)
        assert (s.hf_id, s.download_gb, s.native_ctx, s.precisions, s.gated) == (hf, gb, 1024, ("fp32",), False)


def test_llama_precisions_default_fp16():
    for slug in ("llama32_3b", "llama3_8b", "llama3_70b"):
        assert spec_for(slug).precisions == ("fp16", "8bit", "4bit")
        assert default_precision(slug) == "fp16"


@pytest.mark.parametrize("slug,chosen,cuda,expected", [
    ("gpt2_large", None, True, ("fp32", "cuda")),
    ("gpt2_large", None, False, ("fp32", "cpu")),
    ("llama3_8b", None, True, ("fp16", "cuda")),
    ("llama3_8b", "4bit", True, ("4bit", "cuda")),
    ("llama3_8b", "8bit", True, ("8bit", "cuda")),
    ("llama3_8b", "4bit", False, ("fp32", "cpu")),
    ("llama3_8b", "fp16", False, ("fp32", "cpu")),
])
def test_load_plan_table(slug, chosen, cuda, expected):
    pm = {slug: chosen} if chosen else {}
    assert load_plan(slug, pm, cuda) == expected


def test_effective_precision_ignores_values_the_model_does_not_offer():
    assert effective_precision("gpt2", {"gpt2": "fp16"}) == "fp32"
    assert effective_precision("llama3_8b", {"llama3_8b": "bf16"}) == "fp16"


def test_weights_gb():
    assert weights_gb("llama3_70b", "fp16") == pytest.approx(142.0)
    assert weights_gb("gpt2_xl", "fp32") == pytest.approx(6.0)
