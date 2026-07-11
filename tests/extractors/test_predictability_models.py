import re
import pytest
from extractors.predictability_models import (
    all_specs, spec_for, is_gated, slug_for_hf_id, ALL_SPECS,
)

_SAFE = re.compile(r"^[a-z0-9_]+$")

def test_roster_has_gpt2_and_three_llamas():
    slugs = {s.slug for s in all_specs()}
    assert slugs == {"gpt2", "llama32_3b", "llama3_8b", "llama3_70b"}

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
