import pytest
import yaml
from copy import deepcopy
from utils.preferences import DEFAULT_PREFS
from utils.config_yaml import (
    FEATURES_YAML, RUN_YAML, split_prefs, join_yaml_pair, read_yaml,
    write_features_yaml, write_run_yaml,
    validate_features_yaml, validate_run_yaml,
    retired_keys_present, RETIRED_FEATURE_KEYS, RETIRED_PREDICTABILITY_KEYS,
)


def test_split_prefs_routes_shared_to_run_and_rest_to_features():
    features, run = split_prefs(DEFAULT_PREFS)
    assert set(run) == {"shared"}
    assert run["shared"] == DEFAULT_PREFS["shared"]
    assert "shared" not in features
    assert features["schema_version"] == DEFAULT_PREFS["schema_version"]
    # every non-shared section lands in features.yaml
    assert set(features) == set(DEFAULT_PREFS) - {"shared"}


def test_split_prefs_returns_copies():
    features, run = split_prefs(DEFAULT_PREFS)
    run["shared"]["analyzer_name"] = "mutated"
    features["features"]["master_structural"] = "mutated"
    assert DEFAULT_PREFS["shared"]["analyzer_name"] != "mutated"
    assert DEFAULT_PREFS["features"]["master_structural"] != "mutated"


def test_join_yaml_pair_round_trips_defaults():
    features, run = split_prefs(DEFAULT_PREFS)
    assert join_yaml_pair(features, run) == DEFAULT_PREFS


def test_join_yaml_pair_tolerates_missing_parts():
    features, run = split_prefs(DEFAULT_PREFS)
    assert "shared" not in join_yaml_pair(features, {})
    only_run = join_yaml_pair({}, run)
    assert only_run == {"shared": DEFAULT_PREFS["shared"]}


def test_write_read_round_trip(tmp_path):
    features, run = split_prefs(DEFAULT_PREFS)
    fp, rp = tmp_path / FEATURES_YAML, tmp_path / RUN_YAML
    write_features_yaml(fp, features)
    write_run_yaml(rp, run)
    assert read_yaml(fp) == features
    assert read_yaml(rp) == run


def test_written_files_have_regenerated_comments(tmp_path):
    features, run = split_prefs(DEFAULT_PREFS)
    fp, rp = tmp_path / FEATURES_YAML, tmp_path / RUN_YAML
    write_features_yaml(fp, features)
    write_run_yaml(rp, run)
    ftext = fp.read_text(encoding="utf-8")
    rtext = rp.read_text(encoding="utf-8")
    assert ftext.startswith("#") and rtext.startswith("#")
    # a per-section comment precedes each top-level section
    assert "\n# " in ftext and "predictability" in ftext
    assert "Reset settings" in ftext  # reset-able is documented in the header
    assert "Never touched" in rtext


def test_write_creates_parent_dirs(tmp_path):
    nested = tmp_path / "a" / "b" / FEATURES_YAML
    features, _ = split_prefs(DEFAULT_PREFS)
    write_features_yaml(nested, features)
    assert read_yaml(nested) == features


def test_read_yaml_missing_or_empty_returns_empty_dict(tmp_path):
    assert read_yaml(tmp_path / "nope.yaml") == {}
    p = tmp_path / "empty.yaml"
    p.write_text("", encoding="utf-8")
    assert read_yaml(p) == {}


def test_read_yaml_malformed_raises(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("features: [unclosed", encoding="utf-8")
    with pytest.raises(yaml.YAMLError):
        read_yaml(p)


def test_validate_features_yaml_accepts_default_split():
    features, _ = split_prefs(DEFAULT_PREFS)
    assert validate_features_yaml(features) == []


def test_validate_features_yaml_flags_unknown_and_bad_type():
    features, _ = split_prefs(DEFAULT_PREFS)
    features["features"]["not_a_real_toggle"] = True
    key = next(iter(DEFAULT_PREFS["features"]))
    features["output"]["level_word"] = "yes"          # not a bool
    features["prosody"]["not_a_setting"] = 1
    features["bogus_section"] = {}
    errs = validate_features_yaml(features)
    assert any("not_a_real_toggle" in e for e in errs)
    assert any("level_word" in e and "true/false" in e for e in errs)
    assert any("prosody.not_a_setting" in e for e in errs)
    assert any("bogus_section" in e for e in errs)


def test_validate_run_yaml_accepts_default_and_flags_unknown():
    _, run = split_prefs(DEFAULT_PREFS)
    assert validate_run_yaml(run) == []
    run["shared"]["bogus_key"] = "x"
    run["not_shared"] = {}
    errs = validate_run_yaml(run)
    assert any("bogus_key" in e for e in errs)
    assert any("not_shared" in e for e in errs)


# ---------------------------------------------------------------------------
# Final-review hardening (Fix 3): validators must not raise AttributeError on
# a truthy non-mapping section — flag it as an error and skip iterating it.
# ---------------------------------------------------------------------------

def test_validate_features_yaml_non_mapping_features_section_reports_error():
    errs = validate_features_yaml({"features": [1, 2, 3]})
    assert any("features" in e and "must be a mapping" in e for e in errs)


def test_validate_features_yaml_non_mapping_scalar_section_reports_error():
    errs = validate_features_yaml({"features": 5})
    assert any("features" in e and "must be a mapping" in e for e in errs)


def test_validate_features_yaml_non_mapping_typed_section_reports_error():
    errs = validate_features_yaml({"prosody": 5})
    assert any("prosody" in e and "must be a mapping" in e for e in errs)


def test_validate_features_yaml_non_mapping_semantic_section_reports_error():
    errs = validate_features_yaml({"semantic": 5})
    assert any("semantic" in e and "must be a mapping" in e for e in errs)


def test_validate_features_yaml_non_mapping_output_section_reports_error():
    errs = validate_features_yaml({"output": [1, 2]})
    assert any("output" in e and "must be a mapping" in e for e in errs)


def test_validate_run_yaml_non_mapping_shared_section_reports_error():
    errs = validate_run_yaml({"shared": 5})
    assert any("shared" in e and "must be a mapping" in e for e in errs)


# ---------------------------------------------------------------------------
# Final-review Fix 2: retired keys are tolerated (warned), not errors; a
# genuine typo is still an error.
# ---------------------------------------------------------------------------

def test_validate_features_yaml_tolerates_every_retired_feature_key():
    features, _ = split_prefs(DEFAULT_PREFS)
    for key in RETIRED_FEATURE_KEYS:
        features["features"][key] = True
    assert validate_features_yaml(features) == []


def test_validate_features_yaml_tolerates_retired_predictability_llm_model():
    features, _ = split_prefs(DEFAULT_PREFS)
    features["predictability"]["llm_model"] = "gpt-3.5"
    assert validate_features_yaml(features) == []


def test_validate_features_yaml_still_rejects_a_genuine_typo():
    features, _ = split_prefs(DEFAULT_PREFS)
    features["features"]["feat_phoneme_arpabetX"] = True
    errs = validate_features_yaml(features)
    assert any("feat_phoneme_arpabetX" in e for e in errs)


def test_retired_keys_present_reports_each_retired_feature_key():
    features, _ = split_prefs(DEFAULT_PREFS)
    features["features"]["feat_phoneme_label"] = True
    warnings = retired_keys_present(features)
    assert any("feat_phoneme_label" in w for w in warnings)


def test_retired_keys_present_reports_retired_predictability_key():
    features, _ = split_prefs(DEFAULT_PREFS)
    features["predictability"]["llm_model"] = "gpt-3.5"
    warnings = retired_keys_present(features)
    assert any("predictability.llm_model" in w for w in warnings)


def test_retired_keys_present_empty_for_a_clean_default_config():
    features, _ = split_prefs(DEFAULT_PREFS)
    assert retired_keys_present(features) == []


def test_retired_keys_present_ignores_a_genuine_typo():
    features, _ = split_prefs(DEFAULT_PREFS)
    features["features"]["feat_phoneme_arpabetX"] = True
    assert retired_keys_present(features) == []
