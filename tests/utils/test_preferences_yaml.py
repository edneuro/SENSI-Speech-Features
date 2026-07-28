"""Two-file YAML store behavior: paths/env precedence, save routing,
round-trips, tolerance. Legacy-json import is covered in
test_preferences_migration.py; reset in test_preferences.py."""
import json
import yaml
from copy import deepcopy
from pathlib import Path

from utils.preferences import (
    DEFAULT_PREFS, SCHEMA_VERSION, get_config_dir,
    get_features_yaml_path, get_run_yaml_path,
    load_preferences, save_preferences,
)
from utils.config_yaml import (
    FEATURES_YAML, RUN_YAML, read_yaml, split_prefs,
    write_features_yaml, write_run_yaml,
)


def test_config_dir_env_precedence(tmp_path, monkeypatch):
    a, b = tmp_path / "a", tmp_path / "b"
    monkeypatch.setenv("SENSI_CONFIG_DIR", str(a))
    monkeypatch.setenv("SENSI_PREFS_PATH", str(b / "p.json"))
    assert get_config_dir() == a                       # CONFIG_DIR wins
    monkeypatch.delenv("SENSI_CONFIG_DIR")
    assert get_config_dir() == b                       # falls back to PREFS_PATH parent
    monkeypatch.delenv("SENSI_PREFS_PATH")
    assert get_config_dir().name == "config"           # project default
    assert get_features_yaml_path().name == FEATURES_YAML
    assert get_run_yaml_path().name == RUN_YAML


def test_legacy_json_path_follows_config_dir(tmp_path, monkeypatch):
    # Hermeticity: with only SENSI_CONFIG_DIR set, the legacy json is looked
    # up INSIDE that dir — never at the repo's real config/preferences.json.
    from utils.preferences import get_preferences_path
    monkeypatch.delenv("SENSI_PREFS_PATH", raising=False)
    monkeypatch.setenv("SENSI_CONFIG_DIR", str(tmp_path))
    assert get_preferences_path() == tmp_path / "preferences.json"


def test_load_returns_defaults_when_no_files(tmp_path, monkeypatch):
    monkeypatch.setenv("SENSI_CONFIG_DIR", str(tmp_path))
    prefs = load_preferences()
    assert prefs == DEFAULT_PREFS
    assert not (tmp_path / FEATURES_YAML).exists()     # load never creates files


def test_default_shared_has_hf_token_file():
    assert DEFAULT_PREFS["shared"]["hf_token_file"] == "hf_token.txt"


def test_save_then_load_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("SENSI_CONFIG_DIR", str(tmp_path))
    save_preferences({"features": {"master_structural": False},
                      "shared": {"analyzer_name": "Alice"}})
    prefs = load_preferences()
    assert prefs["features"]["master_structural"] is False
    assert prefs["shared"]["analyzer_name"] == "Alice"
    # sections landed in the right files
    fdata = read_yaml(tmp_path / FEATURES_YAML)
    rdata = read_yaml(tmp_path / RUN_YAML)
    assert fdata["features"]["master_structural"] is False
    assert fdata["schema_version"] == SCHEMA_VERSION
    assert "shared" not in fdata
    assert rdata["shared"]["analyzer_name"] == "Alice"
    assert set(rdata) == {"shared"}


def test_first_save_creates_both_files(tmp_path, monkeypatch):
    monkeypatch.setenv("SENSI_CONFIG_DIR", str(tmp_path))
    save_preferences({"features": {"master_structural": False}})
    assert (tmp_path / FEATURES_YAML).exists()
    assert (tmp_path / RUN_YAML).exists()              # pair completed on first save


def test_save_routes_writes_by_section(tmp_path, monkeypatch):
    monkeypatch.setenv("SENSI_CONFIG_DIR", str(tmp_path))
    save_preferences({"features": {"master_structural": False}})
    features_before = (tmp_path / FEATURES_YAML).read_text(encoding="utf-8")
    save_preferences({"shared": {"analyzer_name": "Bob"}})
    # a shared-only save must not rewrite features.yaml
    assert (tmp_path / FEATURES_YAML).read_text(encoding="utf-8") == features_before
    assert read_yaml(tmp_path / RUN_YAML)["shared"]["analyzer_name"] == "Bob"


def test_yaml_pair_takes_priority_over_legacy_json(tmp_path, monkeypatch):
    import json
    monkeypatch.setenv("SENSI_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("SENSI_PREFS_PATH", str(tmp_path / "preferences.json"))
    (tmp_path / "preferences.json").write_text(
        json.dumps({"schema_version": SCHEMA_VERSION,
                    "shared": {"analyzer_name": "FromJson"}}), encoding="utf-8")
    features, run = split_prefs(deepcopy(DEFAULT_PREFS))
    run["shared"]["analyzer_name"] = "FromYaml"
    write_features_yaml(tmp_path / FEATURES_YAML, features)
    write_run_yaml(tmp_path / RUN_YAML, run)
    assert load_preferences()["shared"]["analyzer_name"] == "FromYaml"
    assert (tmp_path / "preferences.json").exists()    # json untouched when pair exists


def test_missing_schema_version_in_yaml_assumes_current(tmp_path, monkeypatch):
    # A hand-made features.yaml without schema_version must NOT be run through
    # the v1 flat-file migration (which would mangle its sections).
    monkeypatch.setenv("SENSI_CONFIG_DIR", str(tmp_path))
    features, run = split_prefs(deepcopy(DEFAULT_PREFS))
    features.pop("schema_version")
    features["features"]["master_structural"] = False
    write_features_yaml(tmp_path / FEATURES_YAML, features)
    write_run_yaml(tmp_path / RUN_YAML, run)
    prefs = load_preferences()
    assert prefs["features"]["master_structural"] is False
    assert prefs["schema_version"] == SCHEMA_VERSION


def test_missing_schema_version_still_runs_the_v20_finalize_block(
        tmp_path, monkeypatch):
    # Final-review Fix 2: a missing schema_version used to be assumed
    # SCHEMA_VERSION (current), which skipped the v20 finalize block and let
    # a retired feat_phoneme_label survive with `changed` False (never
    # rewritten). It must instead be assumed v19, so the removal-migration
    # runs and the file self-heals on this very load.
    monkeypatch.setenv("SENSI_CONFIG_DIR", str(tmp_path))
    (tmp_path / FEATURES_YAML).write_text(
        "features:\n  feat_phoneme_label: true\n", encoding="utf-8")
    write_run_yaml(tmp_path / RUN_YAML, split_prefs(deepcopy(DEFAULT_PREFS))[1])
    prefs = load_preferences()
    assert "feat_phoneme_label" not in prefs["features"]
    assert prefs["features"]["feat_phoneme_arpabet"] is True
    assert prefs["schema_version"] == SCHEMA_VERSION
    # self-healed: reading the file back off disk shows the same thing, not
    # just the in-memory result.
    on_disk = read_yaml(tmp_path / FEATURES_YAML)
    assert "feat_phoneme_label" not in on_disk["features"]


def test_quoted_schema_version_migrates_like_the_int_equivalent(
        tmp_path, monkeypatch):
    # schema_version: '19' (a YAML string) must not silently fall through the
    # int-keyed version-branch comparison into the v1 flat-json migration
    # path, which would mangle this file's already-sectioned shape.
    monkeypatch.setenv("SENSI_CONFIG_DIR", str(tmp_path))
    (tmp_path / FEATURES_YAML).write_text(
        "schema_version: '19'\nfeatures:\n  feat_phoneme_label: true\n",
        encoding="utf-8")
    write_run_yaml(tmp_path / RUN_YAML, split_prefs(deepcopy(DEFAULT_PREFS))[1])
    prefs = load_preferences()
    assert "feat_phoneme_label" not in prefs["features"]
    assert prefs["features"]["feat_phoneme_arpabet"] is True
    assert prefs["schema_version"] == SCHEMA_VERSION


def test_corrupted_features_yaml_falls_back_to_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("SENSI_CONFIG_DIR", str(tmp_path))
    (tmp_path / FEATURES_YAML).write_text("features: [unclosed", encoding="utf-8")
    prefs = load_preferences()
    assert prefs["features"] == DEFAULT_PREFS["features"]
    # load never overwrites a corrupted file
    assert (tmp_path / FEATURES_YAML).read_text(encoding="utf-8") == "features: [unclosed"


def test_unknown_keys_survive_save_load(tmp_path, monkeypatch):
    # forward-compat: same tolerance the json store had
    monkeypatch.setenv("SENSI_CONFIG_DIR", str(tmp_path))
    features, run = split_prefs(deepcopy(DEFAULT_PREFS))
    features["future_section"] = {"x": 1}
    write_features_yaml(tmp_path / FEATURES_YAML, features)
    write_run_yaml(tmp_path / RUN_YAML, run)
    save_preferences({"features": {"master_structural": False}})
    assert read_yaml(tmp_path / FEATURES_YAML)["future_section"] == {"x": 1}


def test_legacy_json_imports_once_and_renames_to_bak(tmp_path, monkeypatch):
    monkeypatch.delenv("SENSI_CONFIG_DIR", raising=False)
    p = tmp_path / "preferences.json"
    monkeypatch.setenv("SENSI_PREFS_PATH", str(p))
    legacy = deepcopy(DEFAULT_PREFS)
    legacy["shared"]["analyzer_name"] = "Legacy"
    legacy["features"]["master_structural"] = False
    p.write_text(json.dumps(legacy), encoding="utf-8")

    prefs = load_preferences()
    assert prefs["shared"]["analyzer_name"] == "Legacy"
    assert prefs["features"]["master_structural"] is False
    assert not p.exists()
    assert (tmp_path / "preferences.json.bak").exists()
    assert (tmp_path / FEATURES_YAML).exists() and (tmp_path / RUN_YAML).exists()

    # second load reads the YAMLs, not the .bak: mutate features.yaml, reload
    fdata = read_yaml(tmp_path / FEATURES_YAML)
    fdata["features"]["master_structural"] = True
    write_features_yaml(tmp_path / FEATURES_YAML, fdata)
    assert load_preferences()["features"]["master_structural"] is True


def test_legacy_v2_json_runs_full_migration_chain(tmp_path, monkeypatch):
    monkeypatch.delenv("SENSI_CONFIG_DIR", raising=False)
    p = tmp_path / "preferences.json"
    monkeypatch.setenv("SENSI_PREFS_PATH", str(p))
    v2 = {"schema_version": 2,
          "shared": {"analyzer_name": "OldTimer", "dictionary_backend": "liwc"},
          "word_level": {"feature_toggles": {"zipf_freq": False}}}
    p.write_text(json.dumps(v2), encoding="utf-8")
    prefs = load_preferences()
    assert prefs["schema_version"] == SCHEMA_VERSION
    assert prefs["shared"]["analyzer_name"] == "OldTimer"
    assert prefs["semantic"]["backend"] == "empath"      # v11 liwc coercion
    assert prefs["features"]["freq_subtlex"] is False    # v2→v3 toggle mapping
    assert (tmp_path / "preferences.json.bak").exists()
    on_disk = read_yaml(tmp_path / FEATURES_YAML)
    assert on_disk["schema_version"] == SCHEMA_VERSION


def test_corrupted_legacy_json_returns_defaults_without_bak(tmp_path, monkeypatch):
    monkeypatch.delenv("SENSI_CONFIG_DIR", raising=False)
    p = tmp_path / "preferences.json"
    monkeypatch.setenv("SENSI_PREFS_PATH", str(p))
    p.write_text("{not json", encoding="utf-8")
    assert load_preferences() == DEFAULT_PREFS
    assert p.exists()                                   # left in place for inspection
    assert not (tmp_path / "preferences.json.bak").exists()
    assert not (tmp_path / FEATURES_YAML).exists()


def test_reset_rewrites_features_yaml_only(tmp_path, monkeypatch):
    from utils.preferences import reset_to_defaults
    monkeypatch.setenv("SENSI_CONFIG_DIR", str(tmp_path))
    save_preferences({"features": {"master_structural": False},
                      "shared": {"analyzer_name": "Keep Me"}})
    run_before = (tmp_path / RUN_YAML).read_text(encoding="utf-8")
    fresh = reset_to_defaults()                        # keep_shared=True default
    assert fresh["features"]["master_structural"] is True
    assert fresh["shared"]["analyzer_name"] == "Keep Me"
    assert (tmp_path / RUN_YAML).read_text(encoding="utf-8") == run_before
    fdata = read_yaml(tmp_path / FEATURES_YAML)
    assert fdata["features"]["master_structural"] is True


def test_reset_full_wipe_resets_run_yaml_too(tmp_path, monkeypatch):
    from utils.preferences import reset_to_defaults
    monkeypatch.setenv("SENSI_CONFIG_DIR", str(tmp_path))
    save_preferences({"shared": {"analyzer_name": "Wipe Me"}})
    fresh = reset_to_defaults(keep_shared=False)
    assert fresh["shared"]["analyzer_name"] == ""
    assert read_yaml(tmp_path / RUN_YAML)["shared"]["analyzer_name"] == ""


# ---------------------------------------------------------------------------
# Final-review hardening: half-import window (Fix 1) + read-only tolerance
# (Fix 2)
# ---------------------------------------------------------------------------

def test_empty_features_yaml_with_legacy_json_falls_through_to_import(tmp_path, monkeypatch):
    # A crash between the two _write_yaml_pair writes (or a torn write) can
    # leave features.yaml present-but-empty and run.yaml never written, while
    # the original preferences.json is still on disk. Both tolerant YAML
    # reads come back with nothing usable, so load_preferences must not
    # silently hand back defaults and orphan the json — it should complete
    # the import instead.
    monkeypatch.setenv("SENSI_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("SENSI_PREFS_PATH", str(tmp_path / "preferences.json"))
    legacy = deepcopy(DEFAULT_PREFS)
    legacy["shared"]["analyzer_name"] = "HalfImport"
    legacy["features"]["master_structural"] = False
    (tmp_path / "preferences.json").write_text(json.dumps(legacy), encoding="utf-8")
    (tmp_path / FEATURES_YAML).write_text("", encoding="utf-8")   # empty, no run.yaml

    prefs = load_preferences()
    assert prefs["shared"]["analyzer_name"] == "HalfImport"
    assert prefs["features"]["master_structural"] is False
    # import completed: pair written, legacy json renamed to .bak
    assert (tmp_path / FEATURES_YAML).exists() and (tmp_path / RUN_YAML).exists()
    assert not (tmp_path / "preferences.json").exists()
    assert (tmp_path / "preferences.json.bak").exists()


def test_corrupted_features_yaml_with_legacy_json_falls_through_to_import(tmp_path, monkeypatch):
    # Same half-import recovery, but the surviving features.yaml is
    # unparseable rather than empty (also tolerantly reads as {}).
    monkeypatch.setenv("SENSI_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("SENSI_PREFS_PATH", str(tmp_path / "preferences.json"))
    legacy = deepcopy(DEFAULT_PREFS)
    legacy["shared"]["analyzer_name"] = "Recovered"
    (tmp_path / "preferences.json").write_text(json.dumps(legacy), encoding="utf-8")
    (tmp_path / FEATURES_YAML).write_text("features: [unclosed", encoding="utf-8")

    prefs = load_preferences()
    assert prefs["shared"]["analyzer_name"] == "Recovered"
    assert (tmp_path / "preferences.json.bak").exists()
    assert not (tmp_path / "preferences.json").exists()


def test_complete_features_yaml_missing_run_yaml_with_legacy_json_reimports(tmp_path, monkeypatch):
    # The primary half-import variant: the crash lands BETWEEN the two
    # _write_yaml_pair writes, so features.yaml is complete and parseable but
    # run.yaml was never written, and preferences.json is still on disk. The
    # joined user dict is non-empty here, so the "both reads empty" guard
    # alone would miss it — an incomplete pair alongside a surviving json
    # must also fall through to the import (a completed import always
    # renames the json, so this state can only mean an interrupted one).
    monkeypatch.setenv("SENSI_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("SENSI_PREFS_PATH", str(tmp_path / "preferences.json"))
    legacy = deepcopy(DEFAULT_PREFS)
    legacy["shared"]["analyzer_name"] = "SurvivedCrash"
    legacy["features"]["master_structural"] = False
    (tmp_path / "preferences.json").write_text(json.dumps(legacy), encoding="utf-8")
    features, _run = split_prefs(deepcopy(legacy))
    write_features_yaml(tmp_path / FEATURES_YAML, features)   # complete; no run.yaml

    prefs = load_preferences()
    assert prefs["shared"]["analyzer_name"] == "SurvivedCrash"
    assert prefs["features"]["master_structural"] is False
    # import completed: pair now whole, legacy json renamed to .bak
    assert (tmp_path / RUN_YAML).exists()
    assert not (tmp_path / "preferences.json").exists()
    assert (tmp_path / "preferences.json.bak").exists()


def test_readonly_config_dir_does_not_crash_first_launch(tmp_path, monkeypatch):
    # An unwritable config dir must not crash startup: settings still load
    # for the session (from the legacy json), persistence just doesn't
    # happen, and the json is NOT renamed to .bak (it's the only surviving
    # store — destroying it would lose the user's settings).
    import utils.preferences as prefs_mod
    monkeypatch.delenv("SENSI_CONFIG_DIR", raising=False)
    p = tmp_path / "preferences.json"
    monkeypatch.setenv("SENSI_PREFS_PATH", str(p))
    legacy = deepcopy(DEFAULT_PREFS)
    legacy["shared"]["analyzer_name"] = "ReadOnly"
    p.write_text(json.dumps(legacy), encoding="utf-8")

    def _raise(_merged):
        raise OSError("read-only filesystem")
    monkeypatch.setattr(prefs_mod, "_write_yaml_pair", _raise)

    prefs = load_preferences()
    assert prefs["shared"]["analyzer_name"] == "ReadOnly"
    assert p.exists()
    assert not (tmp_path / "preferences.json.bak").exists()
