from copy import deepcopy
from utils.preferences import DEFAULT_PREFS
from utils.config_yaml import prefs_to_config, config_to_prefs, _settings_sections


def test_settings_sections_excludes_structural_sections():
    s = set(_settings_sections())
    assert "predictability" in s and "prosody" in s and "acoustic_ts" in s
    assert not (s & {"schema_version", "shared", "features", "output", "semantic"})


def test_prefs_to_config_shape_and_omits_secrets_and_paths():
    cfg = prefs_to_config(DEFAULT_PREFS)
    assert set(cfg) == {"run", "features", "settings", "semantic_categories"}
    # features block merges features + output toggles
    assert set(cfg["features"]) == set(DEFAULT_PREFS["features"]) | set(DEFAULT_PREFS["output"])
    # run block has empty placeholders + token-file paths, no real shared values
    assert cfg["run"]["transcripts_folder"] == ""
    assert cfg["run"]["analyzer"] == ""
    assert "hf_token_file" in cfg["run"] and "openai_api_key_file" in cfg["run"]
    assert "hf_token" not in cfg["run"]  # never a raw secret


def test_config_to_prefs_round_trips_toggles_settings_categories():
    cfg = prefs_to_config(DEFAULT_PREFS)
    prefs = config_to_prefs(cfg)
    assert prefs["features"] == DEFAULT_PREFS["features"]
    assert prefs["output"] == DEFAULT_PREFS["output"]
    assert prefs["semantic"] == DEFAULT_PREFS["semantic"]
    for s in _settings_sections():
        assert prefs[s] == DEFAULT_PREFS[s]


def test_config_to_prefs_routes_toggles_and_run_folders():
    cfg = prefs_to_config(DEFAULT_PREFS)
    # pick one real feature key and one real output key
    feat_key = next(iter(DEFAULT_PREFS["features"]))
    out_key = next(iter(DEFAULT_PREFS["output"]))
    cfg["features"][feat_key] = False
    cfg["features"][out_key] = False
    cfg["run"].update(transcripts_folder="/scratch/t", audio_folder="/scratch/a",
                       output_folder="/scratch/o", analyzer="me")
    prefs = config_to_prefs(cfg)
    assert prefs["features"][feat_key] is False
    assert prefs["output"][out_key] is False
    assert prefs["shared"]["last_transcripts_folder"] == "/scratch/t"
    assert prefs["shared"]["last_audio_folder"] == "/scratch/a"
    assert prefs["shared"]["last_output_folder"] == "/scratch/o"
    assert prefs["shared"]["analyzer_name"] == "me"


from utils.config_yaml import validate_config


def test_validate_config_accepts_exported_default():
    cfg = prefs_to_config(DEFAULT_PREFS)
    assert validate_config(cfg) == []


def test_validate_config_flags_unknown_toggle():
    cfg = prefs_to_config(DEFAULT_PREFS)
    cfg["features"]["not_a_real_toggle"] = True
    errs = validate_config(cfg)
    assert any("not_a_real_toggle" in e for e in errs)


def test_validate_config_flags_bad_toggle_type():
    cfg = prefs_to_config(DEFAULT_PREFS)
    key = next(iter(DEFAULT_PREFS["features"]))
    cfg["features"][key] = "yes"  # not a bool
    errs = validate_config(cfg)
    assert any(key in e and "true/false" in e for e in errs)


def test_validate_config_flags_unknown_section_and_setting():
    cfg = prefs_to_config(DEFAULT_PREFS)
    cfg["bogus_section"] = {}
    cfg["settings"].setdefault("prosody", {})["not_a_setting"] = 1
    errs = validate_config(cfg)
    assert any("bogus_section" in e for e in errs)
    assert any("not_a_setting" in e for e in errs)


def test_write_then_read_config_round_trips(tmp_path):
    from utils.config_yaml import read_config, write_config
    cfg = prefs_to_config(DEFAULT_PREFS)
    p = tmp_path / "config.yaml"
    write_config(cfg, str(p))
    text = p.read_text(encoding="utf-8")
    assert text.startswith("#")            # header comment present
    back = read_config(str(p))
    assert back == cfg                      # structural round-trip


def test_read_config_empty_file_returns_empty_dict(tmp_path):
    from utils.config_yaml import read_config
    p = tmp_path / "empty.yaml"
    p.write_text("", encoding="utf-8")
    assert read_config(str(p)) == {}


def test_config_to_prefs_expands_user_and_env_in_run_folders(monkeypatch):
    import os
    monkeypatch.setenv("MYSCRATCH", "/scratch/users/x")
    cfg = prefs_to_config(DEFAULT_PREFS)
    cfg["run"]["transcripts_folder"] = "$MYSCRATCH/t"
    cfg["run"]["audio_folder"] = "$MYSCRATCH/a"
    cfg["run"]["output_folder"] = "~/out"
    prefs = config_to_prefs(cfg)
    assert prefs["shared"]["last_transcripts_folder"] == "/scratch/users/x/t"
    assert prefs["shared"]["last_audio_folder"] == "/scratch/users/x/a"
    assert prefs["shared"]["last_output_folder"] == os.path.expanduser("~/out")


def test_features_and_output_key_sets_are_disjoint():
    # config_yaml's flatten (prefs_to_config) and membership routing
    # (config_to_prefs) assume no key appears in both prefs sections. Guard it.
    assert set(DEFAULT_PREFS["features"]) & set(DEFAULT_PREFS["output"]) == set()
