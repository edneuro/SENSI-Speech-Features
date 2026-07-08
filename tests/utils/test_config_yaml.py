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
