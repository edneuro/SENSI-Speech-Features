import json

import pytest

from utils.preferences import (
    DEFAULT_PREFS,
    load_preferences,
    save_preferences,
    reset_to_defaults,
    get_preferences_path,
)


def test_reset_to_defaults_restores_feature_toggles(tmp_path, monkeypatch):
    monkeypatch.setenv('SENSI_PREFS_PATH', str(tmp_path / 'p.json'))
    # Sanity: master_structural's shipped default is True, so flipping it off
    # actually diverges from DEFAULT_PREFS (a vacuous key wouldn't catch a
    # reset that silently no-ops).
    assert DEFAULT_PREFS["features"]["master_structural"] is True
    # User flips a feature and sets an I/O folder.
    save_preferences({
        "features": {"master_structural": False},
        "shared": {"last_transcripts_folder": "/x/y", "analyzer_name": "Alice"},
    })
    assert load_preferences()["features"]["master_structural"] is False

    fresh = reset_to_defaults()  # keep_shared=True by default
    # Feature toggles are back to shipped defaults...
    assert fresh["features"]["master_structural"] is True
    # ...verified via a fresh load_preferences() call, not just the returned dict.
    reloaded = load_preferences()
    assert reloaded["features"]["master_structural"] is True
    # ...but the I/O folder + analyzer are preserved.
    assert fresh["shared"]["last_transcripts_folder"] == "/x/y"
    assert fresh["shared"]["analyzer_name"] == "Alice"
    assert reloaded["shared"]["last_transcripts_folder"] == "/x/y"
    assert reloaded["shared"]["analyzer_name"] == "Alice"


def test_reset_to_defaults_full_wipe_clears_shared(tmp_path, monkeypatch):
    monkeypatch.setenv('SENSI_PREFS_PATH', str(tmp_path / 'p.json'))
    save_preferences({"shared": {"last_transcripts_folder": "/x/y"}})
    fresh = reset_to_defaults(keep_shared=False)
    assert fresh["shared"]["last_transcripts_folder"] == \
        DEFAULT_PREFS["shared"]["last_transcripts_folder"]
    # ...verified via a fresh load_preferences() call, not just the returned dict.
    assert load_preferences()["shared"]["last_transcripts_folder"] == \
        DEFAULT_PREFS["shared"]["last_transcripts_folder"]


def test_load_preferences_returns_default_when_file_missing(tmp_path, monkeypatch):
    monkeypatch.setenv('SENSI_PREFS_PATH', str(tmp_path / 'does-not-exist.json'))
    prefs = load_preferences()
    assert prefs == DEFAULT_PREFS


def test_save_then_load_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv('SENSI_PREFS_PATH', str(tmp_path / 'p.json'))
    save_preferences({
        'shared': {'analyzer_name': 'Alice'},
        'semantic': {'backend': 'empath'},
    })
    loaded = load_preferences()
    assert loaded['shared']['analyzer_name'] == 'Alice'
    # 'empath' round-trips; 'liwc' would be coerced away (disabled, EULA).
    assert loaded['semantic']['backend'] == 'empath'
    # Defaults preserved
    assert 'features' in loaded
    assert 'output' in loaded


def test_get_preferences_path_respects_env_var(tmp_path, monkeypatch):
    custom = tmp_path / 'custom.json'
    monkeypatch.setenv('SENSI_PREFS_PATH', str(custom))
    assert get_preferences_path() == custom


def test_get_preferences_path_defaults_to_config_dir(monkeypatch):
    # The autouse _isolated_config_dir fixture sets SENSI_CONFIG_DIR for every
    # test; undo it here too so this test can see the true project default
    # (same pattern as test_config_dir_env_precedence in test_preferences_yaml.py).
    monkeypatch.delenv('SENSI_CONFIG_DIR', raising=False)
    monkeypatch.delenv('SENSI_PREFS_PATH', raising=False)
    p = get_preferences_path()
    assert p.name == 'preferences.json'
    assert p.parent.name == 'config'


def test_default_prefs_v3_has_expected_top_level_keys():
    from utils.preferences import DEFAULT_PREFS_V3
    assert DEFAULT_PREFS_V3["schema_version"] == 3
    assert set(DEFAULT_PREFS_V3.keys()) == {
        "schema_version", "shared", "features",
        "predictability", "semantic", "output",
    }


def test_default_prefs_v3_features_keys():
    from utils.preferences import DEFAULT_PREFS_V3
    expected = {
        "structural",
        "lexling_tokens", "lexling_pos", "lexling_morph", "lexling_dep",
        "freq_subtlex",
        "freq_ewfg_k2", "freq_ewfg_3to5", "freq_ewfg_6to8",
        "freq_ewfg_9to12", "freq_ewfg_overall",
        "pred_surprisal_gpt2", "pred_surprisal_llm",
        "pred_cloze", "pred_topmatch",
        "sem_indicators", "sem_composite_psych",
        "acoustic_boundaries",
    }
    assert set(DEFAULT_PREFS_V3["features"].keys()) == expected
    assert all(isinstance(v, bool) for v in DEFAULT_PREFS_V3["features"].values())


def test_default_prefs_v3_has_two_category_lists():
    from utils.preferences import DEFAULT_PREFS_V3
    assert "categories_liwc" in DEFAULT_PREFS_V3["semantic"]
    assert "categories_empath" in DEFAULT_PREFS_V3["semantic"]
    assert isinstance(DEFAULT_PREFS_V3["semantic"]["categories_liwc"], list)
    assert isinstance(DEFAULT_PREFS_V3["semantic"]["categories_empath"], list)


# ---------------------------------------------------------------------------
# v4 — active schema (batch 3)
# ---------------------------------------------------------------------------

def test_default_prefs_v4_schema_version():
    from utils.preferences import DEFAULT_PREFS_V4
    assert DEFAULT_PREFS_V4["schema_version"] == 4


def test_default_prefs_v4_has_per_column_structural_keys():
    from utils.preferences import DEFAULT_PREFS_V4
    f = DEFAULT_PREFS_V4["features"]
    assert "structural" not in f
    for k in ("feat_struct_doc_id", "feat_struct_paragraph_id",
              "feat_struct_sentence_id", "feat_struct_sentence_text",
              "feat_struct_char_start", "feat_struct_char_end",
              "feat_struct_term_id", "feat_struct_token_id"):
        assert k in f, k


def test_default_prefs_v4_has_per_column_acoustic_keys():
    from utils.preferences import DEFAULT_PREFS_V4
    f = DEFAULT_PREFS_V4["features"]
    assert "acoustic_boundaries" not in f
    for k in ("feat_acoustic_onset", "feat_acoustic_end",
              "feat_acoustic_word_duration", "feat_acoustic_word_gap"):
        assert k in f, k


def test_default_prefs_v4_has_per_column_psych_keys():
    from utils.preferences import DEFAULT_PREFS_V4
    f = DEFAULT_PREFS_V4["features"]
    assert "sem_composite_psych" not in f
    for k in ("feat_psych_authentic", "feat_psych_analytic",
              "feat_psych_tone", "feat_psych_clout"):
        assert k in f, k


def test_default_prefs_v4_has_per_column_doc_count_keys():
    from utils.preferences import DEFAULT_PREFS_V4
    o = DEFAULT_PREFS_V4["output"]
    assert "doc_counts" not in o
    for k in ("out_count_tokens", "out_count_content_tokens",
              "out_count_sentences", "out_count_paragraphs"):
        assert k in o, k


def test_default_prefs_v4_has_four_category_lists():
    from utils.preferences import DEFAULT_PREFS_V4
    s = DEFAULT_PREFS_V4["semantic"]
    for k in ("word_categories_liwc", "word_categories_empath",
              "doc_categories_liwc", "doc_categories_empath"):
        assert k in s, k
        assert isinstance(s[k], list)
