import json
import os
from pathlib import Path

import pytest

from utils.preferences import (
    DEFAULT_PREFS,
    load_preferences,
    save_preferences,
    SCHEMA_VERSION,
)


def test_default_prefs_has_v4_sections():
    """DEFAULT_PREFS is now v4-shaped (same top-level sections as v3)."""
    assert 'shared' in DEFAULT_PREFS
    assert 'features' in DEFAULT_PREFS
    assert 'predictability' in DEFAULT_PREFS
    assert 'semantic' in DEFAULT_PREFS
    assert 'output' in DEFAULT_PREFS


def test_default_prefs_semantic_has_unified_category_lists():
    """v7 default uses unified categories_<backend> keys (split keys removed)."""
    s = DEFAULT_PREFS['semantic']
    for k in ('categories_liwc', 'categories_empath'):
        assert k in s, k
    assert len(s['categories_empath']) >= 10
    for k in ('word_categories_liwc', 'word_categories_empath',
              'doc_categories_liwc', 'doc_categories_empath'):
        assert k not in s, f"split key {k!r} should not exist in v7 defaults"


def test_default_prefs_features_has_v4_keys(tmp_path, monkeypatch):
    monkeypatch.setenv('SENSI_PREFS_PATH', str(tmp_path / 'p.json'))
    prefs = load_preferences()
    keys = set(prefs['features'].keys())
    # Carried-over keys
    for k in ('lexling_tokens', 'lexling_pos', 'lexling_morph', 'lexling_dep',
              'freq_subtlex', 'freq_ewfg_k2', 'freq_ewfg_3to5',
              'freq_ewfg_6to8', 'freq_ewfg_9to12', 'freq_ewfg_overall',
              'pred_surprisal_gpt2', 'pred_surprisal_llm',
              'pred_cloze', 'pred_topmatch', 'sem_indicators'):
        assert k in keys, k
    # Per-column expansions (batch 3)
    for k in ('feat_struct_doc_id', 'feat_struct_paragraph_id',
              'feat_struct_sentence_id', 'feat_struct_sentence_text',
              'feat_struct_char_start', 'feat_struct_char_end',
              'feat_struct_term_id', 'feat_struct_token_id',
              'feat_acoustic_onset', 'feat_acoustic_end',
              'feat_acoustic_word_duration', 'feat_acoustic_word_gap',
              'feat_psych_authentic', 'feat_psych_analytic',
              'feat_psych_tone', 'feat_psych_clout'):
        assert k in keys, k
    # Collapsed parents are gone
    for k in ('structural', 'acoustic_boundaries', 'sem_composite_psych'):
        assert k not in keys, k


def test_load_v1_flat_schema_migrates_to_v4(tmp_path, monkeypatch):
    """An old (batch-1) flat-shape v1 file is migrated through v2 → v3 → v4 → v5 → v6 → v7."""
    monkeypatch.setenv('SENSI_PREFS_PATH', str(tmp_path / 'p.json'))
    old = {
        'analyzer_name': 'Alice',
        'last_transcripts_folder': '/x/y',
        'llm_model': 'gpt-4o-mini',
        'dictionary_backend': 'liwc',
        'semantic_categories': ['social', 'cognition'],
        'feature_toggles': {'spacy_parse': True, 'zipf_freq': False},
    }
    (tmp_path / 'p.json').write_text(json.dumps(old))

    prefs = load_preferences()
    assert prefs['schema_version'] == SCHEMA_VERSION
    assert prefs['shared']['analyzer_name'] == 'Alice'
    assert prefs['shared']['last_transcripts_folder'] == '/x/y'
    # LIWC is disabled (EULA): the v11 finalize coerces a stored "liwc"
    # backend to "empath".
    assert prefs['semantic']['backend'] == 'empath'
    assert prefs['predictability']['llm_model'] == 'gpt-4o-mini'
    assert prefs['features']['freq_subtlex'] is False   # zipf_freq was False
    # spacy_parse was True → all per-column structural toggles True
    assert prefs['features']['feat_struct_doc_id'] is True
    assert prefs['features']['feat_struct_token_id'] is True


def test_migration_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setenv('SENSI_PREFS_PATH', str(tmp_path / 'p.json'))
    # Save a fresh new-schema file
    save_preferences(DEFAULT_PREFS)
    first = load_preferences()
    second = load_preferences()
    assert first == second


def test_unknown_top_level_keys_preserved(tmp_path, monkeypatch):
    """Forward compat — unknown top-level keys are preserved on round-trip."""
    monkeypatch.setenv('SENSI_PREFS_PATH', str(tmp_path / 'p.json'))
    target = tmp_path / 'p.json'
    target.write_text(json.dumps({
        'schema_version': SCHEMA_VERSION,
        'shared': {'analyzer_name': 'Bob'},
        'future_section': {'x': 1},
    }))
    loaded = load_preferences()
    assert loaded.get('future_section') == {'x': 1}


def test_save_preferences_writes_schema_version(tmp_path, monkeypatch):
    monkeypatch.setenv('SENSI_PREFS_PATH', str(tmp_path / 'p.json'))
    save_preferences({'shared': {'analyzer_name': 'C'}})
    data = json.loads((tmp_path / 'p.json').read_text())
    assert data.get('schema_version') == SCHEMA_VERSION


# ---------------------------------------------------------------------------
# v2 → v3 migration tests
# ---------------------------------------------------------------------------

@pytest.fixture
def v2_prefs_file(tmp_path, monkeypatch):
    """A realistic v2 prefs file on disk, with SENSI_PREFS_PATH pointing at it."""
    p = tmp_path / "preferences.json"
    p.write_text(json.dumps({
        "schema_version": 2,
        "shared": {
            "analyzer_name": "Amilcar",
            "last_transcripts_folder": "/tmp/t",
            "last_audio_folder": "/tmp/a",
            "last_output_folder": "/tmp/o",
            "dictionary_backend": "liwc",
        },
        "word_level": {
            "feature_toggles": {
                "spacy_parse": True,
                "zipf_freq": True,
                "edu_freq": False,
                "sem_domain": True,
                "gpt2_surprisal": False,
                "llm_cloze": True,
                "alignment": True,
            },
            "semantic_categories": ["social", "cognition"],
            "llm_model": "gpt-4o-mini",
        },
        "document_level": {
            "feature_toggles": {
                "counts": True,
                "mean_zipf_freq": True,
                "mean_edu_zipf": False,
                "mean_surprisal": True,
                "mean_llm_cloze": True,
                "ttr": True,
                "mtld": False,
                "flesch_reading_ease": True,
                "flesch_kincaid_grade": True,
                "category_proportions": True,
                "liwc_authentic": True,
                "liwc_analytic": False,
                "liwc_tone": True,
                "liwc_clout": True,
            },
            "semantic_categories": ["social"],
            "llm_model": "gpt-4o-mini",
        },
    }), encoding="utf-8")
    monkeypatch.setenv("SENSI_PREFS_PATH", str(p))
    return p


def test_v2_file_loads_as_v4(v2_prefs_file):
    """v2 file chains through v3 → v4 → v5 → v6 → v7 → v8 → v9 on load."""
    from utils.preferences import load_preferences, SCHEMA_VERSION
    prefs = load_preferences()
    assert prefs["schema_version"] == SCHEMA_VERSION


def test_v2_migration_carries_shared_fields(v2_prefs_file):
    from utils.preferences import load_preferences
    prefs = load_preferences()
    assert prefs["shared"]["analyzer_name"] == "Amilcar"
    assert prefs["shared"]["last_transcripts_folder"] == "/tmp/t"


def test_v2_migration_splits_spacy_parse_through_to_v4(v2_prefs_file):
    """v2.spacy_parse=True → v3.structural=True → v4 8 feat_struct_* all True."""
    from utils.preferences import load_preferences
    prefs = load_preferences()
    f = prefs["features"]
    for k in ("feat_struct_doc_id", "feat_struct_paragraph_id",
              "feat_struct_sentence_id", "feat_struct_sentence_text",
              "feat_struct_char_start", "feat_struct_char_end",
              "feat_struct_term_id", "feat_struct_token_id"):
        assert f[k] is True, k
    assert f["lexling_tokens"] is True
    assert f["lexling_pos"] is True
    assert f["lexling_morph"] is True
    assert f["lexling_dep"] is True


def test_v2_migration_splits_edu_freq_into_five_bands(v2_prefs_file):
    from utils.preferences import load_preferences
    prefs = load_preferences()
    f = prefs["features"]
    for band in ("freq_ewfg_k2", "freq_ewfg_3to5", "freq_ewfg_6to8",
                 "freq_ewfg_9to12", "freq_ewfg_overall"):
        assert f[band] is False, band


def test_v2_migration_splits_llm_cloze_into_three(v2_prefs_file):
    from utils.preferences import load_preferences
    prefs = load_preferences()
    f = prefs["features"]
    assert f["pred_surprisal_llm"] is True
    assert f["pred_cloze"] is True
    assert f["pred_topmatch"] is True


def test_v2_migration_maps_doc_toggles_through_to_v4(v2_prefs_file):
    """v2 counts=True → v3 doc_counts=True → v4 4 out_count_* all True."""
    from utils.preferences import load_preferences
    prefs = load_preferences()
    o = prefs["output"]
    for k in ("out_count_tokens", "out_count_content_tokens",
              "out_count_sentences", "out_count_paragraphs"):
        assert o[k] is True, k
    for k in ("doc_agg_zipf_freq", "doc_agg_edu_zipf",
              "doc_agg_surprisal", "doc_agg_llm"):
        assert o[k] is True, k
    assert o["doc_lex_diversity"] is True
    assert o["doc_readability"] is True
    assert o["doc_category_proportions"] is True


def test_v2_migration_maps_liwc_to_psych_through_to_v4(v2_prefs_file):
    """v2 liwc_* (any True) → v3 sem_composite_psych=True → v4 4 feat_psych_* all True."""
    from utils.preferences import load_preferences
    prefs = load_preferences()
    f = prefs["features"]
    for k in ("feat_psych_authentic", "feat_psych_analytic",
              "feat_psych_tone", "feat_psych_clout"):
        assert f[k] is True, k


def test_v2_migration_categories_unified_in_v7(v2_prefs_file):
    """v2 dictionary_backend=liwc with semantic_categories=['social','cognition']
    chains to v3 categories_liwc=[...] → v4 word/doc split → v7 unified
    categories_liwc=['social','cognition'] (lossless union, same in both)."""
    from utils.preferences import load_preferences
    prefs = load_preferences()
    s = prefs["semantic"]
    assert set(s["categories_liwc"]) == {"social", "cognition"}
    assert "word_categories_liwc" not in s
    assert "doc_categories_liwc" not in s


def test_v2_migration_carries_llm_model(v2_prefs_file):
    from utils.preferences import load_preferences
    prefs = load_preferences()
    assert prefs["predictability"]["llm_model"] == "gpt-4o-mini"


def test_v2_migration_coerces_disabled_liwc_backend_to_empath(v2_prefs_file):
    # v2 stored dictionary_backend="liwc"; LIWC is now disabled under its
    # EULA, so the v11 finalize coerces it to "empath".
    from utils.preferences import load_preferences
    prefs = load_preferences()
    assert prefs["semantic"]["backend"] == "empath"


def test_v2_migration_rewrites_file_as_v4(v2_prefs_file):
    from utils.preferences import load_preferences, SCHEMA_VERSION
    load_preferences()
    on_disk = json.loads(Path(v2_prefs_file).read_text(encoding="utf-8"))
    assert on_disk["schema_version"] == SCHEMA_VERSION
    assert "features" in on_disk
    assert "word_level" not in on_disk
    # v4-specific keys present after migration
    assert "feat_struct_doc_id" in on_disk["features"]
    # v7: unified categories key (split keys removed)
    assert "categories_liwc" in on_disk["semantic"]
    assert "word_categories_liwc" not in on_disk["semantic"]
    # v5-specific keys also present
    assert "feat_prosody_pitch_mean" in on_disk["features"]
    assert "prosody" in on_disk
    # v6-specific keys also present
    assert "feat_syllable_onset" in on_disk["features"]
    assert "syllables_csv" in on_disk["output"]
    # v7-specific master toggle keys present
    assert "master_word_prosody" in on_disk["features"]
    assert "master_doc_prosody" in on_disk["output"]


# ---------------------------------------------------------------------------
# v3 → v4 migration
# ---------------------------------------------------------------------------

@pytest.fixture
def v3_prefs_file(tmp_path, monkeypatch):
    """A realistic v3 file with non-default values for v3→v4 migration tests."""
    p = tmp_path / "preferences.json"
    p.write_text(json.dumps({
        "schema_version": 3,
        "shared": {
            "analyzer_name": "Amilcar",
            "last_transcripts_folder": "/tmp/t",
            "last_audio_folder": "/tmp/a",
            "last_output_folder": "/tmp/o",
            "theme": "system",
        },
        "features": {
            "structural": False,
            "lexling_tokens": True,
            "lexling_pos": True,
            "lexling_morph": False,
            "lexling_dep": True,
            "freq_subtlex": True,
            "freq_ewfg_k2": True,
            "freq_ewfg_3to5": False,
            "freq_ewfg_6to8": False,
            "freq_ewfg_9to12": False,
            "freq_ewfg_overall": True,
            "pred_surprisal_gpt2": True,
            "pred_surprisal_llm": False,
            "pred_cloze": False,
            "pred_topmatch": False,
            "sem_indicators": True,
            "sem_composite_psych": True,
            "acoustic_boundaries": True,
        },
        "predictability": {"llm_model": "gpt-4o-mini"},
        "semantic": {
            "backend": "liwc",
            "categories_liwc": ["affect", "cogproc"],
            "categories_empath": ["social", "money"],
        },
        "output": {
            "word_level_csv": True,
            "document_summary": True,
            "doc_counts": False,
            "doc_aggregates": True,
            "doc_lex_diversity": True,
            "doc_readability": True,
            "doc_category_proportions": True,
        },
    }), encoding="utf-8")
    monkeypatch.setenv("SENSI_PREFS_PATH", str(p))
    return p


def test_v3_file_loads_as_v4(v3_prefs_file):
    prefs = load_preferences()
    assert prefs["schema_version"] == SCHEMA_VERSION


def test_v3_migration_expands_structural_per_column(v3_prefs_file):
    f = load_preferences()["features"]
    assert "structural" not in f
    for k in ("feat_struct_doc_id", "feat_struct_paragraph_id",
              "feat_struct_sentence_id", "feat_struct_sentence_text",
              "feat_struct_char_start", "feat_struct_char_end",
              "feat_struct_term_id", "feat_struct_token_id"):
        assert f[k] is False, k


def test_v3_migration_expands_acoustic_per_column(v3_prefs_file):
    f = load_preferences()["features"]
    for k in ("feat_acoustic_onset", "feat_acoustic_end",
              "feat_acoustic_word_duration", "feat_acoustic_word_gap"):
        assert f[k] is True, k


def test_v3_migration_expands_psych_per_column(v3_prefs_file):
    f = load_preferences()["features"]
    for k in ("feat_psych_authentic", "feat_psych_analytic",
              "feat_psych_tone", "feat_psych_clout"):
        assert f[k] is True, k


def test_v3_migration_expands_doc_counts_per_column(v3_prefs_file):
    o = load_preferences()["output"]
    assert "doc_counts" not in o
    for k in ("out_count_tokens", "out_count_content_tokens",
              "out_count_sentences", "out_count_paragraphs"):
        assert o[k] is False, k


def test_v3_migration_categories_unified_in_v7(v3_prefs_file):
    """v3 categories_liwc/empath → v4 word/doc split → v7 unified (lossless union)."""
    s = load_preferences()["semantic"]
    assert set(s["categories_liwc"]) == {"affect", "cogproc"}
    assert set(s["categories_empath"]) == {"social", "money"}
    assert "word_categories_liwc" not in s
    assert "doc_categories_liwc" not in s
    assert "word_categories_empath" not in s
    assert "doc_categories_empath" not in s


def test_v3_migration_preserves_unchanged_keys(v3_prefs_file):
    prefs = load_preferences()
    assert prefs["features"]["lexling_morph"] is False
    assert prefs["features"]["pred_surprisal_gpt2"] is True
    assert prefs["predictability"]["llm_model"] == "gpt-4o-mini"
    # LIWC disabled (EULA): stored "liwc" backend is coerced to "empath".
    assert prefs["semantic"]["backend"] == "empath"


def test_v3_migration_rewrites_file_as_v4(v3_prefs_file):
    load_preferences()
    on_disk = json.loads(Path(v3_prefs_file).read_text(encoding="utf-8"))
    assert on_disk["schema_version"] == SCHEMA_VERSION


# ---------------------------------------------------------------------------
# v4 → v5 migration
# ---------------------------------------------------------------------------

def test_v4_to_v5_adds_prosody_defaults(tmp_path, monkeypatch):
    from utils import preferences as prefs_mod
    # Write a v4-shape file
    v4 = {
        "schema_version": 4,
        "shared": {"analyzer_name": "Test"},
        "features": {"feat_struct_doc_id": True, "feat_psych_authentic": False},
        "predictability": {"llm_model": ""},
        "semantic": {"backend": "auto"},
        "output": {"word_level_csv": True},
    }
    path = tmp_path / "preferences.json"
    path.write_text(json.dumps(v4))
    monkeypatch.setenv("SENSI_PREFS_PATH", str(path))

    loaded = prefs_mod.load_preferences()

    assert loaded["schema_version"] == prefs_mod.SCHEMA_VERSION
    # Pre-existing keys preserved
    assert loaded["features"]["feat_struct_doc_id"] is True
    assert loaded["features"]["feat_psych_authentic"] is False
    # All 10 new word-level prosody keys default True
    for k in (
        "feat_prosody_pitch_mean", "feat_prosody_pitch_min",
        "feat_prosody_pitch_max", "feat_prosody_pitch_std",
        "feat_prosody_pitch_delta",
        "feat_prosody_volume_mean", "feat_prosody_volume_min",
        "feat_prosody_volume_max", "feat_prosody_volume_std",
        "feat_prosody_volume_delta",
    ):
        assert loaded["features"][k] is True, f"{k} missing or not True"
    # All 16 new doc-level prosody keys default True
    for k in (
        "feat_doc_prosody_pitch_mean", "feat_doc_prosody_pitch_min",
        "feat_doc_prosody_pitch_max", "feat_doc_prosody_pitch_std",
        "feat_doc_prosody_volume_mean", "feat_doc_prosody_volume_min",
        "feat_doc_prosody_volume_max", "feat_doc_prosody_volume_std",
        "feat_doc_prosody_pitch_delta_mean", "feat_doc_prosody_volume_delta_mean",
        "feat_doc_prosody_hnr", "feat_doc_prosody_articulation_rate",
        "feat_doc_prosody_pause_count", "feat_doc_prosody_pause_mean_duration",
        "feat_doc_prosody_pause_total_duration", "feat_doc_prosody_pause_pct_silence",
    ):
        assert loaded["features"][k] is True, f"{k} missing or not True"
    # New top-level prosody config section
    assert loaded["prosody"]["pitch_units"] == "semitones"
    assert loaded["prosody"]["pitch_floor_hz"] == 75
    assert loaded["prosody"]["pitch_ceiling_hz"] == 500
    assert loaded["prosody"]["pause_threshold_s"] == 0.25


def test_v5_round_trip_preserves_user_edits(tmp_path, monkeypatch):
    from utils import preferences as prefs_mod
    path = tmp_path / "preferences.json"
    monkeypatch.setenv("SENSI_PREFS_PATH", str(path))
    # First load creates a v6 file
    prefs_mod.load_preferences()
    # User flips one toggle and changes pitch units
    prefs_mod.save_preferences({
        "features": {"feat_prosody_pitch_mean": False},
        "prosody": {"pitch_units": "hz", "pitch_floor_hz": 60},
    })
    again = prefs_mod.load_preferences()
    assert again["schema_version"] == prefs_mod.SCHEMA_VERSION
    assert again["features"]["feat_prosody_pitch_mean"] is False
    assert again["prosody"]["pitch_units"] == "hz"
    assert again["prosody"]["pitch_floor_hz"] == 60
    # Untouched keys remain default
    assert again["features"]["feat_prosody_pitch_min"] is True


# ---------------------------------------------------------------------------
# v5 → v6 migration
# ---------------------------------------------------------------------------

def test_v5_to_v6_adds_subword_defaults(tmp_path, monkeypatch):
    import json
    from utils.preferences import load_preferences
    path = tmp_path / "preferences.json"
    path.write_text(json.dumps({
        "schema_version": 5,
        "features": {"feat_prosody_pitch_mean": False},  # existing v5 key, non-default
        "output": {"word_level_csv": True},
    }))
    monkeypatch.setenv("SENSI_PREFS_PATH", str(path))
    prefs = load_preferences()
    assert prefs["schema_version"] == SCHEMA_VERSION
    # New v6 defaults (carried through to v9)
    assert prefs["features"]["feat_syllable_onset"] is True
    assert prefs["features"]["feat_syllable_arpabet"] is False  # the lone default-off
    assert prefs["features"]["feat_phoneme_label"] is True
    assert prefs["output"]["syllables_csv"] is True
    assert prefs["output"]["phonemes_csv"] is True
    # Pre-existing v5 setting preserved
    assert prefs["features"]["feat_prosody_pitch_mean"] is False


def test_v4_to_v6_chain_carries_keys_through(tmp_path, monkeypatch):
    import json
    from utils.preferences import load_preferences
    path = tmp_path / "preferences.json"
    path.write_text(json.dumps({
        "schema_version": 4,
        "features": {"feat_psych_clout": False},
    }))
    monkeypatch.setenv("SENSI_PREFS_PATH", str(path))
    prefs = load_preferences()
    assert prefs["schema_version"] == SCHEMA_VERSION
    assert prefs["features"]["feat_psych_clout"] is False
    assert "feat_syllable_onset" in prefs["features"]
    assert "syllables_csv" in prefs["output"]


# ---------------------------------------------------------------------------
# v6 → v7 migration
# ---------------------------------------------------------------------------

def test_v6_to_v7_unions_categories_lossless(tmp_path, monkeypatch):
    from utils.preferences import (
        load_preferences, get_preferences_path,
        SCHEMA_VERSION_V7, _migrate_v6_into_v7, DEFAULT_PREFS_V7,
    )
    import copy
    v6 = {
        "schema_version": 6,
        "semantic": {
            "backend": "empath",
            "word_categories_empath": ["positive_emotion", "work"],
            "doc_categories_empath":  ["family", "money", "work"],  # overlap
            "word_categories_liwc":   [],
            "doc_categories_liwc":    [],
        },
        "features": {}, "output": {},
    }
    target = copy.deepcopy(DEFAULT_PREFS_V7)
    _migrate_v6_into_v7(target, v6)
    assert target["schema_version"] == SCHEMA_VERSION_V7
    cats = target["semantic"]["categories_empath"]
    assert set(cats) == {"positive_emotion", "work", "family", "money"}
    # Order: word entries first, then doc additions (lossless union)
    assert cats.index("positive_emotion") < cats.index("family")
    assert "word_categories_empath" not in target["semantic"]
    assert "doc_categories_empath"  not in target["semantic"]


def test_v6_to_v7_sets_master_defaults_true():
    import copy
    from utils.preferences import _migrate_v6_into_v7, DEFAULT_PREFS_V7
    target = copy.deepcopy(DEFAULT_PREFS_V7)
    _migrate_v6_into_v7(target, {"schema_version": 6, "features": {},
                                  "output": {}, "semantic": {}})
    for k in ("master_word_prosody", "master_psych_composite",
             "master_syllable_intervals", "master_syllable_labels",
             "master_syllable_prosody", "master_phoneme_intervals"):
        assert target["features"][k] is True
    for k in ("master_doc_prosody", "master_doc_articulation_rate"):
        assert target["output"][k] is True


def test_v6_to_v7_preserves_user_feature_toggles():
    """User v6 settings are preserved through the migration."""
    import copy
    from utils.preferences import _migrate_v6_into_v7, DEFAULT_PREFS_V7
    target = copy.deepcopy(DEFAULT_PREFS_V7)
    _migrate_v6_into_v7(target, {
        "schema_version": 6,
        "features": {"freq_subtlex": False, "feat_prosody_pitch_mean": False},
        "output":   {"word_level_csv": False},
        "semantic": {},
    })
    assert target["features"]["freq_subtlex"] is False
    assert target["features"]["feat_prosody_pitch_mean"] is False
    assert target["output"]["word_level_csv"] is False
    # Master defaults still True even though user touched other keys
    assert target["features"]["master_word_prosody"] is True


def test_v7_default_prefs_contain_all_master_keys():
    from utils.preferences import DEFAULT_PREFS_V7
    for k in ("master_word_prosody", "master_psych_composite",
             "master_syllable_intervals", "master_syllable_labels",
             "master_syllable_prosody", "master_phoneme_intervals"):
        assert k in DEFAULT_PREFS_V7["features"]
    for k in ("master_doc_prosody", "master_doc_articulation_rate"):
        assert k in DEFAULT_PREFS_V7["output"]


def test_v6_to_v7_union_includes_preexisting_unified_value():
    """If a v6 file somehow already contains the unified categories_<backend>
    key (e.g. from a forward-compat sniff or a hand-edit), the migration
    must include those values in the union rather than overwriting them."""
    import copy
    from utils.preferences import _migrate_v6_into_v7, DEFAULT_PREFS_V7
    target = copy.deepcopy(DEFAULT_PREFS_V7)
    _migrate_v6_into_v7(target, {
        "schema_version": 6,
        "features": {}, "output": {},
        "semantic": {
            "categories_empath":      ["preexisting_cat"],
            "word_categories_empath": ["word_cat"],
            "doc_categories_empath":  ["doc_cat"],
        },
    })
    cats = target["semantic"]["categories_empath"]
    assert set(cats) >= {"preexisting_cat", "word_cat", "doc_cat"}
    # Preserved order: preexisting first, then word, then doc additions
    assert cats.index("preexisting_cat") < cats.index("word_cat") < cats.index("doc_cat")
    assert "word_categories_empath" not in target["semantic"]
    assert "doc_categories_empath"  not in target["semantic"]


# ---------------------------------------------------------------------------
# v7 → v8 migration
# ---------------------------------------------------------------------------

def test_v7_into_v8_adds_pcenter_defaults():
    """v7 prefs migrate to v8 with pcenter_k=0.3 and new toggles True."""
    from copy import deepcopy
    from utils.preferences import (
        DEFAULT_PREFS_V7,
        DEFAULT_PREFS_V8,
        _migrate_v7_into_v8,
    )
    target = deepcopy(DEFAULT_PREFS_V8)
    _migrate_v7_into_v8(target, deepcopy(DEFAULT_PREFS_V7))
    assert target['prosody']['pcenter_k'] == 0.3
    assert target['features']['feat_syllable_pcenter'] is True
    assert target['features']['master_syllable_pcenter'] is True


def test_v7_into_v8_preserves_user_pcenter_k():
    """If user set pcenter_k in a v7 file (forward-compat), migration preserves it."""
    from copy import deepcopy
    from utils.preferences import (
        DEFAULT_PREFS_V7,
        DEFAULT_PREFS_V8,
        _migrate_v7_into_v8,
    )
    v7 = deepcopy(DEFAULT_PREFS_V7)
    v7.setdefault('prosody', {})['pcenter_k'] = 0.5
    target = deepcopy(DEFAULT_PREFS_V8)
    _migrate_v7_into_v8(target, v7)
    assert target['prosody']['pcenter_k'] == 0.5


def test_load_v7_prefs_file_migrates_to_current(tmp_path, monkeypatch):
    """load_preferences() on a v7 file auto-migrates to current version."""
    import json
    from utils.preferences import DEFAULT_PREFS_V7, load_preferences, SCHEMA_VERSION
    prefs_path = tmp_path / 'p.json'
    prefs_path.write_text(json.dumps(DEFAULT_PREFS_V7))
    monkeypatch.setenv('SENSI_PREFS_PATH', str(prefs_path))
    loaded = load_preferences()
    assert loaded['schema_version'] == SCHEMA_VERSION
    assert loaded['prosody']['pcenter_k'] == 0.3
    on_disk = json.loads(prefs_path.read_text())
    assert on_disk['schema_version'] == SCHEMA_VERSION


# ---------------------------------------------------------------------------
# v8 → v9 migration
# ---------------------------------------------------------------------------

def test_v8_into_v9_adds_sentence_defaults():
    """v8 prefs migrate to v9 with all sentence-level feature toggles True."""
    from copy import deepcopy
    from utils.preferences import (
        DEFAULT_PREFS_V8,
        DEFAULT_PREFS_V9,
        _migrate_v8_into_v9,
    )
    target = deepcopy(DEFAULT_PREFS_V9)
    _migrate_v8_into_v9(target, deepcopy(DEFAULT_PREFS_V8))
    assert target['features']['feat_sentence_onset'] is True
    assert target['features']['feat_sentence_pitch_mean'] is True
    assert target['features']['master_sentence_timing'] is True
    assert target['features']['master_sentence_prosody'] is True
    assert target['output']['sentences_csv'] is True


def test_v8_into_v9_preserves_existing_features():
    """v8 user feature values survive migration to v9."""
    from copy import deepcopy
    from utils.preferences import DEFAULT_PREFS_V8, DEFAULT_PREFS_V9, _migrate_v8_into_v9

    v8 = deepcopy(DEFAULT_PREFS_V8)
    v8['features']['feat_syllable_pcenter'] = False  # a v8 key the user turned off
    target = deepcopy(DEFAULT_PREFS_V9)
    _migrate_v8_into_v9(target, v8)
    assert target['features']['feat_syllable_pcenter'] is False


def test_load_v8_prefs_file_migrates_to_current(tmp_path, monkeypatch):
    """load_preferences() on a v8 file auto-migrates to current version."""
    import json
    from utils.preferences import DEFAULT_PREFS_V8, load_preferences, SCHEMA_VERSION

    prefs_path = tmp_path / 'p.json'
    prefs_path.write_text(json.dumps(DEFAULT_PREFS_V8))
    monkeypatch.setenv('SENSI_PREFS_PATH', str(prefs_path))
    loaded = load_preferences()
    assert loaded['schema_version'] == SCHEMA_VERSION
    assert loaded['features']['feat_sentence_onset'] is True
    assert loaded['output']['sentences_csv'] is True
    on_disk = json.loads(prefs_path.read_text())
    assert on_disk['schema_version'] == SCHEMA_VERSION


# ---------------------------------------------------------------------------
# v9 → v10 migration
# ---------------------------------------------------------------------------

def test_v9_into_v10_adds_embedding_defaults():
    """v9 prefs migrate to v10 with all embedding output and feature toggles True."""
    from copy import deepcopy
    from utils.preferences import (
        DEFAULT_PREFS_V9,
        DEFAULT_PREFS_V10,
        _migrate_v9_into_v10,
    )
    target = deepcopy(DEFAULT_PREFS_V10)
    _migrate_v9_into_v10(target, deepcopy(DEFAULT_PREFS_V9))
    assert target['features']['master_word2vec'] is True
    assert target['features']['master_bert_word'] is True
    assert target['features']['master_bert_sentence'] is True
    assert target['output']['word2vec_mat'] is True
    assert target['output']['bert_word_mat'] is True
    assert target['output']['bert_sentence_mat'] is True


def test_v9_into_v10_preserves_existing_features():
    """v9 user feature values survive migration to v10."""
    from copy import deepcopy
    from utils.preferences import DEFAULT_PREFS_V9, DEFAULT_PREFS_V10, _migrate_v9_into_v10

    v9 = deepcopy(DEFAULT_PREFS_V9)
    v9['features']['feat_sentence_pitch_mean'] = False
    target = deepcopy(DEFAULT_PREFS_V10)
    _migrate_v9_into_v10(target, v9)
    assert target['features']['feat_sentence_pitch_mean'] is False


def test_load_v9_prefs_file_migrates_to_current(tmp_path, monkeypatch):
    """load_preferences() on a v9 file auto-migrates through to the current
    schema (adds the v10 embedding keys along the way)."""
    import json
    from utils.preferences import DEFAULT_PREFS_V9, load_preferences, SCHEMA_VERSION

    prefs_path = tmp_path / 'p.json'
    prefs_path.write_text(json.dumps(DEFAULT_PREFS_V9))
    monkeypatch.setenv('SENSI_PREFS_PATH', str(prefs_path))
    loaded = load_preferences()
    assert loaded['schema_version'] == SCHEMA_VERSION
    assert loaded['output']['word2vec_mat'] is True
    assert loaded['features']['master_word2vec'] is True
    on_disk = json.loads(prefs_path.read_text())
    assert on_disk['schema_version'] == SCHEMA_VERSION


def test_v10_to_v11_coerces_liwc_backend_and_bumps_version(tmp_path, monkeypatch):
    """A v10 file with semantic.backend='liwc' migrates to v11 with the
    backend coerced to 'empath' (LIWC disabled under its EULA), and the
    upgrade is persisted to disk."""
    import json
    from copy import deepcopy
    from utils.preferences import (
        DEFAULT_PREFS_V10, load_preferences,
        SCHEMA_VERSION, SCHEMA_VERSION_V11,
    )

    v10 = deepcopy(DEFAULT_PREFS_V10)
    v10['semantic']['backend'] = 'liwc'
    prefs_path = tmp_path / 'p.json'
    prefs_path.write_text(json.dumps(v10))
    monkeypatch.setenv('SENSI_PREFS_PATH', str(prefs_path))

    loaded = load_preferences()
    # The liwc→empath coercion happens on the way to the current schema
    # (the version itself has since moved past v11).
    assert SCHEMA_VERSION >= SCHEMA_VERSION_V11
    assert loaded['schema_version'] == SCHEMA_VERSION
    assert loaded['semantic']['backend'] == 'empath'
    on_disk = json.loads(prefs_path.read_text())
    assert on_disk['schema_version'] == SCHEMA_VERSION
    assert on_disk['semantic']['backend'] == 'empath'


def test_v11_default_backend_is_empath():
    from utils.preferences import DEFAULT_PREFS
    assert DEFAULT_PREFS['semantic']['backend'] == 'empath'


# ---------------------------------------------------------------------------
# v12 → v13 migration
# ---------------------------------------------------------------------------

def test_v12_prefs_migrate_to_v13_additively(tmp_path, monkeypatch):
    import json
    from utils import preferences
    p = tmp_path / "preferences.json"
    v12 = {"schema_version": 12, "features": {}, "output": {}}
    p.write_text(json.dumps(v12), encoding="utf-8")
    monkeypatch.setenv("SENSI_PREFS_PATH", str(p))
    prefs = preferences.load_preferences()
    assert prefs["schema_version"] == 13
    f = prefs["features"]
    assert f["master_acoustic_ts"] is False
    assert f["feat_acoustic_ts_envelope"] is True
    assert f["feat_acoustic_ts_mfcc"] is False
    assert prefs["output"]["acoustic_ts_mat"] is True
    a = prefs["acoustic_ts"]
    assert a["analysis_rate"] == 128 and a["env_method"] == "rms"
    assert a["mel_preset"] == "speech"
