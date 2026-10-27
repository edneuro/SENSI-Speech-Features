import json
import os

import pytest

from sensi.utils.preferences import (
    DEFAULT_PREFS,
    load_preferences,
    save_preferences,
    SCHEMA_VERSION,
    SCHEMA_VERSION_V20,
)


def _yaml_on_disk(json_path):
    """The store the migrated prefs actually landed in (YAML pair next to the
    legacy json path)."""
    from pathlib import Path
    from sensi.utils.config_yaml import read_yaml, join_yaml_pair, FEATURES_YAML, RUN_YAML
    d = Path(json_path).parent
    return join_yaml_pair(read_yaml(d / FEATURES_YAML), read_yaml(d / RUN_YAML))


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
              'sem_indicators'):
        assert k in keys, k
    # Per-column expansions (batch 3)
    for k in ('feat_struct_doc_id', 'feat_struct_paragraph_id',
              'feat_struct_sentence_id', 'feat_struct_sentence_text',
              'feat_struct_char_start', 'feat_struct_char_end',
              'feat_struct_token_id', 'feat_struct_token_in_sentence_id',
              'feat_acoustic_onset', 'feat_acoustic_end',
              'feat_acoustic_word_duration', 'feat_acoustic_word_gap',
              'feat_psych_authentic', 'feat_psych_analytic',
              'feat_psych_tone', 'feat_psych_clout'):
        assert k in keys, k
    # Collapsed parents and the pre-v14 struct-id key names are gone
    for k in ('structural', 'acoustic_boundaries', 'sem_composite_psych',
              'feat_struct_term_id'):
        assert k not in keys, k
    # v18: model/metric selection moved out of these four feature toggles
    # into predictability.models/metrics
    for k in ('pred_surprisal_gpt2', 'pred_surprisal_llm',
              'pred_cloze', 'pred_topmatch'):
        assert k not in keys, k


def test_load_v1_flat_schema_migrates_to_v4(tmp_path, monkeypatch):
    """An old (batch-1) flat-shape v1 file is migrated through v2 → v3 → v4 → v5 → v6 → v7."""
    monkeypatch.setenv('SENSI_PREFS_PATH', str(tmp_path / 'p.json'))
    old = {
        'analyzer_name': 'Alice',
        'last_transcripts_folder': '/x/y',
        'llm_model': 'meta-llama/Meta-Llama-3-8B',
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
    # v18: the four pred_* toggles were all on (defaults), so the stored
    # llm_model's slug is mapped into predictability.models alongside "gpt2";
    # the old llm_model field itself is dropped.
    assert set(prefs['predictability']['models']) == {'gpt2', 'llama3_8b'}
    assert prefs['predictability']['metrics']['surprisal'] is True
    assert prefs['predictability']['metrics']['cloze'] is True
    assert prefs['predictability']['metrics']['topmatch'] is True
    assert 'llm_model' not in prefs['predictability']
    assert 'pred_surprisal_gpt2' not in prefs['features']
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
    data = _yaml_on_disk(tmp_path / 'p.json')
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
            "llm_model": "meta-llama/Meta-Llama-3-8B",
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
            "llm_model": "meta-llama/Meta-Llama-3-8B",
        },
    }), encoding="utf-8")
    monkeypatch.setenv("SENSI_PREFS_PATH", str(p))
    return p


def test_v2_file_loads_as_v4(v2_prefs_file):
    """v2 file chains through v3 → v4 → v5 → v6 → v7 → v8 → v9 on load."""
    from sensi.utils.preferences import load_preferences, SCHEMA_VERSION
    prefs = load_preferences()
    assert prefs["schema_version"] == SCHEMA_VERSION


def test_v2_migration_carries_shared_fields(v2_prefs_file):
    from sensi.utils.preferences import load_preferences
    prefs = load_preferences()
    assert prefs["shared"]["analyzer_name"] == "Amilcar"
    assert prefs["shared"]["last_transcripts_folder"] == "/tmp/t"


def test_v2_migration_splits_spacy_parse_through_to_v4(v2_prefs_file):
    """v2.spacy_parse=True → v3.structural=True → v4 8 feat_struct_* all True."""
    from sensi.utils.preferences import load_preferences
    prefs = load_preferences()
    f = prefs["features"]
    for k in ("feat_struct_doc_id", "feat_struct_paragraph_id",
              "feat_struct_sentence_id", "feat_struct_sentence_text",
              "feat_struct_char_start", "feat_struct_char_end",
              "feat_struct_token_id", "feat_struct_token_in_sentence_id"):
        assert f[k] is True, k
    assert f["lexling_tokens"] is True
    assert f["lexling_pos"] is True
    assert f["lexling_morph"] is True
    assert f["lexling_dep"] is True


def test_v2_migration_splits_edu_freq_into_five_bands(v2_prefs_file):
    from sensi.utils.preferences import load_preferences
    prefs = load_preferences()
    f = prefs["features"]
    for band in ("freq_ewfg_k2", "freq_ewfg_3to5", "freq_ewfg_6to8",
                 "freq_ewfg_9to12", "freq_ewfg_overall"):
        assert f[band] is False, band


def test_v2_migration_llm_cloze_becomes_llama3_8b_model_and_metrics(v2_prefs_file):
    """v2 llm_cloze=True (gpt2_surprisal=False) migrates through the old
    pred_surprisal_llm/pred_cloze/pred_topmatch=True intermediate shape, then
    the v18 finalize maps that + the stored llm_model onto predictability.
    models/metrics (gpt2 stays off since gpt2_surprisal was False)."""
    from sensi.utils.preferences import load_preferences
    prefs = load_preferences()
    assert prefs["predictability"]["models"] == ["llama3_8b"]
    metrics = prefs["predictability"]["metrics"]
    assert metrics["surprisal"] is True
    assert metrics["cloze"] is True
    assert metrics["topmatch"] is True
    assert "pred_surprisal_llm" not in prefs["features"]


def test_v2_migration_maps_doc_toggles_through_to_v4(v2_prefs_file):
    """v2 counts=True → v3 doc_counts=True → v4 4 out_count_* all True."""
    from sensi.utils.preferences import load_preferences
    prefs = load_preferences()
    o = prefs["output"]
    for k in ("out_count_tokens", "out_count_content_tokens",
              "out_count_sentences", "out_count_paragraphs"):
        assert o[k] is True, k
    for k in ("doc_agg_zipf_freq", "doc_agg_edu_zipf",
              "doc_agg_predictability"):
        assert o[k] is True, k
    assert "doc_agg_surprisal" not in o
    assert "doc_agg_llm" not in o
    assert o["doc_lex_diversity"] is True
    assert o["doc_readability"] is True
    assert o["doc_category_proportions"] is True


def test_v2_migration_maps_liwc_to_psych_through_to_v4(v2_prefs_file):
    """v2 liwc_* (any True) → v3 sem_composite_psych=True → v4 4 feat_psych_* all True."""
    from sensi.utils.preferences import load_preferences
    prefs = load_preferences()
    f = prefs["features"]
    for k in ("feat_psych_authentic", "feat_psych_analytic",
              "feat_psych_tone", "feat_psych_clout"):
        assert f[k] is True, k


def test_v2_migration_categories_unified_in_v7(v2_prefs_file):
    """v2 dictionary_backend=liwc with semantic_categories=['social','cognition']
    chains to v3 categories_liwc=[...] → v4 word/doc split → v7 unified
    categories_liwc=['social','cognition'] (lossless union, same in both)."""
    from sensi.utils.preferences import load_preferences
    prefs = load_preferences()
    s = prefs["semantic"]
    assert set(s["categories_liwc"]) == {"social", "cognition"}
    assert "word_categories_liwc" not in s
    assert "doc_categories_liwc" not in s


def test_v2_migration_llm_model_maps_to_model_slug(v2_prefs_file):
    """The v2 file's stored llm_model no longer survives as
    predictability.llm_model (v18 removes that field) — since a pred_*
    toggle was on (llm_cloze=True), its slug lands in predictability.models."""
    from sensi.utils.preferences import load_preferences
    prefs = load_preferences()
    assert prefs["predictability"]["models"] == ["llama3_8b"]
    assert "llm_model" not in prefs["predictability"]


def test_v2_migration_coerces_disabled_liwc_backend_to_empath(v2_prefs_file):
    # v2 stored dictionary_backend="liwc"; LIWC is now disabled under its
    # EULA, so the v11 finalize coerces it to "empath".
    from sensi.utils.preferences import load_preferences
    prefs = load_preferences()
    assert prefs["semantic"]["backend"] == "empath"


def test_v2_migration_rewrites_file_as_v4(v2_prefs_file):
    from sensi.utils.preferences import load_preferences, SCHEMA_VERSION
    load_preferences()
    on_disk = _yaml_on_disk(v2_prefs_file)
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
    # v21: per-file output toggles became output.level_* switches
    assert "level_subword" in on_disk["output"]
    assert "syllables_csv" not in on_disk["output"]
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
        "predictability": {"llm_model": "meta-llama/Meta-Llama-3-8B"},
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
              "feat_struct_token_id", "feat_struct_token_in_sentence_id"):
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
    # v18: pred_surprisal_gpt2=True (only pred_* toggle on in this fixture)
    # maps to the "gpt2" model slug + metrics.surprisal; the other three
    # toggles were False so llm_model is dropped without producing a model.
    assert prefs["predictability"]["models"] == ["gpt2"]
    assert prefs["predictability"]["metrics"]["surprisal"] is True
    assert "llm_model" not in prefs["predictability"]
    assert "pred_surprisal_gpt2" not in prefs["features"]
    # LIWC disabled (EULA): stored "liwc" backend is coerced to "empath".
    assert prefs["semantic"]["backend"] == "empath"


def test_v3_migration_rewrites_file_as_v4(v3_prefs_file):
    load_preferences()
    on_disk = _yaml_on_disk(v3_prefs_file)
    assert on_disk["schema_version"] == SCHEMA_VERSION


# ---------------------------------------------------------------------------
# v4 → v5 migration
# ---------------------------------------------------------------------------

def test_v4_to_v5_adds_prosody_defaults(tmp_path, monkeypatch):
    from sensi.utils import preferences as prefs_mod
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
    from sensi.utils import preferences as prefs_mod
    path = tmp_path / "preferences.json"
    monkeypatch.setenv("SENSI_PREFS_PATH", str(path))
    # First load creates a v6 file
    prefs_mod.load_preferences()
    # User flips one toggle (to a non-default value) and changes pitch units
    prefs_mod.save_preferences({
        "features": {"feat_prosody_pitch_mean": True},
        "prosody": {"pitch_units": "hz", "pitch_floor_hz": 60},
    })
    again = prefs_mod.load_preferences()
    assert again["schema_version"] == prefs_mod.SCHEMA_VERSION
    assert again["features"]["feat_prosody_pitch_mean"] is True   # user edit preserved
    assert again["prosody"]["pitch_units"] == "hz"
    assert again["prosody"]["pitch_floor_hz"] == 60
    # Untouched key keeps its (opt-in) default, which is off.
    assert again["features"]["feat_prosody_pitch_min"] is False


# ---------------------------------------------------------------------------
# v5 → v6 migration
# ---------------------------------------------------------------------------

def test_v5_to_v6_adds_subword_defaults(tmp_path, monkeypatch):
    import json
    from sensi.utils.preferences import load_preferences
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
    # v20: feat_phoneme_label was replaced by four per-column toggles; the old
    # v6-era default (True) carries forward onto all four.
    assert "feat_phoneme_label" not in prefs["features"]
    for key in ("feat_phoneme_arpabet", "feat_phoneme_stress",
                "feat_phoneme_ipa", "feat_phoneme_pron_source"):
        assert prefs["features"][key] is True, key
    # v21: the v6 syllables/phonemes outputs (default on) map to the
    # sub-word level switch.
    assert prefs["output"]["level_subword"] is True
    # Pre-existing v5 setting preserved
    assert prefs["features"]["feat_prosody_pitch_mean"] is False


def test_v4_to_v6_chain_carries_keys_through(tmp_path, monkeypatch):
    import json
    from sensi.utils.preferences import load_preferences
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
    assert "level_subword" in prefs["output"]


# ---------------------------------------------------------------------------
# v6 → v7 migration
# ---------------------------------------------------------------------------

def test_v6_to_v7_unions_categories_lossless(tmp_path, monkeypatch):
    from sensi.utils.preferences import (
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
    from sensi.utils.preferences import _migrate_v6_into_v7, DEFAULT_PREFS_V7
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
    from sensi.utils.preferences import _migrate_v6_into_v7, DEFAULT_PREFS_V7
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
    from sensi.utils.preferences import DEFAULT_PREFS_V7
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
    from sensi.utils.preferences import _migrate_v6_into_v7, DEFAULT_PREFS_V7
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
    from sensi.utils.preferences import (
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
    from sensi.utils.preferences import (
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
    from sensi.utils.preferences import DEFAULT_PREFS_V7, load_preferences, SCHEMA_VERSION
    prefs_path = tmp_path / 'p.json'
    prefs_path.write_text(json.dumps(DEFAULT_PREFS_V7))
    monkeypatch.setenv('SENSI_PREFS_PATH', str(prefs_path))
    loaded = load_preferences()
    assert loaded['schema_version'] == SCHEMA_VERSION
    assert loaded['prosody']['pcenter_k'] == 0.3
    on_disk = _yaml_on_disk(prefs_path)
    assert on_disk['schema_version'] == SCHEMA_VERSION


# ---------------------------------------------------------------------------
# v8 → v9 migration
# ---------------------------------------------------------------------------

def test_v8_into_v9_adds_sentence_defaults():
    """v8 prefs migrate to v9 with all sentence-level feature toggles True."""
    from copy import deepcopy
    from sensi.utils.preferences import (
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
    from sensi.utils.preferences import DEFAULT_PREFS_V8, DEFAULT_PREFS_V9, _migrate_v8_into_v9

    v8 = deepcopy(DEFAULT_PREFS_V8)
    v8['features']['feat_syllable_pcenter'] = False  # a v8 key the user turned off
    target = deepcopy(DEFAULT_PREFS_V9)
    _migrate_v8_into_v9(target, v8)
    assert target['features']['feat_syllable_pcenter'] is False


def test_load_v8_prefs_file_migrates_to_current(tmp_path, monkeypatch):
    """load_preferences() on a v8 file auto-migrates to current version."""
    import json
    from sensi.utils.preferences import DEFAULT_PREFS_V8, load_preferences, SCHEMA_VERSION

    prefs_path = tmp_path / 'p.json'
    prefs_path.write_text(json.dumps(DEFAULT_PREFS_V8))
    monkeypatch.setenv('SENSI_PREFS_PATH', str(prefs_path))
    loaded = load_preferences()
    assert loaded['schema_version'] == SCHEMA_VERSION
    assert loaded['features']['feat_sentence_onset'] is True
    assert loaded['output']['level_sentence'] is True  # v21: was sentences_csv
    on_disk = _yaml_on_disk(prefs_path)
    assert on_disk['schema_version'] == SCHEMA_VERSION


# ---------------------------------------------------------------------------
# v9 → v10 migration
# ---------------------------------------------------------------------------

def test_v9_into_v10_adds_embedding_defaults():
    """v9 prefs migrate to v10 with all embedding output and feature toggles True."""
    from copy import deepcopy
    from sensi.utils.preferences import (
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
    from sensi.utils.preferences import DEFAULT_PREFS_V9, DEFAULT_PREFS_V10, _migrate_v9_into_v10

    v9 = deepcopy(DEFAULT_PREFS_V9)
    v9['features']['feat_sentence_pitch_mean'] = False
    target = deepcopy(DEFAULT_PREFS_V10)
    _migrate_v9_into_v10(target, v9)
    assert target['features']['feat_sentence_pitch_mean'] is False


def test_load_v9_prefs_file_migrates_to_current(tmp_path, monkeypatch):
    """load_preferences() on a v9 file auto-migrates through to the current
    schema (adds the v10 embedding keys along the way)."""
    import json
    from sensi.utils.preferences import DEFAULT_PREFS_V9, load_preferences, SCHEMA_VERSION

    prefs_path = tmp_path / 'p.json'
    prefs_path.write_text(json.dumps(DEFAULT_PREFS_V9))
    monkeypatch.setenv('SENSI_PREFS_PATH', str(prefs_path))
    loaded = load_preferences()
    assert loaded['schema_version'] == SCHEMA_VERSION
    # Embedding keys are added by the migration; under opt-in defaults both the
    # output and the embedding master (which is the feature on/off) default off.
    assert 'word2vec_mat' not in loaded['output']  # v21: retired
    assert loaded['features']['master_word2vec'] is False
    on_disk = _yaml_on_disk(prefs_path)
    assert on_disk['schema_version'] == SCHEMA_VERSION


def test_v10_to_v11_coerces_liwc_backend_and_bumps_version(tmp_path, monkeypatch):
    """A v10 file with semantic.backend='liwc' migrates to v11 with the
    backend coerced to 'empath' (LIWC disabled under its EULA), and the
    upgrade is persisted to disk."""
    import json
    from copy import deepcopy
    from sensi.utils.preferences import (
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
    on_disk = _yaml_on_disk(prefs_path)
    assert on_disk['schema_version'] == SCHEMA_VERSION
    assert on_disk['semantic']['backend'] == 'empath'


def test_v11_default_backend_is_empath():
    from sensi.utils.preferences import DEFAULT_PREFS
    assert DEFAULT_PREFS['semantic']['backend'] == 'empath'


# ---------------------------------------------------------------------------
# v12 → v13 migration
# ---------------------------------------------------------------------------

def test_v12_prefs_migrate_to_v13_additively(tmp_path, monkeypatch):
    import json
    from sensi.utils import preferences
    p = tmp_path / "preferences.json"
    v12 = {"schema_version": 12, "features": {}, "output": {}}
    p.write_text(json.dumps(v12), encoding="utf-8")
    monkeypatch.setenv("SENSI_PREFS_PATH", str(p))
    prefs = preferences.load_preferences()
    # v12 migrates additively up to the current schema; the v13 acoustic keys
    # below must be present regardless of how many versions have since landed.
    assert prefs["schema_version"] == preferences.SCHEMA_VERSION
    f = prefs["features"]
    assert f["master_acoustic_ts"] is False
    # Opt-in defaults: acoustic features + output default off (added by migration).
    assert f["feat_acoustic_ts_envelope"] is False
    assert f["feat_acoustic_ts_mfcc"] is False
    assert prefs["output"]["level_acoustic"] is False   # opt-in (v21: was acoustic_ts_mat)
    a = prefs["acoustic_ts"]
    assert a["analysis_rate"] == 128 and a["env_method"] == "rms"
    assert a["mel_preset"] == "speech"


# ---------------------------------------------------------------------------
# v13 → v14 migration (structural token-id rename)
# ---------------------------------------------------------------------------

def test_v13_to_v14_renames_struct_token_keys_without_collision(tmp_path, monkeypatch):
    """The two struct token-index toggles swap names. A pre-v14 file with the
    global toggle OFF and the within-sentence toggle ON must migrate so the
    OFF value lands on the new global key (feat_struct_token_id) and the ON
    value on the within-sentence key (feat_struct_token_in_sentence_id) — not
    collide onto the same key."""
    from sensi.utils.preferences import load_preferences, SCHEMA_VERSION
    path = tmp_path / "preferences.json"
    path.write_text(json.dumps({
        "schema_version": 13,
        "features": {
            "feat_struct_term_id": False,   # old GLOBAL toggle, user turned off
            "feat_struct_token_id": True,   # old WITHIN-SENTENCE toggle, on
        },
    }), encoding="utf-8")
    monkeypatch.setenv("SENSI_PREFS_PATH", str(path))

    f = load_preferences()["features"]
    # Old key names are gone.
    assert "feat_struct_term_id" not in f
    # Global value (was False) carried to the new global key.
    assert f["feat_struct_token_id"] is False
    # Within-sentence value (was True) carried to the new within-sentence key.
    assert f["feat_struct_token_in_sentence_id"] is True


def test_v14_defaults_have_renamed_struct_keys():
    from sensi.utils.preferences import DEFAULT_PREFS
    f = DEFAULT_PREFS["features"]
    assert "feat_struct_term_id" not in f
    assert f["feat_struct_token_id"] is True
    assert f["feat_struct_token_in_sentence_id"] is True


# ---------------------------------------------------------------------------
# v15 → v16 migration (phoneme surprisal + cohort entropy)
# ---------------------------------------------------------------------------

def test_v15_prefs_migrate_to_v16_additively(tmp_path, monkeypatch):
    import json
    from sensi.utils import preferences as P
    p = tmp_path / "prefs.json"
    # A minimal v15 file: additive migration must fill the new v16 keys.
    p.write_text(json.dumps({"schema_version": 15, "features": {}, "output": {}}),
                 encoding="utf-8")
    monkeypatch.setenv("SENSI_PREFS_PATH", str(p))
    prefs = P.load_preferences()
    assert prefs["schema_version"] == P.SCHEMA_VERSION   # lands on current
    assert prefs["features"]["master_phoneme_surprisal"] is True
    assert prefs["features"]["feat_phoneme_surprisal"] is False
    assert prefs["features"]["feat_phoneme_cohort_entropy"] is False
    assert prefs["features"]["feat_phoneme_surprisal_mat"] is False  # v21: was output.phoneme_surprisal_mat
    assert prefs["phoneme_surprisal"]["analysis_sr"] == 128
    assert prefs["phoneme_surprisal"]["strip_stress"] is True
    assert prefs["phoneme_surprisal"]["primary_only"] is True


# ---------------------------------------------------------------------------
# v17 — env_comp default becomes the exact mTRF constant log10(2)
# ---------------------------------------------------------------------------

def test_default_env_comp_is_exact_mtrf_constant():
    """The extractor docstring requires comp == log10(2) for the claimed
    bit-for-bit mTRFenvelope.m replication; the prefs default (0.3) silently
    broke that for every GUI run. Guard both the value and cross-module drift."""
    from sensi.extractors.acoustic_ts import MTRF_COMP
    import math
    assert DEFAULT_PREFS["acoustic_ts"]["env_comp"] == MTRF_COMP
    assert MTRF_COMP == pytest.approx(math.log10(2), abs=0)


def test_v16_env_comp_030_coerced_to_mtrf(tmp_path, monkeypatch):
    import json
    from sensi.utils import preferences as P
    from sensi.extractors.acoustic_ts import MTRF_COMP
    p = tmp_path / "prefs.json"
    p.write_text(json.dumps({"schema_version": 16,
                             "acoustic_ts": {"env_comp": 0.3}}),
                 encoding="utf-8")
    monkeypatch.setenv("SENSI_PREFS_PATH", str(p))
    prefs = P.load_preferences()
    assert prefs["schema_version"] == P.SCHEMA_VERSION
    assert prefs["acoustic_ts"]["env_comp"] == MTRF_COMP


def test_v16_custom_env_comp_preserved(tmp_path, monkeypatch):
    """Only the old DEFAULT (exactly 0.3) is upgraded; a deliberately chosen
    exponent is the user's to keep."""
    import json
    from sensi.utils import preferences as P
    p = tmp_path / "prefs.json"
    p.write_text(json.dumps({"schema_version": 16,
                             "acoustic_ts": {"env_comp": 0.45}}),
                 encoding="utf-8")
    monkeypatch.setenv("SENSI_PREFS_PATH", str(p))
    prefs = P.load_preferences()
    assert prefs["acoustic_ts"]["env_comp"] == 0.45


def test_v17_deliberate_030_not_coerced(tmp_path, monkeypatch):
    """Post-v17, a user who intentionally sets 0.3 keeps it — the coercion
    fires only when migrating from pre-v17 files."""
    import json
    from sensi.utils import preferences as P
    p = tmp_path / "prefs.json"
    p.write_text(json.dumps({"schema_version": P.SCHEMA_VERSION_V17,
                             "acoustic_ts": {"env_comp": 0.3}}),
                 encoding="utf-8")
    monkeypatch.setenv("SENSI_PREFS_PATH", str(p))
    prefs = P.load_preferences()
    assert prefs["acoustic_ts"]["env_comp"] == 0.3


def test_v16_stored_openai_llm_model_dropped_by_v18_migration(tmp_path, monkeypatch):
    """v16's gpt-* llm_model was historically coerced to "" by the v17
    finalize (OpenAI cloze scoring no longer works). v18 goes further and
    removes predictability.llm_model outright — with no pred_* toggle
    enabled in this source file, the stored (now-defunct OpenAI) selection
    is simply discarded rather than mapped onto a model slug."""
    import json
    from sensi.utils import preferences as P
    p = tmp_path / "prefs.json"
    p.write_text(json.dumps({"schema_version": 16,
                             "predictability": {"llm_model": "gpt-4o-mini"}}),
                 encoding="utf-8")
    monkeypatch.setenv("SENSI_PREFS_PATH", str(p))
    prefs = P.load_preferences()
    assert "llm_model" not in prefs["predictability"]
    assert prefs["predictability"]["models"] == []


def test_v16_stored_local_llm_model_dropped_when_no_predictability_toggle_on(tmp_path, monkeypatch):
    """Without any pred_* toggle enabled, a stored llm_model (even a valid
    local model id) is discarded by the v18 migration — model selection is
    now tied to which toggles/metrics were on, not merely to whether
    llm_model was ever set."""
    import json
    from sensi.utils import preferences as P
    p = tmp_path / "prefs.json"
    p.write_text(json.dumps({"schema_version": 16,
                             "predictability": {"llm_model": "meta-llama/Meta-Llama-3-8B"}}),
                 encoding="utf-8")
    monkeypatch.setenv("SENSI_PREFS_PATH", str(p))
    prefs = P.load_preferences()
    assert "llm_model" not in prefs["predictability"]
    assert prefs["predictability"]["models"] == []


# ---------------------------------------------------------------------------
# v18 — predictability models/metrics unification (replaces the four pred_*
# feature toggles + predictability.llm_model)
# ---------------------------------------------------------------------------

def test_v17_gpt2_surprisal_toggle_becomes_gpt2_model(tmp_path, monkeypatch):
    import json
    from sensi.utils import preferences as P
    p = tmp_path / "prefs.json"
    p.write_text(json.dumps({"schema_version": 17,
        "features": {"pred_surprisal_gpt2": True},
        "predictability": {"llm_model": ""}}), encoding="utf-8")
    monkeypatch.setenv("SENSI_PREFS_PATH", str(p))
    prefs = P.load_preferences()
    assert prefs["schema_version"] == P.SCHEMA_VERSION
    assert "gpt2" in prefs["predictability"]["models"]
    # Full dict: only surprisal is on (no cloze/topmatch were enabled).
    assert prefs["predictability"]["metrics"] == {
        "surprisal": True, "cloze": False, "topmatch": False}
    assert "pred_surprisal_gpt2" not in prefs["features"]


def test_v17_llm_toggles_become_stored_model_and_metrics(tmp_path, monkeypatch):
    import json
    from sensi.utils import preferences as P
    p = tmp_path / "prefs.json"
    p.write_text(json.dumps({"schema_version": 17,
        "features": {"pred_surprisal_llm": True, "pred_cloze": True,
                     "pred_topmatch": False},
        "predictability": {"llm_model": "meta-llama/Meta-Llama-3-8B"}}),
        encoding="utf-8")
    monkeypatch.setenv("SENSI_PREFS_PATH", str(p))
    prefs = P.load_preferences()
    assert prefs["predictability"]["models"] == ["llama3_8b"]
    assert prefs["predictability"]["metrics"]["surprisal"] is True
    assert prefs["predictability"]["metrics"]["cloze"] is True
    assert prefs["predictability"]["metrics"]["topmatch"] is False
    assert "llm_model" not in prefs["predictability"]


def test_v17_llm_toggle_unknown_model_defaults_llama32_3b(tmp_path, monkeypatch):
    import json
    from sensi.utils import preferences as P
    p = tmp_path / "prefs.json"
    p.write_text(json.dumps({"schema_version": 17,
        "features": {"pred_cloze": True},
        "predictability": {"llm_model": ""}}), encoding="utf-8")
    monkeypatch.setenv("SENSI_PREFS_PATH", str(p))
    prefs = P.load_preferences()
    assert prefs["predictability"]["models"] == ["llama32_3b"]
    # Regression guard: a cloze-only user must NOT be silently given
    # surprisal (the v18 default has surprisal=True; migration must override
    # it from the old toggles). Assert the full dict, not just cloze.
    assert prefs["predictability"]["metrics"] == {
        "surprisal": False, "cloze": True, "topmatch": False}


def test_v18_fresh_default_predictability_off(tmp_path, monkeypatch):
    from sensi.utils import preferences as P
    monkeypatch.setenv("SENSI_PREFS_PATH", str(tmp_path / "none.json"))
    prefs = P.load_preferences()
    # Fully off on a fresh install: no models AND no metric pre-checked, so no
    # spurious day-one "metric on, no model" preflight warning.
    assert prefs["predictability"]["models"] == []
    assert prefs["predictability"]["metrics"] == {
        "surprisal": False, "cloze": False, "topmatch": False}


def test_v17_doc_agg_surprisal_or_llm_unifies_to_doc_agg_predictability(tmp_path, monkeypatch):
    """A real pre-v18 saved file has the literal doc_agg_surprisal /
    doc_agg_llm output keys (shipped v7-v17). Migration ORs them into the
    single doc_agg_predictability toggle and drops both old keys."""
    import json
    from sensi.utils import preferences as P
    p = tmp_path / "prefs.json"
    p.write_text(json.dumps({"schema_version": 17,
        "output": {"doc_agg_surprisal": False, "doc_agg_llm": True}}),
        encoding="utf-8")
    monkeypatch.setenv("SENSI_PREFS_PATH", str(p))
    prefs = P.load_preferences()
    assert prefs["output"]["doc_agg_predictability"] is True
    assert "doc_agg_surprisal" not in prefs["output"]
    assert "doc_agg_llm" not in prefs["output"]


def test_v17_doc_agg_both_off_stays_off(tmp_path, monkeypatch):
    """Both old toggles off must migrate to doc_agg_predictability=False, not
    fall back to whatever the v18 default happens to be."""
    import json
    from sensi.utils import preferences as P
    p = tmp_path / "prefs.json"
    p.write_text(json.dumps({"schema_version": 17,
        "output": {"doc_agg_surprisal": False, "doc_agg_llm": False}}),
        encoding="utf-8")
    monkeypatch.setenv("SENSI_PREFS_PATH", str(p))
    prefs = P.load_preferences()
    assert prefs["output"]["doc_agg_predictability"] is False


def test_v19_fresh_default_has_empty_context_map(tmp_path, monkeypatch):
    from sensi.utils import preferences as P
    monkeypatch.setenv("SENSI_PREFS_PATH", str(tmp_path / "none.json"))
    prefs = P.load_preferences()
    assert prefs["schema_version"] == P.SCHEMA_VERSION
    assert prefs["predictability"]["context"] == {}


def test_v18_file_gains_empty_context_map(tmp_path, monkeypatch):
    """Additive v18→v19: a v18 file (no context key) loads with context={},
    its models/metrics untouched."""
    import json
    from sensi.utils import preferences as P
    p = tmp_path / "prefs.json"
    p.write_text(json.dumps({"schema_version": 18,
        "predictability": {"models": ["gpt2"],
                           "metrics": {"surprisal": True, "cloze": False,
                                       "topmatch": False}}}), encoding="utf-8")
    monkeypatch.setenv("SENSI_PREFS_PATH", str(p))
    prefs = P.load_preferences()
    assert prefs["predictability"]["context"] == {}
    assert prefs["predictability"]["models"] == ["gpt2"]
    assert prefs["predictability"]["metrics"]["surprisal"] is True


def test_v18_file_preserves_existing_context_values(tmp_path, monkeypatch):
    import json
    from sensi.utils import preferences as P
    p = tmp_path / "prefs.json"
    p.write_text(json.dumps({"schema_version": 18,
        "predictability": {"models": ["llama3_8b"],
                           "metrics": {"surprisal": True, "cloze": False,
                                       "topmatch": False},
                           "context": {"llama3_8b": 2048}}}), encoding="utf-8")
    monkeypatch.setenv("SENSI_PREFS_PATH", str(p))
    prefs = P.load_preferences()
    assert prefs["predictability"]["context"] == {"llama3_8b": 2048}


# ---------------------------------------------------------------------------
# v19 -> v20: phoneme label split + syllable IPA + alignment.g2p_oov
# ---------------------------------------------------------------------------

_NEW_PHONEME_KEYS = ("feat_phoneme_arpabet", "feat_phoneme_stress",
                     "feat_phoneme_ipa", "feat_phoneme_pron_source")


def _load_from(tmp_path, monkeypatch, stored: dict) -> dict:
    """Write `stored` as the legacy json and return the migrated prefs."""
    monkeypatch.setenv('SENSI_PREFS_PATH', str(tmp_path / 'p.json'))
    (tmp_path / 'p.json').write_text(json.dumps(stored), encoding='utf-8')
    return load_preferences()


def test_v20_version_constant():
    assert SCHEMA_VERSION_V20 == 20
    assert SCHEMA_VERSION > SCHEMA_VERSION_V20
    assert DEFAULT_PREFS['schema_version'] == SCHEMA_VERSION


def test_defaults_drop_label_and_add_the_split_keys():
    f = DEFAULT_PREFS['features']
    assert 'feat_phoneme_label' not in f
    for key in _NEW_PHONEME_KEYS:
        assert key in f
    assert f['feat_syllable_ipa'] is False
    assert DEFAULT_PREFS['alignment'] == {'g2p_oov': False}


def test_paragraph_toggle_is_untouched():
    """It already existed; the migration must not add or reset it."""
    assert DEFAULT_PREFS['features']['feat_struct_paragraph_id'] is True


def test_v19_label_on_migrates_all_four_replacements_on(tmp_path, monkeypatch):
    prefs = _load_from(tmp_path, monkeypatch, {
        'schema_version': 19,
        'features': {'feat_phoneme_label': True},
    })
    assert prefs['schema_version'] == SCHEMA_VERSION
    assert 'feat_phoneme_label' not in prefs['features']
    for key in _NEW_PHONEME_KEYS:
        assert prefs['features'][key] is True, key


def test_v19_label_off_migrates_all_four_replacements_off(tmp_path, monkeypatch):
    prefs = _load_from(tmp_path, monkeypatch, {
        'schema_version': 19,
        'features': {'feat_phoneme_label': False},
    })
    for key in _NEW_PHONEME_KEYS:
        assert prefs['features'][key] is False, key


def test_g2p_oov_defaults_off_after_migration(tmp_path, monkeypatch):
    prefs = _load_from(tmp_path, monkeypatch,
                       {'schema_version': 19, 'features': {}})
    assert prefs['alignment']['g2p_oov'] is False


def test_post_v20_user_values_survive_a_reload(tmp_path, monkeypatch):
    """A v20 source is not re-migrated; the user's own values win."""
    prefs = _load_from(tmp_path, monkeypatch, {
        'schema_version': 20,
        'features': {'feat_phoneme_ipa': False, 'feat_phoneme_arpabet': True},
        'alignment': {'g2p_oov': True},
    })
    assert prefs['features']['feat_phoneme_ipa'] is False
    assert prefs['features']['feat_phoneme_arpabet'] is True
    assert prefs['alignment']['g2p_oov'] is True


def test_alignment_section_merges_by_inner_key_not_wholesale():
    """Regression for a missed registration step: `alignment` must be listed
    in _CURRENT_SECTIONS (utils/preferences.py) so it merges by inner key like
    every other section (acoustic_ts, phoneme_surprisal, ...). An unregistered
    section is wholesale-replaced, so saving one alignment.* toggle would
    silently erase any sibling key the instant a second one exists — this
    pins the merge behavior itself, not just that g2p_oov is present."""
    from sensi.utils.preferences import _deep_merge_current
    cur = {"alignment": {"g2p_oov": False, "future_key": True}}
    usr = {"alignment": {"g2p_oov": True}}
    _deep_merge_current(cur, usr)
    assert cur["alignment"] == {"g2p_oov": True, "future_key": True}, (
        "alignment must merge by inner key: a partial user dict must not "
        "erase sibling keys"
    )


def test_partial_alignment_section_on_disk_default_fills_g2p_oov(tmp_path, monkeypatch):
    """A features.yaml carrying a partial (empty) `alignment` section must
    still load with g2p_oov present and False, not an empty dict — the
    end-to-end counterpart of the _deep_merge_current test above."""
    import yaml
    monkeypatch.setenv("SENSI_CONFIG_DIR", str(tmp_path))
    (tmp_path / "features.yaml").write_text(
        yaml.safe_dump({"schema_version": 20, "alignment": {}}),
        encoding="utf-8")
    prefs = load_preferences()
    assert prefs["alignment"] == {"g2p_oov": False}
