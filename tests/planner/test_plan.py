"""plan_from_prefs algorithm tests.

The contract: given a prefs dict (output + features sections), produce
a Plan whose stages_to_run satisfies every active cluster's
requires_stages plus all transitive consumes-dependencies.
"""
import pytest

from gui.planner.plan import plan_from_prefs


# v7 master toggle defaults (all True) — included in every test prefs dict
# so individual tests that don't care about master gates still see clusters
# as active (matching real v7 default behaviour).
_MASTER_DEFAULTS_FEATURES = {
    "master_structural":         True,
    "master_lexling":            True,
    "master_frequency":          True,
    "master_predictability":     True,
    "master_acoustic":           True,
    "master_word_prosody":       True,
    "master_psych_composite":    True,
    "master_syllable_intervals": True,
    "master_syllable_labels":    True,
    "master_syllable_prosody":   True,
    "master_phoneme_intervals":  True,
}
_MASTER_DEFAULTS_OUTPUT = {
    "master_doc_prosody":           True,
    "master_doc_articulation_rate": True,
    "master_doc_counts":            True,
}


def _prefs(output: dict, features: dict, *, semantic=None, prosody=None,
           predictability=None, shared=None) -> dict:
    """Build a minimal prefs dict with sensible defaults for unspecified sections.

    Master toggle keys (v7) are included at True so tests that don't exercise
    master gates don't inadvertently block clusters.  Individual tests may
    override a master key by passing it explicitly in the output/features dicts.
    """
    merged_features = {**_MASTER_DEFAULTS_FEATURES, **features}
    merged_output   = {**_MASTER_DEFAULTS_OUTPUT,   **output}
    return {
        "output":   merged_output,
        "features": merged_features,
        "semantic": semantic or {"backend": "empath", "categories_empath": []},
        "prosody":  prosody or {"pitch_units": "semitones", "pitch_floor_hz": 75.0,
                                 "pitch_ceiling_hz": 500.0, "pause_threshold_s": 0.25},
        "predictability": predictability or {"llm_model": ""},
        "shared":   shared or {},
    }


def test_empty_selection_produces_empty_plan():
    """All output toggles off → no stages, no CSVs, no work."""
    p = plan_from_prefs(_prefs(
        output={"word_level_csv": False, "document_summary": False,
                "syllables_csv": False, "phonemes_csv": False,
                "doc_category_proportions": False, "doc_lex_diversity": False,
                "doc_readability": False, "doc_aggregates": False,
                "out_count_tokens": False, "out_count_content_tokens": False,
                "out_count_sentences": False, "out_count_paragraphs": False},
        features={},
    ))
    assert p.stages_to_run == ()
    assert p.output_csvs == frozenset()


def test_subtlex_only_word_run():
    """Word CSV + SUBTLEX toggle on → spacy_parse + word_freq."""
    p = plan_from_prefs(_prefs(
        output={"word_level_csv": True, "syllables_csv": False, "phonemes_csv": False,
                "document_summary": False},
        features={"freq_subtlex": True},
    ))
    assert "spacy_parse" in p.stages_to_run
    assert "word_freq" in p.stages_to_run
    assert "mfa_align" not in p.stages_to_run
    assert "features" in p.output_csvs


def test_subword_features_force_parse_and_align():
    """Regression: sub-word features on, all word toggles off, sidecars wanted
    → spacy_parse and mfa_align must fire so syllabify_pairs has the columns
    it needs. (This is exactly the bug that shipped without a test before.)"""
    p = plan_from_prefs(_prefs(
        output={"word_level_csv": False, "syllables_csv": True, "phonemes_csv": True,
                "document_summary": False},
        features={"feat_syllable_onset": True, "feat_syllable_end": True,
                  "feat_phoneme_label": True,
                  # All struct + acoustic OFF (matches the user's reported config)
                  "feat_struct_doc_id": False, "feat_struct_sentence_id": False,
                  "feat_struct_token_id": False, "feat_struct_token_in_sentence_id": False,
                  "feat_acoustic_onset": False, "feat_acoustic_end": False},
    ))
    assert "spacy_parse" in p.stages_to_run, \
        "spaCy parse must fire for syllabify_pairs to have doc_id/sentence_id/token_in_sentence_id"
    assert "mfa_align" in p.stages_to_run, \
        "MFA must fire to produce phones_df"
    assert "syllabify_pairs" in p.stages_to_run
    assert {"syllables", "phonemes"} <= p.output_csvs
    assert "features" not in p.output_csvs


def test_syllable_prosody_forces_contours():
    """Regression: enabling syllable prosody must pull praat_contours into the plan."""
    p = plan_from_prefs(_prefs(
        output={"word_level_csv": False, "syllables_csv": True, "phonemes_csv": False,
                "document_summary": False},
        features={"feat_syllable_pitch_mean": True},
    ))
    assert "praat_contours" in p.stages_to_run
    assert "syllable_prosody" in p.stages_to_run


def test_word_prosody_forces_align_and_contours():
    p = plan_from_prefs(_prefs(
        output={"word_level_csv": True, "syllables_csv": False, "phonemes_csv": False,
                "document_summary": False},
        features={"feat_prosody_pitch_mean": True},
    ))
    assert "mfa_align" in p.stages_to_run
    assert "praat_contours" in p.stages_to_run
    assert "word_prosody" in p.stages_to_run


def test_doc_prosody_pulls_align_and_contours_only_for_doc():
    p = plan_from_prefs(_prefs(
        output={"word_level_csv": False, "syllables_csv": False, "phonemes_csv": False,
                "document_summary": True},
        features={"feat_doc_prosody_pitch_mean": True},
    ))
    assert "mfa_align" in p.stages_to_run
    assert "praat_contours" in p.stages_to_run
    assert "doc_prosody" in p.stages_to_run
    assert "documents_summary" in p.output_csvs


def test_inactive_cluster_doesnt_pull_its_stage():
    """If a cluster is NOT active (no toggle on or its csv not wanted),
    the stages it would have required are not pulled unless some OTHER
    active cluster also needs them."""
    p = plan_from_prefs(_prefs(
        output={"word_level_csv": False, "syllables_csv": False, "phonemes_csv": False,
                "document_summary": True,
                "doc_lex_diversity": True},  # output-keyed toggle; only lex diversity on
        features={},
    ))
    assert "spacy_parse" in p.stages_to_run    # lex diversity needs it
    assert "doc_lex_diversity" in p.stages_to_run
    assert "mfa_align" not in p.stages_to_run  # nothing needs it
    assert "praat_contours" not in p.stages_to_run
    assert "word_freq" not in p.stages_to_run


def test_topological_order_dependencies_first():
    """If both spacy_parse and word_freq are in stages_to_run, spacy_parse
    must come before word_freq (because word_freq consumes word_df)."""
    p = plan_from_prefs(_prefs(
        output={"word_level_csv": True, "syllables_csv": False, "phonemes_csv": False,
                "document_summary": False},
        features={"freq_subtlex": True},
    ))
    order = list(p.stages_to_run)
    assert order.index("spacy_parse") < order.index("word_freq")


def test_column_drops_for_inactive_cluster_columns_in_active_csv():
    """If features.csv is wanted and SUBTLEX is on but prosody is off,
    column_drops['features'] should include all prosody columns."""
    p = plan_from_prefs(_prefs(
        output={"word_level_csv": True, "syllables_csv": False, "phonemes_csv": False,
                "document_summary": False},
        features={"freq_subtlex": True},  # prosody all off (default)
    ))
    drops = p.column_drops.get("features", frozenset())
    assert "Pitch_Mean" in drops
    assert "Volume_Mean" in drops
    assert "Zipf_Freq" not in drops  # this IS active


def test_stage_configs_carries_prosody_params():
    p = plan_from_prefs(_prefs(
        output={"word_level_csv": True, "syllables_csv": False, "phonemes_csv": False,
                "document_summary": False},
        features={"feat_prosody_pitch_mean": True},
        prosody={"pitch_units": "Hz", "pitch_floor_hz": 50.0,
                 "pitch_ceiling_hz": 600.0, "pause_threshold_s": 0.3},
    ))
    cfg = p.stage_configs.get("praat_contours", {})
    assert cfg.get("pitch_floor") == 50.0
    assert cfg.get("pitch_ceiling") == 600.0
    # word_prosody is in the plan, so its config is present and uses Hz.
    assert p.stage_configs.get("word_prosody", {}).get("pitch_units") == "Hz"
    # Non-running prosody stages must NOT leak phantom config entries.
    assert "syllable_prosody" not in p.stage_configs
    assert "doc_prosody" not in p.stage_configs


def test_stage_configs_carries_llm_model():
    p = plan_from_prefs(_prefs(
        output={"word_level_csv": True, "syllables_csv": False, "phonemes_csv": False,
                "document_summary": False},
        features={"pred_cloze": True},
        predictability={"llm_model": "gpt-4o-mini"},
    ))
    assert "llm_predict" in p.stages_to_run
    assert p.stage_configs.get("llm_predict", {}).get("llm_model") == "gpt-4o-mini"


# ---------------------------------------------------------------------------
# master_toggle and available_when activation tests
# ---------------------------------------------------------------------------

import gui.planner.plan as _plan_mod
from gui.planner.stages import FeatureCluster, Stage


def _install_test_cluster(monkeypatch, cluster: FeatureCluster, extra_stages=()):
    """Replace CLUSTERS and STAGES with a minimal synthetic registry."""
    test_stage = Stage(name="t_stage", consumes=("text",),
                       produces=("text.out",), run=lambda *a, **kw: None)
    monkeypatch.setattr(_plan_mod, "CLUSTERS", [cluster])
    monkeypatch.setattr(_plan_mod, "STAGES", [test_stage, *extra_stages])


def test_master_toggle_gates_cluster_inactive_when_false(monkeypatch):
    cluster = FeatureCluster(
        name="t", toggle_keys=("child",), output_csv="features",
        requires_stages=("t_stage",), emits_columns=("X",),
        master_toggle="master_t",
    )
    _install_test_cluster(monkeypatch, cluster)
    p = plan_from_prefs({
        "output": {"word_level_csv": True},
        "features": {"child": True, "master_t": False},  # child on, master off
    })
    assert p.stages_to_run == ()  # master off = cluster inactive


def test_master_toggle_active_when_true(monkeypatch):
    cluster = FeatureCluster(
        name="t", toggle_keys=("child",), output_csv="features",
        requires_stages=("t_stage",), emits_columns=("X",),
        master_toggle="master_t",
    )
    _install_test_cluster(monkeypatch, cluster)
    p = plan_from_prefs({
        "output": {"word_level_csv": True},
        "features": {"child": True, "master_t": True},
    })
    assert "t_stage" in p.stages_to_run


def test_available_when_gates_cluster_inactive_on_mismatch(monkeypatch):
    cluster = FeatureCluster(
        name="t", toggle_keys=("child",), output_csv="features",
        requires_stages=("t_stage",), emits_columns=("X",),
        available_when=("semantic.backend", "liwc"),
    )
    _install_test_cluster(monkeypatch, cluster)
    p = plan_from_prefs({
        "output": {"word_level_csv": True},
        "features": {"child": True},
        "semantic": {"backend": "empath"},  # mismatch
    })
    assert p.stages_to_run == ()


def test_available_when_active_on_match(monkeypatch):
    cluster = FeatureCluster(
        name="t", toggle_keys=("child",), output_csv="features",
        requires_stages=("t_stage",), emits_columns=("X",),
        available_when=("semantic.backend", "liwc"),
    )
    _install_test_cluster(monkeypatch, cluster)
    p = plan_from_prefs({
        "output": {"word_level_csv": True},
        "features": {"child": True},
        "semantic": {"backend": "liwc"},
    })
    assert "t_stage" in p.stages_to_run


def test_master_and_available_when_both_required(monkeypatch):
    """Both gates must pass for the cluster to be active."""
    cluster = FeatureCluster(
        name="t", toggle_keys=("child",), output_csv="features",
        requires_stages=("t_stage",), emits_columns=("X",),
        master_toggle="master_t",
        available_when=("semantic.backend", "liwc"),
    )
    _install_test_cluster(monkeypatch, cluster)
    # master True, available_when False
    p = plan_from_prefs({
        "output": {"word_level_csv": True},
        "features": {"child": True, "master_t": True},
        "semantic": {"backend": "empath"},
    })
    assert p.stages_to_run == ()
    # master False, available_when True
    p = plan_from_prefs({
        "output": {"word_level_csv": True},
        "features": {"child": True, "master_t": False},
        "semantic": {"backend": "liwc"},
    })
    assert p.stages_to_run == ()
    # both pass
    p = plan_from_prefs({
        "output": {"word_level_csv": True},
        "features": {"child": True, "master_t": True},
        "semantic": {"backend": "liwc"},
    })
    assert "t_stage" in p.stages_to_run
