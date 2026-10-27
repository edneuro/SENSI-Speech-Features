from dataclasses import FrozenInstanceError
import pytest

from sensi.gui.planner.stages import Stage, FeatureCluster, Plan


def test_stage_is_frozen_dataclass():
    s = Stage(name="x", consumes=("a",), produces=("b",), run=lambda: None)
    with pytest.raises(FrozenInstanceError):
        s.name = "y"  # pyright: ignore[reportAttributeAccessIssue]


def test_stage_fields():
    fn = lambda x: x
    s = Stage(name="word_freq", consumes=("word_df",), produces=("Zipf_Freq",), run=fn)
    assert s.name == "word_freq"
    assert s.consumes == ("word_df",)
    assert s.produces == ("Zipf_Freq",)
    assert s.run is fn


def test_feature_cluster_is_frozen():
    c = FeatureCluster(
        name="SUBTLEX",
        toggle_keys=("freq_subtlex",),
        output_csv="features",
        requires_stages=("word_freq",),
        emits_columns=("Zipf_Freq",),
    )
    with pytest.raises(FrozenInstanceError):
        c.name = "y"  # pyright: ignore[reportAttributeAccessIssue]


def test_feature_cluster_fields():
    c = FeatureCluster(
        name="word prosody",
        toggle_keys=("feat_prosody_pitch_mean", "feat_prosody_volume_mean"),
        output_csv="features",
        requires_stages=("word_prosody",),
        emits_columns=("Pitch_Mean", "Volume_Mean"),
    )
    assert c.output_csv == "features"
    assert "feat_prosody_pitch_mean" in c.toggle_keys


def test_plan_is_frozen_dataclass():
    p = Plan(
        stages_to_run=("a", "b"),
        output_csvs=frozenset({"features"}),
        column_drops={"features": frozenset({"X"})},
        stage_configs={"a": {"k": "v"}},
    )
    with pytest.raises(FrozenInstanceError):
        p.stages_to_run = ("c",)  # pyright: ignore[reportAttributeAccessIssue]


def test_plan_field_reassignment_blocked():
    p = Plan(stages_to_run=(), output_csvs=frozenset(),
             column_drops={}, stage_configs={})
    with pytest.raises(FrozenInstanceError):
        p.column_drops = {}  # pyright: ignore[reportAttributeAccessIssue]


def test_feature_cluster_accepts_master_toggle_field():
    c = FeatureCluster(
        name="word prosody",
        toggle_keys=("feat_prosody_pitch_mean",),
        output_csv="features",
        requires_stages=("word_prosody",),
        emits_columns=("Pitch_Mean",),
        master_toggle="master_word_prosody",
    )
    assert c.master_toggle == "master_word_prosody"
    assert c.available_when is None


def test_feature_cluster_accepts_available_when_field():
    c = FeatureCluster(
        name="psych composite",
        toggle_keys=("feat_psych_authentic",),
        output_csv="documents_summary",
        requires_stages=("doc_liwc",),
        emits_columns=("psych_authentic",),
        available_when=("semantic.backend", "liwc"),
    )
    assert c.available_when == ("semantic.backend", "liwc")
    assert c.master_toggle is None


def test_feature_cluster_defaults_both_to_none():
    c = FeatureCluster(
        name="default",
        toggle_keys=("k",),
        output_csv="features",
        requires_stages=(),
        emits_columns=(),
    )
    assert c.master_toggle is None
    assert c.available_when is None
