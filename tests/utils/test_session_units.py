"""Tests for utils.session_units."""
import pytest
from gui.planner.stages import Plan
from utils.session_units import build_session_units


def _mk_plan(stages: tuple[str, ...] = (),
             csvs: frozenset[str] = frozenset({"features"}),
             stage_configs: dict | None = None) -> Plan:
    return Plan(
        stages_to_run=stages,
        output_csvs=csvs,
        column_drops={c: frozenset() for c in csvs},
        stage_configs=stage_configs or {},
    )


def test_predictability_unit_rows_from_dynamic_columns():
    # The <slug>_<Metric> columns are dynamic (depend on the selected model
    # roster), so their unit rows come from plan.predictability_columns, not a
    # static table. One row per column, unit keyed off the metric suffix.
    plan = Plan(
        stages_to_run=("predictability",),
        output_csvs=frozenset({"features"}),
        column_drops={"features": frozenset()},
        stage_configs={},
        predictability_columns=("gpt2_Surprisal", "gpt2_Cloze",
                                "llama3_8b_TopMatch"),
    )
    rows = build_session_units(plan)
    by_col = {r["column"]: r["unit"] for r in rows
              if r["column"] in {"gpt2_Surprisal", "gpt2_Cloze",
                                 "llama3_8b_TopMatch"}}
    assert by_col == {
        "gpt2_Surprisal": "bits",
        "gpt2_Cloze": "probability (0-1)",
        "llama3_8b_TopMatch": "boolean",
    }
    assert all(r["output_csv"] == "features" for r in rows
               if r["column"] in by_col)


def test_predictability_unit_rows_suppressed_without_features_csv():
    plan = Plan(
        stages_to_run=("predictability",),
        output_csvs=frozenset({"documents_summary"}),
        column_drops={"documents_summary": frozenset()},
        stage_configs={},
        predictability_columns=("gpt2_Surprisal",),
    )
    rows = build_session_units(plan)
    assert not any(r["column"] == "gpt2_Surprisal" for r in rows)


def test_no_rows_for_inactive_csv():
    plan = _mk_plan(csvs=frozenset({"features"}))
    rows = build_session_units(plan)
    assert not any(r["output_csv"] == "syllables" for r in rows)
    assert not any(r["output_csv"] == "documents_summary" for r in rows)


def test_volume_rows_emit_db():
    plan = _mk_plan(csvs=frozenset({"features"}))
    rows = build_session_units(plan)
    vol = [r for r in rows if r["column"].startswith("Volume_")]
    assert vol, "expected Volume_* rows"
    assert all(r["unit"] == "dB" for r in vol)


def test_pitch_units_propagate_from_praat_contours_config():
    plan = _mk_plan(
        csvs=frozenset({"features"}),
        stage_configs={"praat_contours": {"pitch_units": "Hz"}},
    )
    rows = build_session_units(plan)
    pitch = [r for r in rows if r["column"].startswith("Pitch_")]
    assert pitch
    assert all(r["unit"] == "Hz" for r in pitch)


def test_pitch_units_default_to_semitones():
    plan = _mk_plan(csvs=frozenset({"features"}))
    rows = build_session_units(plan)
    pitch = [r for r in rows if r["column"].startswith("Pitch_")]
    assert all(r["unit"] == "semitones" for r in pitch)


def test_doc_summary_includes_articulation_and_pause_units():
    plan = _mk_plan(csvs=frozenset({"documents_summary"}))
    rows = build_session_units(plan)
    by_col = {r["column"]: r["unit"] for r in rows}
    assert by_col["doc_articulation_rate"] == "syllables/second"
    assert by_col["doc_pause_count"] == "count"
    assert by_col["doc_pause_pct_silence"] == "percent"
    # Spot-checks to catch typos across unrelated unit strings (a swap
    # between two doc-summary entries shouldn't slip through).
    assert by_col["doc_hnr"] == "dB"
    assert by_col["doc_pause_mean_duration"] == "seconds"


def test_multiple_csvs_active():
    plan = _mk_plan(csvs=frozenset({"features", "syllables", "phonemes"}))
    rows = build_session_units(plan)
    csvs = {r["output_csv"] for r in rows}
    assert csvs == {"features", "syllables", "phonemes"}


def test_no_rows_when_no_csvs_active():
    plan = _mk_plan(csvs=frozenset())
    rows = build_session_units(plan)
    assert rows == []
