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
