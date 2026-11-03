"""Planner wiring for the word / syllable event .mat artifacts (v22)."""
import copy

from sensi.gui.planner import plan_from_prefs, to_namespace
from sensi.utils.preferences import DEFAULT_PREFS
from sensi.utils.session_units import build_session_units


def _prefs(output=None, features=None, **sections):
    p = copy.deepcopy(DEFAULT_PREFS)
    p["output"].update(output or {})
    p["features"].update(features or {})
    for k, v in sections.items():
        p[k] = {**p.get(k, {}), **v}
    return p


def test_word_events_mat_planned_with_alignment():
    plan = plan_from_prefs(_prefs({"level_word": True},
                                  {"feat_word_events_mat": True}))
    assert "word_events_mat" in plan.output_csvs
    assert "mfa_align" in plan.stages_to_run


def test_word_events_mat_absent_when_level_or_master_off():
    on = {"feat_word_events_mat": True}
    assert "word_events_mat" not in plan_from_prefs(
        _prefs({"level_word": False}, on)).output_csvs
    assert "word_events_mat" not in plan_from_prefs(
        _prefs({"level_word": True}, {**on, "master_word_events": False})).output_csvs


def test_syllable_events_mat_plans_syllabify_without_csv():
    prefs = _prefs({"level_subword": True}, {"feat_syllable_events_mat": True})
    plan = plan_from_prefs(prefs)
    assert "syllable_events_mat" in plan.output_csvs
    assert "syllabify_pairs" in plan.stages_to_run
    ns = to_namespace(plan, prefs, "word")
    assert ns.no_syllable_events_mat is False
    assert ns.no_word_events_mat is True


def test_event_units_carry_predictability_even_without_features_csv():
    prefs = _prefs(
        {"level_word": True},
        {"feat_word_events_mat": True, "master_structural": False,
         "master_predictability": True},
        predictability={"models": ["gpt2"],
                        "metrics": {"surprisal": True, "cloze": False,
                                    "topmatch": False}})
    import dataclasses
    plan = plan_from_prefs(prefs)
    plan = dataclasses.replace(plan, output_csvs=plan.output_csvs - {"features"})
    ns = to_namespace(plan, prefs, "word")
    units = ns.event_units["word_events_mat"]
    assert units["gpt2_Surprisal"] == "bits"
    assert units["onset"] == "seconds" and units["duration"] == "seconds"
    assert ns.event_units["syllable_events_mat"]["pcenter"] == "seconds"
    assert to_namespace(plan, prefs, "doc").event_units == {}


def test_session_units_document_live_event_files():
    plan = plan_from_prefs(_prefs({"level_word": True},
                                  {"feat_word_events_mat": True}))
    rows = build_session_units(plan)
    ev = {r["column"] for r in rows if r["output_csv"] == "word_events_mat"}
    assert ev == {"onset", "offset", "duration"}
