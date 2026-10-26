"""v21 output levels: LEVEL_ARTIFACTS gating, uniform section switches, and
document predictability means independent of the word-level section."""
import json
from copy import deepcopy
from pathlib import Path

import pytest

from gui.planner.emit import to_namespace
from gui.planner.plan import plan_from_prefs
from utils.preferences import DEFAULT_PREFS, DEFAULT_PREFS_V20, _merged_from_user

_CASES = json.loads(
    (Path(__file__).resolve().parents[1] / "fixtures" / "output_switch_snapshots"
     / "v20_cases.json").read_text(encoding="utf-8"))
_CASE_BY_NAME = {c["name"]: c for c in _CASES}


def _deep_merge(base: dict, over: dict) -> dict:
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = deepcopy(v)
    return base


def _migrated(name: str) -> dict:
    v20 = _deep_merge(deepcopy(DEFAULT_PREFS_V20), _CASE_BY_NAME[name]["prefs_overrides"])
    merged, _ = _merged_from_user(v20)
    return merged


@pytest.mark.parametrize(
    "case", [c for c in _CASES if c["name"] != "doc_only_with_pred_means"],
    ids=lambda c: c["name"])
def test_migrated_v20_cases_plan_identically(case):
    plan = plan_from_prefs(_migrated(case["name"]))
    assert sorted(plan.output_csvs) == case["output_csvs"]
    assert list(plan.stages_to_run) == case["stages_to_run"]


def _doc_means_prefs() -> dict:
    prefs = _migrated("doc_only_with_pred_means")
    prefs["features"]["master_predictability"] = False
    return prefs


def test_doc_means_compute_with_word_predictability_off():
    plan = plan_from_prefs(_doc_means_prefs())
    assert "predictability" in plan.stages_to_run
    assert "features" not in plan.output_csvs
    assert plan.predictability_columns == ()


def test_doc_namespace_carries_predictability():
    prefs = _doc_means_prefs()
    ns = to_namespace(plan_from_prefs(prefs), prefs, level="doc")
    assert ns.predictability["models"] == ["gpt2"]
    assert ns.no_doc_predictability is False


def test_section_off_drops_artifact():
    prefs = deepcopy(DEFAULT_PREFS)
    prefs["output"]["level_subword"] = True
    for k in prefs["features"]:
        if k.startswith("feat_syllable_"):
            prefs["features"][k] = True
    assert "syllables" in plan_from_prefs(prefs).output_csvs  # control
    for k in ("master_syllable_intervals", "master_syllable_labels",
              "master_syllable_prosody", "master_syllable_pcenter"):
        prefs["features"][k] = False
    assert "syllables" not in plan_from_prefs(prefs).output_csvs


def test_level_off_drops_all_its_artifacts():
    prefs = _migrated("all_on")
    on = plan_from_prefs(prefs).output_csvs
    assert {"features", "word2vec", "bert_word"} <= on  # control
    prefs["output"]["level_word"] = False
    out = plan_from_prefs(prefs).output_csvs
    assert not out & {"features", "word2vec", "bert_word"}
