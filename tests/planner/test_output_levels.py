"""v21 output levels: LEVEL_ARTIFACTS gating, uniform section switches, and
document predictability means independent of the word-level section."""
import json
from copy import deepcopy
from pathlib import Path

import pytest

from sensi.gui.planner.emit import to_namespace
from sensi.gui.planner.plan import plan_from_prefs
from sensi.utils.preferences import DEFAULT_PREFS, DEFAULT_PREFS_V20, _merged_from_user

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


def _word_pred_prefs(master_on: bool) -> dict:
    prefs = deepcopy(DEFAULT_PREFS)
    prefs["output"]["level_word"] = True
    prefs["features"]["master_predictability"] = master_on
    prefs["predictability"]["models"] = ["gpt2"]
    prefs["predictability"]["metrics"] = {"surprisal": True}
    return prefs


def test_word_namespace_has_no_models_when_predictability_off():
    prefs = _word_pred_prefs(master_on=False)
    ns = to_namespace(plan_from_prefs(prefs), prefs, level="word")
    assert ns.predictability["models"] == []
    assert not any(ns.predictability["metrics"].values())


def test_doc_namespace_has_models_on_doc_path():
    prefs = _doc_means_prefs()
    plan = plan_from_prefs(prefs)
    assert to_namespace(plan, prefs, level="doc").predictability["models"] == ["gpt2"]
    assert to_namespace(plan, prefs, level="word").predictability["models"] == []


def test_word_namespace_has_models_on_word_path():
    prefs = _word_pred_prefs(master_on=True)
    ns = to_namespace(plan_from_prefs(prefs), prefs, level="word")
    assert ns.predictability["models"] == ["gpt2"]
    assert ns.predictability["metrics"]["surprisal"] is True


# --- Migration cases with non-default v20 masters -------------------------
# Expected plans were taken by running the base commit's (0587fef, v20)
# planner on the same v20 prefs: (output_csvs, stages_to_run, sidecar
# column_drops). They cover the clusters that had no section master in v20
# (phoneme-surprisal .mat, P-center) and a master that was off.

def _feat_v20(*prefixes: str) -> dict:
    return {k: True for k, v in DEFAULT_PREFS_V20["features"].items()
            if isinstance(v, bool) and k.startswith(prefixes)}


_PSR_TICKS = {"feat_phoneme_surprisal": True, "feat_phoneme_cohort_entropy": True}
_NONDEFAULT_MASTER_CASES = {
    "psr_mat_master_off": {
        "output": {"phonemes_csv": False, "phoneme_surprisal_mat": True},
        "features": {"master_phoneme_surprisal": False, **_PSR_TICKS}},
    "psr_mat_master_off_with_phonemes": {
        "output": {"phonemes_csv": True, "phoneme_surprisal_mat": True},
        "features": {**_feat_v20("feat_phoneme_"),
                     "master_phoneme_surprisal": False, **_PSR_TICKS}},
    "psr_mat_master_on_no_phonemes": {
        "output": {"phonemes_csv": False, "phoneme_surprisal_mat": True},
        "features": {"master_phoneme_surprisal": True, **_PSR_TICKS}},
    "syllables_only_psr_ticked": {
        "output": {"syllables_csv": True, "phonemes_csv": False,
                   "phoneme_surprisal_mat": False},
        "features": {**_feat_v20("feat_syllable_", "feat_phoneme_"),
                     "master_phoneme_surprisal": True}},
    "bert_sentence_master_off": {
        "output": {"sentences_csv": False, "bert_sentence_mat": True},
        "features": {"master_bert_sentence": False, **_feat_v20("feat_sentence_")}},
    "bert_sentence_master_off_with_csv": {
        "output": {"sentences_csv": True, "bert_sentence_mat": True},
        "features": {"master_bert_sentence": False, **_feat_v20("feat_sentence_")}},
    "pcenter_master_off_syllables_on": {
        "output": {"syllables_csv": True, "phonemes_csv": False},
        "features": {"master_syllable_pcenter": False, **_feat_v20("feat_syllable_")}},
}

_BASE_PLANS = {
    "psr_mat_master_off": (
        ['features', 'phoneme_surprisal_mat'],
        ['spacy_parse', 'mfa_align', 'syllabify_pairs', 'phoneme_surprisal'],
        {'phoneme_surprisal_mat': []}),
    "psr_mat_master_off_with_phonemes": (
        ['features', 'phoneme_surprisal_mat', 'phonemes'],
        ['spacy_parse', 'mfa_align', 'syllabify_pairs', 'phoneme_surprisal'],
        {'phoneme_surprisal_mat': [], 'phonemes': ['cohort_entropy', 'phoneme_surprisal']}),
    "psr_mat_master_on_no_phonemes": (
        ['features', 'phoneme_surprisal_mat'],
        ['spacy_parse', 'mfa_align', 'syllabify_pairs', 'phoneme_surprisal'],
        {'phoneme_surprisal_mat': []}),
    "syllables_only_psr_ticked": (
        ['features', 'syllables'],
        ['spacy_parse', 'mfa_align', 'praat_contours', 'syllabify_pairs', 'syllable_prosody', 'pcenter'],
        {'syllables': []}),
    "bert_sentence_master_off": (
        ['features'],
        ['spacy_parse'],
        {}),
    "bert_sentence_master_off_with_csv": (
        ['features', 'sentences'],
        ['spacy_parse', 'mfa_align', 'praat_contours', 'sentence_prosody'],
        {'sentences': []}),
    "pcenter_master_off_syllables_on": (
        ['features', 'syllables'],
        ['spacy_parse', 'mfa_align', 'praat_contours', 'syllabify_pairs', 'syllable_prosody', 'pcenter'],
        {'syllables': []}),
}


@pytest.mark.parametrize("name", sorted(_NONDEFAULT_MASTER_CASES))
def test_migrated_nondefault_masters_plan_like_v20(name):
    v20 = _deep_merge(deepcopy(DEFAULT_PREFS_V20), _NONDEFAULT_MASTER_CASES[name])
    merged, _ = _merged_from_user(v20)
    plan = plan_from_prefs(merged)
    csvs, stages, sidecar_drops = _BASE_PLANS[name]
    assert sorted(plan.output_csvs) == csvs
    assert list(plan.stages_to_run) == stages
    got = {k: sorted(v) for k, v in plan.column_drops.items() if k != "features"}
    assert got == sidecar_drops


def test_migration_keeps_psr_ticks_when_nothing_phoneme_was_written():
    # Syllables only: the section is switched off instead of unticking.
    v20 = _deep_merge(deepcopy(DEFAULT_PREFS_V20),
                      _NONDEFAULT_MASTER_CASES["syllables_only_psr_ticked"])
    f = _merged_from_user(v20)[0]["features"]
    assert f["master_phoneme_surprisal"] is False
    assert f["feat_phoneme_surprisal"] is True
    assert f["feat_phoneme_cohort_entropy"] is True


# --- Document aggregates pull in their source extractor --------------------

def _doc_only(**output) -> dict:
    prefs = deepcopy(DEFAULT_PREFS)
    prefs["output"].update(level_word=False, level_document=True, **output)
    return prefs


@pytest.mark.parametrize("toggle,stage", [
    ("doc_agg_zipf_freq", "word_freq"),
    ("doc_agg_lexical_norms", "lexical_norms"),
    ("doc_category_proportions", "sem_domain"),
])
def test_doc_aggregate_runs_source_with_word_level_off(toggle, stage):
    prefs = _doc_only(**{toggle: True})
    plan = plan_from_prefs(prefs)
    assert stage in plan.stages_to_run
    assert "features" not in plan.output_csvs
    assert plan.output_csvs == {"documents_summary"}
    ns = to_namespace(plan, prefs, level="doc")
    flag = {"word_freq": "no_freq", "lexical_norms": "no_lexical_norms",
            "sem_domain": "no_sem_domain"}[stage]
    assert getattr(ns, flag) is False


def test_doc_zipf_source_does_not_leak_into_word_csv():
    # Word level on with the Frequency section off: the doc pass computes
    # Zipf_Freq for its mean, the word pass neither runs nor keeps it.
    prefs = _doc_only(doc_agg_zipf_freq=True)
    prefs["output"]["level_word"] = True
    prefs["features"]["master_frequency"] = False
    plan = plan_from_prefs(prefs)
    assert "Zipf_Freq" in plan.column_drops["features"]
    assert to_namespace(plan, prefs, level="word").no_freq is True
    assert to_namespace(plan, prefs, level="doc").no_freq is False


def test_doc_edu_zipf_without_bands_skips_ewfg():
    prefs = _doc_only(doc_agg_edu_zipf=True)
    plan = plan_from_prefs(prefs)
    assert "ewfg_freq" in plan.stages_to_run
    ns = to_namespace(plan, prefs, level="doc")
    assert ns.no_doc_mean_edu_zipf is True and ns.no_edu_freq is True
    prefs["features"]["freq_ewfg_k2"] = True
    ns = to_namespace(plan_from_prefs(prefs), prefs, level="doc")
    assert ns.no_doc_mean_edu_zipf is False and ns.no_edu_freq is False
