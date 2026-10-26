"""Run-time estimate (utils/run_estimates.estimate_time / format_duration)."""
import copy

import pytest

from gui.planner.plan import plan_from_prefs
from utils import estimate_constants as K
from utils.preferences import DEFAULT_PREFS
from utils.run_estimates import InputStats, estimate_time, format_duration
from utils.timing_history import STEP_NAMES


def _all_on(models=("gpt2",)) -> dict:
    prefs = copy.deepcopy(DEFAULT_PREFS)
    for section in ("features", "output"):
        for k, v in list(prefs[section].items()):
            if isinstance(v, bool):
                prefs[section][k] = True
    prefs["predictability"]["models"] = list(models)
    prefs["predictability"]["metrics"] = {"surprisal": True, "cloze": True,
                                          "topmatch": True}
    return prefs


def _stats(**kw) -> InputStats:
    base = dict(n_stimuli=4, n_with_audio=4, words=2_000,
                words_with_audio=2_000, audio_seconds=600.0)
    base.update(kw)
    return InputStats(**base)


def _history(rates: dict, device="cpu") -> dict:
    """3 samples per key at the given s/unit; key = step or (step, model)."""
    samples = []
    for key, r in rates.items():
        step, model = key if isinstance(key, tuple) else (key, None)
        dev = device if step == "predictability" else "cpu"
        samples += [{"step": step, "model": model, "device": dev,
                     "units": 10.0, "seconds": 10.0 * r}] * 3
    return {"version": 1, "samples": samples, "load_seconds": {}}


def _every_step(rate=0.001, over=None) -> dict:
    rates = {s: rate for s in STEP_NAMES if s != "predictability"}
    rates[("predictability", "gpt2")] = rate
    rates.update(over or {})
    return rates


def test_first_estimate_uses_wide_spread():
    prefs = _all_on()
    e = estimate_time(plan_from_prefs(prefs), prefs, _stats(), {}, False)
    assert not e.learned
    lo, hi = K.FIRST_ESTIMATE_SPREAD
    assert e.low_s > 0
    assert e.high_s / e.low_s == pytest.approx(hi / lo)


def test_learned_rates_narrow_the_range():
    prefs = _all_on()
    e = estimate_time(plan_from_prefs(prefs), prefs, _stats(),
                      _history(_every_step()), False)
    assert e.learned
    lo, hi = K.LEARNED_SPREAD
    assert e.high_s / e.low_s == pytest.approx(hi / lo)


def test_cuda_history_not_used_for_cpu():
    prefs = _all_on(models=("llama32_3b",))
    rates = {s: 0.001 for s in STEP_NAMES if s != "predictability"}
    rates[("predictability", "llama32_3b")] = 0.001
    hist = _history(rates, device="cuda")     # predictability only on cuda
    plan = plan_from_prefs(prefs)
    assert not estimate_time(plan, prefs, _stats(), hist, False).learned
    assert estimate_time(plan, prefs, _stats(), hist, True).learned


def test_doc_pass_does_not_recharge_mfa_or_predictability():
    prefs = _all_on()
    no_doc = copy.deepcopy(prefs)
    no_doc["output"]["level_document"] = False
    big = 100.0
    small = 0.001
    hist = _history(_every_step(small, {"mfa_align": big,
                                ("predictability", "gpt2"): big}))
    hist["load_seconds"] = {k: 0.0 for k in K.FIRST_LOAD_SECONDS}
    s = _stats()
    with_doc = estimate_time(plan_from_prefs(prefs), prefs, s, hist, False)
    without = estimate_time(plan_from_prefs(no_doc), no_doc, s, hist, False)
    # The doc pass re-runs spaCy, the Praat contours and the doc summary
    # only (MFA from _align_cache / disk, predictability from _SCORE_CACHE).
    doc_pass = small * (s.words + s.audio_seconds + s.words)
    lo, _ = K.LEARNED_SPREAD
    assert with_doc.low_s - without.low_s == pytest.approx(lo * doc_pass)


def test_doc_pass_recharges_predictability_past_score_cache():
    # _SCORE_CACHE holds 256 (model, stimulus) entries (FIFO): a larger run
    # evicts every word-pass entry before the doc pass reads it.
    prefs = _all_on()
    no_doc = copy.deepcopy(prefs)
    no_doc["output"]["level_document"] = False
    hist = _history(_every_step(0.0, {("predictability", "gpt2"): 1.0}))
    hist["load_seconds"] = {k: 0.0 for k in K.FIRST_LOAD_SECONDS}
    s = _stats(n_stimuli=300, n_with_audio=300)
    with_doc = estimate_time(plan_from_prefs(prefs), prefs, s, hist, False)
    without = estimate_time(plan_from_prefs(no_doc), no_doc, s, hist, False)
    assert with_doc.low_s == pytest.approx(2 * without.low_s)


def test_text_only_items_not_charged_mfa():
    prefs = _all_on()
    rates = _every_step()
    del rates["mfa_align"]                     # no MFA history at all
    hist = _history(rates)
    e = estimate_time(plan_from_prefs(prefs), prefs,
                      _stats(n_with_audio=0, words_with_audio=0,
                             audio_seconds=0.0, n_audio_files=0,
                             all_audio_seconds=0.0),
                      hist, False)
    assert e.learned                           # MFA was never charged


def test_acoustic_pass_charges_every_audio_file():
    prefs = _all_on()
    prefs["output"] = {k: (k == "level_acoustic") if isinstance(v, bool) else v
                       for k, v in prefs["output"].items()}
    hist = _history({"acoustic_ts": 1.0})
    e = estimate_time(plan_from_prefs(prefs), prefs,
                      _stats(audio_seconds=10.0, all_audio_seconds=50.0),
                      hist, False)
    assert e.learned
    assert e.low_s == pytest.approx(K.LEARNED_SPREAD[0] * 50.0)


def test_format_duration_examples():
    assert format_duration(600, 2400) == "≈ 10–40 min"
    assert format_duration(1075, 1090) == "≈ 18 min"
    assert format_duration(20, 45) == "≈ 20–45 s"
    assert format_duration(3600, 4 * 3600) == "≈ 1–4 h"
    assert format_duration(5000, 21600) == "≈ 1.4–6 h"
    assert format_duration(5400, 5400) == "≈ 90 min"
    assert format_duration(20, 300) == "≈ 1–5 min"
