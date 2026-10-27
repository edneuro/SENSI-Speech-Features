"""Run-time estimate (utils/run_estimates.estimate_time / format_duration)."""
import copy

import pytest

from sensi.gui.planner.plan import plan_from_prefs
from sensi.utils import estimate_constants as K
from sensi.utils.preferences import DEFAULT_PREFS
from sensi.utils.run_estimates import InputStats, estimate_time, format_duration
from sensi.utils.timing_history import STEP_NAMES


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


_REAL_STARTUP = K.STARTUP_SECONDS


@pytest.fixture(autouse=True)
def _no_startup(monkeypatch):
    """Exact-value tests below exclude the fixed start-up allowance
    (test_startup_allowance_added_once checks it)."""
    monkeypatch.setattr(K, "STARTUP_SECONDS", 0.0)


def _history(rates: dict, device="cpu") -> dict:
    """3 samples per key at the given s/unit; key = step or (step, model).
    mfa_align gets batch sizes 10 / 1000 / 10 audio s, so its overhead+slope
    fit is learned (overhead 0, slope = the rate) and covers the test corpora."""
    samples = []
    for key, r in rates.items():
        step, model = key if isinstance(key, tuple) else (key, None)
        dev = device if step == "predictability" else "cpu"
        sizes = (10.0, 1000.0, 10.0) if step == "mfa_align" else (10.0,) * 3
        samples += [{"step": step, "model": model, "device": dev,
                     "units": u, "seconds": u * r} for u in sizes]
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
    s = _stats(n_stimuli=300, n_with_audio=300,
               audio_seconds_to_align=0.0)    # MFA cost is not the point here
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
    assert format_duration(20, 300) == "≈ 20 s–5 min"


def test_fresh_alignment_not_charged_mfa():
    # audio_seconds_to_align = 0: every paired stimulus already has a fresh
    # TextGrid on disk (extractors.aligner.is_alignment_fresh), so the run
    # never starts MFA; None = unknown, charge every paired audio second.
    prefs = _all_on()
    plan = plan_from_prefs(prefs)
    hist = _history(_every_step(0.0, {"mfa_align": 1.0}))
    hist["load_seconds"] = {k: 0.0 for k in K.FIRST_LOAD_SECONDS}
    lo, _ = K.LEARNED_SPREAD
    unknown = estimate_time(plan, prefs, _stats(), hist, False)
    assert unknown.low_s == pytest.approx(lo * 600.0)
    fresh = estimate_time(plan, prefs, _stats(audio_seconds_to_align=0.0),
                          hist, False)
    assert fresh.low_s == pytest.approx(0.0)
    part = estimate_time(plan, prefs, _stats(audio_seconds_to_align=150.0),
                         hist, False)
    assert part.low_s == pytest.approx(lo * 150.0)


def test_scan_inputs_audio_seconds_to_align(tmp_path, monkeypatch):
    import numpy as np
    import soundfile as sf
    from sensi.extractors import aligner
    from sensi.utils.run_estimates import scan_inputs

    tdir, adir, out = tmp_path / "t", tmp_path / "a", tmp_path / "out"
    for d in (tdir, adir, out):
        d.mkdir()
    (tdir / "s1.txt").write_text("hello there", encoding="utf-8")
    (tdir / "s2.txt").write_text("general kenobi", encoding="utf-8")
    for name, secs in (("s1", 2.0), ("s2", 3.0)):
        sf.write(str(adir / f"{name}.wav"),
                 np.zeros(int(secs * 8000), dtype=np.float32), 8000)
    # Fake TextGrids aren't parseable; treat them as readable here.
    monkeypatch.setattr(aligner, "_readable", lambda p: True)

    assert scan_inputs(str(tdir), str(adir)).audio_seconds_to_align is None
    st = scan_inputs(str(tdir), str(adir), output_folder=str(out))
    assert st.audio_seconds_to_align == pytest.approx(5.0)   # nothing on disk

    persist = aligner.alignment_dir(out)
    persist.mkdir()
    (persist / "s1.TextGrid").write_text("fake tg", encoding="utf-8")
    aligner._write_sidecar(persist, "s1", str(adir / "s1.wav"), "hello there")
    st = scan_inputs(str(tdir), str(adir), output_folder=str(out))
    assert st.audio_seconds_to_align == pytest.approx(3.0)   # s1 fresh
    # A different g2p setting makes the saved alignment stale.
    st = scan_inputs(str(tdir), str(adir), output_folder=str(out), g2p=True)
    assert st.audio_seconds_to_align == pytest.approx(5.0)
    # Editing the transcript makes it stale too.
    (tdir / "s1.txt").write_text("hello again", encoding="utf-8")
    st = scan_inputs(str(tdir), str(adir), output_folder=str(out))
    assert st.audio_seconds_to_align == pytest.approx(5.0)


# --- MFA: fixed overhead per aligning run + per-audio-second slope ----------


def _mfa_hist(points):
    hist = _history(_every_step(0.0))
    hist["samples"] = [s for s in hist["samples"] if s["step"] != "mfa_align"]
    hist["samples"] += [{"step": "mfa_align", "model": None, "device": "cpu",
                         "units": float(u), "seconds": float(s)} for u, s in points]
    hist["load_seconds"] = {k: 0.0 for k in K.FIRST_LOAD_SECONDS}
    return hist


def test_small_mfa_batches_do_not_blow_up_large_corpus():
    # Top-up batches of 5 and 20 s, overhead 80 s + 0.2 s/s: the old median
    # rate (~8 s per audio s) would charge 10 h of audio ~80 h. 10 h is far
    # beyond the 20 s batches the fit saw, so the range stays wide.
    hist = _mfa_hist([(5, 81), (20, 84), (5, 81)])
    prefs = _all_on()
    e = estimate_time(plan_from_prefs(prefs), prefs,
                      _stats(audio_seconds=36_000.0,
                             audio_seconds_to_align=36_000.0), hist, False)
    assert not e.learned
    lo, _ = K.FIRST_ESTIMATE_SPREAD
    assert e.low_s == pytest.approx(lo * (80.0 + 0.2 * 36_000.0))


def test_mfa_overhead_charged_once_for_a_small_run():
    hist = _mfa_hist([(10, 82), (100, 100), (50, 90)])
    prefs = _all_on()
    e = estimate_time(plan_from_prefs(prefs), prefs,
                      _stats(audio_seconds_to_align=5.0), hist, False)
    lo, _ = K.LEARNED_SPREAD
    assert e.learned
    assert e.low_s == pytest.approx(lo * (80.0 + 0.2 * 5.0))


def test_single_size_mfa_history_stays_unlearned():
    hist = _mfa_hist([(100, 110), (110, 111), (100, 109)])
    prefs = _all_on()
    e = estimate_time(plan_from_prefs(prefs), prefs,
                      _stats(audio_seconds_to_align=100.0), hist, False)
    assert not e.learned
    lo, _ = K.FIRST_ESTIMATE_SPREAD
    assert e.low_s == pytest.approx(lo * (K.FIRST_MFA_OVERHEAD_SECONDS
                                      + K.FIRST_ESTIMATE_RATES["mfa_align"] * 100.0))


def test_unlearned_mfa_not_charged_keeps_learned():
    hist = _mfa_hist([(100, 110)])
    e = estimate_time(plan_from_prefs(_all_on()), _all_on(),
                      _stats(audio_seconds_to_align=0.0), hist, False)
    assert e.learned


def test_startup_allowance_added_once(monkeypatch):
    monkeypatch.setattr(K, "STARTUP_SECONDS", _REAL_STARTUP)
    prefs = _all_on()
    hist = _history(_every_step(0.0))
    hist["load_seconds"] = {k: 0.0 for k in K.FIRST_LOAD_SECONDS}
    e = estimate_time(plan_from_prefs(prefs), prefs,
                      _stats(audio_seconds_to_align=0.0), hist, False)
    lo, hi = K.LEARNED_SPREAD
    assert K.STARTUP_SECONDS == pytest.approx(10.0)
    assert e.low_s == pytest.approx(lo * K.STARTUP_SECONDS)
    assert e.high_s == pytest.approx(hi * K.STARTUP_SECONDS)


def test_format_duration_keeps_range_across_units():
    assert format_duration(20, 61) == "≈ 20 s–1 min"
    assert format_duration(59.6, 59.8) == "≈ 1 min"
    assert format_duration(1800, 6 * 3600) == "≈ 30 min–6 h"
    assert format_duration(20 * 3600, 3 * 86400) == "≈ 20 h–3 days"
    assert format_duration(3 * 86400, 5 * 86400) == "≈ 3–5 days"
    assert format_duration(40 * 3600, 48 * 3600) == "≈ 40–48 h"
