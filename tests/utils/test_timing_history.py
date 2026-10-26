"""Machine-local timing history: per-(step, model, device) rate maths + store."""
import json

import pytest

from utils import timing_history as th
from utils.timing_history import Sample


def _samples(n, *, step="spacy_parse", model=None, device="cpu", rate=0.5):
    return [Sample(step=step, model=model, device=device,
                   units=10.0, seconds=10.0 * rate) for _ in range(n)]


def test_rate_needs_three_samples(tmp_path):
    path = tmp_path / "h.json"
    th.append_samples(_samples(2), path=path)
    assert th.rate(th.load_history(path), "spacy_parse", None, "cpu") is None
    th.append_samples(_samples(1), path=path)
    assert th.rate(th.load_history(path), "spacy_parse", None, "cpu") == pytest.approx(0.5)


def test_rate_is_median_per_key(tmp_path):
    path = tmp_path / "h.json"
    cpu = [Sample("predictability", "gpt2", "cpu", units=100.0, seconds=s)
           for s in (10.0, 20.0, 90.0)]          # per-unit 0.1, 0.2, 0.9 -> median 0.2
    cuda = [Sample("predictability", "gpt2", "cuda", units=100.0, seconds=s)
            for s in (1.0, 2.0, 3.0, 4.0)]       # per-unit 0.01..0.04 -> median 0.025
    th.append_samples(cpu + cuda, path=path)
    h = th.load_history(path)
    assert th.rate(h, "predictability", "gpt2", "cpu") == pytest.approx(0.2)
    assert th.rate(h, "predictability", "gpt2", "cuda") == pytest.approx(0.025)
    assert th.rate(h, "predictability", "llama-3.2-1b", "cpu") is None
    assert th.rate(h, "predictability", None, "cpu") is None


def test_corrupt_history_file_reads_as_empty(tmp_path):
    path = tmp_path / "h.json"
    path.write_bytes(b"\x00\xff{not json at all")
    h = th.load_history(path)
    assert h == {}
    assert th.rate(h, "spacy_parse", None, "cpu") is None
    # A later append recovers rather than raising.
    th.append_samples(_samples(3), path=path)
    assert th.rate(th.load_history(path), "spacy_parse", None, "cpu") == pytest.approx(0.5)


def test_missing_history_file_reads_as_empty(tmp_path):
    assert th.load_history(tmp_path / "absent.json") == {}


def test_history_caps_200_per_key(tmp_path):
    path = tmp_path / "h.json"
    old = _samples(150, rate=9.0)
    new = _samples(150, rate=1.0)
    other = _samples(5, step="bert")
    th.append_samples(old, path=path)
    th.append_samples(new + other, path=path)
    h = th.load_history(path)
    spacy = [s for s in h["samples"] if s["step"] == "spacy_parse"]
    assert len(spacy) == 200
    # The most recent 200 are kept: all 150 new ones + the newest 50 old ones.
    assert sum(1 for s in spacy if s["seconds"] == 10.0) == 150
    assert len([s for s in h["samples"] if s["step"] == "bert"]) == 5


def test_history_respects_config_dir(tmp_path, monkeypatch):
    cfg = tmp_path / "cfgdir"
    monkeypatch.setenv("SENSI_CONFIG_DIR", str(cfg))
    th.append_samples(_samples(3))
    target = cfg / "timing_history.json"
    assert target.is_file()
    assert json.loads(target.read_text(encoding="utf-8"))["samples"]
    assert th.rate(th.load_history(), "spacy_parse", None, "cpu") == pytest.approx(0.5)


def test_load_seconds_are_stored_separately(tmp_path):
    path = tmp_path / "h.json"
    th.append_samples(_samples(1), path=path,
                      load_seconds={"predictability:gpt2": 4.5})
    h = th.load_history(path)
    assert h["load_seconds"] == {"predictability:gpt2": 4.5}
    assert all(s["step"] == "spacy_parse" for s in h["samples"])


def test_stage_timer_records_sample_and_skips_on_error():
    timer = th.StageTimer()
    with timer.stage("spacy_parse", 12, device="cpu"):
        pass
    with pytest.raises(ValueError):
        with timer.stage("bert", 12):
            raise ValueError("boom")
    assert [s.step for s in timer.samples] == ["spacy_parse"]
    s = timer.samples[0]
    assert s.units == 12 and s.model is None and s.device == "cpu" and s.seconds >= 0


def test_model_device_rule():
    # GPT-2 is pinned to full-precision CPU by the extractor even with CUDA.
    assert th.model_device("predictability", "gpt2", True) == "cpu"
    # Other roster models go to the GPU exactly when CUDA is available.
    from extractors.predictability_models import ALL_SPECS
    other = next(s.slug for s in ALL_SPECS if s.hf_id != "gpt2")
    assert th.model_device("predictability", other, True) == "cuda"
    assert th.model_device("predictability", other, False) == "cpu"
    # BERT is never moved to CUDA by extractors/bert_embed.py.
    assert th.model_device("bert", None, True) == "cpu"
    for step in ("spacy_parse", "mfa_align", "word2vec", "acoustic_ts"):
        assert th.model_device(step, None, True) == "cpu"
    # Unknown slug: no GPU claim.
    assert th.model_device("predictability", "no-such-model", True) == "cpu"


# --- MFA fixed overhead + per-audio-second slope (fit_overhead_slope) -------

def _mfa(units, seconds):
    return {"step": "mfa_align", "model": None, "device": "cpu",
            "units": float(units), "seconds": float(seconds)}


def test_fit_recovers_known_overhead_and_slope():
    hist = {"samples": [_mfa(u, 80.0 + 0.25 * u) for u in (10, 40, 120, 400)]}
    overhead, slope = th.fit_overhead_slope(hist, "mfa_align")
    assert overhead == pytest.approx(80.0)
    assert slope == pytest.approx(0.25)


def test_fit_needs_two_clearly_different_sizes():
    one = {"samples": [_mfa(100, 110)]}
    same = {"samples": [_mfa(100, 110), _mfa(120, 112), _mfa(100, 108)]}
    assert th.fit_overhead_slope(one, "mfa_align") is None
    assert th.fit_overhead_slope(same, "mfa_align") is None      # max/min < 2
    two = {"samples": [_mfa(100, 110), _mfa(200, 135)]}
    assert th.fit_overhead_slope(two, "mfa_align") is None       # < 3 samples
    three = {"samples": [_mfa(100, 110), _mfa(200, 135), _mfa(150, 122.5)]}
    assert th.fit_overhead_slope(three, "mfa_align") == pytest.approx((85.0, 0.25))


def test_clamped_fit_is_unlearned():
    # Noisy top-ups: a bigger batch was faster (slope <= 0), or the line runs
    # below the origin -- noise, not a rate, so the fit stays unlearned.
    neg_slope = {"samples": [_mfa(10, 90), _mfa(100, 80), _mfa(50, 85)]}
    assert th.fit_overhead_slope(neg_slope, "mfa_align") is None
    neg_icpt = {"samples": [_mfa(10, 1), _mfa(100, 100), _mfa(50, 50)]}
    assert th.fit_overhead_slope(neg_icpt, "mfa_align") is None


def test_max_units():
    hist = {"samples": [_mfa(10, 90), _mfa(100, 80)]}
    assert th.max_units(hist, "mfa_align") == 100.0
    assert th.max_units({}, "mfa_align") == 0.0


def test_fit_reads_old_format_history(tmp_path):
    # Version-1 files (one mfa_align Sample per MFA batch, units = audio s)
    # load unchanged and feed the fit.
    path = tmp_path / "h.json"
    path.write_text(json.dumps({"version": 1, "samples": [
        _mfa(113.8, 114.4), _mfa(16.0, 90.6), _mfa(10.1, 88.9)],
        "load_seconds": {}}),
        encoding="utf-8")
    fit = th.fit_overhead_slope(th.load_history(path), "mfa_align")
    assert fit is not None and fit[0] > 0 and fit[1] > 0


def test_count_words_is_raw_whitespace_split():
    assert th.count_words("I won 1,500   points\n") == 4
