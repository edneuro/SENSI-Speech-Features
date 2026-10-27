"""Batch runners record per-step timings of successful stimuli only."""
import pytest

from sensi import pipeline
from tests.batch_args import build_parser
from sensi.utils import timing_history as th

_TEXT = "The cat sat on the mat. It was very happy today."


def _batch_args(tmp_path):
    tdir = tmp_path / "transcripts"
    tdir.mkdir()
    (tdir / "stim.txt").write_text(_TEXT, encoding="utf-8")
    return build_parser().parse_args([
        "--transcripts-folder", str(tdir),
        "--output-folder", str(tmp_path / "out"),
        "--analyzer", "t", "--no-align", "--no-freq", "--no-sem-domain",
    ])


def test_successful_pair_records_samples(tmp_path):
    pipeline.run_word_level_batch(_batch_args(tmp_path))
    hist = th.load_history()   # SENSI_CONFIG_DIR -> tmp_path (conftest)
    assert (tmp_path / "timing_history.json").is_file()
    spacy = [s for s in hist["samples"] if s["step"] == "spacy_parse"]
    assert len(spacy) == 1
    assert spacy[0]["units"] == len(_TEXT.split())
    assert spacy[0]["device"] == "cpu" and spacy[0]["model"] is None
    assert spacy[0]["seconds"] >= 0
    assert {s["step"] for s in hist["samples"]} <= set(th.STEP_NAMES)


def test_failed_pair_records_nothing(tmp_path, monkeypatch):
    def _boom(args):
        raise RuntimeError("extractor failed")
    monkeypatch.setattr(pipeline, "run", _boom)
    pipeline.run_word_level_batch(_batch_args(tmp_path))
    assert th.load_history() == {}


def test_no_timer_means_no_history(tmp_path):
    """Direct run() callers (CLI single run) carry no _stage_timer: no-op."""
    args = _batch_args(tmp_path)
    args.transcript = str(tmp_path / "transcripts" / "stim.txt")
    args.audio = ""
    pipeline.run(args)
    assert not (tmp_path / "timing_history.json").exists()


def test_timed_predictability_matches_single_call_and_times_load(monkeypatch):
    """The per-model split used when timing yields the same frame as the
    original multi-model call; first loads go to load_seconds; a doc-pass
    score-cache hit records no sample."""
    import types
    import pandas as pd
    from sensi.extractors import predictability as P
    from sensi.extractors.predictability_models import ALL_SPECS

    class _Fake:
        def __init__(self, model_id, max_context=None):
            self.k = len(model_id)

        def score(self, text, tokens):
            return [{"surprisal": float(i + self.k), "p": 0.5 / (i + 1),
                     "top_match": i % 2} for i in range(len(tokens))]

    monkeypatch.setattr(P, "_make_default_backend", _Fake)
    monkeypatch.setattr(P, "_BACKEND_CACHE", {})
    monkeypatch.setattr(P, "_SCORE_CACHE", type(P._SCORE_CACHE)())
    slugs = [s.slug for s in ALL_SPECS[:2]]
    metrics = ["surprisal", "cloze", "topmatch"]
    df = pd.DataFrame({"token": ["a", "b", "."], "upos": ["DET", "NOUN", "PUNCT"]})
    want = P.add_predictability(df, "a b.", models=slugs, metrics=metrics)

    monkeypatch.setattr(P, "_BACKEND_CACHE", {})
    monkeypatch.setattr(P, "_SCORE_CACHE", type(P._SCORE_CACHE)())
    args = types.SimpleNamespace(_stage_timer=th.StageTimer())
    got = pipeline._timed_predictability(args, df, "a b.", slugs, metrics, None)
    pd.testing.assert_frame_equal(got, want)
    assert [s.model for s in args._stage_timer.samples] == slugs
    assert all(s.step == "predictability" and s.units == 2
               for s in args._stage_timer.samples)
    assert set(args._stage_timer.load_seconds) == {
        th.load_key("predictability", s) for s in slugs}

    again = types.SimpleNamespace(_stage_timer=th.StageTimer())
    got2 = pipeline._timed_predictability(again, df, "a b.", slugs, metrics, None)
    pd.testing.assert_frame_equal(got2, want)
    assert again._stage_timer.samples == [] and again._stage_timer.load_seconds == {}


def test_mfa_sample_counts_only_aligned_audio(tmp_path):
    import types
    import numpy as np
    import soundfile as sf
    wavs = []
    for stem, secs in (("a", 2.0), ("b", 3.0)):
        p = tmp_path / f"{stem}.wav"
        sf.write(str(p), np.zeros(int(16000 * secs)), 16000)
        wavs.append(str(p))
    items = [(w, "x") for w in wavs]
    args = types.SimpleNamespace(_stage_timer=th.StageTimer())
    pipeline._record_mfa_timing(args, items, {"aligned": ["a"], "cached": ["b"],
                                          "failed": []}, 7.0)
    (s,) = th.load_history()["samples"]
    assert (s["step"], s["units"], s["seconds"]) == ("mfa_align", 2.0, 7.0)
    # Any failure, an all-cached pre-pass, or no timer: no sample.
    pipeline._record_mfa_timing(args, items, {"aligned": ["a"], "failed": ["b"]}, 7.0)
    pipeline._record_mfa_timing(args, items, {"aligned": [], "cached": ["a", "b"]}, 1.0)
    pipeline._record_mfa_timing(types.SimpleNamespace(), items, {"aligned": ["a"]}, 1.0)
    assert len(th.load_history()["samples"]) == 1
