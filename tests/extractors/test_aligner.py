"""Tests for extractors.aligner caching (memoization across passes).

The MFA subprocess is never invoked here — we monkeypatch the uncached compute
so the tests run without Kaldi/MFA and stay fast.
"""
import json

import pandas as pd

from extractors import aligner


def _fake_result(tag):
    words = pd.DataFrame({"Transcript": ["a"], "Onset": [0.0], "End": [0.1]})
    phones = pd.DataFrame({"phone": ["AH0"], "onset": [0.0], "offset": [0.1]})
    words.attrs["tag"] = tag  # marker to prove identity vs copy
    return words, phones


def test_no_cache_calls_mfa_every_time(monkeypatch):
    calls = []
    monkeypatch.setattr(aligner, "_align_uncached",
                        lambda a, t, g2p=False: (calls.append((a, t)), _fake_result("x"))[1])
    aligner.align_with_phones("a.wav", "hello")
    aligner.align_with_phones("a.wav", "hello")
    assert len(calls) == 2, "without a cache, MFA must run on every call"


def test_cache_runs_mfa_once_per_document(monkeypatch):
    calls = []
    monkeypatch.setattr(aligner, "_align_uncached",
                        lambda a, t, g2p=False: (calls.append((a, t)), _fake_result("x"))[1])
    cache: dict = {}
    # Same (audio, transcript) twice — simulates the word pass then the doc pass.
    w1, p1 = aligner.align_with_phones("stim.wav", "hello world", cache=cache)
    w2, p2 = aligner.align_with_phones("stim.wav", "hello world", cache=cache)
    assert len(calls) == 1, "MFA must run exactly once per document when cached"
    # Same data served both times.
    pd.testing.assert_frame_equal(w1, w2)
    pd.testing.assert_frame_equal(p1, p2)


def test_cache_distinguishes_documents(monkeypatch):
    calls = []
    monkeypatch.setattr(aligner, "_align_uncached",
                        lambda a, t, g2p=False: (calls.append((a, t)), _fake_result("x"))[1])
    cache: dict = {}
    aligner.align_with_phones("one.wav", "text one", cache=cache)
    aligner.align_with_phones("two.wav", "text two", cache=cache)
    assert len(calls) == 2, "different documents must each align once"


def test_cache_returns_independent_copies(monkeypatch):
    monkeypatch.setattr(aligner, "_align_uncached",
                        lambda a, t, g2p=False: _fake_result("original"))
    cache: dict = {}
    w1, _ = aligner.align_with_phones("s.wav", "t", cache=cache)
    # Mutating the returned frame must not corrupt the cached entry.
    w1.loc[0, "Onset"] = 999.0
    w2, _ = aligner.align_with_phones("s.wav", "t", cache=cache)
    assert w2.loc[0, "Onset"] == 0.0, "cache hit must return an uncorrupted copy"


def test_sidecar_records_g2p_and_version_2(tmp_path, monkeypatch):
    audio = tmp_path / "s.wav"
    audio.write_bytes(b"\0" * 16)
    monkeypatch.setattr(aligner, "_sha256_file", lambda p, _chunk=1: "deadbeef")

    meta_off = aligner._freshness_dict(str(audio), "hello", g2p=False)
    meta_on = aligner._freshness_dict(str(audio), "hello", g2p=True)

    assert meta_off["version"] == 2
    assert meta_off["mfa_g2p"] is False
    assert meta_on["mfa_g2p"] is True


def test_cached_alignment_is_stale_when_g2p_setting_differs(tmp_path, monkeypatch):
    audio = tmp_path / "s.wav"
    audio.write_bytes(b"\0" * 16)
    monkeypatch.setattr(aligner, "_sha256_file", lambda p, _chunk=1: "deadbeef")
    (tmp_path / "s.TextGrid").write_text("x", encoding="utf-8")
    (tmp_path / "s.json").write_text(
        json.dumps(aligner._freshness_dict(str(audio), "hello", g2p=False)),
        encoding="utf-8")

    assert aligner._is_fresh(tmp_path, "s", str(audio), "hello", g2p=False)
    assert not aligner._is_fresh(tmp_path, "s", str(audio), "hello", g2p=True)


def test_run_mfa_passes_g2p_flag_only_when_requested(monkeypatch):
    calls = []

    class _Result:
        returncode = 0
        stdout = ""
        stderr = ""

    monkeypatch.setattr(aligner.subprocess, "run",
                        lambda cmd, **kw: calls.append(cmd) or _Result())
    monkeypatch.setattr(aligner, "g2p_model_available", lambda: True)

    aligner._run_mfa("corpus", "out", g2p=False)
    assert "--g2p_model_path" not in calls[0]

    aligner._run_mfa("corpus", "out", g2p=True)
    assert "--g2p_model_path" in calls[1]
    assert calls[1][calls[1].index("--g2p_model_path") + 1].endswith(".zip")


def test_g2p_requested_but_model_missing_is_a_warning_not_a_crash(monkeypatch, capsys):
    class _Result:
        returncode = 0
        stdout = ""
        stderr = ""

    calls = []
    monkeypatch.setattr(aligner.subprocess, "run",
                        lambda cmd, **kw: calls.append(cmd) or _Result())
    monkeypatch.setattr(aligner, "g2p_model_available", lambda: False)

    aligner._run_mfa("corpus", "out", g2p=True)
    assert "--g2p_model_path" not in calls[0]
    assert "g2p" in capsys.readouterr().err.lower()
