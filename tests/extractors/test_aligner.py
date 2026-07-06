"""Tests for extractors.aligner caching (memoization across passes).

The MFA subprocess is never invoked here — we monkeypatch the uncached compute
so the tests run without Kaldi/MFA and stay fast.
"""
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
                        lambda a, t: (calls.append((a, t)), _fake_result("x"))[1])
    aligner.align_with_phones("a.wav", "hello")
    aligner.align_with_phones("a.wav", "hello")
    assert len(calls) == 2, "without a cache, MFA must run on every call"


def test_cache_runs_mfa_once_per_document(monkeypatch):
    calls = []
    monkeypatch.setattr(aligner, "_align_uncached",
                        lambda a, t: (calls.append((a, t)), _fake_result("x"))[1])
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
                        lambda a, t: (calls.append((a, t)), _fake_result("x"))[1])
    cache: dict = {}
    aligner.align_with_phones("one.wav", "text one", cache=cache)
    aligner.align_with_phones("two.wav", "text two", cache=cache)
    assert len(calls) == 2, "different documents must each align once"


def test_cache_returns_independent_copies(monkeypatch):
    monkeypatch.setattr(aligner, "_align_uncached",
                        lambda a, t: _fake_result("original"))
    cache: dict = {}
    w1, _ = aligner.align_with_phones("s.wav", "t", cache=cache)
    # Mutating the returned frame must not corrupt the cached entry.
    w1.loc[0, "Onset"] = 999.0
    w2, _ = aligner.align_with_phones("s.wav", "t", cache=cache)
    assert w2.loc[0, "Onset"] == 0.0, "cache hit must return an uncorrupted copy"
