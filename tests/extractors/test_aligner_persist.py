"""Tests for persistent on-disk MFA alignment (freshness + batch pre-pass).

The MFA subprocess is never invoked here: freshness tests use real temp files,
and the batch tests monkeypatch _run_mfa to synthesize TextGrids.
"""
import json
import os

import pandas as pd
import pytest

from extractors import aligner


def _make_audio(tmp_path, name="stim.wav", data=b"RIFFfake-audio-bytes"):
    p = tmp_path / name
    p.write_bytes(data)
    return str(p)


def test_alignment_dir_appends_named_subfolder(tmp_path):
    d = aligner.alignment_dir(tmp_path)
    assert d.name == "Montreal Forced Alignment"
    assert d.parent == tmp_path


def test_stem_from_audio_basename():
    assert aligner._stem("/a/b/story_01.wav") == "story_01"


def test_sha256_text_is_stable_and_prepared():
    # Hash is over the prepared (whitespace-normalized, numbers-spelled) form.
    a = aligner._sha256_text(aligner_prepared("There were 2  dogs."))
    b = aligner._sha256_text(aligner_prepared("There were two dogs."))
    assert a == b


def aligner_prepared(t):
    from utils.transcript_prep import prepare_for_aligner
    return prepare_for_aligner(t)


def test_write_and_is_fresh_roundtrip(tmp_path):
    persist = tmp_path / "align"
    persist.mkdir()
    audio = _make_audio(tmp_path)
    (persist / "stim.TextGrid").write_text("fake tg", encoding="utf-8")
    aligner._write_sidecar(persist, "stim", audio, "hello world")
    assert aligner._is_fresh(persist, "stim", audio, "hello world") is True


def test_is_fresh_false_when_textgrid_missing(tmp_path):
    persist = tmp_path / "align"
    persist.mkdir()
    audio = _make_audio(tmp_path)
    aligner._write_sidecar(persist, "stim", audio, "hello")  # sidecar only
    assert aligner._is_fresh(persist, "stim", audio, "hello") is False


def test_is_fresh_false_on_transcript_change(tmp_path):
    persist = tmp_path / "align"
    persist.mkdir()
    audio = _make_audio(tmp_path)
    (persist / "stim.TextGrid").write_text("fake tg", encoding="utf-8")
    aligner._write_sidecar(persist, "stim", audio, "hello")
    assert aligner._is_fresh(persist, "stim", audio, "goodbye") is False


def test_is_fresh_false_on_audio_change(tmp_path):
    persist = tmp_path / "align"
    persist.mkdir()
    audio = _make_audio(tmp_path)
    (persist / "stim.TextGrid").write_text("fake tg", encoding="utf-8")
    aligner._write_sidecar(persist, "stim", audio, "hello")
    # Rewrite audio bytes (same length => exercises the content hash, not size).
    with open(audio, "wb") as f:
        f.write(b"RIFFdifferent-bytes!")
    assert aligner._is_fresh(persist, "stim", audio, "hello") is False


def test_is_fresh_false_on_model_change(tmp_path, monkeypatch):
    persist = tmp_path / "align"
    persist.mkdir()
    audio = _make_audio(tmp_path)
    (persist / "stim.TextGrid").write_text("fake tg", encoding="utf-8")
    aligner._write_sidecar(persist, "stim", audio, "hello")
    monkeypatch.setattr(aligner, "MFA_ACOUSTIC_MODEL", "some_other_model")
    assert aligner._is_fresh(persist, "stim", audio, "hello") is False


def _fake_frames():
    words = pd.DataFrame({"Transcript": ["a"], "Onset": [0.0], "End": [0.1]})
    phones = pd.DataFrame({"phone": ["AH0"], "onset": [0.0], "offset": [0.1]})
    return words, phones


def test_disk_hit_parses_without_running_mfa(tmp_path, monkeypatch):
    persist = tmp_path / "align"
    persist.mkdir()
    (persist / "stim.TextGrid").write_text("fake tg", encoding="utf-8")
    called = []
    monkeypatch.setattr(aligner, "_align_uncached",
                        lambda a, t: called.append(1))
    monkeypatch.setattr(aligner, "_parse_textgrid", lambda p: _fake_frames())
    w, p = aligner.align_with_phones(
        str(tmp_path / "stim.wav"), "hello", persist_dir=persist)
    assert called == [], "disk hit must not invoke MFA"
    assert list(w["Onset"]) == [0.0]


def test_run_if_missing_false_raises_on_disk_miss(tmp_path, monkeypatch):
    persist = tmp_path / "align"
    persist.mkdir()  # empty: no TextGrid for this stem
    monkeypatch.setattr(aligner, "_align_uncached",
                        lambda a, t: pytest.fail("MFA must not run"))
    with pytest.raises(RuntimeError):
        aligner.align_with_phones(
            str(tmp_path / "stim.wav"), "hello",
            persist_dir=persist, run_if_missing=False)


def test_run_if_missing_true_falls_back_to_mfa(tmp_path, monkeypatch):
    persist = tmp_path / "align"
    persist.mkdir()
    monkeypatch.setattr(aligner, "_align_uncached", lambda a, t: _fake_frames())
    w, _ = aligner.align_with_phones(
        str(tmp_path / "stim.wav"), "hello",
        persist_dir=persist, run_if_missing=True)
    assert list(w["Onset"]) == [0.0]
