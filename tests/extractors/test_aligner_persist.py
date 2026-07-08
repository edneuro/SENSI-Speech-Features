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
