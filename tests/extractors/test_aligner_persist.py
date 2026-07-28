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
                        lambda a, t, g2p=False: called.append(1))
    monkeypatch.setattr(aligner, "_parse_textgrid", lambda p: _fake_frames())
    w, p = aligner.align_with_phones(
        str(tmp_path / "stim.wav"), "hello", persist_dir=persist)
    assert called == [], "disk hit must not invoke MFA"
    assert list(w["Onset"]) == [0.0]


def test_run_if_missing_false_raises_on_disk_miss(tmp_path, monkeypatch):
    persist = tmp_path / "align"
    persist.mkdir()  # empty: no TextGrid for this stem
    monkeypatch.setattr(aligner, "_align_uncached",
                        lambda a, t, g2p=False: pytest.fail("MFA must not run"))
    with pytest.raises(RuntimeError):
        aligner.align_with_phones(
            str(tmp_path / "stim.wav"), "hello",
            persist_dir=persist, run_if_missing=False)


def test_run_if_missing_true_falls_back_to_mfa(tmp_path, monkeypatch):
    persist = tmp_path / "align"
    persist.mkdir()
    monkeypatch.setattr(aligner, "_align_uncached",
                        lambda a, t, g2p=False: _fake_frames())
    w, _ = aligner.align_with_phones(
        str(tmp_path / "stim.wav"), "hello",
        persist_dir=persist, run_if_missing=True)
    assert list(w["Onset"]) == [0.0]


def _install_fake_mfa(monkeypatch, aligned_stems=None):
    """Replace _run_mfa with one that writes a TextGrid per staged .lab.

    aligned_stems=None => align everything staged; otherwise only those stems
    get a TextGrid (simulating per-stimulus alignment failure).
    """
    runs = {"count": 0}

    def fake(corpus_dir, output_dir, g2p=False):
        runs["count"] += 1
        for name in os.listdir(corpus_dir):
            if not name.endswith(".lab"):
                continue
            stem = name[:-4]
            if aligned_stems is not None and stem not in aligned_stems:
                continue
            with open(os.path.join(output_dir, f"{stem}.TextGrid"), "w") as f:
                f.write(f"tg for {stem}")

    monkeypatch.setattr(aligner, "_run_mfa", fake)
    return runs


def test_ensure_aligns_all_missing_in_one_run(tmp_path, monkeypatch):
    persist = tmp_path / "align"
    a1 = _make_audio(tmp_path, "one.wav", b"aaaa")
    a2 = _make_audio(tmp_path, "two.wav", b"bbbb")
    runs = _install_fake_mfa(monkeypatch)
    res = aligner.ensure_alignments([(a1, "hi one"), (a2, "hi two")], persist)
    assert runs["count"] == 1, "one MFA invocation for the whole batch"
    assert set(res["aligned"]) == {"one", "two"}
    assert (persist / "one.TextGrid").exists()
    assert (persist / "two.json").exists()


def test_ensure_skips_mfa_when_all_fresh(tmp_path, monkeypatch):
    persist = tmp_path / "align"
    a1 = _make_audio(tmp_path, "one.wav", b"aaaa")
    _install_fake_mfa(monkeypatch)  # first call populates disk
    aligner.ensure_alignments([(a1, "hi one")], persist)
    runs2 = _install_fake_mfa(monkeypatch)  # fresh counter
    res = aligner.ensure_alignments([(a1, "hi one")], persist)
    assert runs2["count"] == 0, "no MFA when everything is fresh"
    assert res["cached"] == ["one"] and res["aligned"] == []


def test_ensure_deletes_stale_before_realign(tmp_path, monkeypatch):
    persist = tmp_path / "align"
    a1 = _make_audio(tmp_path, "one.wav", b"aaaa")
    _install_fake_mfa(monkeypatch)
    aligner.ensure_alignments([(a1, "old transcript")], persist)
    # New transcript => stale. MFA now "fails" for this stem (produces nothing).
    _install_fake_mfa(monkeypatch, aligned_stems=set())
    res = aligner.ensure_alignments([(a1, "new transcript")], persist)
    assert res["failed"] == ["one"]
    assert not (persist / "one.TextGrid").exists(), \
        "stale TextGrid must be removed, not left for reuse"


def test_ensure_alignments_forwards_g2p_value_to_run_mfa(tmp_path, monkeypatch):
    """Final-review Fix 4: pins VALUE-correct g2p forwarding from the public
    ensure_alignments API down to _run_mfa (existing fakes accept g2p=False
    but never record what they received)."""
    persist = tmp_path / "align"
    a1 = _make_audio(tmp_path, "one.wav", b"aaaa")
    a2 = _make_audio(tmp_path, "two.wav", b"bbbb")
    received = {}

    def _fake_run_mfa(corpus_dir, output_dir, g2p=False):
        received["g2p"] = g2p
        for name in os.listdir(corpus_dir):
            if name.endswith(".lab"):
                stem = name[:-4]
                with open(os.path.join(output_dir, f"{stem}.TextGrid"), "w") as f:
                    f.write("tg")

    monkeypatch.setattr(aligner, "_run_mfa", _fake_run_mfa)

    aligner.ensure_alignments([(a1, "hi one")], persist, g2p=True)
    assert received["g2p"] is True

    aligner.ensure_alignments([(a2, "hi two")], persist, g2p=False)
    assert received["g2p"] is False


def test_ensure_only_realigns_stale_subset(tmp_path, monkeypatch):
    persist = tmp_path / "align"
    a1 = _make_audio(tmp_path, "one.wav", b"aaaa")
    a2 = _make_audio(tmp_path, "two.wav", b"bbbb")
    _install_fake_mfa(monkeypatch)
    aligner.ensure_alignments([(a1, "hi one"), (a2, "hi two")], persist)
    # Change only two's transcript.
    staged = []

    def spy(corpus_dir, output_dir, g2p=False):
        staged.extend(n[:-4] for n in os.listdir(corpus_dir)
                      if n.endswith(".lab"))
        for s in staged:
            with open(os.path.join(output_dir, f"{s}.TextGrid"), "w") as f:
                f.write("tg")
    monkeypatch.setattr(aligner, "_run_mfa", spy)
    res = aligner.ensure_alignments([(a1, "hi one"), (a2, "CHANGED")], persist)
    assert staged == ["two"], "only the stale stimulus is re-aligned"
    assert res["cached"] == ["one"] and res["aligned"] == ["two"]
