"""Tests for persistent on-disk MFA alignment (freshness + batch pre-pass).

The MFA subprocess is never invoked here: freshness tests use real temp files,
and the batch tests monkeypatch _run_mfa to synthesize TextGrids.
"""
import json
import os

import pandas as pd
import pytest

from extractors import aligner


def _staged_files(corpus_dir):
    """File names staged for MFA, across the per-stimulus speaker subfolders."""
    return [n for _, _, names in os.walk(corpus_dir) for n in names]


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
    audio = tmp_path / "stim.wav"
    audio.write_bytes(b"RIFFfake")
    aligner._write_sidecar(persist, "stim", str(audio), "hello")
    called = []
    monkeypatch.setattr(aligner, "_align_uncached",
                        lambda a, t, g2p=False, **kw: called.append(1))
    monkeypatch.setattr(aligner, "_parse_textgrid", lambda p: _fake_frames())
    w, p = aligner.align_with_phones(
        str(tmp_path / "stim.wav"), "hello", persist_dir=persist)
    assert called == [], "disk hit must not invoke MFA"
    assert list(w["Onset"]) == [0.0]


def test_run_if_missing_false_raises_on_disk_miss(tmp_path, monkeypatch):
    persist = tmp_path / "align"
    persist.mkdir()  # empty: no TextGrid for this stem
    monkeypatch.setattr(aligner, "_align_uncached",
                        lambda a, t, g2p=False, **kw: pytest.fail("MFA must not run"))
    with pytest.raises(RuntimeError):
        aligner.align_with_phones(
            str(tmp_path / "stim.wav"), "hello",
            persist_dir=persist, run_if_missing=False)


def test_run_if_missing_true_falls_back_to_mfa(tmp_path, monkeypatch):
    persist = tmp_path / "align"
    persist.mkdir()
    monkeypatch.setattr(aligner, "_align_uncached",
                        lambda a, t, g2p=False, **kw: _fake_frames())
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

    def fake(corpus_dir, output_dir, g2p=False, **kw):
        runs["count"] += 1
        for name in _staged_files(corpus_dir):
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
    # Fake TextGrids aren't parseable; treat them as readable here.
    monkeypatch.setattr(aligner, "_readable", lambda p: True)
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

    def _fake_run_mfa(corpus_dir, output_dir, g2p=False, **kw):
        received["g2p"] = g2p
        for name in _staged_files(corpus_dir):
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
    # Fake TextGrids aren't parseable; treat them as readable here.
    monkeypatch.setattr(aligner, "_readable", lambda p: True)
    a1 = _make_audio(tmp_path, "one.wav", b"aaaa")
    a2 = _make_audio(tmp_path, "two.wav", b"bbbb")
    _install_fake_mfa(monkeypatch)
    aligner.ensure_alignments([(a1, "hi one"), (a2, "hi two")], persist)
    # Change only two's transcript.
    staged = []

    def spy(corpus_dir, output_dir, g2p=False, **kw):
        staged.extend(n[:-4] for n in _staged_files(corpus_dir)
                      if n.endswith(".lab"))
        for s in staged:
            with open(os.path.join(output_dir, f"{s}.TextGrid"), "w") as f:
                f.write("tg")
    monkeypatch.setattr(aligner, "_run_mfa", spy)
    res = aligner.ensure_alignments([(a1, "hi one"), (a2, "CHANGED")], persist)
    assert staged == ["two"], "only the stale stimulus is re-aligned"
    assert res["cached"] == ["one"] and res["aligned"] == ["two"]


def test_fallback_mfa_persists_textgrid_and_sidecar(tmp_path, monkeypatch):
    """A per-document MFA run (fallback after a failed batch pre-pass) writes
    its TextGrid + sidecar to persist_dir, so a later pass reads it from disk
    instead of re-aligning, and the folder matches the CSVs built from it."""
    persist = tmp_path / "align"
    audio = tmp_path / "stim.wav"
    audio.write_bytes(b"RIFFfake")

    def fake(corpus_dir, output_dir, g2p=False, **kw):
        with open(os.path.join(output_dir, "stim.TextGrid"), "w") as f:
            f.write("tg")

    monkeypatch.setattr(aligner, "_run_mfa", fake)
    monkeypatch.setattr(aligner, "_parse_textgrid", lambda p: _fake_frames())

    aligner.align_with_phones(str(audio), "hello", persist_dir=persist)
    assert (persist / "stim.TextGrid").exists()
    assert aligner._is_fresh(persist, "stim", str(audio), "hello")

    monkeypatch.setattr(aligner, "_run_mfa",
                        lambda *a, **kw: pytest.fail("MFA must not run again"))
    aligner.align_with_phones(str(audio), "hello", persist_dir=persist,
                              run_if_missing=False)


def test_disk_hit_requires_fresh_sidecar(tmp_path, monkeypatch):
    """A TextGrid whose sidecar no longer matches the transcript is not reused."""
    persist = tmp_path / "align"
    persist.mkdir()
    audio = tmp_path / "stim.wav"
    audio.write_bytes(b"RIFFfake")
    (persist / "stim.TextGrid").write_text("fake tg", encoding="utf-8")
    aligner._write_sidecar(persist, "stim", str(audio), "old transcript")
    monkeypatch.setattr(aligner, "_parse_textgrid", lambda p: _fake_frames())
    with pytest.raises(RuntimeError, match="No alignment on disk"):
        aligner.align_with_phones(str(audio), "new transcript",
                                  persist_dir=persist, run_if_missing=False)


def test_every_mfa_call_gets_its_own_temporary_directory(tmp_path, monkeypatch):
    """MFA's working folder (corpus.db) is per call, never the shared
    $MFA_ROOT_DIR/corpus/, so two runs at once can't collide."""
    cmds = []

    class _Result:
        returncode = 0
        stdout = stderr = ""

    def fake_subprocess_run(cmd, **kw):
        cmds.append(cmd)
        corpus_dir, output_dir = cmd[2], cmd[5]
        for name in _staged_files(corpus_dir):
            if name.endswith(".lab"):
                with open(os.path.join(output_dir, f"{name[:-4]}.TextGrid"),
                          "w") as f:
                    f.write("tg")
        return _Result()

    monkeypatch.setattr(aligner.subprocess, "run", fake_subprocess_run)
    monkeypatch.setattr(aligner, "_parse_textgrid", lambda p: _fake_frames())
    audio = tmp_path / "one.wav"
    audio.write_bytes(b"RIFFone")

    aligner.align_with_phones(str(audio), "hello")
    aligner.ensure_alignments([(str(audio), "hello")], tmp_path / "persist")

    temp_dirs = [c[c.index("--temporary_directory") + 1] for c in cmds]
    assert len(temp_dirs) == 2 and temp_dirs[0] != temp_dirs[1]
    mfa_root = os.environ.get("MFA_ROOT_DIR")
    if mfa_root:
        for d in temp_dirs:
            assert not os.path.abspath(d).startswith(os.path.abspath(mfa_root))


def test_batch_stages_one_speaker_folder_per_stimulus(tmp_path, monkeypatch):
    """Each recording gets its own speaker subfolder, so MFA's per-speaker
    CMVN / fMLLR never pools audio across stimuli. MFA mirrors the subfolder
    in its output; those nested TextGrids must be collected."""
    seen = {}

    def fake(corpus_dir, output_dir, g2p=False, **kw):
        for spk in os.listdir(corpus_dir):
            seen[spk] = sorted(os.listdir(os.path.join(corpus_dir, spk)))
            os.makedirs(os.path.join(output_dir, spk))
            with open(os.path.join(output_dir, spk, f"{spk}.TextGrid"), "w") as f:
                f.write("tg")

    monkeypatch.setattr(aligner, "_run_mfa", fake)
    items = []
    for stem in ("one", "two"):
        wav = tmp_path / f"{stem}.wav"
        wav.write_bytes(stem.encode())
        items.append((str(wav), f"hi {stem}"))
    res = aligner.ensure_alignments(items, tmp_path / "persist")
    assert seen == {"one": ["one.lab", "one.wav"], "two": ["two.lab", "two.wav"]}
    assert res["aligned"] == ["one", "two"] and res["failed"] == []
    assert (tmp_path / "persist" / "one.TextGrid").exists()


def test_pooled_speaker_sidecars_are_stale(tmp_path):
    """TextGrids written under the pooled-speaker layout (sidecar v2) are
    re-aligned once rather than reused."""
    persist = tmp_path / "align"
    persist.mkdir()
    audio = tmp_path / "stim.wav"
    audio.write_bytes(b"RIFFfake")
    (persist / "stim.TextGrid").write_text("tg", encoding="utf-8")
    aligner._write_sidecar(persist, "stim", str(audio), "hello")
    side = persist / "stim.json"
    meta = json.loads(side.read_text(encoding="utf-8"))
    meta["version"] = 2
    side.write_text(json.dumps(meta), encoding="utf-8")
    assert not aligner._is_fresh(persist, "stim", str(audio), "hello")


def test_unreadable_textgrid_with_fresh_sidecar_is_realigned(tmp_path, monkeypatch):
    """A damaged TextGrid whose sidecar still matches is re-aligned and
    reported as 'corrupt', instead of being trusted and failing later."""
    persist = tmp_path / "align"
    persist.mkdir()
    audio = tmp_path / "stim.wav"
    audio.write_bytes(b"RIFFfake")
    (persist / "stim.TextGrid").write_text("garbage", encoding="utf-8")
    aligner._write_sidecar(persist, "stim", str(audio), "hello")
    ran = []

    def fake(corpus_dir, output_dir, g2p=False, **kw):
        ran.append(1)
        with open(os.path.join(output_dir, "stim.TextGrid"), "w") as f:
            f.write("tg")

    monkeypatch.setattr(aligner, "_run_mfa", fake)
    res = aligner.ensure_alignments([(str(audio), "hello")], persist)
    assert ran == [1]
    assert res["corrupt"] == ["stim"] and res["aligned"] == ["stim"]
