"""run() in a pre-aligned batch reads TextGrids from disk, never spawns MFA.

Also covers the `_prealign_batch` fallback: when the batch MFA invocation
fails outright, the helper must swallow the exception and leave
`_batch_prealigned` False so run() falls back to per-document alignment;
and per-pair isolation: an unreadable transcript skips only its own stimulus.
"""
import types

import pandas as pd
import pytest

import main
from extractors import aligner


def test_run_reads_disk_and_never_calls_mfa(tmp_path, monkeypatch):
    out = tmp_path / "out"
    persist = aligner.alignment_dir(out)
    persist.mkdir(parents=True)
    (persist / "stim.TextGrid").write_text("tg", encoding="utf-8")

    transcript = tmp_path / "stim.txt"
    transcript.write_text("hello world", encoding="utf-8")
    audio = tmp_path / "stim.wav"
    audio.write_bytes(b"RIFFfake")

    # Disk parse returns canned frames; MFA must never be invoked.
    monkeypatch.setattr(
        aligner, "_parse_textgrid",
        lambda p: (pd.DataFrame({"Transcript": ["hello", "world"],
                                 "Onset": [0.0, 0.5], "End": [0.4, 0.9]}),
                   pd.DataFrame({"phone": ["HH"], "onset": [0.0],
                                 "offset": [0.1]})))
    monkeypatch.setattr(aligner, "_align_uncached",
                        lambda a, t: pytest.fail("MFA must not run"))

    args = types.SimpleNamespace(
        transcript=str(transcript), audio=str(audio), output_folder=str(out),
        _batch_prealigned=True, doc_id=1,
        no_parse=True, no_freq=True, no_edu_freq=True, no_lexical_norms=True,
        no_sem_domain=True, no_surprisal=True, no_llm_cloze=True,
        no_align=False, no_prosody=True, sem_categories="",
        save_prefs=False,
    )
    df = main.run(args)
    assert "Onset" in df.columns
    assert list(df["Onset"]) == [0.0, 0.5]


def test_prealign_batch_fallback_on_mfa_failure(tmp_path, monkeypatch):
    """A batch-wide MFA failure must not raise, and must leave the batch
    unmarked so run() falls back to per-document alignment (run_if_missing)."""

    class FakePair:
        def __init__(self, audio_path, transcript_path):
            self.audio_path = audio_path
            self.transcript_path = transcript_path

    transcript = tmp_path / "stim.txt"
    transcript.write_text("hello world", encoding="utf-8")
    audio = tmp_path / "stim.wav"
    audio.write_bytes(b"RIFFfake")

    pairs = [FakePair(audio, transcript)]

    def _boom(items, persist_dir, g2p=False):
        raise RuntimeError("MFA not installed")

    monkeypatch.setattr(aligner, "ensure_alignments", _boom)

    ns = types.SimpleNamespace(no_align=False)
    main._prealign_batch(ns, pairs, tmp_path / "out")

    assert getattr(ns, "_batch_prealigned", False) is False


def test_prealign_batch_skips_unreadable_transcript(tmp_path, monkeypatch):
    """An unreadable transcript skips only its own stimulus; the readable ones
    still get pre-aligned and the batch is not aborted."""

    class FakePair:
        def __init__(self, stimulus_id, audio_path, transcript_path):
            self.stimulus_id = stimulus_id
            self.audio_path = audio_path
            self.transcript_path = transcript_path

    good_txt = tmp_path / "good.txt"
    good_txt.write_text("hello world", encoding="utf-8")
    good_wav = tmp_path / "good.wav"
    good_wav.write_bytes(b"RIFFgood")
    bad_wav = tmp_path / "bad.wav"
    bad_wav.write_bytes(b"RIFFbad")
    missing_txt = tmp_path / "does_not_exist.txt"  # open() will raise

    pairs = [
        FakePair("bad", bad_wav, missing_txt),
        FakePair("good", good_wav, good_txt),
    ]

    captured = {}

    def _capture(items, persist_dir, g2p=False):
        captured["items"] = list(items)
        captured["g2p"] = g2p

    monkeypatch.setattr(aligner, "ensure_alignments", _capture)

    ns = types.SimpleNamespace(no_align=False)
    main._prealign_batch(ns, pairs, tmp_path / "out")  # must not raise

    audios = [a for a, _ in captured["items"]]
    assert audios == [str(good_wav)], "only the readable stimulus is staged"
    assert getattr(ns, "_batch_prealigned", False) is True
    assert captured["g2p"] is False, "g2p_oov defaults off when unset on args"


def test_prealign_batch_threads_g2p_oov_true(tmp_path, monkeypatch):
    """When args.g2p_oov is True, _prealign_batch must pass g2p=True through
    to ensure_alignments — this is the first real caller of that plumbing."""

    class FakePair:
        def __init__(self, audio_path, transcript_path):
            self.audio_path = audio_path
            self.transcript_path = transcript_path

    transcript = tmp_path / "stim.txt"
    transcript.write_text("hello world", encoding="utf-8")
    audio = tmp_path / "stim.wav"
    audio.write_bytes(b"RIFFfake")

    pairs = [FakePair(audio, transcript)]

    captured = {}

    def _capture(items, persist_dir, g2p=False):
        captured["g2p"] = g2p

    monkeypatch.setattr(aligner, "ensure_alignments", _capture)

    ns = types.SimpleNamespace(no_align=False, g2p_oov=True)
    main._prealign_batch(ns, pairs, tmp_path / "out")

    assert captured["g2p"] is True
