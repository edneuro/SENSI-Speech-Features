"""run() in a pre-aligned batch reads TextGrids from disk, never spawns MFA.

Also covers the `_prealign_batch` fallback: when the batch MFA invocation
fails outright, the helper must swallow the exception and leave
`_batch_prealigned` unset so run() falls back to per-document alignment.
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

    def _boom(items, persist_dir):
        raise RuntimeError("MFA not installed")

    monkeypatch.setattr(aligner, "ensure_alignments", _boom)

    ns = types.SimpleNamespace(no_align=False)
    main._prealign_batch(ns, pairs, tmp_path / "out")

    assert getattr(ns, "_batch_prealigned", False) is False
