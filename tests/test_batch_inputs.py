"""Batch runners fail fast on missing or mistyped input folders and record
transcripts that ran text-only while alignment was on (audit F2)."""
import types

import pytest

import main
from utils.file_pairing import find_pairs


def test_missing_transcripts_folder_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="Transcripts folder"):
        main._check_input_folders(tmp_path / "nope", "")


def test_configured_but_missing_audio_folder_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="Audio folder"):
        main._check_input_folders(tmp_path, str(tmp_path / "typo"))


def test_no_audio_folder_is_a_valid_text_only_run(tmp_path):
    main._check_input_folders(tmp_path, "")


@pytest.mark.parametrize("runner", ["run_word_level_batch", "run_doc_level_batch"])
def test_empty_transcripts_folder_raises(tmp_path, runner):
    empty = tmp_path / "empty"
    empty.mkdir()
    ns = types.SimpleNamespace(
        transcripts_folder=str(empty), audio_folder="",
        output_folder=str(tmp_path / "out"), analyzer="t",
        text_extensions="txt", audio_extensions="wav")
    with pytest.raises(FileNotFoundError, match="No transcripts"):
        getattr(main, runner)(ns)


def test_acoustic_batch_requires_audio_folder(tmp_path):
    ns = types.SimpleNamespace(audio_folder=str(tmp_path / "typo"),
                               output_folder=str(tmp_path / "out"))
    with pytest.raises(FileNotFoundError, match="Audio folder"):
        main.run_acoustic_level_batch(ns)


class _Logger:
    def __init__(self):
        self.warnings = []

    def add_warning(self, msg):
        self.warnings.append(msg)


def _pairing(tmp_path):
    t = tmp_path / "t"
    a = tmp_path / "a"
    t.mkdir()
    a.mkdir()
    (t / "one.txt").write_text("x", encoding="utf-8")
    (t / "two.txt").write_text("x", encoding="utf-8")
    (a / "one.wav").write_bytes(b"RIFF")
    return a, find_pairs(t, a, text_exts=["txt"], audio_exts=["wav"])


def test_unmatched_transcripts_warned_when_aligning(tmp_path):
    a, pairing = _pairing(tmp_path)
    log = _Logger()
    ns = types.SimpleNamespace(audio_folder=str(a), no_align=False)
    main._warn_pairing_gaps(log, ns, pairing)
    assert len(log.warnings) == 1 and "two.txt" in log.warnings[0]


@pytest.mark.parametrize("audio_folder,no_align", [("", False), ("x", True)])
def test_no_warning_for_intended_text_only_runs(tmp_path, audio_folder, no_align):
    _, pairing = _pairing(tmp_path)
    log = _Logger()
    ns = types.SimpleNamespace(audio_folder=audio_folder, no_align=no_align)
    main._warn_pairing_gaps(log, ns, pairing)
    assert log.warnings == []
