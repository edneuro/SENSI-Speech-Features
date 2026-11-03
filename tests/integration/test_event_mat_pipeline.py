"""Event .mat files written by run_word_level_batch (spec 2026-11-03-event-mat)."""
import numpy as np
import pandas as pd
import pytest
import scipy.io
import soundfile as sf

from sensi import pipeline
from sensi.pipeline import run_word_level_batch, run_doc_level_batch
from tests.batch_args import build_parser

WORD_DF = pd.DataFrame({
    "doc_id": [1, 1], "sentence_id": [0, 0], "token_in_sentence_id": [0, 1],
    "token": ["hello", "world"], "upos": ["INTJ", "NOUN"],
    "Onset": [0.0, 0.5], "End": [0.4, 0.9],
})
PHONE_DF = pd.DataFrame({
    "phone": ["HH", "EH1", "L", "OW1", "W", "ER1", "L", "D"],
    "onset":  [0.00, 0.10, 0.20, 0.30, 0.50, 0.60, 0.75, 0.85],
    "offset": [0.10, 0.20, 0.30, 0.40, 0.60, 0.75, 0.85, 0.90],
})


def _setup(tmp_path, monkeypatch, *, audio=True, word_df=WORD_DF, **flags):
    txt = tmp_path / "txt"
    txt.mkdir()
    out = tmp_path / "out"
    out.mkdir()
    (txt / "story.txt").write_text("hello world")
    aud = tmp_path / "audio"
    aud.mkdir()
    if audio:
        sf.write(str(aud / "story.wav"), np.zeros(16000, dtype=np.float32), 16000)

    def fake_run(args):
        if audio:
            args._phones_df = PHONE_DF
        args._contours = None
        return word_df.copy()

    monkeypatch.setattr(pipeline, "run", fake_run)
    args = build_parser().parse_args([
        "--transcripts-folder", str(txt), "--audio-folder", str(aud),
        "--output-folder", str(out), "--analyzer", "tester",
        "--no-align", "--no-edu-freq",
    ])
    args.no_syllables = True
    args.no_phonemes = True
    args.no_sentences = True
    args.no_word_events_mat = False
    args.event_units = {"word_events_mat": {}, "syllable_events_mat": {}}
    for k, v in flags.items():
        setattr(args, k, v)
    return args, out


def _manifest_ok(out):
    import json
    m = json.loads(next((out / "logs").glob("word_run_*_manifest.json")).read_text())
    assert not m["errors"], m["errors"]
    assert m["n_pairs_processed"] == 1


def test_word_events_written_and_matches_csv(tmp_path, monkeypatch):
    args, out = _setup(tmp_path, monkeypatch)
    run_word_level_batch(args)
    _manifest_ok(out)
    mat = scipy.io.loadmat(str(out / "word" / "story_word_events.mat"),
                           simplify_cells=True)
    assert len(mat["onset"]) == len(mat["offset"]) == len(mat["token"]) == 2
    header = set(pd.read_csv(out / "word" / "story_features.csv").columns)
    assert set(np.atleast_1d(mat["columns"])) <= header


def test_word_events_without_onset_in_csv(tmp_path, monkeypatch):
    args, out = _setup(tmp_path, monkeypatch)
    args.feature_toggles = {"feat_acoustic_onset": False, "feat_acoustic_end": False}
    run_word_level_batch(args)
    _manifest_ok(out)
    assert "Onset" not in pd.read_csv(out / "word" / "story_features.csv").columns
    mat = scipy.io.loadmat(str(out / "word" / "story_word_events.mat"),
                           simplify_cells=True)
    assert list(mat["onset"]) == [0.0, 0.5]


def test_word_events_when_csv_off(tmp_path, monkeypatch):
    args, out = _setup(tmp_path, monkeypatch, no_word_csv=True)
    run_word_level_batch(args)
    _manifest_ok(out)
    assert (out / "word" / "story_word_events.mat").exists()
    assert not (out / "word" / "story_features.csv").exists()


def test_text_only_skips(tmp_path, monkeypatch, capsys):
    df = WORD_DF.drop(columns=["Onset", "End"])
    args, out = _setup(tmp_path, monkeypatch, audio=False, word_df=df,
                       no_syllable_events_mat=False)
    run_word_level_batch(args)
    _manifest_ok(out)
    assert not list(out.rglob("*_events.mat"))
    err = capsys.readouterr().err
    assert "[word_events] skipped for story: no aligned words" in err
    assert "[syllable_events] skipped for story: no aligned syllables" in err


def test_syllable_events_with_syllable_csv_off(tmp_path, monkeypatch):
    args, out = _setup(tmp_path, monkeypatch, no_syllable_events_mat=False)
    run_word_level_batch(args)
    _manifest_ok(out)
    assert not list((out / "syllable").glob("*_syllables.csv"))
    mat = scipy.io.loadmat(str(out / "syllable" / "story_syllable_events.mat"),
                           simplify_cells=True)
    assert "pcenter" in mat
    assert "Pitch_Mean" in mat


def test_doc_pass_never_writes(tmp_path, monkeypatch):
    args, out = _setup(tmp_path, monkeypatch)
    args.no_word_events_mat = False
    run_doc_level_batch(args)
    assert list((out / "document").glob("*.csv"))
    assert not list(out.rglob("*_events.mat"))
