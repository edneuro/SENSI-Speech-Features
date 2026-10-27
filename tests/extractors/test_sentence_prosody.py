"""Tests for sensi.extractors.sentence_prosody.add_sentence_prosody.

All tests use synthetic Contours — no audio file needed.
"""
import numpy as np
import pandas as pd
import pytest

from sensi.extractors.audio_features import Contours


def make_contours(pitch_hz, intensity_db, frame_rate=100, hnr=15.0,
                   hnr_frame_db=None, hnr_frame_times=None):
    """Build a Contours instance. times derived from frame rate."""
    pitch_hz = np.asarray(pitch_hz, dtype=float)
    intensity_db = np.asarray(intensity_db, dtype=float)
    times = np.arange(len(pitch_hz)) / frame_rate
    voiced = pitch_hz[~np.isnan(pitch_hz)]
    median = float(np.median(voiced)) if voiced.size else float('nan')
    if hnr_frame_db is None:
        hnr_frame_db = np.full(len(times), hnr)
        hnr_frame_times = times.copy()
    return Contours(
        pitch_hz=pitch_hz, intensity_db=intensity_db, times=times,
        frame_rate_hz=frame_rate, speaker_median_pitch_hz=median, hnr_db=hnr,
        hnr_frame_db=hnr_frame_db, hnr_frame_times=hnr_frame_times,
    )


def make_word_df(rows):
    """rows = list of (doc_id, sentence_id, sentence_text, token, onset, end)."""
    return pd.DataFrame([
        {
            "doc_id": doc_id,
            "sentence_id": sent_id,
            "sentence": sent_text,
            "token": tok,
            "Onset": on,
            "End": en,
            "upos": "NOUN",
        }
        for doc_id, sent_id, sent_text, tok, on, en in rows
    ])


def test_basic_sentence_timing():
    """Sentence onset = min word Onset, end = max word End."""
    from sensi.extractors.sentence_prosody import add_sentence_prosody

    df = make_word_df([
        ("doc1", 0, "Hello world", "Hello", 0.1, 0.5),
        ("doc1", 0, "Hello world", "world", 0.6, 1.0),
        ("doc1", 1, "Goodbye",     "Goodbye", 1.2, 1.8),
    ])
    contours = make_contours(np.full(200, 200.0), np.full(200, 60.0))
    result = add_sentence_prosody(df, contours)

    assert len(result) == 2
    assert result.iloc[0]["Onset"] == pytest.approx(0.1)
    assert result.iloc[0]["End"] == pytest.approx(1.0)
    assert result.iloc[1]["Onset"] == pytest.approx(1.2)
    assert result.iloc[1]["End"] == pytest.approx(1.8)


def test_sentence_text_is_preserved():
    from sensi.extractors.sentence_prosody import add_sentence_prosody

    df = make_word_df([
        ("doc1", 0, "The cat sat.", "The", 0.0, 0.3),
        ("doc1", 0, "The cat sat.", "cat", 0.4, 0.7),
    ])
    contours = make_contours(np.full(100, 200.0), np.full(100, 60.0))
    result = add_sentence_prosody(df, contours)

    assert result.iloc[0]["sentence"] == "The cat sat."


def test_pitch_mean_aggregates_over_sentence_window():
    """Uniform 200 Hz pitch across 0-2 s -> Pitch_Mean = 200 Hz."""
    from sensi.extractors.sentence_prosody import add_sentence_prosody

    contours = make_contours(np.full(200, 200.0), np.full(200, 60.0))
    df = make_word_df([
        ("doc1", 0, "Hello world", "Hello", 0.0, 1.0),
        ("doc1", 0, "Hello world", "world", 1.0, 2.0),
    ])
    result = add_sentence_prosody(df, contours, pitch_units="hz")
    assert result.iloc[0]["Pitch_Mean"] == pytest.approx(200.0)


def test_pitch_delta_nan_for_first_sentence():
    from sensi.extractors.sentence_prosody import add_sentence_prosody

    contours = make_contours(
        np.concatenate([np.full(100, 200.0), np.full(100, 250.0)]),
        np.full(200, 60.0),
    )
    df = make_word_df([
        ("doc1", 0, "sent A", "A", 0.0, 1.0),
        ("doc1", 1, "sent B", "B", 1.0, 2.0),
    ])
    result = add_sentence_prosody(df, contours, pitch_units="hz")
    assert pd.isna(result.iloc[0]["Pitch_Delta"])
    assert result.iloc[1]["Pitch_Delta"] == pytest.approx(50.0)


def test_nan_onset_yields_nan_prosody():
    """Words with NaN alignment -> sentence onset/end NaN -> NaN prosody."""
    from sensi.extractors.sentence_prosody import add_sentence_prosody

    contours = make_contours(np.full(100, 200.0), np.full(100, 60.0))
    df = make_word_df([
        ("doc1", 0, "unaligned", "word", float("nan"), float("nan")),
    ])
    result = add_sentence_prosody(df, contours)
    assert pd.isna(result.iloc[0]["Onset"])
    assert pd.isna(result.iloc[0]["Pitch_Mean"])
    assert pd.isna(result.iloc[0]["HNR"])


def test_pitch_delta_skips_unaligned_sentence_not_just_first():
    """Sentence B fails to align entirely; sentence C's delta must be
    measured back to sentence A, not NaN'd out by B (regression for the
    fix mirroring Word_Gap's ffill behavior)."""
    from sensi.extractors.sentence_prosody import add_sentence_prosody

    contours = make_contours(
        np.concatenate([np.full(100, 200.0), np.full(100, 250.0)]),
        np.full(200, 60.0),
    )
    df = make_word_df([
        ("doc1", 0, "sent A", "A", 0.0, 1.0),
        ("doc1", 1, "sent B", "b", float("nan"), float("nan")),
        ("doc1", 2, "sent C", "C", 1.0, 2.0),
    ])
    result = add_sentence_prosody(df, contours, pitch_units="hz")
    assert pd.isna(result.iloc[0]["Pitch_Delta"])   # first sentence
    assert pd.isna(result.iloc[1]["Pitch_Delta"])   # unaligned sentence: no Mean of its own
    assert result.iloc[2]["Pitch_Delta"] == pytest.approx(50.0)  # C vs A, not NaN


def test_none_contours_yields_nan_prosody():
    """If contours is None (no audio), timing comes from word_df but prosody is NaN."""
    from sensi.extractors.sentence_prosody import add_sentence_prosody

    df = make_word_df([
        ("doc1", 0, "hello", "hello", 0.0, 1.0),
    ])
    result = add_sentence_prosody(df, contours=None)
    assert result.iloc[0]["Onset"] == pytest.approx(0.0)
    assert result.iloc[0]["End"] == pytest.approx(1.0)
    assert pd.isna(result.iloc[0]["Pitch_Mean"])


def test_hnr_aggregated_per_sentence_window():
    """HNR frames 0-1 s = 12 dB, frames 1-2 s = 20 dB. Two sentences should differ."""
    from sensi.extractors.sentence_prosody import add_sentence_prosody

    times = np.arange(200) / 100.0       # 0.00, 0.01, ..., 1.99
    hnr_vals = np.where(times < 1.0, 12.0, 20.0)
    contours = make_contours(
        np.full(200, 200.0), np.full(200, 60.0),
        hnr_frame_db=hnr_vals, hnr_frame_times=times,
    )
    df = make_word_df([
        ("doc1", 0, "first",  "first",  0.0, 1.0),
        ("doc1", 1, "second", "second", 1.0, 2.0),
    ])
    result = add_sentence_prosody(df, contours, pitch_units="hz")
    assert result.iloc[0]["HNR"] == pytest.approx(12.0, abs=0.5)
    assert result.iloc[1]["HNR"] == pytest.approx(20.0, abs=0.5)


def test_output_columns_present():
    from sensi.extractors.sentence_prosody import add_sentence_prosody

    contours = make_contours(np.full(100, 200.0), np.full(100, 60.0))
    df = make_word_df([("doc1", 0, "hi", "hi", 0.0, 1.0)])
    result = add_sentence_prosody(df, contours)

    expected_cols = {
        "doc_id", "sentence_id", "sentence", "Onset", "End",
        "Pitch_Mean", "Pitch_Min", "Pitch_Max", "Pitch_Std", "Pitch_Delta",
        "Volume_Mean", "Volume_Min", "Volume_Max", "Volume_Std", "Volume_Delta",
        "HNR",
    }
    assert expected_cols.issubset(set(result.columns))
