"""Tests for extractors.prosody.add_word_prosody and compute_doc_prosody.

Uses synthetic Contours fixtures — no audio file needed.
"""
import numpy as np
import pandas as pd
import pytest

from extractors.prosody import add_word_prosody, compute_doc_prosody
from extractors.audio_features import Contours


def make_contours(pitch_hz, intensity_db, frame_rate=100, hnr=15.0):
    """Build a Contours instance from 1-D arrays. times derived from frame rate."""
    pitch_hz = np.asarray(pitch_hz, dtype=float)
    intensity_db = np.asarray(intensity_db, dtype=float)
    times = np.arange(len(pitch_hz)) / frame_rate
    voiced = pitch_hz[~np.isnan(pitch_hz)]
    median = float(np.median(voiced)) if voiced.size else float('nan')
    return Contours(
        pitch_hz=pitch_hz, intensity_db=intensity_db, times=times,
        frame_rate_hz=frame_rate, speaker_median_pitch_hz=median, hnr_db=hnr,
    )


def make_df(rows):
    """rows = list of (token, onset, end); upos defaults to NOUN."""
    return pd.DataFrame([
        {"token": tok, "Onset": on, "End": en, "upos": "NOUN"}
        for tok, on, en in rows
    ])


def test_pitch_mean_min_max_std_on_known_distribution():
    # 100 frames at 100 Hz = 1 second of audio, voiced pitch = np.arange(100) Hz.
    contours = make_contours(np.arange(100, dtype=float),
                             np.full(100, 60.0))
    df = make_df([("word", 0.0, 1.0)])
    out = add_word_prosody(df, contours, pitch_units="hz")
    row = out.iloc[0]
    # np.nanpercentile(np.arange(100), 5) == 4.95; (..., 95) == 94.05
    assert row["Pitch_Min"] == pytest.approx(4.95)
    assert row["Pitch_Max"] == pytest.approx(94.05)
    assert row["Pitch_Mean"] == pytest.approx(49.5)
    assert row["Pitch_Std"] == pytest.approx(np.std(np.arange(100)))


def test_volume_columns_same_logic():
    contours = make_contours(np.full(100, 200.0),
                             np.arange(100, dtype=float))
    df = make_df([("word", 0.0, 1.0)])
    out = add_word_prosody(df, contours, pitch_units="hz")
    row = out.iloc[0]
    assert row["Volume_Mean"] == pytest.approx(49.5)
    assert row["Volume_Min"] == pytest.approx(4.95)
    assert row["Volume_Max"] == pytest.approx(94.05)


def test_pitch_delta_first_word_nan():
    contours = make_contours(np.full(200, 200.0), np.full(200, 60.0))
    df = make_df([("w1", 0.0, 1.0), ("w2", 1.0, 2.0)])
    out = add_word_prosody(df, contours, pitch_units="hz")
    assert pd.isna(out.iloc[0]["Pitch_Delta"])
    # Both words have identical contours, so delta = 0
    assert out.iloc[1]["Pitch_Delta"] == pytest.approx(0.0)


def test_pitch_delta_varies_between_words():
    contours = make_contours(
        np.concatenate([np.full(100, 200.0), np.full(100, 250.0)]),
        np.full(200, 60.0),
    )
    df = make_df([("w1", 0.0, 1.0), ("w2", 1.0, 2.0)])
    out = add_word_prosody(df, contours, pitch_units="hz")
    assert out.iloc[1]["Pitch_Delta"] == pytest.approx(50.0)


def test_nan_onset_yields_all_nan_prosody():
    contours = make_contours(np.full(100, 200.0), np.full(100, 60.0))
    df = pd.DataFrame([
        {"token": "punct", "Onset": np.nan, "End": np.nan, "upos": "PUNCT"},
        {"token": "w1", "Onset": 0.0, "End": 1.0, "upos": "NOUN"},
    ])
    out = add_word_prosody(df, contours, pitch_units="hz")
    for col in ("Pitch_Mean", "Pitch_Min", "Pitch_Max", "Pitch_Std",
                "Volume_Mean", "Volume_Min", "Volume_Max", "Volume_Std"):
        assert pd.isna(out.iloc[0][col]), f"{col} should be NaN for NaN-Onset row"


def test_pitch_delta_skips_unaligned_row_not_just_first():
    # w1 (aligned) -> punct (unaligned, NaN Onset/End) -> w2 (aligned).
    # w2's delta must be measured back to w1, not NaN'd out by the
    # intervening unaligned row (regression for the release-hardening-style
    # fix mirroring Word_Gap's ffill behavior).
    contours = make_contours(
        np.concatenate([np.full(100, 200.0), np.full(100, 250.0)]),
        np.full(200, 60.0),
    )
    df = pd.DataFrame([
        {"token": "w1", "Onset": 0.0, "End": 1.0, "upos": "NOUN"},
        {"token": ",", "Onset": np.nan, "End": np.nan, "upos": "PUNCT"},
        {"token": "w2", "Onset": 1.0, "End": 2.0, "upos": "NOUN"},
    ])
    out = add_word_prosody(df, contours, pitch_units="hz")
    assert pd.isna(out.iloc[0]["Pitch_Delta"])  # first word: no prior word
    assert pd.isna(out.iloc[1]["Pitch_Delta"])  # unaligned row: no Mean of its own
    assert out.iloc[2]["Pitch_Delta"] == pytest.approx(50.0)  # w2 vs w1, not NaN


def test_unvoiced_word_pitch_nan_volume_valid():
    pitch = np.full(100, np.nan)  # entire stimulus unvoiced
    intensity = np.full(100, 50.0)
    contours = make_contours(pitch, intensity)
    df = make_df([("w", 0.0, 1.0)])
    out = add_word_prosody(df, contours, pitch_units="hz")
    row = out.iloc[0]
    assert pd.isna(row["Pitch_Mean"])
    assert pd.isna(row["Pitch_Std"])
    assert row["Volume_Mean"] == pytest.approx(50.0)


def test_semitone_conversion():
    # Median pitch will be 300 Hz (median of 50×200 + 50×400); word's frames at 400 Hz.
    pitch = np.concatenate([np.full(50, 200.0), np.full(50, 400.0)])
    contours = make_contours(pitch, np.full(100, 60.0))
    df = make_df([("w", 0.5, 1.0)])  # second half, all 400 Hz frames
    out = add_word_prosody(df, contours, pitch_units="semitones")
    expected = 12 * np.log2(400.0 / 300.0)
    assert out.iloc[0]["Pitch_Mean"] == pytest.approx(expected, abs=1e-6)


def test_single_frame_word_std_is_zero():
    contours = make_contours(np.full(100, 200.0), np.full(100, 60.0))
    df = make_df([("w", 0.0, 0.01)])  # exactly one frame (10 ms at 100 Hz)
    out = add_word_prosody(df, contours, pitch_units="hz")
    row = out.iloc[0]
    assert row["Pitch_Std"] == pytest.approx(0.0)
    assert row["Pitch_Min"] == row["Pitch_Max"] == row["Pitch_Mean"]


# ---------------------------------------------------------------------------
# compute_doc_prosody tests
# ---------------------------------------------------------------------------

def _aligned_df_with_prosody(rows):
    """rows = list of (token, onset, end). Prosody columns set to NaN — the
    caller fills them via add_word_prosody before compute_doc_prosody."""
    return pd.DataFrame([
        {"token": tok, "Onset": on, "End": en, "upos": "NOUN"}
        for tok, on, en in rows
    ])


def test_doc_prosody_basic_aggregates():
    contours = make_contours(np.arange(100, dtype=float), np.full(100, 60.0))
    df = _aligned_df_with_prosody([("a", 0.0, 1.0)])
    df = add_word_prosody(df, contours, pitch_units="hz")
    doc = compute_doc_prosody(df, contours, text="cat",
                              pause_threshold_s=0.25, pitch_units="hz")
    assert doc["doc_pitch_mean"] == pytest.approx(49.5)  # mean(arange 100)
    assert doc["doc_volume_mean"] == pytest.approx(60.0)
    assert doc["doc_pitch_min"] == pytest.approx(4.95)
    assert doc["doc_pitch_max"] == pytest.approx(94.05)


def test_doc_pause_stats_three_words_one_pause():
    # words at [0, 0.5], [0.5, 1.0], [1.5, 2.0] — gap between w2 and w3 is 0.5s
    contours = make_contours(np.full(200, 200.0), np.full(200, 60.0))
    df = _aligned_df_with_prosody([("a", 0.0, 0.5), ("b", 0.5, 1.0), ("c", 1.5, 2.0)])
    df = add_word_prosody(df, contours, pitch_units="hz")
    doc = compute_doc_prosody(df, contours, text="a b c",
                              pause_threshold_s=0.25, pitch_units="hz")
    assert doc["doc_pause_count"] == 1
    assert doc["doc_pause_mean_duration"] == pytest.approx(0.5)
    assert doc["doc_pause_total_duration"] == pytest.approx(0.5)
    # Total stimulus duration = end[last] - onset[first] = 2.0 - 0.0 = 2.0
    assert doc["doc_pause_pct_silence"] == pytest.approx(0.25)


def test_doc_pause_threshold_strict_greater_than():
    # gap exactly equal to threshold should NOT count as a pause
    contours = make_contours(np.full(200, 200.0), np.full(200, 60.0))
    df = _aligned_df_with_prosody([("a", 0.0, 0.5), ("b", 0.75, 1.0)])
    df = add_word_prosody(df, contours, pitch_units="hz")
    doc = compute_doc_prosody(df, contours, text="a b",
                              pause_threshold_s=0.25, pitch_units="hz")
    assert doc["doc_pause_count"] == 0


def test_doc_articulation_rate_uses_surface_form():
    contours = make_contours(np.full(300, 200.0), np.full(300, 60.0))
    # "running" via pyphen = 2 syllables; "run" = 1. Two 1-second words.
    df = pd.DataFrame([
        {"token": "running", "Onset": 0.0, "End": 1.0, "upos": "VERB"},
        {"token": "run",     "Onset": 1.0, "End": 2.0, "upos": "VERB"},
    ])
    df = add_word_prosody(df, contours, pitch_units="hz")
    doc = compute_doc_prosody(df, contours, text="running run",
                              pause_threshold_s=0.25, pitch_units="hz")
    # 3 syllables / 2 seconds = 1.5
    assert doc["doc_articulation_rate"] == pytest.approx(1.5)


def test_doc_single_word_no_pauses():
    contours = make_contours(np.full(100, 200.0), np.full(100, 60.0))
    df = _aligned_df_with_prosody([("only", 0.0, 1.0)])
    df = add_word_prosody(df, contours, pitch_units="hz")
    doc = compute_doc_prosody(df, contours, text="only",
                              pause_threshold_s=0.25, pitch_units="hz")
    assert doc["doc_pause_count"] == 0
    assert doc["doc_pause_total_duration"] == pytest.approx(0.0)


def test_doc_hnr_passed_through_from_contours():
    contours = make_contours(np.full(100, 200.0), np.full(100, 60.0), hnr=22.5)
    df = _aligned_df_with_prosody([("w", 0.0, 1.0)])
    df = add_word_prosody(df, contours, pitch_units="hz")
    doc = compute_doc_prosody(df, contours, text="w",
                              pause_threshold_s=0.25, pitch_units="hz")
    assert doc["doc_hnr"] == pytest.approx(22.5)


def test_doc_returns_all_16_keys():
    contours = make_contours(np.full(100, 200.0), np.full(100, 60.0))
    df = _aligned_df_with_prosody([("w", 0.0, 1.0)])
    df = add_word_prosody(df, contours, pitch_units="hz")
    doc = compute_doc_prosody(df, contours, text="w",
                              pause_threshold_s=0.25, pitch_units="hz")
    expected = {
        "doc_pitch_mean", "doc_pitch_min", "doc_pitch_max", "doc_pitch_std",
        "doc_volume_mean", "doc_volume_min", "doc_volume_max", "doc_volume_std",
        "doc_pitch_delta_mean", "doc_volume_delta_mean",
        "doc_hnr", "doc_articulation_rate",
        "doc_pause_count", "doc_pause_mean_duration",
        "doc_pause_total_duration", "doc_pause_pct_silence",
    }
    assert expected.issubset(set(doc.keys()))
