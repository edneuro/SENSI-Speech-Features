"""Unit tests for syllable_prosody.add_syllable_prosody."""
import numpy as np
import pandas as pd
import pytest

from sensi.extractors.syllable_prosody import add_syllable_prosody


class _FakeContours:
    """Minimal fake of extractors.audio_features.Contours for testing."""
    def __init__(self, *, pitch_hz: np.ndarray, intensity_db: np.ndarray,
                 times: np.ndarray, speaker_median_pitch_hz: float = 100.0):
        self.pitch_hz = pitch_hz
        self.intensity_db = intensity_db
        self.times = times
        self.speaker_median_pitch_hz = speaker_median_pitch_hz


def _flat_contours(n_frames: int = 100, pitch_hz_value: float = 120.0,
                   intensity_value: float = 60.0, frame_step: float = 0.01
                   ) -> _FakeContours:
    return _FakeContours(
        pitch_hz=np.full(n_frames, pitch_hz_value),
        intensity_db=np.full(n_frames, intensity_value),
        times=np.arange(n_frames) * frame_step,
    )


def _two_syllables() -> pd.DataFrame:
    return pd.DataFrame([
        {'doc_id': 'd', 'sentence_id': 1, 'token_in_sentence_id': 1, 'token': 'below',
         'syllable_id': 1, 'Onset': 0.10, 'End': 0.30, 'Duration': 0.20},
        {'doc_id': 'd', 'sentence_id': 1, 'token_in_sentence_id': 1, 'token': 'below',
         'syllable_id': 2, 'Onset': 0.30, 'End': 0.50, 'Duration': 0.20},
    ])


def test_constant_pitch_yields_constant_pitch_mean_in_semitones():
    """Flat 120 Hz contour with speaker median 100 Hz -> Pitch_Mean ~= 12 * log2(1.2)."""
    df = add_syllable_prosody(_two_syllables(), _flat_contours(pitch_hz_value=120.0),
                              pitch_units='semitones')
    expected = 12.0 * np.log2(120.0 / 100.0)
    assert abs(df.Pitch_Mean.iloc[0] - expected) < 0.01
    assert abs(df.Pitch_Mean.iloc[1] - expected) < 0.01


def test_pitch_units_hz_emits_raw_hz():
    df = add_syllable_prosody(_two_syllables(), _flat_contours(pitch_hz_value=120.0),
                              pitch_units='hz')
    assert abs(df.Pitch_Mean.iloc[0] - 120.0) < 0.01


def test_volume_mean_is_intensity_mean_db():
    df = add_syllable_prosody(_two_syllables(),
                              _flat_contours(intensity_value=55.0))
    assert abs(df.Volume_Mean.iloc[0] - 55.0) < 0.01


def test_delta_skips_an_unvoiced_syllable():
    """A syllable with no pitch doesn't blank the next delta: it is measured
    back to the most recent syllable that had a value."""
    times = np.arange(100) * 0.01
    pitch = np.where(times < 0.30, 100.0, 130.0)
    pitch[(times >= 0.30) & (times < 0.50)] = np.nan       # syllable 2 unvoiced
    contours = _FakeContours(pitch_hz=pitch, intensity_db=np.full(100, 60.0),
                             times=times, speaker_median_pitch_hz=100.0)
    syl = pd.concat([_two_syllables(), _two_syllables().iloc[[1]].assign(
        syllable_id=3, Onset=0.50, End=0.70)], ignore_index=True)
    df = add_syllable_prosody(syl, contours, pitch_units='hz')
    assert pd.isna(df.Pitch_Mean.iloc[1])
    assert df.Pitch_Delta.iloc[2] == pytest.approx(30.0)   # 130 - 100


def test_pitch_delta_is_nan_on_first_syllable():
    df = add_syllable_prosody(_two_syllables(), _flat_contours())
    assert pd.isna(df.Pitch_Delta.iloc[0])
    assert pd.isna(df.Volume_Delta.iloc[0])


def test_delta_is_difference_between_consecutive_means():
    """Two-region contour: syllable 1 over a flat region, syllable 2 over
    a higher region. Delta = mean_2 - mean_1."""
    times = np.arange(100) * 0.01
    pitch = np.where(times < 0.30, 100.0, 130.0)
    intensity = np.where(times < 0.30, 50.0, 60.0)
    contours = _FakeContours(pitch_hz=pitch, intensity_db=intensity, times=times,
                             speaker_median_pitch_hz=100.0)

    df = add_syllable_prosody(_two_syllables(), contours, pitch_units='hz')
    # Syllable 1: 0.10-0.30 covers the 100 Hz region -> mean 100
    # Syllable 2: 0.30-0.50 covers the 130 Hz region -> mean 130
    assert abs(df.Pitch_Mean.iloc[0] - 100.0) < 0.5
    assert abs(df.Pitch_Mean.iloc[1] - 130.0) < 0.5
    # Delta = 130 - 100 = 30
    assert pd.isna(df.Pitch_Delta.iloc[0])
    assert abs(df.Pitch_Delta.iloc[1] - 30.0) < 0.5
    # Volume delta similarly
    assert abs(df.Volume_Delta.iloc[1] - 10.0) < 0.5


def test_contours_none_emits_nan_columns_without_raising():
    df = add_syllable_prosody(_two_syllables(), None)
    assert df.Pitch_Mean.isna().all()
    assert df.Volume_Mean.isna().all()
    assert df.Pitch_Delta.isna().all()
    assert df.Volume_Delta.isna().all()


def test_empty_syllables_df_returns_empty_df_with_prosody_cols():
    empty = pd.DataFrame(columns=['doc_id', 'sentence_id', 'token_in_sentence_id', 'token',
                                  'syllable_id', 'Onset', 'End', 'Duration'])
    df = add_syllable_prosody(empty, _flat_contours())
    assert len(df) == 0
    for col in ('Pitch_Mean', 'Volume_Mean', 'Pitch_Delta', 'Volume_Delta'):
        assert col in df.columns
