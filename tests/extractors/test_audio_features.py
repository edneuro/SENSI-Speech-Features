"""Tests for extractors.audio_features (Parselmouth wrapper).

The integration test synthesizes a 200 Hz sine WAV in-process so the test
suite has no external dependencies.
"""
import numpy as np
import pytest

parselmouth = pytest.importorskip("parselmouth")
from extractors.audio_features import extract, Contours, ProsodyUnavailable


def _write_sine_wav(path, freq_hz=200.0, duration_s=2.0, sr=16000, amplitude=0.5):
    """Synthesize a sine wave and write a 16-bit mono WAV. No external deps."""
    import wave, struct
    t = np.arange(int(sr * duration_s)) / sr
    samples = (amplitude * np.sin(2 * np.pi * freq_hz * t) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
        wf.writeframes(samples.tobytes())


def test_extract_returns_contours_dataclass(tmp_path):
    wav = tmp_path / "sine.wav"
    _write_sine_wav(wav, freq_hz=200.0)
    contours = extract(str(wav), pitch_floor_hz=75, pitch_ceiling_hz=500)
    assert isinstance(contours, Contours)
    assert contours.pitch_hz.ndim == 1
    assert contours.intensity_db.ndim == 1
    assert contours.times.ndim == 1
    assert contours.pitch_hz.shape == contours.times.shape
    assert contours.intensity_db.shape == contours.times.shape
    assert contours.frame_rate_hz > 0


def test_extract_tracks_200hz_sine_within_5pct(tmp_path):
    wav = tmp_path / "sine.wav"
    _write_sine_wav(wav, freq_hz=200.0)
    contours = extract(str(wav), pitch_floor_hz=75, pitch_ceiling_hz=500)
    voiced = contours.pitch_hz[~np.isnan(contours.pitch_hz)]
    assert len(voiced) > 0, "Sine should have voiced frames"
    # Allow 5% tolerance — Praat may dither slightly even on synthetic input
    assert 190 < np.median(voiced) < 210


def test_extract_speaker_median_is_finite(tmp_path):
    wav = tmp_path / "sine.wav"
    _write_sine_wav(wav, freq_hz=200.0)
    contours = extract(str(wav), pitch_floor_hz=75, pitch_ceiling_hz=500)
    assert np.isfinite(contours.speaker_median_pitch_hz)
    assert 75 < contours.speaker_median_pitch_hz < 500


def test_extract_hnr_is_finite(tmp_path):
    wav = tmp_path / "sine.wav"
    _write_sine_wav(wav, freq_hz=200.0)
    contours = extract(str(wav), pitch_floor_hz=75, pitch_ceiling_hz=500)
    # A pure sine has very high HNR (~60+ dB); not testing exact value, just finite.
    assert np.isfinite(contours.hnr_db)
    assert contours.hnr_db > 0


def test_extract_silence_has_no_voiced_frames(tmp_path):
    wav = tmp_path / "silence.wav"
    _write_sine_wav(wav, freq_hz=200.0, amplitude=0.0)  # all zeros
    contours = extract(str(wav), pitch_floor_hz=75, pitch_ceiling_hz=500)
    # All pitch frames should be NaN (unvoiced)
    assert np.isnan(contours.pitch_hz).all()


def test_extract_hnr_excludes_unvoiced_sentinel(tmp_path):
    """Regression: Praat marks unvoiced frames as -200 dB. The mean must
    exclude them, otherwise real speech with silences gets a wildly wrong HNR.
    """
    import wave
    wav = tmp_path / "tone_then_silence.wav"
    sr = 16000
    tone = (0.5 * np.sin(2 * np.pi * 200.0 * np.arange(sr) / sr) * 32767).astype(np.int16)
    silence = np.zeros(sr, dtype=np.int16)
    samples = np.concatenate([tone, silence])
    with wave.open(str(wav), "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
        wf.writeframes(samples.tobytes())

    contours = extract(str(wav), pitch_floor_hz=75, pitch_ceiling_hz=500)
    # Voiced HNR for a pure sine is ~100 dB. Bug would give ~-50.
    assert contours.hnr_db > 0, (
        f"HNR mean {contours.hnr_db} suggests unvoiced sentinels (-200 dB) "
        "are still being included in the average"
    )


def test_extract_populates_hnr_frame_arrays(tmp_path):
    wav = tmp_path / "sine.wav"
    _write_sine_wav(wav, freq_hz=200.0)
    contours = extract(str(wav), pitch_floor_hz=75, pitch_ceiling_hz=500)
    assert contours.hnr_frame_db is not None
    assert contours.hnr_frame_times is not None
    assert contours.hnr_frame_db.ndim == 1
    assert len(contours.hnr_frame_db) == len(contours.hnr_frame_times)
    finite_mask = np.isfinite(contours.hnr_frame_db)
    assert finite_mask.any(), "Sine wave should have some voiced (finite HNR) frames"
