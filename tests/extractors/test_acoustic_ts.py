import numpy as np
import pytest

from extractors import acoustic_ts


def _tone(sr=16000, dur=1.0, f=200.0, amp=0.5):
    t = np.arange(int(sr * dur)) / sr
    return (amp * np.sin(2 * np.pi * f * t)).astype(np.float64), sr


def test_rms_envelope_length_matches_grid():
    y, sr = _tone(dur=1.0)
    rate = 128
    env = acoustic_ts.amplitude_envelope(y, sr, rate, method="rms", comp=0.3)
    assert env.shape == (round(1.0 * rate),)


def test_rms_envelope_constant_tone_is_flat():
    # A steady tone -> envelope roughly constant across the interior frames.
    y, sr = _tone(dur=1.0, amp=0.5)
    env = acoustic_ts.amplitude_envelope(y, sr, 128, method="rms", comp=1.0)
    interior = env[5:-5]
    assert np.std(interior) < 0.05 * np.mean(interior)


def test_rms_envelope_compression_monotonic():
    # Lower exponent compresses the dynamic range: a quiet vs loud ratio
    # moves toward 1 as comp -> 0.
    y_loud, sr = _tone(amp=0.8)
    y_quiet, _ = _tone(amp=0.1)
    def ratio(comp):
        a = acoustic_ts.amplitude_envelope(y_loud, sr, 128, method="rms", comp=comp).mean()
        b = acoustic_ts.amplitude_envelope(y_quiet, sr, 128, method="rms", comp=comp).mean()
        return a / b
    assert ratio(0.3) < ratio(1.0)
