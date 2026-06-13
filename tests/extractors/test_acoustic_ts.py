import numpy as np

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


def test_rms_envelope_exact_length_non_integer_ratio():
    # sr/rate is non-integer here (44100/128); length must still be exactly N
    # with no drift/padding.
    y, sr = _tone(sr=44100, dur=2.0)
    rate = 128
    env = acoustic_ts.amplitude_envelope(y, sr, rate, method="rms", comp=0.3)
    assert env.shape == (round(2.0 * rate),)


def test_fit_to_grid_trims_pads_and_handles_empty():
    a = np.arange(5, dtype=np.float64)
    np.testing.assert_array_equal(acoustic_ts._fit_to_grid(a, 3), [0, 1, 2])
    np.testing.assert_array_equal(acoustic_ts._fit_to_grid(a, 7), [0, 1, 2, 3, 4, 4, 4])
    out = acoustic_ts._fit_to_grid(np.zeros((0, 4)), 3)
    assert out.shape == (3, 4)


def test_hilbert_envelope_length_and_shape_matches_rms():
    y, sr = _tone(dur=1.0)
    rate = 128
    env_h = acoustic_ts.amplitude_envelope(y, sr, rate, method="hilbert", comp=0.3)
    env_r = acoustic_ts.amplitude_envelope(y, sr, rate, method="rms", comp=0.3)
    assert env_h.shape == env_r.shape == (round(1.0 * rate),)


def test_hilbert_and_rms_agree_on_steady_tone():
    # For a pure sine, hilbert gives peak amplitude and rms gives peak/sqrt(2).
    # Both should be within sqrt(2) (≈1.41) of each other; we allow up to 50%
    # tolerance so the test passes with correct implementations of both methods.
    y, sr = _tone(dur=1.0, amp=0.5)
    h = acoustic_ts.amplitude_envelope(y, sr, 128, method="hilbert", comp=1.0)[5:-5]
    r = acoustic_ts.amplitude_envelope(y, sr, 128, method="rms", comp=1.0)[5:-5]
    assert abs(h.mean() - r.mean()) < 0.50 * r.mean()


def test_onset_envelope_nonnegative_and_length_preserved():
    env = np.array([0.0, 1.0, 1.0, 0.5, 2.0, 2.0], dtype=np.float64)
    onset = acoustic_ts.onset_envelope(env)
    assert onset.shape == env.shape
    assert np.all(onset >= 0)


def test_onset_envelope_keeps_rises_drops_falls():
    env = np.array([0.0, 1.0, 1.0, 0.5, 2.0], dtype=np.float64)
    onset = acoustic_ts.onset_envelope(env)
    # diff = [_, +1, 0, -0.5, +1.5]; HWR keeps positives, first elem 0.
    np.testing.assert_allclose(onset, [0.0, 1.0, 0.0, 0.0, 1.5])


def test_mel_range_for_preset_returns_speech_and_general():
    assert acoustic_ts.mel_range("speech", sr=16000) == (50.0, 8000.0)
    fmin, fmax = acoustic_ts.mel_range("music", sr=16000)
    assert fmin == 0.0 and fmax == 8000.0  # Nyquist for sr=16000


def test_log_mel_shape_and_center_freqs():
    y, sr = _tone(dur=1.0)
    rate, n_mels = 128, 16
    logmel, freqs = acoustic_ts.log_mel_spectrogram(
        y, sr, rate, n_mels=n_mels, fmin=50.0, fmax=8000.0)
    assert logmel.shape == (round(1.0 * rate), n_mels)
    assert freqs.shape == (n_mels,)


def test_log_mel_tone_energy_in_expected_band():
    y, sr = _tone(dur=1.0, f=200.0)
    logmel, freqs = acoustic_ts.log_mel_spectrogram(
        y, sr, 128, n_mels=16, fmin=50.0, fmax=8000.0)
    peak_band = int(np.argmax(logmel.mean(axis=0)))
    assert peak_band < 8  # lower half of the 16 bands
