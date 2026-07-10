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


def test_log_mel_tone_lands_in_nearest_center_band():
    # A pure tone must peak in the band whose center frequency is nearest the
    # tone (within one band of it), proving the mel mapping places energy at the
    # right frequency — not merely "somewhere low".
    sr, n_mels = 16000, 32
    for f in (200.0, 800.0, 3000.0):
        y, _ = _tone(sr=sr, dur=1.0, f=f)
        logmel, freqs = acoustic_ts.log_mel_spectrogram(
            y, sr, 128, n_mels=n_mels, fmin=50.0, fmax=8000.0)
        peak = int(np.argmax(logmel.mean(axis=0)))
        nearest = int(np.argmin(np.abs(freqs - f)))
        assert abs(peak - nearest) <= 1, f"f={f}: peak {peak} vs nearest {nearest}"


def test_log_mel_center_freqs_are_monotonic_and_span_range():
    _, freqs = acoustic_ts.log_mel_spectrogram(
        _tone()[0], 16000, 128, n_mels=16, fmin=50.0, fmax=8000.0)
    assert np.all(np.diff(freqs) > 0)            # strictly increasing
    assert abs(freqs[0] - 50.0) < 1.0            # first band at fmin
    assert abs(freqs[-1] - 8000.0) < 1.0         # last band at fmax


def test_log_mel_equals_hand_composed_stft_filterbank_db():
    # Validate the full pipeline our wrapper claims to run: |STFT|^2 -> mel
    # filterbank -> absolute dB. Rebuilt from librosa primitives with the same
    # parameters our wrapper derives, so it must match bit-for-bit.
    import librosa
    y, sr = _tone(dur=1.0, f=440.0)
    rate, n_mels, fmin, fmax = 128, 16, 50.0, 8000.0
    hop = max(1, int(round(sr / rate)))
    win = max(hop, int(round(0.025 * sr)))
    n_fft = 1 << (win - 1).bit_length()
    power = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop,
                               win_length=win)) ** 2
    fb = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels,
                             fmin=fmin, fmax=fmax)
    expected = librosa.power_to_db(fb @ power, ref=1.0, top_db=None).T
    n = acoustic_ts._grid_len(y.size, sr, rate)
    expected = acoustic_ts._fit_to_grid(expected, n)
    ours, _ = acoustic_ts.log_mel_spectrogram(
        y, sr, rate, n_mels=n_mels, fmin=fmin, fmax=fmax)
    np.testing.assert_allclose(ours, expected, atol=1e-9, rtol=0)


def _slaney_mel_filterbank(sr, n_fft, n_mels, fmin, fmax):
    """Independent numpy port of the Slaney-scale mel filterbank algorithm.

    Written from the published Slaney auditory-toolbox design (linear below
    1 kHz, log above) with area normalization — NOT copied from librosa — so
    agreement with librosa.filters.mel confirms librosa builds the standard
    filterbank our features rely on.
    """
    def hz_to_mel(f):
        f = np.asarray(f, dtype=np.float64)
        f_sp = 200.0 / 3.0
        mels = f / f_sp
        min_log_hz, logstep = 1000.0, np.log(6.4) / 27.0
        min_log_mel = min_log_hz / f_sp
        hi = f >= min_log_hz
        mels[hi] = min_log_mel + np.log(f[hi] / min_log_hz) / logstep
        return mels

    def mel_to_hz(m):
        m = np.asarray(m, dtype=np.float64)
        f_sp = 200.0 / 3.0
        freqs = f_sp * m
        min_log_hz, logstep = 1000.0, np.log(6.4) / 27.0
        min_log_mel = min_log_hz / f_sp
        hi = m >= min_log_mel
        freqs[hi] = min_log_hz * np.exp(logstep * (m[hi] - min_log_mel))
        return freqs

    n_bins = 1 + n_fft // 2
    fftfreqs = np.linspace(0.0, sr / 2.0, n_bins)
    mel_pts = np.linspace(hz_to_mel([fmin])[0], hz_to_mel([fmax])[0], n_mels + 2)
    freqs = mel_to_hz(mel_pts)                       # band edges, length n_mels+2
    fdiff = np.diff(freqs)
    ramps = freqs[:, None] - fftfreqs[None, :]
    weights = np.zeros((n_mels, n_bins))
    for i in range(n_mels):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        weights[i] = np.maximum(0.0, np.minimum(lower, upper))
    enorm = 2.0 / (freqs[2:n_mels + 2] - freqs[:n_mels])
    return weights * enorm[:, None]


def test_mel_filterbank_matches_independent_slaney_port():
    import librosa
    sr, n_fft, n_mels, fmin, fmax = 16000, 512, 32, 50.0, 8000.0
    # librosa.filters.mel defaults to float32; ask for float64 to compare at
    # full precision against the independent port.
    ours = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels,
                               fmin=fmin, fmax=fmax, dtype=np.float64)
    ref = _slaney_mel_filterbank(sr, n_fft, n_mels, fmin, fmax)
    np.testing.assert_allclose(ours, ref, atol=1e-12, rtol=0)


def test_log_mel_silence_hits_fixed_floor_not_neg_inf():
    # Absolute dB with amin=1e-10 floors silence at ~ -100 dB (finite), not
    # -inf, and not the per-file-relative 0 the old ref=np.max produced.
    y = np.zeros(16000, dtype=np.float64)
    logmel, _ = acoustic_ts.log_mel_spectrogram(
        y, 16000, 128, n_mels=16, fmin=50.0, fmax=8000.0)
    assert np.all(np.isfinite(logmel))
    np.testing.assert_allclose(logmel, -100.0, atol=1e-6)


def test_log_mel_is_absolute_db_not_per_file_max():
    # Scaling the whole signal by a constant gain must shift the log-mel by a
    # uniform 20*log10(g) dB everywhere (absolute dB). Under the old ref=np.max
    # the shift would be zero, because the per-file max scales with the signal.
    # Broadband noise keeps every band well above the fixed amin floor, so the
    # shift is uniform (a pure tone leaves silent bands pinned at the floor).
    rng = np.random.default_rng(0)
    sr = 16000
    y = 0.1 * rng.standard_normal(sr)   # 1 s of broadband noise
    g = 2.0
    base, _ = acoustic_ts.log_mel_spectrogram(y, sr, 128, n_mels=16,
                                              fmin=50.0, fmax=8000.0)
    louder, _ = acoustic_ts.log_mel_spectrogram(g * y, sr, 128, n_mels=16,
                                                fmin=50.0, fmax=8000.0)
    diff = louder - base
    np.testing.assert_allclose(diff, 20.0 * np.log10(g), atol=1e-6)


def test_mfcc_shape_on_grid():
    y, sr = _tone(dur=1.0)
    rate, n_mfcc = 128, 13
    m = acoustic_ts.mfcc(y, sr, rate, n_mfcc=n_mfcc, n_mels=16,
                         fmin=50.0, fmax=8000.0)
    assert m.shape == (round(1.0 * rate), n_mfcc)


def test_mfcc_is_dct_of_log_mel():
    # An MFCC *is* the type-II orthonormal DCT of the log-mel spectrogram,
    # truncated to n_mfcc coefficients. Rebuild that directly and assert our
    # mfcc() matches — validates the transform end to end.
    import librosa
    from scipy.fftpack import dct
    y, sr = _tone(dur=1.0, f=300.0)
    rate, n_mfcc, n_mels, fmin, fmax = 128, 13, 16, 50.0, 8000.0
    hop = max(1, int(round(sr / rate)))
    win = max(hop, int(round(0.025 * sr)))
    n_fft = 1 << (win - 1).bit_length()
    # librosa.feature.mfcc's internal log-mel uses power_to_db defaults (top_db=80).
    logmel = librosa.power_to_db(librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop, win_length=win,
        n_mels=n_mels, fmin=fmin, fmax=fmax, power=2.0))
    expected = dct(logmel, axis=0, type=2, norm="ortho")[:n_mfcc]
    n = acoustic_ts._grid_len(y.size, sr, rate)
    expected = acoustic_ts._fit_to_grid(expected.T, n)
    ours = acoustic_ts.mfcc(y, sr, rate, n_mfcc=n_mfcc, n_mels=n_mels,
                            fmin=fmin, fmax=fmax)
    np.testing.assert_allclose(ours, expected, atol=1e-9, rtol=0)


def test_mfcc_gain_changes_only_c0():
    # A gain g adds a constant 20*log10(g) to every log-mel band; the DCT puts a
    # constant entirely into the 0th coefficient. So C0 shifts by exactly
    # 20*log10(g)*sqrt(n_mels) and all higher coefficients are unchanged.
    rng = np.random.default_rng(2)
    sr, n_mels, n_mfcc = 16000, 16, 13
    y = 0.1 * rng.standard_normal(sr)        # broadband, well above the floor
    g = 2.0
    base = acoustic_ts.mfcc(y, sr, 128, n_mfcc=n_mfcc, n_mels=n_mels,
                            fmin=50.0, fmax=8000.0)
    louder = acoustic_ts.mfcc(g * y, sr, 128, n_mfcc=n_mfcc, n_mels=n_mels,
                              fmin=50.0, fmax=8000.0)
    np.testing.assert_allclose(louder[:, 1:], base[:, 1:], atol=1e-6)
    expected_dc = 20.0 * np.log10(g) * np.sqrt(n_mels)
    np.testing.assert_allclose(louder[:, 0] - base[:, 0], expected_dc, atol=1e-6)


def test_extract_acoustic_all_features_share_grid(tmp_path):
    import soundfile as sf
    y, sr = _tone(dur=1.0)
    wav = tmp_path / "stim.wav"
    sf.write(str(wav), y, sr)
    rate = 128
    out = acoustic_ts.extract_acoustic(str(wav), rate, {
        "envelope": True, "onset_env": True, "logmel": True, "mfcc": True,
        "env_method": "rms", "env_comp": 0.3, "n_mels": 16, "n_mfcc": 13,
        "mel_preset": "speech",
    })
    n = out["t"].shape[0]
    assert out["fs"] == rate
    assert out["envelope"].shape == (n,)
    assert out["onset_envelope"].shape == (n,)
    assert out["logmel"].shape[0] == n and out["logmel"].shape[1] == 16
    assert out["mfcc"].shape[0] == n and out["mfcc"].shape[1] == 13
    assert out["mel_center_freqs"].shape == (16,)
    assert "config" in out


def test_extract_acoustic_respects_disabled_features(tmp_path):
    import soundfile as sf
    y, sr = _tone(dur=0.5)
    wav = tmp_path / "stim.wav"
    sf.write(str(wav), y, sr)
    out = acoustic_ts.extract_acoustic(str(wav), 100, {
        "envelope": True, "onset_env": False, "logmel": False, "mfcc": False,
        "env_method": "rms", "env_comp": 0.3, "n_mels": 16, "n_mfcc": 13,
        "mel_preset": "speech",
    })
    assert "envelope" in out
    assert "onset_envelope" not in out
    assert "logmel" not in out
    assert "mfcc" not in out


# ---------------------------------------------------------------------------
# Fidelity to mTRFenvelope.m / mTRFresample.m (mTRF-Toolbox).
# The reference below is a direct transcription of those two .m files
# (no buffer, dim=1): square -> centered moving-average resample -> sqrt ->
# power-law compress. amplitude_envelope(method="rms") must match it exactly.
# ---------------------------------------------------------------------------

def _matlab_round(v: float) -> int:
    # MATLAB round(): half away from zero (numpy/Python round to even).
    return int(np.floor(np.abs(v) + 0.5) * np.sign(v))


def _mtrf_reference(x, fsin, fsout, window=1.0, comp=np.log10(2)):
    x = np.asarray(x, dtype=np.float64).ravel()
    x2 = x ** 2                                   # mTRFenvelope: square first
    nin = x2.size
    nout = _matlab_round(nin / fsin * fsout)
    t = np.arange(nout) / fsout                   # mTRFresample: (0:nout-1)/fsout
    n = 0.5 * window / fsout                       # half-window in seconds
    y = np.empty(nout)
    for i in range(nout):
        idx1 = max(0, _matlab_round(fsin * (t[i] - n)))           # 0-based start
        idx2 = min(nin, _matlab_round(fsin * (t[i] + n)) + 1)     # 0-based end (excl.)
        y[i] = x2[idx1:idx2].mean()
    return np.sqrt(y) ** comp                       # mTRFenvelope: sqrt then ^comp


def test_rms_matches_mtrfenvelope_reference():
    rng = np.random.default_rng(0)
    fsin, fsout = 22050, 128
    tt = np.arange(int(fsin * 0.8)) / fsin
    x = (0.4 * np.sin(2 * np.pi * 180 * tt) * (1 + 0.5 * np.sin(2 * np.pi * 4 * tt))
         + 0.02 * rng.standard_normal(tt.size))
    ours = acoustic_ts.amplitude_envelope(x, fsin, fsout, method="rms", comp=np.log10(2))
    ref = _mtrf_reference(x, fsin, fsout)
    assert ours.shape == ref.shape
    np.testing.assert_allclose(ours, ref, atol=1e-9, rtol=0)


def test_rms_window_param_matches_reference_and_smooths():
    rng = np.random.default_rng(1)
    fsin, fsout = 16000, 100
    x = rng.standard_normal(int(fsin * 0.5))
    e1 = acoustic_ts.amplitude_envelope(x, fsin, fsout, method="rms", comp=1.0, window=1.0)
    e4 = acoustic_ts.amplitude_envelope(x, fsin, fsout, method="rms", comp=1.0, window=4.0)
    assert e1.shape == e4.shape == (round(0.5 * fsout),)
    # A wider averaging window (overlap) reduces frame-to-frame variability.
    assert np.std(np.diff(e4)) < np.std(np.diff(e1))
    # window>1 must also match the mTRF reference exactly.
    np.testing.assert_allclose(
        e4, _mtrf_reference(x, fsin, fsout, window=4.0, comp=1.0), atol=1e-9, rtol=0)
