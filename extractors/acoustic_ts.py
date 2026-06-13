"""Audio-only continuous acoustic features for TRF / cortical-tracking analyses.

Pure functions of the audio signal (no transcript, no EEG). Every feature is
returned as a slow time series at a caller-chosen analysis rate on a shared
canonical grid of N = round(duration * rate) samples, so multiple features
stack directly as TRF predictors.

The `rms` amplitude-envelope method faithfully replicates mTRFenvelope.m from
the mTRF-Toolbox (square -> moving-average downsample -> sqrt -> power-law
compress); `hilbert` is an alternative analytic-signal envelope.
"""
from __future__ import annotations

import numpy as np


def _grid_len(n_samples: int, sr: float, rate: float) -> int:
    """Canonical number of output frames for a signal of n_samples at sr."""
    return int(round((n_samples / sr) * rate))


def _fit_to_grid(arr: np.ndarray, n: int) -> np.ndarray:
    """Trim or edge-pad `arr` along axis 0 to exactly `n` rows."""
    if arr.shape[0] == n:
        return arr
    if arr.shape[0] > n:
        return arr[:n]
    if arr.shape[0] == 0:
        return np.zeros((n,) + arr.shape[1:], dtype=arr.dtype)
    pad_width = [(0, n - arr.shape[0])] + [(0, 0)] * (arr.ndim - 1)
    return np.pad(arr, pad_width, mode="edge")


def amplitude_envelope(y: np.ndarray, sr: float, rate: float, *,
                       method: str = "rms", comp: float = 0.3) -> np.ndarray:
    """Broadband amplitude envelope, power-law compressed, at `rate` Hz.

    method="rms" replicates mTRFenvelope.m: square -> frame-average every
    round(sr/rate) samples -> sqrt -> raise to `comp`.
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    n = _grid_len(y.size, sr, rate)
    if n <= 0:
        return np.zeros(0, dtype=np.float64)
    if method == "rms":
        # Average the squared signal over exactly N variable-width bins that
        # span the whole signal. This is the boxcar anti-alias + decimation of
        # mTRFenvelope.m (average the ~sr/rate nearest samples per output
        # sample), placed directly on the canonical grid so there is no
        # cumulative drift for non-integer sr/rate.
        edges = np.linspace(0, y.size, n + 1).round().astype(int)
        power = np.empty(n, dtype=np.float64)
        for i in range(n):
            seg = y[edges[i]:edges[i + 1]]
            power[i] = np.mean(seg ** 2) if seg.size else 0.0
        env = np.sqrt(power) ** comp
    elif method == "hilbert":
        from scipy.signal import hilbert, resample_poly
        from math import gcd
        analytic = np.abs(hilbert(y))
        # Resample the magnitude envelope from sr to rate with the built-in
        # FIR anti-alias filter. Use integer up/down derived from the ratio.
        target = int(round(rate))
        source = int(round(sr))
        g = gcd(target, source)
        env = resample_poly(analytic, up=target // g, down=source // g) ** comp
    else:
        raise ValueError(f"unknown envelope method: {method!r}")
    return _fit_to_grid(env, n)


def onset_envelope(env: np.ndarray) -> np.ndarray:
    """Half-wave-rectified first difference of the amplitude envelope.

    Keeps only positive-going changes (acoustic onsets); length is preserved
    by prepending 0 so it stays on the same grid as `env`.
    """
    env = np.asarray(env, dtype=np.float64).ravel()
    diff = np.diff(env, prepend=env[:1])
    return np.clip(diff, 0.0, None)


def mel_range(preset: str, sr: float) -> tuple[float, float]:
    """Resolve a preset name to (fmin, fmax) in Hz for a given sample rate."""
    nyq = sr / 2.0
    if preset == "speech":
        return (50.0, min(8000.0, nyq))
    if preset in ("music", "general"):
        return (0.0, nyq)
    raise ValueError(f"unknown mel preset: {preset!r}")


def log_mel_spectrogram(y: np.ndarray, sr: float, rate: float, *,
                        n_mels: int = 16, fmin: float = 50.0,
                        fmax: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Log-mel (dB) spectrogram on the canonical grid, plus mel center freqs.

    hop_length is set so the frame rate equals `rate`; a ~25 ms analysis
    window (next power of two >= 0.025*sr) is used.
    """
    import librosa
    y = np.asarray(y, dtype=np.float64).ravel()
    if fmax is None:
        fmax = sr / 2.0
    hop = max(1, int(round(sr / rate)))
    win = max(hop, int(round(0.025 * sr)))
    n_fft = 1 << (win - 1).bit_length()   # next power of two >= win
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop, win_length=win,
        n_mels=n_mels, fmin=fmin, fmax=fmax, power=2.0)
    logmel = librosa.power_to_db(mel, ref=np.max).T   # (frames, n_mels)
    freqs = librosa.mel_frequencies(n_mels=n_mels, fmin=fmin, fmax=fmax)
    n = _grid_len(y.size, sr, rate)
    return _fit_to_grid(logmel, n), np.asarray(freqs, dtype=np.float64)
