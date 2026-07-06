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

# mTRFenvelope.m's default power-law compression exponent (Stevens 1955).
# Using exactly log10(2) keeps the `rms` envelope bit-for-bit with mTRF defaults;
# any other value (e.g. a rounded 0.3) introduces a systematic, value-dependent
# offset in the final `sqrt(meanSquare) ** comp` step.
MTRF_COMP = float(np.log10(2))


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


def _mround(v: float) -> int:
    """MATLAB-style round: half away from zero (numpy/Python round to even)."""
    return int(np.floor(np.abs(v) + 0.5) * np.sign(v))


def amplitude_envelope(y: np.ndarray, sr: float, rate: float, *,
                       method: str = "rms", comp: float = MTRF_COMP,
                       window: float = 1.0) -> np.ndarray:
    """Broadband amplitude envelope, power-law compressed, at `rate` Hz.

    method="rms" is a faithful port of mTRFenvelope.m -> mTRFresample.m from the
    mTRF-Toolbox (no buffer, dim=1): square -> centered moving-average resample
    to `rate` Hz -> sqrt -> raise to `comp`. Output frame i is centered at
    t = i/rate over a (window/rate)-wide averaging window; the first frame sits
    at t=0 as a clamped half-window, matching mTRF bit-for-bit. `window` > 1
    overlaps adjacent frames (extra smoothing) without changing the frame rate.
    mTRF's default compression is log10(2) ~= 0.301 (Stevens 1955); we use that
    exact value as the `comp` default (`MTRF_COMP`) so the output matches
    mTRFenvelope.m bit-for-bit. A rounded 0.3 would, on normalized audio (values
    < 1), make every frame slightly larger than MATLAB's.
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    n = _grid_len(y.size, sr, rate)
    if n <= 0:
        return np.zeros(0, dtype=np.float64)
    if method == "rms":
        # mTRFenvelope: x = x.^2; y = mTRFresample(...); y = sqrt(y).^comp.
        # mTRFresample centers frame i at t=i/rate and averages the squared
        # samples in [round(sr*(t-h)), round(sr*(t+h))] (h = half window in s),
        # clamped to the signal bounds. Reproduced exactly here.
        x2 = y ** 2
        nin = y.size
        half = 0.5 * window / rate
        env = np.empty(n, dtype=np.float64)
        for i in range(n):
            ti = i / rate
            start = max(0, _mround(sr * (ti - half)))
            end = min(nin, _mround(sr * (ti + half)) + 1)
            env[i] = x2[start:end].mean() if end > start else 0.0
        env = np.sqrt(env) ** comp
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
    # Absolute dB (ref=1.0): 10*log10(power) on a fixed reference, so values are
    # comparable across files. top_db=None disables the per-file-max dynamic-range
    # clip (which would otherwise reintroduce per-file coupling at the noise
    # floor); the floor is then the fixed amin-based level (~ -100 dB).
    logmel = librosa.power_to_db(mel, ref=1.0, top_db=None).T   # (frames, n_mels)
    freqs = librosa.mel_frequencies(n_mels=n_mels, fmin=fmin, fmax=fmax)
    n = _grid_len(y.size, sr, rate)
    return _fit_to_grid(logmel, n), np.asarray(freqs, dtype=np.float64)


def mfcc(y: np.ndarray, sr: float, rate: float, *, n_mfcc: int = 13,
         n_mels: int = 16, fmin: float = 50.0,
         fmax: float | None = None) -> np.ndarray:
    """MFCCs on the canonical grid.

    NOTE: for TRF / cortical-tracking work the log-mel spectrogram is usually
    preferred over MFCCs — the DCT that produces MFCCs removes the tonotopic
    (frequency-band) interpretability that makes mel features map onto
    auditory-cortex organization. MFCCs are provided for completeness, not as
    the primary feature.
    """
    import librosa
    y = np.asarray(y, dtype=np.float64).ravel()
    if fmax is None:
        fmax = sr / 2.0
    hop = max(1, int(round(sr / rate)))
    win = max(hop, int(round(0.025 * sr)))
    n_fft = 1 << (win - 1).bit_length()
    m = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop,
        win_length=win, n_mels=n_mels, fmin=fmin, fmax=fmax)
    n = _grid_len(y.size, sr, rate)
    return _fit_to_grid(m.T, n)


def extract_acoustic(audio_path: str, rate: float, cfg: dict) -> dict:
    """Compute all enabled acoustic time-series features for one audio file.

    Returns a dict of named arrays (only the enabled ones) plus `fs`, `t`,
    `mel_center_freqs` (when logmel/mfcc enabled), and a `config` provenance
    dict. All feature arrays share exactly N = round(duration*rate) rows.
    """
    import librosa
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    y = np.asarray(y, dtype=np.float64)
    n = _grid_len(y.size, sr, rate)
    fmin, fmax = mel_range(cfg.get("mel_preset", "speech"), sr)

    out: dict = {"fs": float(rate), "t": (np.arange(n) / rate).astype(np.float64)}

    env = None
    if cfg.get("envelope", True) or cfg.get("onset_env", True):
        env = amplitude_envelope(y, sr, rate, method=cfg.get("env_method", "rms"),
                                 comp=float(cfg.get("env_comp", MTRF_COMP)),
                                 window=float(cfg.get("env_window", 1.0)))
    if cfg.get("envelope", True):
        out["envelope"] = env
    if cfg.get("onset_env", True):
        out["onset_envelope"] = onset_envelope(env)
    if cfg.get("logmel", True):
        logmel, freqs = log_mel_spectrogram(
            y, sr, rate, n_mels=int(cfg.get("n_mels", 16)), fmin=fmin, fmax=fmax)
        out["logmel"] = logmel
        out["mel_center_freqs"] = freqs
    if cfg.get("mfcc", False):
        out["mfcc"] = mfcc(y, sr, rate, n_mfcc=int(cfg.get("n_mfcc", 13)),
                           n_mels=int(cfg.get("n_mels", 16)), fmin=fmin, fmax=fmax)
        if "mel_center_freqs" not in out:
            out["mel_center_freqs"] = librosa_mel_freqs(int(cfg.get("n_mels", 16)), fmin, fmax)

    out["config"] = {
        "analysis_rate": float(rate), "env_method": cfg.get("env_method", "rms"),
        "env_comp": float(cfg.get("env_comp", MTRF_COMP)),
        "env_window": float(cfg.get("env_window", 1.0)),
        "n_mels": int(cfg.get("n_mels", 16)), "n_mfcc": int(cfg.get("n_mfcc", 13)),
        "fmin": float(fmin), "fmax": float(fmax),
        "mel_preset": cfg.get("mel_preset", "speech"),
        "source_sr": float(sr), "duration_s": float(y.size / sr),
    }
    return out


def librosa_mel_freqs(n_mels: int, fmin: float, fmax: float) -> np.ndarray:
    """Return the mel filterbank center frequencies as a float64 array."""
    import librosa
    return np.asarray(librosa.mel_frequencies(n_mels=n_mels, fmin=fmin, fmax=fmax),
                      dtype=np.float64)
