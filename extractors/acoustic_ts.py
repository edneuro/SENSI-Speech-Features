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
        raise NotImplementedError  # Task 2
    else:
        raise ValueError(f"unknown envelope method: {method!r}")
    return _fit_to_grid(env, n)
