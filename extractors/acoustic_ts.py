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
    if method == "rms":
        L = max(1, int(round(sr / rate)))
        n_frames = y.size // L
        framed = y[:n_frames * L].reshape(n_frames, L)
        power = (framed ** 2).mean(axis=1)        # boxcar avg = anti-alias
        env = np.sqrt(power) ** comp
    elif method == "hilbert":
        raise NotImplementedError  # Task 2
    else:
        raise ValueError(f"unknown envelope method: {method!r}")
    return _fit_to_grid(env, n)
