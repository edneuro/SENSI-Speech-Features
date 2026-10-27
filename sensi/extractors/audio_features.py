"""Parselmouth (Praat) wrapper — one acoustic-analysis pass per stimulus.

extract(audio_path, ...) -> Contours dataclass holding:
    pitch_hz: 1-D np.ndarray, NaN at unvoiced frames
    intensity_db: 1-D np.ndarray, dB SPL
    times: 1-D np.ndarray, frame timestamps (seconds, aligned to pitch frames)
    frame_rate_hz: int (default 100, i.e. 10 ms time step)
    speaker_median_pitch_hz: float — semitone reference
    hnr_db: float — mean Harmonics-to-Noise Ratio over voiced frames
    hnr_frame_db: np.ndarray | None — per-frame HNR (NaN=unvoiced), harmonicity time grid
    hnr_frame_times: np.ndarray | None — timestamps matching hnr_frame_db

Raises ProsodyUnavailable if parselmouth isn't importable.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

try:
    import parselmouth  # type: ignore
    _PARSELMOUTH_OK = True
except ImportError:
    parselmouth = None  # sentinel: extract() raises ProsodyUnavailable
    _PARSELMOUTH_OK = False


class ProsodyUnavailable(RuntimeError):
    """Raised when parselmouth is not importable; callers emit NaN cols."""


@dataclass(frozen=True)
class Contours:
    pitch_hz: np.ndarray
    intensity_db: np.ndarray
    times: np.ndarray
    frame_rate_hz: int
    speaker_median_pitch_hz: float
    hnr_db: float
    hnr_frame_db: np.ndarray | None = None    # per-frame HNR, NaN=unvoiced
    hnr_frame_times: np.ndarray | None = None  # timestamps matching hnr_frame_db


def extract(audio_path: str, *,
            pitch_floor_hz: float = 75.0,
            pitch_ceiling_hz: float = 500.0,
            time_step_s: float = 0.01) -> Contours:
    """Single Parselmouth pass; returns Contours."""
    if parselmouth is None:
        raise ProsodyUnavailable(
            "praat-parselmouth is not installed. Run: pip install praat-parselmouth")

    snd = parselmouth.Sound(audio_path)

    pitch = snd.to_pitch_ac(
        time_step=time_step_s,
        pitch_floor=pitch_floor_hz,
        pitch_ceiling=pitch_ceiling_hz,
    )
    # np.asarray with explicit dtype guards against NumPy 2.0's stricter
    # ufunc casting rules when parselmouth returns non-standard array types.
    pitch_raw = np.asarray(pitch.selected_array['frequency'], dtype=np.float64)
    # selected_array['frequency'] may be 2D (n_frames, n_candidates) in some
    # parselmouth versions; take the first column (best candidate).
    if pitch_raw.ndim > 1:
        pitch_raw = pitch_raw[:, 0]
    # Unvoiced frames come back as 0.0 in Praat; convert to NaN.
    pitch_values = np.where(pitch_raw == 0.0, np.nan, pitch_raw)
    pitch_times = np.asarray(pitch.xs(), dtype=np.float64).ravel()

    intensity = snd.to_intensity(
        minimum_pitch=pitch_floor_hz,
        time_step=time_step_s,
    )
    int_values = np.asarray(intensity.values, dtype=np.float64).ravel()
    int_times = np.asarray(intensity.xs(), dtype=np.float64).ravel()

    # Resample intensity onto pitch's time grid (handles minor edge offsets).
    intensity_db = np.interp(pitch_times, int_times, int_values,
                              left=np.nan, right=np.nan)

    # Speaker median pitch (used as semitone reference).
    voiced = pitch_values[~np.isnan(pitch_values)]
    speaker_median = float(np.median(voiced)) if voiced.size else float('nan')

    # HNR (mean over voiced frames). Praat marks unvoiced frames with the
    # sentinel value -200 dB; drop them before averaging.
    hnr_mean = float('nan')
    hnr_frame_db_arr: np.ndarray | None = None
    hnr_frame_times_arr: np.ndarray | None = None
    try:
        harmonicity = snd.to_harmonicity_cc(
            time_step=time_step_s,
            minimum_pitch=pitch_floor_hz,
        )
        hnr_vals_raw = np.asarray(harmonicity.values, dtype=np.float64).ravel()
        hnr_times_raw = np.asarray(harmonicity.xs(), dtype=np.float64).ravel()
        hnr_vals_clean = np.where(hnr_vals_raw <= -200.0, np.nan, hnr_vals_raw)
        voiced_hnr = hnr_vals_clean[np.isfinite(hnr_vals_clean)]
        hnr_mean = float(np.mean(voiced_hnr)) if voiced_hnr.size else float('nan')
        hnr_frame_db_arr = hnr_vals_clean
        hnr_frame_times_arr = hnr_times_raw
    except Exception:
        pass  # hnr_mean stays nan; frame arrays stay None

    return Contours(
        pitch_hz=pitch_values,
        intensity_db=intensity_db,
        times=pitch_times,
        frame_rate_hz=int(round(1.0 / time_step_s)),
        speaker_median_pitch_hz=speaker_median,
        hnr_db=hnr_mean,
        hnr_frame_db=hnr_frame_db_arr,
        hnr_frame_times=hnr_frame_times_arr,
    )
