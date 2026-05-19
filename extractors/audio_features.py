"""Parselmouth (Praat) wrapper — one acoustic-analysis pass per stimulus.

extract(audio_path, ...) -> Contours dataclass holding:
    pitch_hz: 1-D np.ndarray, NaN at unvoiced frames
    intensity_db: 1-D np.ndarray, dB SPL
    times: 1-D np.ndarray, frame timestamps (seconds, aligned to pitch frames)
    frame_rate_hz: int (default 100, i.e. 10 ms time step)
    speaker_median_pitch_hz: float — semitone reference
    hnr_db: float — mean Harmonics-to-Noise Ratio over voiced frames

Raises ProsodyUnavailable if parselmouth isn't importable.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

try:
    import parselmouth  # type: ignore
    _PARSELMOUTH_OK = True
except ImportError:
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


def extract(audio_path: str, *,
            pitch_floor_hz: float = 75.0,
            pitch_ceiling_hz: float = 500.0,
            time_step_s: float = 0.01) -> Contours:
    """Single Parselmouth pass; returns Contours."""
    if not _PARSELMOUTH_OK:
        raise ProsodyUnavailable(
            "praat-parselmouth is not installed. Run: pip install praat-parselmouth")

    snd = parselmouth.Sound(audio_path)

    pitch = snd.to_pitch_ac(
        time_step=time_step_s,
        pitch_floor=pitch_floor_hz,
        pitch_ceiling=pitch_ceiling_hz,
    )
    pitch_values = pitch.selected_array['frequency'].astype(float)
    # Unvoiced frames come back as 0.0 in Praat; convert to NaN.
    pitch_values[pitch_values == 0.0] = np.nan
    pitch_times = pitch.xs().astype(float)

    intensity = snd.to_intensity(
        minimum_pitch=pitch_floor_hz,
        time_step=time_step_s,
    )
    int_values = intensity.values.flatten().astype(float)
    int_times = intensity.xs().astype(float)

    # Resample intensity onto pitch's time grid (handles minor edge offsets).
    intensity_db = np.interp(pitch_times, int_times, int_values,
                              left=np.nan, right=np.nan)

    # Speaker median pitch (used as semitone reference).
    voiced = pitch_values[~np.isnan(pitch_values)]
    speaker_median = float(np.median(voiced)) if voiced.size else float('nan')

    # HNR (mean over voiced frames). Praat marks unvoiced frames with the
    # sentinel value -200 dB; np.isfinite would let those through and drag
    # the mean down to ~-50 on real speech with silences. Drop them.
    try:
        harmonicity = snd.to_harmonicity_cc(
            time_step=time_step_s,
            minimum_pitch=pitch_floor_hz,
        )
        hnr_vals = harmonicity.values.flatten()
        hnr_vals = hnr_vals[np.isfinite(hnr_vals) & (hnr_vals > -200.0)]
        hnr_mean = float(np.mean(hnr_vals)) if hnr_vals.size else float('nan')
    except Exception:
        hnr_mean = float('nan')

    return Contours(
        pitch_hz=pitch_values,
        intensity_db=intensity_db,
        times=pitch_times,
        frame_rate_hz=int(round(1.0 / time_step_s)),
        speaker_median_pitch_hz=speaker_median,
        hnr_db=hnr_mean,
    )
