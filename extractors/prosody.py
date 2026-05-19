"""Word- and document-level prosody aggregators.

Consumes a Contours object from extractors.audio_features and an MFA-aligned
word DataFrame. Pure pandas/numpy; no Parselmouth calls happen here.

See docs/superpowers/specs/2026-05-12-prosody-design.md §5–7 for the column
definitions and edge-case rules.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from extractors.audio_features import Contours


def _semitones_from_hz(hz_array: np.ndarray, ref_hz: float) -> np.ndarray:
    """Convert Hz array to semitones relative to ref_hz. NaN-safe."""
    if not np.isfinite(ref_hz) or ref_hz <= 0:
        return np.full_like(hz_array, np.nan)
    return 12.0 * np.log2(hz_array / ref_hz)


def _slice(times: np.ndarray, values: np.ndarray,
           onset: float, end: float) -> np.ndarray:
    """Return values whose times fall in [onset, end)."""
    if not np.isfinite(onset) or not np.isfinite(end):
        return np.array([], dtype=float)
    mask = (times >= onset) & (times < end)
    return values[mask]


def _agg(values: np.ndarray) -> tuple[float, float, float, float]:
    """(mean, p5, p95, std) over non-NaN entries; all NaN if no valid frames."""
    finite = values[~np.isnan(values)]
    if finite.size == 0:
        return (np.nan, np.nan, np.nan, np.nan)
    if finite.size == 1:
        v = float(finite[0])
        return (v, v, v, 0.0)
    return (
        float(np.mean(finite)),
        float(np.percentile(finite, 5)),
        float(np.percentile(finite, 95)),
        float(np.std(finite)),
    )


def add_word_prosody(df: pd.DataFrame, contours: Contours,
                     *, pitch_units: str = "semitones") -> pd.DataFrame:
    """Append 10 prosody columns to df. df is not modified in place."""
    df = df.copy()
    pitch = contours.pitch_hz
    if pitch_units == "semitones":
        pitch = _semitones_from_hz(pitch, contours.speaker_median_pitch_hz)
    elif pitch_units != "hz":
        raise ValueError(f"pitch_units must be 'hz' or 'semitones', got {pitch_units!r}")

    onsets = df["Onset"].to_numpy() if "Onset" in df.columns else np.full(len(df), np.nan)
    ends = df["End"].to_numpy() if "End" in df.columns else np.full(len(df), np.nan)

    pitch_aggs, vol_aggs = [], []
    for on, en in zip(onsets, ends):
        if not (np.isfinite(on) and np.isfinite(en)):
            pitch_aggs.append((np.nan,) * 4)
            vol_aggs.append((np.nan,) * 4)
            continue
        pitch_aggs.append(_agg(_slice(contours.times, pitch, on, en)))
        vol_aggs.append(_agg(_slice(contours.times, contours.intensity_db, on, en)))

    p_mean, p_min, p_max, p_std = zip(*pitch_aggs) if pitch_aggs else ([], [], [], [])
    v_mean, v_min, v_max, v_std = zip(*vol_aggs) if vol_aggs else ([], [], [], [])

    df["Pitch_Mean"]  = list(p_mean)
    df["Pitch_Min"]   = list(p_min)
    df["Pitch_Max"]   = list(p_max)
    df["Pitch_Std"]   = list(p_std)
    df["Volume_Mean"] = list(v_mean)
    df["Volume_Min"]  = list(v_min)
    df["Volume_Max"]  = list(v_max)
    df["Volume_Std"]  = list(v_std)

    # Word-to-word deltas (NaN for first word).
    df["Pitch_Delta"] = df["Pitch_Mean"].diff()
    df["Volume_Delta"] = df["Volume_Mean"].diff()
    return df


def _count_syllables(token: str) -> int:
    """Surface-form syllable count via pyphen. Returns 1 on lookup failure
    or 0 for empty tokens. Returns -1 sentinel if pyphen is unavailable."""
    try:
        import pyphen  # type: ignore
    except ImportError:
        return -1
    if not hasattr(_count_syllables, "_dic"):
        _count_syllables._dic = pyphen.Pyphen(lang="en_US")
    if not token:
        return 0
    return _count_syllables._dic.inserted(token).count("-") + 1


def compute_doc_prosody(df: pd.DataFrame, contours: Contours, *,
                        text: str, pause_threshold_s: float = 0.25,
                        pitch_units: str = "semitones") -> dict[str, float]:
    """Return 16 doc-level prosody columns as a flat dict.

    `text` is currently unused — reserved for a future syllable-from-orthography
    fallback when MFA-aligned word forms differ from the transcript (e.g.,
    contractions). Callers in build_doc_summary already forward the transcript
    text, so the wiring is in place.
    """
    del text  # see docstring
    # Whole-stimulus pitch/volume aggregates (apply unit conversion).
    pitch_arr = contours.pitch_hz
    if pitch_units == "semitones":
        pitch_arr = _semitones_from_hz(pitch_arr, contours.speaker_median_pitch_hz)
    p_mean, p_min, p_max, p_std = _agg(pitch_arr)
    v_mean, v_min, v_max, v_std = _agg(contours.intensity_db)

    # Mean absolute deltas from per-word columns.
    pitch_delta_mean = float(df["Pitch_Delta"].abs().mean()) if "Pitch_Delta" in df else float("nan")
    volume_delta_mean = float(df["Volume_Delta"].abs().mean()) if "Volume_Delta" in df else float("nan")

    # Pause statistics over MFA word rows.
    aligned = df.dropna(subset=["Onset", "End"]).sort_values("Onset").reset_index(drop=True)
    pauses = []
    if len(aligned) >= 2:
        gaps = aligned["Onset"].to_numpy()[1:] - aligned["End"].to_numpy()[:-1]
        pauses = gaps[gaps > pause_threshold_s].tolist()
    pause_count = len(pauses)
    pause_total = float(sum(pauses))
    pause_mean = float(np.mean(pauses)) if pauses else 0.0
    if len(aligned) >= 1:
        stim_duration = float(aligned["End"].iloc[-1] - aligned["Onset"].iloc[0])
    else:
        stim_duration = float("nan")
    pause_pct = (pause_total / stim_duration) if stim_duration and stim_duration > 0 else float("nan")

    # Articulation rate: syllables / spoken time (excludes pauses by definition).
    syllable_total = 0
    spoken_time = 0.0
    pyphen_ok = True
    for _, row in aligned.iterrows():
        if row.get("upos") == "PUNCT":
            continue
        syl = _count_syllables(str(row["token"]))
        if syl < 0:
            pyphen_ok = False
            break
        syllable_total += syl
        spoken_time += float(row["End"]) - float(row["Onset"])
    if not pyphen_ok or spoken_time <= 0:
        articulation = float("nan")
    else:
        articulation = syllable_total / spoken_time

    return {
        "doc_pitch_mean": p_mean,
        "doc_pitch_min": p_min,
        "doc_pitch_max": p_max,
        "doc_pitch_std": p_std,
        "doc_volume_mean": v_mean,
        "doc_volume_min": v_min,
        "doc_volume_max": v_max,
        "doc_volume_std": v_std,
        "doc_pitch_delta_mean": pitch_delta_mean,
        "doc_volume_delta_mean": volume_delta_mean,
        "doc_hnr": float(contours.hnr_db),
        "doc_articulation_rate": articulation,
        "doc_pause_count": pause_count,
        "doc_pause_mean_duration": pause_mean,
        "doc_pause_total_duration": pause_total,
        "doc_pause_pct_silence": pause_pct,
    }
