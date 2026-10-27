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

    # pd.to_numeric + float cast guards against pandas.NA (object dtype) in
    # the Onset/End columns when MFA alignment produced no matches — pandas.NA
    # is not numeric and np.isfinite(pandas.NA) raises TypeError in NumPy 2.0.
    onsets = (pd.to_numeric(df["Onset"], errors='coerce').to_numpy(dtype=float, na_value=np.nan)
              if "Onset" in df.columns else np.full(len(df), np.nan))
    ends = (pd.to_numeric(df["End"], errors='coerce').to_numpy(dtype=float, na_value=np.nan)
            if "End" in df.columns else np.full(len(df), np.nan))

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

    # Word-to-word deltas, measured back to the most recent ALIGNED word
    # (ffill skips NaN-Mean rows — PUNCT and unaligned words — instead of
    # poisoning the next aligned word's delta with NaN; mirrors the
    # Word_Gap fix in extractors/timing.py). NaN for the first aligned word.
    df["Pitch_Delta"] = df["Pitch_Mean"] - df["Pitch_Mean"].ffill().shift(1)
    df["Volume_Delta"] = df["Volume_Mean"] - df["Volume_Mean"].ffill().shift(1)
    return df


_syllable_dic = None  # lazily-built pyphen dictionary (module-level cache)


def _count_syllables(token: str) -> int:
    """Surface-form syllable count via pyphen. Returns 1 on lookup failure
    or 0 for empty tokens. Returns -1 sentinel if pyphen is unavailable."""
    global _syllable_dic
    try:
        import pyphen  # type: ignore
    except ImportError:
        return -1
    if _syllable_dic is None:
        _syllable_dic = pyphen.Pyphen(lang="en_US")
    if not token:
        return 0
    return _syllable_dic.inserted(token).count("-") + 1


def _count_spoken_syllables(aligned: pd.DataFrame,
                            phones_df: pd.DataFrame | None) -> int:
    """Syllables actually spoken in the aligned words.

    With an MFA phone tier: one syllable per vowel phone (ARPABET vowels carry
    a stress digit), so the count follows the pronunciation MFA aligned, not
    the spelling, and a contraction sharing one MFA interval is counted once.
    An OOV word aligned to a single `spn` phone has no vowels, so it falls back
    to the pyphen count of the word whose interval holds it. Without a phone
    tier, pyphen over every aligned non-PUNCT token. Returns -1 if pyphen is
    needed but unavailable.
    """
    words = aligned[aligned["upos"] != "PUNCT"] if "upos" in aligned else aligned
    if phones_df is None or len(phones_df) == 0:
        total = 0
        for token in words["token"].astype(str):
            syl = _count_syllables(token)
            if syl < 0:
                return -1
            total += syl
        return total
    labels = phones_df["phone"].astype(str)
    total = int(labels.str[-1].str.isdigit().sum())
    for onset in phones_df.loc[labels == "spn", "onset"]:
        holder = words[(words["Onset"] <= onset) & (words["End"] > onset)]
        if len(holder):
            syl = _count_syllables(str(holder["token"].iloc[0]))
            if syl < 0:
                return -1
            total += syl
    return total


def compute_doc_prosody(df: pd.DataFrame, contours: Contours, *,
                        text: str, pause_threshold_s: float = 0.25,
                        pitch_units: str = "semitones",
                        phones_df: pd.DataFrame | None = None) -> dict[str, float]:
    """Return 16 doc-level prosody columns as a flat dict.

    `phones_df` is the MFA phone tier (phone / onset / offset); when given,
    articulation rate counts syllables from it (see _count_spoken_syllables).
    `text` is unused (kept for call-site compatibility).
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
    # No pauses -> the mean pause length is undefined, not 0 s.
    pause_mean = float(np.mean(pauses)) if pauses else float("nan")
    if len(aligned) >= 1:
        stim_duration = float(aligned["End"].iloc[-1] - aligned["Onset"].iloc[0])
    else:
        stim_duration = float("nan")
    # Percent (0-100) of the first-word-to-last-word span spent in pauses;
    # leading/trailing file padding is deliberately not counted.
    pause_pct = (100.0 * pause_total / stim_duration
                 if stim_duration and stim_duration > 0 else float("nan"))

    # Articulation rate: syllables per second of speaking time, where speaking
    # time is the speech span minus pauses longer than the threshold. Short
    # gaps between words count as speaking (the standard definition), so the
    # pause threshold sets what is excluded.
    speaking_time = (stim_duration - pause_total
                     if np.isfinite(stim_duration) else float("nan"))
    syllable_total = _count_spoken_syllables(aligned, phones_df)
    if syllable_total < 0 or not (speaking_time > 0):
        articulation = float("nan")
    else:
        articulation = syllable_total / speaking_time

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
