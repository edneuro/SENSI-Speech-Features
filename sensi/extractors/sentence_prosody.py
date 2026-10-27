"""Sentence-level timing and prosody aggregator.

Given an MFA-aligned word DataFrame and Praat Contours, produces a
sentence-level DataFrame with onset/end timing (min/max over words)
and 11 prosody columns: Pitch_{Mean,Min,Max,Std,Delta},
Volume_{Mean,Min,Max,Std,Delta}, HNR.

The 11 columns mirror doc-level prosody but without pause statistics.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from sensi.extractors.audio_features import Contours
from sensi.extractors.prosody import _agg, _slice, _semitones_from_hz

_COLUMN_ORDER = [
    "doc_id", "sentence_id", "sentence", "Onset", "End",
    "Pitch_Mean", "Pitch_Min", "Pitch_Max", "Pitch_Std", "Pitch_Delta",
    "Volume_Mean", "Volume_Min", "Volume_Max", "Volume_Std", "Volume_Delta",
    "HNR",
]


def add_sentence_prosody(
    word_df: pd.DataFrame,
    contours: Contours | None = None,
    *,
    pitch_units: str = "semitones",
) -> pd.DataFrame:
    """Return a sentence-level DataFrame with timing + 11 prosody columns.

    Requires word_df to have: doc_id, sentence_id, sentence, Onset, End.
    If contours is None or onset/end are NaN, prosody columns are NaN.
    """
    required = {"doc_id", "sentence_id", "sentence", "Onset", "End"}
    missing = required - set(word_df.columns)
    if missing:
        raise ValueError(f"word_df missing columns: {sorted(missing)}")

    pitch_arr: np.ndarray | None = None
    if contours is not None:
        pitch_arr = contours.pitch_hz.copy()
        if pitch_units == "semitones":
            pitch_arr = _semitones_from_hz(pitch_arr, contours.speaker_median_pitch_hz)

    onsets = pd.to_numeric(word_df["Onset"], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    ends   = pd.to_numeric(word_df["End"],   errors="coerce").to_numpy(dtype=float, na_value=np.nan)

    tmp = word_df.copy()
    tmp["_onset_f"] = onsets
    tmp["_end_f"]   = ends

    rows: list[dict] = []
    for (doc_id, sent_id), grp in tmp.groupby(["doc_id", "sentence_id"], sort=False):
        sent_text = grp["sentence"].iloc[0]
        onset = float(grp["_onset_f"].min())
        end   = float(grp["_end_f"].max())

        has_timing = np.isfinite(onset) and np.isfinite(end)
        row: dict = {
            "doc_id":      doc_id,
            "sentence_id": sent_id,
            "sentence":    sent_text,
            "Onset": onset if has_timing else np.nan,
            "End":   end   if has_timing else np.nan,
        }

        if contours is not None and pitch_arr is not None and has_timing:
            p_mean, p_min, p_max, p_std = _agg(
                _slice(contours.times, pitch_arr, onset, end))
            v_mean, v_min, v_max, v_std = _agg(
                _slice(contours.times, contours.intensity_db, onset, end))

            if (contours.hnr_frame_db is not None
                    and contours.hnr_frame_times is not None):
                hnr_frames = _slice(
                    contours.hnr_frame_times, contours.hnr_frame_db, onset, end)
                finite = hnr_frames[np.isfinite(hnr_frames)]
                hnr = float(np.mean(finite)) if finite.size else np.nan
            else:
                hnr = np.nan
        else:
            p_mean = p_min = p_max = p_std = np.nan
            v_mean = v_min = v_max = v_std = np.nan
            hnr = np.nan

        row.update({
            "Pitch_Mean": p_mean, "Pitch_Min": p_min,
            "Pitch_Max":  p_max,  "Pitch_Std": p_std,
            "Volume_Mean": v_mean, "Volume_Min": v_min,
            "Volume_Max":  v_max,  "Volume_Std": v_std,
            "HNR": hnr,
        })
        rows.append(row)

    df = pd.DataFrame(rows)

    if len(df) > 1:
        # Measured back to the most recent ALIGNED sentence (ffill skips
        # NaN-Mean rows -- sentences where no word aligned -- instead of
        # poisoning the next aligned sentence's delta with NaN; mirrors the
        # Word_Gap fix in extractors/timing.py and the word-prosody fix in
        # extractors/prosody.py).
        df["Pitch_Delta"]  = df["Pitch_Mean"] - df["Pitch_Mean"].ffill().shift(1)
        df["Volume_Delta"] = df["Volume_Mean"] - df["Volume_Mean"].ffill().shift(1)
    elif len(df) == 1:
        df["Pitch_Delta"]  = np.nan
        df["Volume_Delta"] = np.nan
    else:
        df["Pitch_Delta"]  = pd.Series(dtype=float)
        df["Volume_Delta"] = pd.Series(dtype=float)

    present = [c for c in _COLUMN_ORDER if c in df.columns]
    return df[present]
