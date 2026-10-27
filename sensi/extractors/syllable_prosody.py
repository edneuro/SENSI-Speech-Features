"""Syllable-level prosody -- mean pitch/volume + delta within each syllable
interval. Reuses the Contours object built once per stimulus by
sensi.extractors.audio_features.extract(). No second audio decode.

Mirrors sensi/extractors/prosody.py:add_word_prosody for the per-row math;
the only difference is the row granularity (syllables vs words).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

PROSODY_COLUMNS = ('Pitch_Mean', 'Volume_Mean', 'Pitch_Delta', 'Volume_Delta')


def _slice_mean(values: np.ndarray, times: np.ndarray,
                onset: float, end: float) -> float:
    if values is None or times is None or len(values) == 0:
        return float('nan')
    mask = (times >= onset) & (times < end)
    if not mask.any():
        return float('nan')
    sub = values[mask]
    finite = sub[np.isfinite(sub)]
    if finite.size == 0:
        return float('nan')
    return float(finite.mean())


def _hz_to_semitones(hz: float, reference_hz: float) -> float:
    if (not np.isfinite(hz) or hz <= 0
            or not np.isfinite(reference_hz) or reference_hz <= 0):
        return float('nan')
    return 12.0 * float(np.log2(hz / reference_hz))


def add_syllable_prosody(syll_df: pd.DataFrame, contours: Any,
                         *, pitch_units: str = 'semitones') -> pd.DataFrame:
    """Append Pitch_Mean / Volume_Mean / Pitch_Delta / Volume_Delta columns
    to syll_df. NaN-fills when contours is None (alignment-only run).

    `syll_df` must contain syllables from a single stimulus — Pitch_Delta
    / Volume_Delta are computed via row-positional `.diff()` and would
    produce a meaningless cross-stimulus delta on the first row of any
    appended stimulus otherwise.
    """
    if pitch_units not in ('hz', 'semitones'):
        raise ValueError(
            f"pitch_units must be 'hz' or 'semitones', got {pitch_units!r}"
        )
    df = syll_df.copy()

    if contours is None or len(df) == 0:
        for col in PROSODY_COLUMNS:
            df[col] = pd.Series([float('nan')] * len(df), dtype=float)
        return df

    pitch_hz = contours.pitch_hz
    intensity_db = contours.intensity_db
    times = contours.times
    median_pitch = float(getattr(contours, 'speaker_median_pitch_hz', 0.0))

    pitch_means: list[float] = []
    volume_means: list[float] = []
    for row in df.itertuples(index=False):
        p_hz = _slice_mean(pitch_hz, times, row.Onset, row.End)
        if pitch_units == 'semitones':
            pitch_means.append(_hz_to_semitones(p_hz, median_pitch))
        else:
            pitch_means.append(p_hz)
        volume_means.append(_slice_mean(intensity_db, times, row.Onset, row.End))

    df['Pitch_Mean'] = pitch_means
    df['Volume_Mean'] = volume_means
    # Delta: difference vs the previous syllable in the stimulus (across word
    # boundaries), measured back to the most recent syllable WITH a value --
    # ffill skips unvoiced/empty syllables instead of blanking the next delta,
    # as the word-level deltas do. NaN for the first syllable with a value.
    df['Pitch_Delta'] = df['Pitch_Mean'] - df['Pitch_Mean'].ffill().shift(1)
    df['Volume_Delta'] = df['Volume_Mean'] - df['Volume_Mean'].ffill().shift(1)
    return df
