"""Perceptual center (P-center) for syllables.

Formula: P_Center = nucleus_onset - k * onset_consonant_duration
where nucleus = first vowel phoneme (ARPABET label ending in 0/1/2)
and onset_consonant_duration = sum of durations of preceding consonants.

Reference: Marcus (1981), Scott (1993).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_FK = ['doc_id', 'sentence_id', 'token_in_sentence_id', 'syllable_id']


def _pcenter_raw(group: pd.DataFrame, k: float) -> float:
    """P_Center for one syllable's phoneme group (rows sorted by phoneme_id).
    Returns NaN when no vowel nucleus is present; caller applies fallback.
    """
    is_vowel = group['Label'].str[-1:].str.isdigit().tolist()
    try:
        first_vowel = is_vowel.index(True)
    except ValueError:
        return float('nan')
    nucleus_onset = float(group['Onset'].iat[first_vowel])
    onset_dur = float(group['Duration'].iloc[:first_vowel].sum())
    return round(nucleus_onset - k * onset_dur, 4)


def add_pcenter(
    syllables_df: pd.DataFrame,
    phonemes_df: pd.DataFrame,
    *,
    k: float = 0.3,
) -> pd.DataFrame:
    """Append P_Center column to syllables_df.

    P_Center = nucleus_onset - k * onset_consonant_duration.
    Falls back to syllable Onset when no vowel nucleus exists.
    NaN when MFA alignment is missing (syllable Onset is NaN).
    """
    df = syllables_df.copy()

    if len(df) == 0 or phonemes_df is None or len(phonemes_df) == 0:
        df['P_Center'] = pd.Series([float('nan')] * len(df), dtype=float)
        return df

    phon_sorted = phonemes_df.sort_values(_FK + ['phoneme_id'])
    pcenter_map: dict[tuple, float] = {}
    for key, group in phon_sorted.groupby(_FK, sort=False):
        pcenter_map[key] = _pcenter_raw(group, k)

    values: list[float] = []
    for row in df.itertuples(index=False):
        onset = getattr(row, 'Onset', None)
        if onset is None or pd.isna(onset):
            values.append(float('nan'))
            continue
        key = (row.doc_id, row.sentence_id, row.token_in_sentence_id, row.syllable_id)
        raw = pcenter_map.get(key, float('nan'))
        if np.isnan(raw):
            values.append(round(float(onset), 4))
        else:
            values.append(raw)

    df['P_Center'] = values
    return df
