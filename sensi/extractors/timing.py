"""Derived timing columns: Word_Duration, Word_Gap and Word_Gap_After.

All are computed from the existing MFA Onset / End columns. No new
dependencies. If alignment hasn't been run, the columns are emitted as
all-NaN to keep the output schema stable.

Word_Duration[i]  = End[i] - Onset[i]
Word_Gap[i]       = Onset[i] - End[j], where j is the most recent PRIOR row
                    with a non-NaN End: the silence before the word.
Word_Gap_After[i] = Onset[k] - End[i], where k is the next LATER row with a
                    non-NaN Onset: the silence after the word, so it equals
                    the next word's Word_Gap.

Punctuation rows (no acoustic interval) are skipped over rather than
breaking the chain, so the silence across a comma is still measured. If an
unaligned word sits between two aligned ones, the gap spans it. NaN for
rows whose own Onset/End is NaN, before the first and after the last
aligned word.

A contraction that spaCy splits ("He'd" -> He + 'd) is one MFA word, so its
parts carry the same Onset/End. The first part holds the word's gaps; the
later parts get NaN (there is no silence between them).
"""

import numpy as np
import pandas as pd


def add_timing_columns(token_df: pd.DataFrame) -> pd.DataFrame:
    df = token_df.copy()

    if 'Onset' not in df.columns or 'End' not in df.columns:
        df['Word_Duration'] = np.nan
        df['Word_Gap'] = np.nan
        df['Word_Gap_After'] = np.nan
        return df

    onset = pd.to_numeric(df['Onset'], errors='coerce')
    end = pd.to_numeric(df['End'], errors='coerce')

    df['Word_Duration'] = end - onset

    # One entry per spoken word: aligned rows, minus continuation parts that
    # repeat the previous aligned row's Onset/End window.
    aligned = onset.notna() & end.notna()
    prev_on = onset.where(aligned).ffill().shift(1)
    prev_end = end.where(aligned).ffill().shift(1)
    continuation = aligned & (onset == prev_on) & (end == prev_end)
    word = aligned & ~continuation

    w_on, w_end = onset.where(word), end.where(word)
    # Most recent / next known word boundary (ffill / bfill skip the NaN rows
    # — punctuation, unaligned words, continuation parts).
    df['Word_Gap'] = (onset - w_end.ffill().shift(1)).where(~continuation)
    df['Word_Gap_After'] = (w_on.bfill().shift(-1) - end).where(~continuation)
    return df
