"""Derived timing columns: Word_Duration and Word_Gap.

Both are computed from the existing MFA Onset / End columns. No new
dependencies. If alignment hasn't been run, both columns are emitted as
all-NaN to keep the output schema stable.

Word_Duration[i] = End[i] - Onset[i]
Word_Gap[i]      = Onset[i] - End[j], where j is the most recent PRIOR row
                   with a non-NaN End. Punctuation rows (no acoustic
                   interval) are skipped over rather than breaking the
                   chain, so the silence across a comma is still measured.
                   If an unaligned word sits between two aligned ones, the
                   gap spans it. NaN for the first aligned word and for
                   rows whose own Onset is NaN.
"""

import numpy as np
import pandas as pd


def add_timing_columns(token_df: pd.DataFrame) -> pd.DataFrame:
    df = token_df.copy()

    if 'Onset' not in df.columns or 'End' not in df.columns:
        df['Word_Duration'] = np.nan
        df['Word_Gap'] = np.nan
        return df

    onset = pd.to_numeric(df['Onset'], errors='coerce')
    end = pd.to_numeric(df['End'], errors='coerce')

    df['Word_Duration'] = end - onset

    # Most recent known End strictly before each row (ffill skips the
    # NaN-End rows — punctuation and unaligned words — instead of
    # poisoning the following word's gap).
    prev_end = end.ffill().shift(1)
    df['Word_Gap'] = onset - prev_end
    return df
