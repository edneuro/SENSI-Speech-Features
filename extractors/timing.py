"""Derived timing columns: Word_Duration and Word_Gap.

Both are computed from the existing MFA Onset / End columns. No new
dependencies. If alignment hasn't been run, both columns are emitted as
all-NaN to keep the output schema stable.

Word_Duration[i] = End[i] - Onset[i]
Word_Gap[i]      = Onset[i] - End[i-1]   (NaN for i=0 or if either neighbor is NaN)
"""

import numpy as np
import pandas as pd


def add_timing_columns(token_df: pd.DataFrame) -> pd.DataFrame:
    df = token_df.copy()

    if 'Onset' not in df.columns or 'End' not in df.columns:
        df['Word_Duration'] = np.nan
        df['Word_Gap'] = np.nan
        return df

    df['Word_Duration'] = df['End'] - df['Onset']

    prev_end = df['End'].shift(1)
    df['Word_Gap'] = df['Onset'] - prev_end
    return df
