"""
Word frequency extractor — wordfreq (SUBTLEX-US, Zipf scale).

Input:  pandas DataFrame with at least 'lemma' and 'upos' columns
        (output of text_parser.parse())
Output: same DataFrame with a new 'Zipf_Freq' column appended

Zipf scale:
    8 = most frequent English words (the, be, …)
    5 = typical content words
    2 = rare / low-frequency words
    0 = out-of-vocabulary (wordfreq returns 0.0 for unknown lemmas)

Punctuation tokens (upos == 'PUNCT') receive NaN — they have no
meaningful frequency value in SUBTLEX.
"""

import pandas as pd
from wordfreq import zipf_frequency


def add_word_freq(token_df: pd.DataFrame) -> pd.DataFrame:
    """Append a Zipf_Freq column to a token-level DataFrame.

    Args:
        token_df: DataFrame produced by text_parser.parse().
                  Must contain 'lemma' and 'upos' columns.

    Returns:
        A copy of token_df with an added 'Zipf_Freq' column (float).
        Punctuation rows have NaN. OOV lemmas have 0.0.
    """
    df = token_df.copy()
    df['Zipf_Freq'] = df.apply(_freq_for_row, axis=1)
    return df


def _freq_for_row(row) -> float | None:
    if row['upos'] == 'PUNCT':
        return None
    return zipf_frequency(row['lemma'], 'en')
