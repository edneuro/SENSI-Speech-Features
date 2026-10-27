"""
Word frequency extractor — Zipf scale via the `wordfreq` library.

Input:  pandas DataFrame with at least 'lemma' and 'upos' columns
        (output of text_parser.parse())
Output: same DataFrame with a new 'Zipf_Freq' column appended

Provenance (report this accurately in methods sections): values come from
wordfreq's combined English frequency list, which blends several corpora —
SUBTLEX-US/UK subtitles (a major component), Wikipedia, books, news, web,
and social-media text — NOT from SUBTLEX-US alone. See
https://github.com/rspeer/wordfreq for the exact source inventory.

The lookup is keyed on the LEMMA (spaCy), so inflectional variants share
their lexeme's frequency ("ran" carries the frequency of "run"). If your
analysis needs surface-form frequency, derive it from the `token` column
with `wordfreq.zipf_frequency` directly.

Zipf scale:
    8 = most frequent English words (the, be, …)
    5 = typical content words
    2 = rare / low-frequency words

Punctuation tokens (upos == 'PUNCT') receive NaN, and so do
out-of-vocabulary lemmas: wordfreq returns 0.0 as an OOV *sentinel*, not a
real frequency, and propagating it would silently bias document-level
means. NaN matches the pipeline-wide "no value, never imputed" convention.
"""

import numpy as np
import pandas as pd
from wordfreq import zipf_frequency


def add_word_freq(token_df: pd.DataFrame) -> pd.DataFrame:
    """Append a Zipf_Freq column to a token-level DataFrame.

    Args:
        token_df: DataFrame produced by text_parser.parse().
                  Must contain 'lemma' and 'upos' columns.

    Returns:
        A copy of token_df with an added 'Zipf_Freq' column (float).
        Punctuation rows and out-of-vocabulary lemmas have NaN.
    """
    df = token_df.copy()
    df['Zipf_Freq'] = df.apply(_freq_for_row, axis=1)
    return df


def _freq_for_row(row) -> float:
    if row['upos'] == 'PUNCT':
        return np.nan
    zipf = zipf_frequency(row['lemma'], 'en')
    # 0.0 is wordfreq's out-of-vocabulary sentinel, not a Zipf value.
    return zipf if zipf > 0.0 else np.nan
