"""
Parses MFA output TextGrid files into a simple list of word intervals.

MFA produces a Praat TextGrid with two tiers:
    'words'  — one interval per word, with start/end times
    'phones' — one interval per phoneme (finer resolution)

This module reads the 'words' tier by default.
Phoneme-level data can also be extracted if needed in the future.
"""

import tgt
import pandas as pd


def parse_words(textgrid_path: str) -> pd.DataFrame:
    """Extract word-level alignment from an MFA TextGrid file.

    Returns a DataFrame with columns:
        word   — the word as MFA aligned it (lowercase, numbers spelled out)
        onset  — word start time in seconds
        offset — word end time in seconds

    Silence intervals (empty text) are excluded.
    """
    tg = tgt.read_textgrid(textgrid_path)
    word_tier = tg.get_tier_by_name('words')

    rows = []
    for interval in word_tier:
        word = interval.text.strip()
        if not word:
            continue
        rows.append({
            'word': word,
            'onset': round(interval.start_time, 4),
            'offset': round(interval.end_time, 4),
        })

    return pd.DataFrame(rows)


def parse_phones(textgrid_path: str) -> pd.DataFrame:
    """Extract phoneme-level alignment from an MFA TextGrid file.

    Returns a DataFrame with columns:
        phone  — ARPABET phoneme label (e.g. 'B', 'EH1', 'L', 'AH0')
        onset  — phoneme start time in seconds
        offset — phoneme end time in seconds

    Silence intervals (SIL, sp, '') are excluded.
    """
    tg = tgt.read_textgrid(textgrid_path)
    phone_tier = tg.get_tier_by_name('phones')

    silence_labels = {'', 'SIL', 'sp', 'spn'}
    rows = []
    for interval in phone_tier:
        phone = interval.text.strip()
        if phone in silence_labels:
            continue
        rows.append({
            'phone': phone,
            'onset': round(interval.start_time, 4),
            'offset': round(interval.end_time, 4),
        })

    return pd.DataFrame(rows)
