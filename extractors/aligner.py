"""
Forced alignment extractor — Montreal Forced Aligner (MFA).

Input:  audio file path (.wav), transcript text (str)
Output: pandas DataFrame, one row per word (punctuation excluded)

Columns produced:
    Transcript — the word form sent to MFA (numbers spelled out)
    Onset      — word start time in audio (seconds)
    End        — word end time in audio (seconds)

MFA prerequisite (one-time setup):
    conda install -c conda-forge montreal-forced-aligner
    mfa model download acoustic english_us_arpa
    mfa model download dictionary english_us_arpa

MFA runs as a subprocess. Audio and transcript are written to a temporary
directory, MFA is invoked, and the output TextGrid is parsed. The temp
directory is cleaned up automatically.

Note on accuracy:
    MFA uses Kaldi-based acoustic models and produces phoneme-level
    precision, making it suitable for EEG word-onset analysis.
    The 'Onset' and 'End' values are the true acoustic boundaries,
    not heuristic estimates.
"""

import os
import shutil
import subprocess
import tempfile

import pandas as pd

from utils.transcript_prep import prepare_for_aligner
from utils.textgrid_parser import parse_words, parse_phones

MFA_ACOUSTIC_MODEL = 'english_us_arpa'
MFA_DICTIONARY = 'english_us_arpa'


def align_with_phones(
    audio_path: str, transcript: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run MFA forced alignment and return both word and phone tiers.

    Args:
        audio_path:  Path to a mono or stereo .wav file.
        transcript:  Raw transcript text (digits OK; spelling-out happens here).

    Returns:
        Tuple of (words_df, phones_df).
        words_df: columns Transcript, Onset, End — one row per word.
        phones_df: columns phone, onset, offset — one row per phoneme.
        Both exclude silence intervals.

    Raises:
        RuntimeError: if MFA is not installed or alignment fails.
    """
    prepared = prepare_for_aligner(transcript)

    with tempfile.TemporaryDirectory() as tmpdir:
        corpus_dir = os.path.join(tmpdir, 'corpus')
        output_dir = os.path.join(tmpdir, 'output')
        os.makedirs(corpus_dir)
        os.makedirs(output_dir)

        stem = os.path.splitext(os.path.basename(audio_path))[0]

        # MFA expects audio + transcript with matching filenames
        shutil.copy(audio_path, os.path.join(corpus_dir, f'{stem}.wav'))
        lab_path = os.path.join(corpus_dir, f'{stem}.lab')
        with open(lab_path, 'w', encoding='utf-8') as f:
            f.write(prepared)

        _run_mfa(corpus_dir, output_dir)

        tg_path = os.path.join(output_dir, f'{stem}.TextGrid')
        if not os.path.exists(tg_path):
            raise RuntimeError(
                f"MFA did not produce a TextGrid for {stem}. "
                "Check that the audio and transcript match, and that MFA models are downloaded."
            )

        word_df = parse_words(tg_path)
        phone_df = parse_phones(tg_path)

    # Rename columns to match pipeline convention
    word_df = word_df.rename(columns={
        'word': 'Transcript',
        'onset': 'Onset',
        'offset': 'End',
    })
    return word_df, phone_df


def align(audio_path: str, transcript: str) -> pd.DataFrame:
    """Run MFA forced alignment on one audio+transcript pair.

    Returns a DataFrame with columns Transcript / Onset / End, one row per
    word; punctuation and silence excluded; words MFA couldn't align are
    omitted (the merge step in main.py flags them NaN downstream).

    Thin wrapper over `align_with_phones` that discards the phone tier.
    Kept for callers that don't need phoneme alignment.
    """
    word_df, _ = align_with_phones(audio_path, transcript)
    return word_df


def _run_mfa(corpus_dir: str, output_dir: str) -> None:
    """Invoke MFA as a subprocess. Raises RuntimeError on failure."""
    cmd = [
        'mfa', 'align',
        corpus_dir,
        MFA_DICTIONARY,
        MFA_ACOUSTIC_MODEL,
        output_dir,
        '--clean',
        '--quiet',
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"MFA alignment failed.\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        )
