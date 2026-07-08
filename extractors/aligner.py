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

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

from utils.transcript_prep import prepare_for_aligner
from utils.textgrid_parser import parse_words, parse_phones

MFA_ACOUSTIC_MODEL = 'english_us_arpa'
MFA_DICTIONARY = 'english_us_arpa'

ALIGNMENT_SUBDIR = 'Montreal Forced Alignment'
SIDECAR_VERSION = 1


def alignment_dir(output_folder) -> Path:
    """Return <output_folder>/Montreal Forced Alignment (not created here)."""
    return Path(output_folder) / ALIGNMENT_SUBDIR


def _stem(audio_path: str) -> str:
    """Stimulus id used for on-disk artifact names: the audio basename sans ext.

    Both the batch pre-pass and align_with_phones derive the stem this way, so
    the TextGrid a pass writes is the one a later pass looks up.
    """
    return os.path.splitext(os.path.basename(audio_path))[0]


def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode('utf-8')).hexdigest()


def _sha256_file(path: str, _chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(_chunk), b''):
            h.update(block)
    return h.hexdigest()


def _freshness_dict(audio_path: str, transcript_text: str) -> dict:
    """The sidecar contents identifying the inputs a TextGrid was aligned from."""
    return {
        'transcript_sha256': _sha256_text(prepare_for_aligner(transcript_text)),
        'audio_sha256': _sha256_file(audio_path),
        'audio_size': os.path.getsize(audio_path),
        'mfa_acoustic': MFA_ACOUSTIC_MODEL,
        'mfa_dictionary': MFA_DICTIONARY,
        'version': SIDECAR_VERSION,
    }


def _write_sidecar(persist_dir, stem: str, audio_path: str,
                   transcript_text: str) -> None:
    meta = _freshness_dict(audio_path, transcript_text)
    (Path(persist_dir) / f'{stem}.json').write_text(
        json.dumps(meta, indent=2, sort_keys=True), encoding='utf-8')


def _is_fresh(persist_dir, stem: str, audio_path: str,
              transcript_text: str) -> bool:
    """True iff a reusable, current TextGrid exists for this stimulus.

    Requires the TextGrid and sidecar to exist and every key in the sidecar to
    match the current inputs. audio_size is a cheap pre-filter: a size mismatch
    short-circuits to stale without hashing the audio.
    """
    persist_dir = Path(persist_dir)
    tg = persist_dir / f'{stem}.TextGrid'
    side = persist_dir / f'{stem}.json'
    if not tg.exists() or not side.exists():
        return False
    try:
        meta = json.loads(side.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return False
    if meta.get('version') != SIDECAR_VERSION:
        return False
    if meta.get('mfa_acoustic') != MFA_ACOUSTIC_MODEL:
        return False
    if meta.get('mfa_dictionary') != MFA_DICTIONARY:
        return False
    if meta.get('transcript_sha256') != _sha256_text(
            prepare_for_aligner(transcript_text)):
        return False
    try:
        if meta.get('audio_size') != os.path.getsize(audio_path):
            return False
    except OSError:
        return False
    return meta.get('audio_sha256') == _sha256_file(audio_path)


def align_with_phones(
    audio_path: str, transcript: str, cache: dict | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run MFA forced alignment and return both word and phone tiers.

    Args:
        audio_path:  Path to a mono or stereo .wav file.
        transcript:  Raw transcript text (digits OK; spelling-out happens here).
        cache:       Optional dict for memoizing alignment across calls. MFA
                     output depends only on (audio, transcript), so when the
                     same document is processed by more than one pass (e.g. the
                     word batch and the doc batch in a single GUI run), passing
                     a shared dict makes MFA run at most once per document.
                     When None (CLI single runs, tests), no caching is done.

    Returns:
        Tuple of (words_df, phones_df).
        words_df: columns Transcript, Onset, End — one row per word.
        phones_df: columns phone, onset, offset — one row per phoneme.
        Both exclude silence intervals.

        When served via `cache`, both DataFrames are returned as fresh copies so
        a caller mutating them can't corrupt the cached entry for later passes.

    Raises:
        RuntimeError: if MFA is not installed or alignment fails.
    """
    if cache is None:
        return _align_uncached(audio_path, transcript)
    key = (os.path.abspath(audio_path), transcript)
    if key not in cache:
        cache[key] = _align_uncached(audio_path, transcript)
    word_df, phone_df = cache[key]
    return word_df.copy(), phone_df.copy()


def _align_uncached(
    audio_path: str, transcript: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run MFA once for (audio_path, transcript). See align_with_phones."""
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
