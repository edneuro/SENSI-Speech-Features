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
import sys
import tempfile
from pathlib import Path

import pandas as pd

from sensi.paths import mfa_pretrained
from sensi.utils.transcript_prep import prepare_for_aligner
from sensi.utils.textgrid_parser import parse_words, parse_phones

MFA_ACOUSTIC_MODEL = 'english_us_arpa'
MFA_DICTIONARY = 'english_us_arpa'

ALIGNMENT_SUBDIR = 'Montreal Forced Alignment'
# 3: one speaker per stimulus (see _stage) — TextGrids from the pooled-speaker
# layout (v2) are re-aligned once.
SIDECAR_VERSION = 3

G2P_MODEL_PATH = mfa_pretrained('g2p', 'english_us_arpa.zip')


def g2p_model_available() -> bool:
    """True when the vendored MFA G2P model is present."""
    return G2P_MODEL_PATH.exists()


def cache_key(audio_path: str, transcript: str, g2p: bool = False) -> tuple:
    """The align_with_phones memo key.

    One definition, shared by align_with_phones (which stores/reads the
    cache) and any caller that needs to check for a cache hit without
    running alignment (e.g. main.py's cache-hit log message) — so a new
    parameter here (like `g2p`) cannot silently desync a hand-rebuilt tuple
    at the call site.
    """
    return (os.path.abspath(audio_path), transcript, bool(g2p))


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


def _freshness_dict(audio_path: str, transcript_text: str, *,
                     g2p: bool = False) -> dict:
    """The sidecar contents identifying the inputs a TextGrid was aligned from."""
    return {
        'transcript_sha256': _sha256_text(prepare_for_aligner(transcript_text)),
        'audio_sha256': _sha256_file(audio_path),
        'audio_size': os.path.getsize(audio_path),
        'mfa_acoustic': MFA_ACOUSTIC_MODEL,
        'mfa_dictionary': MFA_DICTIONARY,
        'mfa_g2p': bool(g2p),
        'version': SIDECAR_VERSION,
    }


def _write_sidecar(persist_dir, stem: str, audio_path: str,
                   transcript_text: str, *, g2p: bool = False) -> None:
    meta = _freshness_dict(audio_path, transcript_text, g2p=g2p)
    (Path(persist_dir) / f'{stem}.json').write_text(
        json.dumps(meta, indent=2, sort_keys=True), encoding='utf-8')


def _is_fresh(persist_dir, stem: str, audio_path: str,
              transcript_text: str, *, g2p: bool = False) -> bool:
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
    if bool(meta.get('mfa_g2p', False)) != bool(g2p):
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
    audio_path: str,
    transcript: str,
    cache: dict | None = None,
    persist_dir=None,
    run_if_missing: bool = True,
    *,
    g2p: bool = False,
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
        persist_dir: Optional directory that may hold a reusable
                     `<stem>.TextGrid` from a prior run (see `alignment_dir` /
                     the batch pre-alignment pass). When set and a FRESH
                     TextGrid for this stimulus exists there (sidecar matches
                     the current inputs, see `_is_fresh`), it is parsed directly
                     and MFA is not invoked. When MFA does run here (the
                     per-document fallback after a failed batch pre-pass), its
                     TextGrid + sidecar are persisted to `persist_dir`, so a
                     later pass in the same dispatch finds it on disk instead
                     of re-aligning, and the folder always matches the CSVs
                     produced from it.
        run_if_missing: When True (default), a disk miss under `persist_dir`
                     falls back to running MFA via `_align_uncached`. When
                     False, a disk miss raises RuntimeError instead — used by
                     batch passes that run after pre-alignment, where a miss
                     means the stimulus already failed to align and MFA should
                     not be retried silently.
        g2p:         Keyword-only. When True and the vendored MFA G2P model is
                     present, MFA generates a pronunciation for out-of-dictionary
                     words instead of aligning them to a single `spn` phone (see
                     `_run_mfa`). Only affects the `_align_uncached` fallback
                     path — a disk hit under `persist_dir` is parsed as-is.
                     Included in the memo `cache` key so a process can't serve
                     an alignment cached under the opposite setting.

    Returns:
        Tuple of (words_df, phones_df).
        words_df: columns Transcript, Onset, End — one row per word.
        phones_df: columns phone, onset, offset — one row per phoneme.
        Both exclude silence intervals.

        When served via `cache`, both DataFrames are returned as fresh copies so
        a caller mutating them can't corrupt the cached entry for later passes.

    Raises:
        RuntimeError: if MFA is not installed or alignment fails, or if
                     `run_if_missing` is False and no disk alignment exists.
    """
    key = cache_key(audio_path, transcript, g2p)
    if cache is not None and key in cache:
        word_df, phone_df = cache[key]
        return word_df.copy(), phone_df.copy()

    result = None
    stem = _stem(audio_path)
    if persist_dir is not None and _is_fresh(persist_dir, stem, audio_path,
                                             transcript, g2p=g2p):
        result = _parse_textgrid(str(Path(persist_dir) / f'{stem}.TextGrid'))
    if result is None:
        if not run_if_missing:
            raise RuntimeError(
                f"No alignment on disk for {_stem(audio_path)!r} under "
                f"{persist_dir} and MFA is disabled for this pass. The batch "
                "pre-alignment step should have produced it; this stimulus "
                "likely failed to align."
            )
        result = _align_uncached(audio_path, transcript, g2p=g2p,
                                 persist_dir=persist_dir)

    if cache is not None:
        cache[key] = result
        word_df, phone_df = result
        return word_df.copy(), phone_df.copy()
    return result


def _align_uncached(
    audio_path: str, transcript: str, *, g2p: bool = False, persist_dir=None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run MFA once for (audio_path, transcript). See align_with_phones.

    When `persist_dir` is set, the TextGrid is moved there with a freshness
    sidecar (stale artifacts for this stem are removed first) — the same
    on-disk contract `ensure_alignments` writes.
    """
    prepared = prepare_for_aligner(transcript)

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        corpus_dir = os.path.join(tmpdir, 'corpus')
        output_dir = os.path.join(tmpdir, 'output')
        os.makedirs(corpus_dir)
        os.makedirs(output_dir)

        stem = _stem(audio_path)
        _stage(corpus_dir, stem, audio_path, prepared)

        _run_mfa(corpus_dir, output_dir, g2p=g2p,
                 temp_dir=os.path.join(tmpdir, 'mfa_tmp'))

        tg_path = _produced_textgrid(output_dir, stem)
        if tg_path is None:
            raise RuntimeError(
                f"MFA did not produce a TextGrid for {stem}. "
                "Check that the audio and transcript match, and that MFA models are downloaded."
            )
        if persist_dir is None:
            return _parse_textgrid(tg_path)
        persist_dir = Path(persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)
        (persist_dir / f'{stem}.json').unlink(missing_ok=True)
        dest = persist_dir / f'{stem}.TextGrid'
        shutil.move(tg_path, str(dest))
        _write_sidecar(persist_dir, stem, audio_path, transcript, g2p=g2p)
        return _parse_textgrid(str(dest))


def _readable(tg_path) -> bool:
    """True if the TextGrid parses (see _parse_textgrid)."""
    try:
        _parse_textgrid(str(tg_path))
        return True
    except Exception:
        return False


def _stage(corpus_dir: str, stem: str, audio_path: str, prepared: str) -> None:
    """Copy one stimulus into `corpus_dir/<stem>/` as `<stem>.wav` + `.lab`.

    MFA names the speaker after the folder a file sits in, and computes CMVN
    and the fMLLR speaker-adaptation transform per speaker. One subfolder per
    stimulus makes every recording its own speaker, so a batch aligns each
    file exactly as if it were aligned alone: its onsets depend only on its
    own audio, never on which other recordings share the batch. (A flat
    corpus folder pooled every file into one speaker, shifting some onsets by
    up to ~80 ms with batch composition.)
    """
    spk_dir = os.path.join(corpus_dir, stem)
    os.makedirs(spk_dir, exist_ok=True)
    shutil.copy(audio_path, os.path.join(spk_dir, f'{stem}.wav'))
    with open(os.path.join(spk_dir, f'{stem}.lab'), 'w', encoding='utf-8') as f:
        f.write(prepared)


def _produced_textgrid(output_dir: str, stem: str) -> str | None:
    """Path of MFA's TextGrid for `stem` (MFA mirrors the speaker subfolder
    in its output; the flat location is accepted too), or None if absent."""
    for path in (os.path.join(output_dir, stem, f'{stem}.TextGrid'),
                 os.path.join(output_dir, f'{stem}.TextGrid')):
        if os.path.exists(path):
            return path
    return None


def _parse_textgrid(tg_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse an MFA TextGrid into (word_df, phone_df) with pipeline columns.

    word_df columns: Transcript / Onset / End. phone_df columns: phone / onset /
    offset. Shared by both the fresh-MFA path and the disk-reuse path so the two
    return identical schemas.
    """
    word_df = parse_words(tg_path)
    phone_df = parse_phones(tg_path)
    word_df = word_df.rename(columns={
        'word': 'Transcript',
        'onset': 'Onset',
        'offset': 'End',
    })
    return word_df, phone_df


def saved_alignment_files(output_folder, audio_path: str) -> tuple[Path, Path]:
    """(TextGrid, freshness sidecar) paths a run writing to `output_folder`
    keeps for this stimulus (they may not exist)."""
    persist_dir = alignment_dir(output_folder)
    stem = _stem(audio_path)
    return persist_dir / f'{stem}.TextGrid', persist_dir / f'{stem}.json'


def is_alignment_fresh(output_folder, audio_path: str, transcript_text: str,
                       *, g2p: bool = False) -> bool:
    """True iff a run writing to `output_folder` would reuse the saved
    alignment for this stimulus instead of running MFA on it.

    The same test ensure_alignments applies per item (_is_fresh, then
    _readable) on the folder main._prealign_batch passes it
    (alignment_dir(output_folder)). Read-only: creates, deletes and writes
    nothing. Hashes the audio file when a matching sidecar exists.
    """
    textgrid, _ = saved_alignment_files(output_folder, audio_path)
    return (_is_fresh(textgrid.parent, textgrid.stem, audio_path,
                      transcript_text, g2p=g2p)
            and _readable(textgrid))


def ensure_alignments(items, persist_dir, *, g2p: bool = False) -> dict:
    """Ensure a fresh <stem>.TextGrid exists under persist_dir for each item.

    items: iterable of (audio_path, transcript_text). The stem is derived from
    the audio basename. Fresh stimuli (matching sidecar, including the `g2p`
    setting) are left untouched; stale/missing ones have their old artifacts
    deleted, then the whole stale set is aligned in a SINGLE `mfa align`
    invocation and each result is moved into persist_dir with a sidecar. If
    nothing is stale, no subprocess runs.

    Returns {'cached': [...], 'aligned': [...], 'failed': [...],
    'corrupt': [...]} of stems; 'corrupt' lists fresh-looking TextGrids that
    could not be parsed and were therefore re-aligned (also in 'aligned' or
    'failed').

    Note: a per-stimulus alignment miss is reported in 'failed' (no exception),
    but a whole-subprocess failure raises from `_run_mfa` and propagates — with
    the stale stems' artifacts already deleted. Callers that must degrade
    gracefully should catch it (see main._prealign_batch).
    """
    persist_dir = Path(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    cached: list[str] = []
    corrupt: list[str] = []
    # Dedup by stem (unique stimulus ids expected); last transcript wins.
    stale_by_stem: dict[str, tuple[str, str]] = {}
    for audio_path, transcript_text in items:
        stem = _stem(audio_path)
        if _is_fresh(persist_dir, stem, audio_path, transcript_text, g2p=g2p):
            if _readable(persist_dir / f'{stem}.TextGrid'):
                cached.append(stem)
                continue
            # Sidecar matches but the TextGrid itself is damaged (partial
            # copy, disk error, hand edit): re-align instead of failing later.
            print(f"[mfa] WARNING: saved alignment for {stem!r} is unreadable; "
                  "re-aligning it")
            corrupt.append(stem)
            stale_by_stem[stem] = (audio_path, transcript_text)
        else:
            if stem in stale_by_stem:
                print(f"[mfa] WARNING: duplicate stem {stem!r} in batch; "
                      "last transcript wins")
            stale_by_stem[stem] = (audio_path, transcript_text)

    if not stale_by_stem:
        print(f"[mfa] {len(cached)} stimuli: all cached, 0 aligned")
        return {'cached': cached, 'aligned': [], 'failed': [], 'corrupt': []}

    # Delete stale artifacts first so a failed realign can't leave a stale file
    # for run() to pick up.
    for stem in stale_by_stem:
        (persist_dir / f'{stem}.TextGrid').unlink(missing_ok=True)
        (persist_dir / f'{stem}.json').unlink(missing_ok=True)

    aligned: list[str] = []
    failed: list[str] = []
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        corpus_dir = os.path.join(tmpdir, 'corpus')
        output_dir = os.path.join(tmpdir, 'output')
        os.makedirs(corpus_dir)
        os.makedirs(output_dir)
        for stem, (audio_path, transcript_text) in stale_by_stem.items():
            _stage(corpus_dir, stem, audio_path,
                   prepare_for_aligner(transcript_text))

        # single invocation for the batch
        _run_mfa(corpus_dir, output_dir, g2p=g2p,
                 temp_dir=os.path.join(tmpdir, 'mfa_tmp'))

        for stem, (audio_path, transcript_text) in stale_by_stem.items():
            produced = _produced_textgrid(output_dir, stem)
            if produced is not None:
                shutil.move(produced, str(persist_dir / f'{stem}.TextGrid'))
                _write_sidecar(persist_dir, stem, audio_path, transcript_text,
                               g2p=g2p)
                aligned.append(stem)
            else:
                failed.append(stem)

    msg = (f"[mfa] {len(cached) + len(stale_by_stem)} stimuli: "
           f"{len(cached)} cached, {len(aligned)} aligned in 1 run")
    if failed:
        msg += f", {len(failed)} failed ({', '.join(failed)})"
    print(msg)
    return {'cached': cached, 'aligned': aligned, 'failed': failed,
            'corrupt': corrupt}


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


def _run_mfa(corpus_dir: str, output_dir: str, *, g2p: bool = False,
             temp_dir: str | None = None) -> None:
    """Invoke MFA as a subprocess. Raises RuntimeError on failure.

    `temp_dir` becomes MFA's `--temporary_directory` (its working folder,
    including `corpus.db`). Without it MFA works in `$MFA_ROOT_DIR/<corpus
    name>/`, which every call shares (the corpus dir is always named
    `corpus`): two runs at once collide on the database lock, and a read-only
    MFA_ROOT_DIR (the Sherlock .sif bakes it under /opt) can't be written at
    all. Callers pass a folder inside their own TemporaryDirectory.

    When `g2p` is set and the vendored G2P model is present, MFA generates a
    pronunciation for out-of-dictionary words instead of aligning them to a
    single `spn` phone. This changes alignment output, so callers must record it
    in the freshness sidecar (see _freshness_dict).
    """
    cmd = [
        'mfa', 'align',
        corpus_dir,
        MFA_DICTIONARY,
        MFA_ACOUSTIC_MODEL,
        output_dir,
        '--clean',
        '--quiet',
    ]
    if temp_dir is not None:
        cmd += ['--temporary_directory', temp_dir]
    if g2p:
        if g2p_model_available():
            cmd += ['--g2p_model_path', str(G2P_MODEL_PATH)]
        else:
            print(f'[mfa] WARNING: g2p requested but no G2P model at '
                  f'{G2P_MODEL_PATH}; out-of-dictionary words will align to '
                  f'spn. Run: python utils/download_models.py',
                  file=sys.stderr)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"MFA alignment failed.\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        )
