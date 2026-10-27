"""
SensiExtractor — stimulus feature extraction pipeline.

Called by the two launchers, never run directly: gui.py (Run tab, via
sensi/gui/run_dispatch.py) and run.py (headless, from config/features.yaml
+ run.yaml). Both build the argument namespace with the planner
(sensi/gui/planner: plan_from_prefs -> to_namespace) and call the three
batch runners below. The old `python main.py --flags` command line was
retired on 2026-10-27.

Predictability (surprisal / cloze / top-match) is configured via
config/features.yaml's "predictability" section (roster of models x
metrics), passed in as ns.predictability by the planner. See
sensi/extractors/predictability.py.

Output:
    One CSV file with one row per word token. Punctuation tokens are
    included (they carry spaCy annotations) but have NaN for numeric
    features (Zipf_Freq, Onset, End).

Column order in output CSV:
    --- spaCy structural ---
    doc_id, paragraph_id, sentence_id, sentence, start, end,
    token_id, token_in_sentence_id
    --- token-level linguistic (spaCy) ---
    token, lemma, upos, xpos, feats, dep_rel, head_token_id
    --- computed features ---
    Zipf_Freq, then predictability columns <slug>_<Metric> (one per selected
    model x metric, e.g. gpt2_Surprisal / gpt2_Cloze / llama3_8b_Surprisal)
    --- MFA alignment ---
    Transcript, Onset, End

Merge strategy (alignment + parse):
    After alignment, Transcript words are merged into the token DataFrame
    by monotonic sequence matching (sensi/extractors/align_merge.py): both word
    streams are consumed left to right, with local repair for tokenization
    differences (contractions, digits vs. spelled-out numbers, hyphenated
    words). Tokens that cannot be paired get NaN Onset/End — never the
    timing of a same-form word elsewhere in the document.
"""

import contextlib
import os
import re
import sys
import time
from pathlib import Path

# MKL on AMD CPUs (Windows/Ryzen) raises a fatal exception (0xc06d007f) during
# BLAS matmul — notably librosa's mel-filterbank einsum in the acoustic level —
# under the default Intel threading layer. Force sequential threading BEFORE
# numpy imports and initialises MKL. Mirrors tests/conftest.py so real GUI/run.py
# runs behave like the (passing) test environment instead of hard-crashing.
os.environ.setdefault("MKL_THREADING_LAYER", "sequential")

import numpy as np
import pandas as pd

# Keep all model/cache downloads inside the project folder
from sensi.utils.local_caches import use_project_caches
use_project_caches()


# ---------------------------------------------------------------------------
# Column ordering
# ---------------------------------------------------------------------------

_COLUMN_ORDER = [
    'doc_id', 'paragraph_id', 'sentence_id', 'sentence', 'start', 'end',
    'token_id', 'token_in_sentence_id',
    'token', 'lemma', 'upos', 'xpos', 'feats', 'dep_rel', 'head_token_id',
    'Zipf_Freq',
    'Edu_Zipf_K2', 'Edu_Zipf_3to5', 'Edu_Zipf_6to8', 'Edu_Zipf_9to12', 'Edu_Zipf_Overall',
    # Sem_<Dict>_<cat> columns get appended dynamically by _reorder
    'Sem_Categories',
    # Predictability <slug>_<Metric> columns (dynamic; one per selected model x
    # metric) are inserted here by _reorder, in the slot the old Surprisal /
    # LLM_* columns used to occupy.
    '_PREDICTABILITY_',
    'Transcript', 'Onset', 'End',
    'Word_Duration', 'Word_Gap',
    # Prosody (Task 5)
    'Pitch_Mean', 'Pitch_Min', 'Pitch_Max', 'Pitch_Std',
    'Volume_Mean', 'Volume_Min', 'Volume_Max', 'Volume_Std',
    'Pitch_Delta', 'Volume_Delta',
]


_PREDICTABILITY_COL_RE = re.compile(r'^[a-z0-9_]+_(Surprisal|Cloze|TopMatch)$')


def _reorder(df: pd.DataFrame) -> pd.DataFrame:
    sem_cols = sorted([c for c in df.columns if c.startswith('Sem_') and c != 'Sem_Categories'])
    # Predictability columns are named <slug>_<Metric> (one per selected model x
    # metric); sort for a deterministic, model-grouped order.
    pred_cols = sorted(c for c in df.columns if _PREDICTABILITY_COL_RE.match(c))
    canonical: list[str] = []
    for c in _COLUMN_ORDER:
        if c == 'Sem_Categories':
            canonical.extend(sem_cols)
            canonical.append('Sem_Categories')
        elif c == '_PREDICTABILITY_':
            canonical.extend(pred_cols)
        else:
            canonical.append(c)
    present = [c for c in canonical if c in df.columns]
    extra = [c for c in df.columns if c not in present]
    return df[present + extra]


# ---------------------------------------------------------------------------
# Per-step timing (Run-tab estimates; sensi/utils/timing_history.py)
# ---------------------------------------------------------------------------
# The batch runners attach a StageTimer as args._stage_timer; every helper
# below is a no-op without one (direct run() callers, e.g. tests), so
# timing never touches what a stage computes or writes.

def _stage(args, step, units, model=None, device='cpu'):
    """`args._stage_timer.stage(...)` when a batch timer is attached, else a
    no-op context. `units` may be a zero-arg callable so its cost (e.g. an
    audio-header read) is paid only when timing; if it fails, the stage just
    runs untimed."""
    timer = getattr(args, '_stage_timer', None)
    if timer is None:
        return contextlib.nullcontext()
    try:
        u = units() if callable(units) else units
    except Exception:   # noqa: BLE001 — timing must never fail a stimulus
        return contextlib.nullcontext()
    return timer.stage(step, u, model=model, device=device)


def _preload_timed(args, key, cache, cache_key, load) -> None:
    """Load a process-cached model up front, timing it into load_seconds, so
    the first stimulus's sample holds compute only. No-op without a timer or
    when the model is already cached (the extractor would reuse it anyway)."""
    timer = getattr(args, '_stage_timer', None)
    if timer is None or cache_key in cache:
        return
    t0 = time.perf_counter()
    load()
    timer.load_seconds[key] = time.perf_counter() - t0


def _cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:   # noqa: BLE001
        return False


def _timed_predictability(args, df, text, models, metrics, context):
    """add_predictability one model at a time, each timed as its own sample.

    Same columns, values and order as the single multi-model call (the
    extractor scores and writes each slug independently, in roster order).
    A model loaded for the first time is loaded first and timed into
    load_seconds; a stimulus whose scores are already cached (the doc pass
    reusing the word pass's scores) records no sample.
    """
    from sensi.extractors import predictability as _pred
    from sensi.extractors.predictability import add_predictability
    from sensi.extractors.predictability_models import effective_context, spec_for
    from sensi.utils.timing_history import load_key, model_device
    if not [m for m in metrics if m in _pred._METRIC_COLS]:
        return add_predictability(df, text, models=models, metrics=metrics,
                                  context=context)
    for slug in models:
        spec_for(slug)   # validate the whole roster first, like the extractor
    timer = args._stage_timer
    tokens = [str(t) for t in df['token'].tolist()]
    digest = _pred._text_tokens_digest(text, tokens)
    for slug in models:
        hf_id = spec_for(slug).hf_id
        eff = effective_context(slug, context or {})
        if (hf_id, eff, digest) in _pred._SCORE_CACHE:
            df = add_predictability(df, text, models=[slug], metrics=metrics,
                                    context=context)
            continue
        _preload_timed(args, load_key('predictability', slug),
                       _pred._BACKEND_CACHE, (hf_id, eff),
                       lambda: _pred._get_backend(slug, eff))
        device = model_device('predictability', slug, _cuda_available())
        with timer.stage('predictability', count_words(text), model=slug,
                         device=device):
            df = add_predictability(df, text, models=[slug], metrics=metrics,
                                    context=context)
    return df


def count_words(text) -> int:
    """Words of a raw transcript — the unit for text steps, shared with the
    Run-tab estimator (sensi.utils.timing_history.count_words)."""
    from sensi.utils.timing_history import count_words as _count
    return _count(text)


def _transcript_words(path) -> int:
    """count_words of a transcript file."""
    with open(path, 'r', encoding='utf-8') as f:
        return count_words(f.read())


def _start_pair_timings(args) -> None:
    timer = getattr(args, '_stage_timer', None)
    if timer is not None:
        timer.samples.clear()


def _record_pair_timings(args) -> None:
    """Append this (successful) stimulus's samples to the machine-local
    timing history. Never raises: history is disposable learning data."""
    timer = getattr(args, '_stage_timer', None)
    if timer is None:
        return
    samples, timer.samples = timer.samples, []
    if not samples and not timer.load_seconds:
        return
    try:
        from sensi.utils.timing_history import append_samples
        append_samples(samples, load_seconds=dict(timer.load_seconds))
        timer.load_seconds.clear()
    except Exception as exc:   # noqa: BLE001
        print(f'[timing] could not update timing history: {exc}', file=sys.stderr)


def _new_stage_timer():
    from sensi.utils.timing_history import StageTimer
    return StageTimer()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args) -> pd.DataFrame:
    from sensi.utils.preferences import load_preferences

    prefs = load_preferences()
    if args.sem_categories:
        sem_categories = args.sem_categories.split(',')
    else:
        # v7 prefs schema unifies categories across levels into a single
        # `categories_<backend>` key. This fallback is reached by direct
        # run() callers; the planner (GUI / run.py) always sets
        # ns.sem_categories explicitly.
        backend_hint = prefs.get('semantic', {}).get('backend', 'auto')
        suffix = 'liwc' if backend_hint == 'liwc' else 'empath'
        sem_categories = prefs.get('semantic', {}).get(
            f'categories_{suffix}', [])

    with open(args.transcript, 'r', encoding='utf-8') as f:
        text = f.read()

    df = None

    if not args.no_parse:
        from sensi.extractors.text_parser import parse as spacy_parse
        print('[spacy] Running spaCy annotation…')
        with _stage(args, 'spacy_parse', lambda: count_words(text)):
            df = spacy_parse(text, doc_id=int(getattr(args, 'doc_id', 1) or 1))

    if not args.no_freq and df is not None:
        from sensi.extractors.word_freq import add_word_freq
        print('[freq] Computing SUBTLEX Zipf…')
        df = add_word_freq(df)

    if not args.no_edu_freq and df is not None:
        from sensi.extractors.educator_freq import add_edu_freq
        print('[edu] Computing EWFG grade-band frequencies…')
        edu_bands = (
            [b.strip() for b in args.edu_bands.split(',') if b.strip()]
            if getattr(args, 'edu_bands', None)
            else None
        )
        df = add_edu_freq(df, lookup_path=args.edu_freq_lookup, bands=edu_bands)

    if not getattr(args, 'no_lexical_norms', True) and df is not None:
        from sensi.extractors.lexical_norms import add_lexical_norms
        print('[norms] Computing lexical norms (AoA / concreteness / imageability '
              '/ familiarity / valence-arousal-dominance)…')
        df = add_lexical_norms(df, paths=getattr(args, 'lexical_norms_paths', None))

    if not args.no_sem_domain and df is not None:
        from sensi.extractors.semantic_domain import add_semantic_domain
        backend, dict_path = _resolve_sem_backend(args.sem_dict)
        print(f'[sem] Computing semantic-domain ({backend})…')
        df = add_semantic_domain(df, backend=backend, categories=sem_categories,
                                 dictionary_path=dict_path)

    # Predictability (surprisal / cloze / top-match) — unified, roster-driven
    # extractor (replaces the old separate add_surprisal / add_llm_cloze
    # blocks). Config comes from ns.predictability when a caller (GUI batch
    # dispatch) set it explicitly; a direct run() caller that leaves
    # args.predictability unset falls back to the config store (mirrors
    # how sem_categories falls back to prefs above).
    pconf = getattr(args, 'predictability', None)
    if pconf is None:
        pconf = prefs.get('predictability', {})
    pred_models = list(pconf.get('models') or [])
    pred_metrics = [m for m, on in (pconf.get('metrics') or {}).items() if on]
    if pred_models and pred_metrics and df is not None:
        from sensi.extractors.predictability import add_predictability
        print(f'[predict] {len(pred_models)} model(s) x {len(pred_metrics)} metric(s)…')
        if getattr(args, '_stage_timer', None) is None:
            df = add_predictability(df, text, models=pred_models, metrics=pred_metrics,
                                    context=pconf.get('context'))
        else:
            df = _timed_predictability(args, df, text, pred_models, pred_metrics,
                                       pconf.get('context'))

    args._phones_df = None  # reset each run; populated below when alignment runs
    if not args.no_align and args.audio:
        from sensi.extractors.aligner import (
            align_with_phones as mfa_align, alignment_dir, cache_key,
        )
        # Reuse a per-document alignment across passes (word + doc batches in
        # one GUI run share `_align_cache`), so MFA runs at most once per doc.
        _align_cache = getattr(args, '_align_cache', None)
        # In batch mode the pre-pass (run_*_level_batch) has already aligned
        # every stimulus to <output_folder>/Montreal Forced Alignment/, so run()
        # reads TextGrids from disk and must NOT spawn MFA (run_if_missing=False).
        # A direct run() call with no output_folder: persist_dir=None, MFA runs once.
        persist_dir = (alignment_dir(args.output_folder)
                       if getattr(args, 'output_folder', None) else None)
        run_if_missing = not getattr(args, '_batch_prealigned', False)
        g2p_oov = bool(getattr(args, 'g2p_oov', False))
        # Built via the SAME helper align_with_phones uses to store the memo
        # entry (sensi.extractors.aligner.cache_key), so this log-only check can
        # never desync from the real lookup inside mfa_align — a hand-rebuilt
        # tuple here previously omitted the `g2p` element added for OOV
        # support, so the "Reusing cached alignment" line never fired.
        if _align_cache is not None and (
                cache_key(args.audio, text, g2p_oov) in _align_cache):
            print('[mfa] Reusing cached alignment…')
        else:
            print('[mfa] Loading alignment…')
        align_df, phones_df = mfa_align(
            args.audio, text, cache=_align_cache,
            persist_dir=persist_dir, run_if_missing=run_if_missing,
            g2p=g2p_oov)
        args._phones_df = phones_df
        if df is not None:
            from sensi.extractors.align_merge import merge_alignment
            df = merge_alignment(df, align_df)
        else:
            df = align_df

    # Prosody (Task 6) — needs Onset/End from alignment, so it runs here.
    args._contours = None  # initialized on every run; consumed by doc-level batch
    if (not getattr(args, 'no_prosody', False) and df is not None
            and 'Onset' in df.columns and args.audio):
        try:
            from sensi.extractors.audio_features import extract as extract_contours
            from sensi.extractors.prosody import add_word_prosody
            print('[prosody] Extracting pitch/intensity contours…')
            with _stage(args, 'prosody_contours',
                        lambda: _audio_duration(args.audio)):
                contours = extract_contours(
                    args.audio,
                    pitch_floor_hz=getattr(args, 'pitch_floor', 75.0),
                    pitch_ceiling_hz=getattr(args, 'pitch_ceiling', 500.0),
                )
                df = add_word_prosody(
                    df, contours,
                    pitch_units=getattr(args, 'pitch_units', 'semitones'),
                )
            args._contours = contours
        except Exception as exc:
            print(f'[prosody] failed: {exc}', file=sys.stderr)
            # Columns remain absent; downstream _reorder is tolerant.

    if df is not None:
        from sensi.extractors.timing import add_timing_columns
        df = add_timing_columns(df)
        df = _reorder(df)

    if df is None:
        # Don't sys.exit here — batch mode calls run() inside a per-pair
        # try/except that only catches Exception. SystemExit would bypass
        # it, kill the worker thread, and leave the GUI's Run button stuck
        # in "running" state (done_cb never fires). Raise a regular
        # RuntimeError instead so the batch handler records it as a pair
        # error and the run completes cleanly.
        raise RuntimeError('All extractors are disabled — nothing to extract.')

    if args.save_prefs and args.sem_categories:
        from sensi.utils.preferences import save_preferences
        # v7 unified: a single categories_<backend> key per backend. Write to
        # both backends so the saved selection survives a later backend switch.
        # Only direct callers set save_prefs; the GUI saves via
        # category_picker._on_toggle which already writes the right key.
        save_preferences({'semantic': {
            'categories_empath': sem_categories,
            'categories_liwc':   sem_categories,
        }})

    return df


import time

from sensi.utils.file_pairing import find_pairs, Pair


class InputFolderError(FileNotFoundError):
    """A batch input folder is missing or holds nothing to process."""


def _check_input_folders(transcripts_dir, audio_folder) -> None:
    """Fail fast on a missing transcripts folder or a configured-but-missing
    audio folder. Without this a typo'd transcripts path processes 0 files and
    a typo'd audio path silently runs every transcript text-only (no MFA,
    prosody or syllables), both reported as a successful run. Raises
    InputFolderError so the GUI dispatcher surfaces it as a run error.
    """
    if not Path(transcripts_dir).is_dir():
        raise InputFolderError(
            f'Transcripts folder does not exist: {transcripts_dir}')
    if audio_folder and not Path(audio_folder).is_dir():
        raise InputFolderError(f'Audio folder does not exist: {audio_folder}')


def _warn_pairing_gaps(logger, args, pairing) -> None:
    """Record, in the manifest, transcripts with no matching audio that run
    text-only while alignment is on (a partial name mismatch or a wrong audio
    folder would otherwise only show up as missing columns)."""
    if (args.audio_folder and not getattr(args, 'no_align', False)
            and pairing.unmatched_text):
        names = ', '.join(p.name for p in pairing.unmatched_text)
        logger.add_warning(
            f'{len(pairing.unmatched_text)} transcript(s) have no matching '
            f'audio and ran text-only (no timing, prosody, syllable or '
            f'phoneme output): {names}')


def _batch_work_items(pairing) -> list:
    """Ordered work list for the word/doc batches: every matched (text, audio)
    pair, followed by each audio-less transcript as a text-only item.

    Text-only items reuse the Pair shape with an empty audio_path so the loop
    body and logging stay uniform; main.run() guards its audio stages with
    `and args.audio`, so an empty path simply skips MFA/prosody and the file
    is processed for whatever non-audio features are selected. This is what
    keeps standalone transcripts (no matched .wav) from being dropped — see
    the 2026-06-24 bug report.
    """
    items = list(pairing.pairs)
    items.extend(
        Pair(stimulus_id=p.stem, transcript_path=p, audio_path="")
        for p in pairing.unmatched_text
    )
    return items
from sensi.utils.run_logger import RunLogger
from sensi.utils.doc_summary import build_doc_summary


# Per-column toggle → CSV column name mappings (schema v4).
# Used by run_word_level_batch and run_doc_level_batch to drop unselected
# columns before writing the CSV. The toggles are passed via args.features
# (and args.output for doc-level toggles) by the planner; a direct caller
# that leaves them unset keeps all columns.

_WORD_FEATURE_COLUMN_MAP = {
    # Structural
    "feat_struct_doc_id": "doc_id",
    "feat_struct_paragraph_id": "paragraph_id",
    "feat_struct_sentence_id": "sentence_id",
    "feat_struct_sentence_text": "sentence",
    "feat_struct_char_start": "start",
    "feat_struct_char_end": "end",
    "feat_struct_token_id": "token_id",
    "feat_struct_token_in_sentence_id": "token_in_sentence_id",
    # Acoustic
    "feat_acoustic_onset": "Onset",
    "feat_acoustic_end": "End",
    "feat_acoustic_word_duration": "Word_Duration",
    "feat_acoustic_word_gap": "Word_Gap",
    # Prosody
    "feat_prosody_pitch_mean":  "Pitch_Mean",
    "feat_prosody_pitch_min":   "Pitch_Min",
    "feat_prosody_pitch_max":   "Pitch_Max",
    "feat_prosody_pitch_std":   "Pitch_Std",
    "feat_prosody_pitch_delta": "Pitch_Delta",
    "feat_prosody_volume_mean":  "Volume_Mean",
    "feat_prosody_volume_min":   "Volume_Min",
    "feat_prosody_volume_max":   "Volume_Max",
    "feat_prosody_volume_std":   "Volume_Std",
    "feat_prosody_volume_delta": "Volume_Delta",
    # Lexical norms (user-supplied; one column per toggle)
    "feat_norm_aoa_mean":     "AoA_Mean",
    "feat_norm_aoa_sd":       "AoA_SD",
    "feat_norm_concreteness": "Concreteness",
    "feat_norm_imageability": "Imageability",
    "feat_norm_familiarity":  "Familiarity",
    "feat_norm_valence":      "Valence",
    "feat_norm_arousal":      "Arousal",
    "feat_norm_dominance":    "Dominance",
}

_DOC_OUTPUT_COLUMN_MAP = {
    "out_count_tokens": "n_tokens",
    "out_count_content_tokens": "n_content_tokens",
    "out_count_sentences": "n_sentences",
    "out_count_paragraphs": "n_paragraphs",
}

_DOC_PSYCH_COLUMN_MAP = {
    "feat_psych_authentic": "psych_authentic",
    "feat_psych_analytic": "psych_analytic",
    "feat_psych_tone": "psych_tone",
    "feat_psych_clout": "psych_clout",
}

_DOC_PROSODY_COLUMN_MAP = {
    "feat_doc_prosody_pitch_mean":  "doc_pitch_mean",
    "feat_doc_prosody_pitch_min":   "doc_pitch_min",
    "feat_doc_prosody_pitch_max":   "doc_pitch_max",
    "feat_doc_prosody_pitch_std":   "doc_pitch_std",
    "feat_doc_prosody_volume_mean": "doc_volume_mean",
    "feat_doc_prosody_volume_min":  "doc_volume_min",
    "feat_doc_prosody_volume_max":  "doc_volume_max",
    "feat_doc_prosody_volume_std":  "doc_volume_std",
    "feat_doc_prosody_pitch_delta_mean":  "doc_pitch_delta_mean",
    "feat_doc_prosody_volume_delta_mean": "doc_volume_delta_mean",
    "feat_doc_prosody_hnr":                "doc_hnr",
    "feat_doc_prosody_articulation_rate":  "doc_articulation_rate",
    "feat_doc_prosody_pause_count":         "doc_pause_count",
    "feat_doc_prosody_pause_mean_duration": "doc_pause_mean_duration",
    "feat_doc_prosody_pause_total_duration":"doc_pause_total_duration",
    "feat_doc_prosody_pause_pct_silence":   "doc_pause_pct_silence",
}

_SYLLABLE_COLUMN_MAP = {
    "feat_syllable_onset":        "Onset",
    "feat_syllable_end":          "End",
    "feat_syllable_duration":     "Duration",
    "feat_syllable_label":        "Label",
    "feat_syllable_arpabet":      "ARPABET",
    "feat_syllable_ipa":          "IPA",
    "feat_syllable_pitch_mean":   "Pitch_Mean",
    "feat_syllable_volume_mean":  "Volume_Mean",
    "feat_syllable_pitch_delta":  "Pitch_Delta",
    "feat_syllable_volume_delta": "Volume_Delta",
    "feat_syllable_pcenter":      "P_Center",
}

_PHONEME_COLUMN_MAP = {
    "feat_phoneme_arpabet":     "ARPABET",
    "feat_phoneme_stress":      "Stress",
    "feat_phoneme_ipa":         "IPA",
    "feat_phoneme_pron_source": "Pron_Source",
    "feat_phoneme_onset":    "Onset",
    "feat_phoneme_end":      "End",
    "feat_phoneme_duration": "Duration",
    "feat_phoneme_surprisal":      "phoneme_surprisal",
    "feat_phoneme_cohort_entropy": "cohort_entropy",
}


def _syllable_columns_to_drop(features: dict) -> set:
    """Return the set of syllable-CSV columns to drop based on v6 toggles."""
    return {col for key, col in _SYLLABLE_COLUMN_MAP.items()
            if not features.get(key, True)}


def _phoneme_columns_to_drop(features: dict) -> set:
    """Return the set of phoneme-CSV columns to drop based on v6 toggles."""
    return {col for key, col in _PHONEME_COLUMN_MAP.items()
            if not features.get(key, True)}


_SENTENCE_COLUMN_MAP = {
    "feat_sentence_text":          "sentence",
    "feat_sentence_onset":         "Onset",
    "feat_sentence_end":           "End",
    "feat_sentence_pitch_mean":    "Pitch_Mean",
    "feat_sentence_pitch_min":     "Pitch_Min",
    "feat_sentence_pitch_max":     "Pitch_Max",
    "feat_sentence_pitch_std":     "Pitch_Std",
    "feat_sentence_pitch_delta":   "Pitch_Delta",
    "feat_sentence_volume_mean":   "Volume_Mean",
    "feat_sentence_volume_min":    "Volume_Min",
    "feat_sentence_volume_max":    "Volume_Max",
    "feat_sentence_volume_std":    "Volume_Std",
    "feat_sentence_volume_delta":  "Volume_Delta",
    "feat_sentence_hnr":           "HNR",
}


def _sentence_columns_to_drop(features: dict) -> set:
    return {col for key, col in _SENTENCE_COLUMN_MAP.items()
            if not features.get(key, True)}


# Lexical-linguistic per-column toggles → the columns each one owns. Unlike
# the 1:1 _WORD_FEATURE_COLUMN_MAP, one toggle can own several columns. These
# columns are produced by spaCy (§0.1) regardless, so the only way to honour a
# lexling toggle being off is to drop its columns from the features CSV here.
_LEXLING_COLUMN_MAP = {
    "lexling_tokens": ("token", "lemma"),
    "lexling_pos":    ("upos", "xpos"),
    "lexling_morph":  ("feats",),
    "lexling_dep":    ("dep_rel", "head_token_id"),
}


def _word_columns_to_drop(features: dict) -> set:
    """Return the set of word-level CSV columns to drop based on v4 toggles."""
    drop = {col for key, col in _WORD_FEATURE_COLUMN_MAP.items()
            if not features.get(key, True)}
    for key, cols in _LEXLING_COLUMN_MAP.items():
        if not features.get(key, True):
            drop.update(cols)
    # head_token_id references the syntactic head by its global token_id, so
    # token_id is its anchor. Keep token_id whenever the dependency parse is
    # emitted, even if its own structural toggle is off — otherwise
    # head_token_id would dangle with no id column to resolve against.
    if features.get("lexling_dep", True):
        drop.discard("token_id")
    return drop


def _doc_columns_to_drop(output: dict, features: dict) -> set:
    """Return the set of doc-level CSV columns to drop based on v5 toggles."""
    drop = set()
    for key, col in _DOC_OUTPUT_COLUMN_MAP.items():
        if not output.get(key, True):
            drop.add(col)
    for key, col in _DOC_PSYCH_COLUMN_MAP.items():
        if not features.get(key, True):
            drop.add(col)
    for key, col in _DOC_PROSODY_COLUMN_MAP.items():
        if not features.get(key, True):
            drop.add(col)
    return drop


def _level_dir(output_dir: Path, level: str) -> Path:
    """Return output_dir/level, creating it if needed (lazy mkdir)."""
    d = output_dir / level
    d.mkdir(parents=True, exist_ok=True)
    return d


def _audio_duration(audio_path) -> float:
    """Duration in seconds via soundfile (no full decode)."""
    import soundfile as sf
    info = sf.info(str(audio_path))
    return float(info.frames) / float(info.samplerate)


def _word_embedding_vars(emb: np.ndarray, df: pd.DataFrame,
                         is_valid: np.ndarray) -> dict:
    """Flat top-level .mat variables for a word-level embedding sidecar.

    `embeddings` stays a top-level key so existing consumers keep working.
    Labels are taken from df (row-aligned to emb). `tokens` uses an object
    array so savemat renders it as a MATLAB cellstr (unpadded, variable
    length) rather than a space-padded char matrix.
    """
    if len(df) != emb.shape[0]:
        raise ValueError(
            f"df ({len(df)} rows) and emb ({emb.shape[0]} rows) must be row-aligned"
        )
    return {
        "embeddings":  emb,
        "tokens":      np.array(df["token"], dtype=object),
        "word_id":     df["token_id"].to_numpy(dtype=np.int64),
        "sentence_id": df["sentence_id"].to_numpy(dtype=np.int64),
        "is_valid":    np.asarray(is_valid, dtype=bool),
    }


def _sentence_embedding_vars(emb: np.ndarray, df: pd.DataFrame,
                              is_valid: np.ndarray) -> dict:
    """Flat top-level .mat variables for the sentence-level embedding sidecar.

    Sentence rows follow the same first-seen (doc_id, sentence_id) grouping the
    BERT extractor uses, so they stay aligned to `emb`. `sentence` uses an
    object array so savemat renders it as a MATLAB cellstr.
    """
    groups = list(df.groupby(["doc_id", "sentence_id"], sort=False))
    if len(groups) != emb.shape[0]:
        raise ValueError(
            f"sentence groups ({len(groups)}) and emb ({emb.shape[0]} rows) "
            "must be aligned"
        )
    sent_ids = [key[1] for key, _ in groups]
    sent_texts = [g["sentence"].iloc[0] for _, g in groups]
    return {
        "embeddings":  emb,
        "sentence_id": np.asarray(sent_ids, dtype=np.int64),
        "sentence":    np.array(sent_texts, dtype=object),
        "is_valid":    np.asarray(is_valid, dtype=bool),
    }


def _prealign_batch(run_args, work_items, output_dir, logger=None) -> None:
    """Align every audio-bearing stimulus in the batch once, up front.

    Populates <output_dir>/Montreal Forced Alignment/ so the per-document run()
    calls read TextGrids from disk instead of each spawning MFA. On success sets
    run_args._batch_prealigned so run() reads disk only (a disk miss then means
    that stimulus failed to align). If the batch MFA invocation fails outright,
    fall back to per-document alignment: leave _batch_prealigned False so run()
    uses run_if_missing=True and the existing per-pair error handling degrades
    gracefully instead of aborting the whole batch. The fallback persists each
    alignment to the same folder, so a later pass in this dispatch finds it on
    disk; stimuli already in the shared in-memory `_align_cache` are skipped
    here too, so MFA runs at most once per document even after a failure.

    Both a whole-batch failure and per-stimulus misses are recorded on
    `logger` (when given) so they reach the run manifest, not only stderr.

    No-op when alignment is disabled or no pair has audio.
    """
    # Default to False so a later pass on a copied namespace can't inherit a
    # stale True; only a fully-successful pre-pass below flips it on.
    run_args._batch_prealigned = False
    if getattr(run_args, 'no_align', False):
        return
    from sensi.extractors.aligner import alignment_dir, cache_key, ensure_alignments
    g2p = bool(getattr(run_args, 'g2p_oov', False))
    align_cache = getattr(run_args, '_align_cache', None) or {}
    items = []
    for pair in work_items:
        if not pair.audio_path:
            continue
        # Read transcripts per-pair defensively: an unreadable transcript (bad
        # encoding, permissions) skips only its own stimulus, mirroring the
        # batch loop's per-pair error isolation — it must not abort the whole
        # pre-pass (and thus the whole dispatch).
        try:
            with open(pair.transcript_path, 'r', encoding='utf-8') as f:
                transcript_text = f.read()
        except OSError as exc:
            print(f'[mfa] skipping {pair.stimulus_id} in pre-alignment '
                  f'(cannot read transcript: {exc})', file=sys.stderr)
            continue
        if cache_key(str(pair.audio_path), transcript_text, g2p) in align_cache:
            continue  # aligned earlier in this dispatch (e.g. by the fallback)
        items.append((str(pair.audio_path), transcript_text))
    if not items:
        # Nothing left to align: every audio item is already in memory. Leave
        # _batch_prealigned False so a cache miss could still run MFA, but
        # none is expected.
        return
    t0 = time.perf_counter()
    try:
        result = ensure_alignments(items, alignment_dir(output_dir), g2p=g2p) or {}
    except Exception as exc:
        first_line = (str(exc).strip().splitlines() or [type(exc).__name__])[0]
        print(f'[mfa] batch pre-alignment failed ({exc}); falling back to '
              'per-document alignment', file=sys.stderr)
        if logger is not None:
            logger.add_warning(
                f'MFA batch pre-alignment failed ({first_line}); fell back to '
                'per-document alignment')
        return
    if logger is not None:
        for stem in result.get('corrupt', []):
            logger.add_warning(f'Saved alignment for {stem} was unreadable; '
                               're-aligned')
        for stem in result.get('failed', []):
            logger.add_warning(f'MFA produced no alignment for {stem}')
    _record_mfa_timing(run_args, items, result, time.perf_counter() - t0)
    run_args._batch_prealigned = True


def _record_mfa_timing(run_args, items, result, seconds) -> None:
    """One mfa_align sample per MFA invocation: units = audio seconds of the
    stimuli it actually aligned. Recorded only when every stale stimulus
    aligned (a run with failures is not a clean rate sample) and at least
    one was aligned (an all-cached pre-pass did no MFA work)."""
    if getattr(run_args, '_stage_timer', None) is None:
        return
    aligned = set(result.get('aligned', []))
    if not aligned or result.get('failed'):
        return
    try:
        from sensi.extractors.aligner import _stem
        from sensi.utils.timing_history import Sample, append_samples
        audio_s = sum(_audio_duration(a) for a, _ in items if _stem(a) in aligned)
        append_samples([Sample('mfa_align', None, 'cpu', audio_s, seconds)])
    except Exception as exc:   # noqa: BLE001 — timing must never fail a run
        print(f'[timing] could not record MFA timing: {exc}', file=sys.stderr)


def run_word_level_batch(args, *, on_pair_done=None, should_cancel=None) -> None:
    """Process all matched (transcript, audio) pairs in the configured folders."""
    transcripts_dir = Path(args.transcripts_folder)
    output_dir = Path(args.output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)
    # audio-folder is optional (text-only batch): a non-existent sentinel path
    # makes find_pairs pair no audio, so every transcript runs text-only.
    audio_dir = (Path(args.audio_folder) if args.audio_folder
                 else output_dir / '__no_audio__')

    if not args.analyzer:
        raise SystemExit('An analyzer name is required (run.yaml: shared.analyzer_name, or the Inputs page).')
    _check_input_folders(transcripts_dir, args.audio_folder)

    text_exts = [e.strip() for e in args.text_extensions.split(',') if e.strip()]
    audio_exts = [e.strip() for e in args.audio_extensions.split(',') if e.strip()]

    pairing = find_pairs(transcripts_dir, audio_dir, text_exts=text_exts, audio_exts=audio_exts)
    if not pairing.pairs and not pairing.unmatched_text:
        raise InputFolderError(
            f'No transcripts ({", ".join(text_exts)}) found in {transcripts_dir}')

    config_snapshot = {
        'no_parse': args.no_parse, 'no_freq': args.no_freq, 'no_edu_freq': args.no_edu_freq,
        'no_sem_domain': args.no_sem_domain, 'no_align': args.no_align,
        'sem_dict': args.sem_dict, 'sem_categories': args.sem_categories,
        'predictability': getattr(args, 'predictability', None),
        'edu_freq_lookup': args.edu_freq_lookup,
        'edu_bands': getattr(args, 'edu_bands', None),
    }

    logger = RunLogger(
        output_folder=output_dir,
        analyzer=args.analyzer,
        config=config_snapshot,
        input_transcripts_folder=transcripts_dir,
        input_audio_folder=audio_dir,
        text_extensions=text_exts,
        audio_extensions=audio_exts,
        filename_prefix='word_run',
        run_ts=getattr(args, 'run_ts', None),
        skip_no_match=getattr(args, 'skip_no_match', False),
    )

    for warning in pairing.warnings:
        logger.add_warning(warning)
    _warn_pairing_gaps(logger, args, pairing)
    # Audio-less transcripts are now processed as text-only items (see
    # _batch_work_items), so they are no longer recorded as unmatched_text.
    # Only audio with no transcript stays unprocessable here.
    for unmatched in pairing.unmatched_audio:
        logger.record_unmatched_audio(unmatched)

    work_items = _batch_work_items(pairing)
    args._stage_timer = _new_stage_timer()
    _prealign_batch(args, work_items, output_dir, logger)
    for i, pair in enumerate(work_items):
        if should_cancel and should_cancel():
            logger.add_warning('Run cancelled by user before processing complete')
            break
        start = time.time()
        _start_pair_timings(args)
        try:
            args.transcript = str(pair.transcript_path)
            args.audio = str(pair.audio_path) if pair.audio_path else ""
            args.doc_id = i + 1
            df = run(args)

            # Sub-word sidecars (Task 7) — emit BEFORE the per-column drop.
            # The drop step below removes columns the user toggled off for the
            # features CSV (e.g. doc_id, sentence_id, token_in_sentence_id, Onset, End when
            # the matching feat_struct_*/feat_acoustic_* keys are False). But
            # syllabify_pairs requires exactly those columns on word_df, so
            # dropping them first silently produces empty syll/phon DataFrames.
            phones_df = getattr(args, '_phones_df', None)
            _want_psr = (not getattr(args, 'no_phoneme_surprisal', True)
                         or not getattr(args, 'no_phoneme_surprisal_mat', True))
            if phones_df is not None and len(phones_df) > 0 and (
                not getattr(args, 'no_syllables', False)
                or not getattr(args, 'no_phonemes', False)
                or _want_psr
            ):
                from sensi.extractors.syllabify import syllabify_pairs
                from sensi.extractors.syllable_prosody import add_syllable_prosody

                with _stage(args, 'syllabify',
                            lambda: _transcript_words(pair.transcript_path)):
                    syll_df, phon_df = syllabify_pairs(df, phones_df)

                if _want_psr and len(phon_df) > 0:
                    from sensi.extractors.phoneme_surprisal import (
                        add_phoneme_surprisal, get_cohort_trie, phoneme_surprisal_mat_vars)
                    _pcfg = getattr(args, 'phoneme_surprisal', {}) or {}
                    _strip_stress = _pcfg.get('strip_stress', True)
                    _primary_only = _pcfg.get('primary_only', True)
                    with _stage(args, 'phoneme_surprisal',
                                lambda: _audio_duration(pair.audio_path)):
                        phon_df = add_phoneme_surprisal(
                            phon_df,
                            strip_stress=_strip_stress,
                            primary_only=_primary_only,
                        )
                    if not getattr(args, 'no_phoneme_surprisal_mat', True) and pair.audio_path:
                        import scipy.io
                        sr = float(_pcfg.get('analysis_sr', 128))
                        # add_phoneme_surprisal() already built (and memoized) the
                        # trie for this exact (dict_path, strip_stress, primary_only)
                        # combo, so this is an O(1) cache hit, not a recompute.
                        try:
                            _cap = float(get_cohort_trie(
                                strip_stress=_strip_stress, primary_only=_primary_only).cap)
                        except Exception:
                            # Defensive fallback only: NaN, not None, because
                            # scipy.io.savemat cannot serialize a bare None
                            # inside a nested struct field.
                            _cap = float('nan')
                        _duration_s = _audio_duration(pair.audio_path)
                        mat_vars = phoneme_surprisal_mat_vars(
                            phon_df, _duration_s, sr,
                            {'analysis_sr': sr,
                             'strip_stress': _strip_stress,
                             'primary_only': _primary_only,
                             'context_independent': True,
                             'first_phoneme_empty_context': True,
                             'cap': _cap, 'n_phonemes': int(len(phon_df)),
                             'duration_s': _duration_s,
                             'lexicon': 'english_us_arpa'})
                        mat_path = (_level_dir(output_dir, 'phoneme')
                                    / f'{pair.stimulus_id}_phoneme_surprisal.mat')
                        scipy.io.savemat(str(mat_path), mat_vars)

                if not getattr(args, 'no_syllables', False) and len(syll_df) > 0:
                    syll_df = add_syllable_prosody(
                        syll_df,
                        getattr(args, '_contours', None),
                        pitch_units=getattr(args, 'pitch_units', 'semitones'),
                    )
                    if len(phon_df) > 0:
                        from sensi.extractors.pcenter import add_pcenter
                        syll_df = add_pcenter(
                            syll_df, phon_df,
                            k=getattr(args, 'pcenter_k', 0.3),
                        )
                    drop = _syllable_columns_to_drop(
                        getattr(args, 'feature_toggles', {}) or {})
                    if drop:
                        syll_df = syll_df.drop(columns=[c for c in drop if c in syll_df.columns])
                    syll_path = _level_dir(output_dir, 'syllable') / f'{pair.stimulus_id}_syllables.csv'
                    syll_df.to_csv(syll_path, index=False)

                if not getattr(args, 'no_phonemes', False) and len(phon_df) > 0:
                    drop = _phoneme_columns_to_drop(
                        getattr(args, 'feature_toggles', {}) or {})
                    if drop:
                        phon_df = phon_df.drop(columns=[c for c in drop if c in phon_df.columns])
                    phon_path = _level_dir(output_dir, 'phoneme') / f'{pair.stimulus_id}_phonemes.csv'
                    phon_df.to_csv(phon_path, index=False)

            # Sentence-level sidecar — emits sentence/ subfolder CSV.
            if not getattr(args, 'no_sentences', False):
                try:
                    from sensi.extractors.sentence_prosody import add_sentence_prosody
                    with _stage(args, 'sentence_prosody',
                                lambda: _transcript_words(pair.transcript_path)):
                        sent_df = add_sentence_prosody(
                            df,
                            getattr(args, '_contours', None),
                            pitch_units=getattr(args, 'pitch_units', 'semitones'),
                        )
                    if len(sent_df) > 0:
                        drop = _sentence_columns_to_drop(
                            getattr(args, 'feature_toggles', {}) or {})
                        if drop:
                            sent_df = sent_df.drop(
                                columns=[c for c in drop if c in sent_df.columns])
                        sent_path = (_level_dir(output_dir, 'sentence')
                                     / f'{pair.stimulus_id}_sentences.csv')
                        sent_df.to_csv(sent_path, index=False)
                except Exception as exc:
                    print(f'[sentence] failed for {pair.stimulus_id}: {exc}',
                          file=sys.stderr)
                    logger.add_warning(
                        f'{pair.stimulus_id}: sentence CSV not written ({exc})')

            # Word2Vec embedding (.mat)
            # getattr default True → skip unless emit.py explicitly sets False.
            if not getattr(args, 'no_word2vec', True):
                # Import OUTSIDE the try: `except Word2VecUnavailable` must not
                # evaluate an unbound name if this import itself fails.
                from sensi.extractors.word2vec_embed import embed_word2vec, Word2VecUnavailable
                try:
                    import scipy.io
                    from sensi.extractors import word2vec_embed as _w2v
                    _preload_timed(args, 'word2vec', _w2v._MODEL_CACHE,
                                   str(Path(_w2v._DEFAULT_MODEL_PATH)),
                                   lambda: _w2v._load_model(_w2v._DEFAULT_MODEL_PATH))
                    with _stage(args, 'word2vec',
                                lambda: _transcript_words(pair.transcript_path)):
                        w2v_mat, w2v_valid = embed_word2vec(df)
                    mat_path = _level_dir(output_dir, 'word') / f'{pair.stimulus_id}_word2vec.mat'
                    scipy.io.savemat(str(mat_path), _word_embedding_vars(w2v_mat, df, w2v_valid))
                except Word2VecUnavailable as exc:
                    print(f'[word2vec] skipped for {pair.stimulus_id}: {exc}',
                          file=sys.stderr)
                except Exception as exc:
                    print(f'[word2vec] failed for {pair.stimulus_id}: {exc}',
                          file=sys.stderr)

            # BERT embeddings (.mat) — shared forward pass for word + sentence.
            # getattr default True → skip unless emit.py explicitly sets False.
            run_bert_word = not getattr(args, 'no_bert_word', True)
            run_bert_sent = not getattr(args, 'no_bert_sentence', True)
            if run_bert_word or run_bert_sent:
                try:
                    import scipy.io
                    from sensi.extractors.bert_embed import embed_bert_words_and_sentences
                    from sensi.extractors import bert_embed as _bert
                    _preload_timed(args, 'bert', _bert._BERT_CACHE,
                                   _bert._DEFAULT_MODEL_NAME,
                                   lambda: _bert._load_bert(_bert._DEFAULT_MODEL_NAME))
                    with _stage(args, 'bert',
                                lambda: _transcript_words(pair.transcript_path)):
                        word_emb, word_valid, sent_emb, sent_valid = \
                            embed_bert_words_and_sentences(df)
                    if run_bert_word:
                        mat_path = (_level_dir(output_dir, 'word')
                                    / f'{pair.stimulus_id}_bert_word.mat')
                        scipy.io.savemat(str(mat_path),
                                         _word_embedding_vars(word_emb, df, word_valid))
                    if run_bert_sent:
                        mat_path = (_level_dir(output_dir, 'sentence')
                                    / f'{pair.stimulus_id}_bert_sentence.mat')
                        scipy.io.savemat(str(mat_path),
                                         _sentence_embedding_vars(sent_emb, df, sent_valid))
                except Exception as exc:
                    print(f'[bert] failed for {pair.stimulus_id}: {exc}',
                          file=sys.stderr)

            # Drop per-column toggled-off output columns (schema v4 word-level)
            # — only affects the features CSV write below; sidecars already ran.
            drop_cols = _word_columns_to_drop(
                getattr(args, 'feature_toggles', {}) or {})
            if drop_cols:
                df = df.drop(columns=[c for c in drop_cols if c in df.columns])

            # Gate the features.csv write on the no_word_csv flag so a
            # sidecars-only run doesn't emit the word-level CSV.
            if not getattr(args, 'no_word_csv', False):
                features_path = _level_dir(output_dir, 'word') / f'{pair.stimulus_id}_features.csv'
                df.to_csv(features_path, index=False)
            else:
                features_path = None

            logger.record_pair_success(
                stimulus_id=pair.stimulus_id,
                transcript_path=pair.transcript_path,
                audio_path=pair.audio_path or "",
                runtime_seconds=time.time() - start,
                output_csv_path=features_path,
                output_summary_path=None,
            )
            _record_pair_timings(args)
        except Exception as exc:
            import traceback
            tb = ''.join(traceback.format_exception_only(type(exc), exc)).strip()
            logger.record_pair_error(
                stimulus_id=pair.stimulus_id,
                transcript_path=pair.transcript_path,
                audio_path=pair.audio_path or "",
                runtime_seconds=time.time() - start,
                error_message=tb,
            )
        if on_pair_done:
            on_pair_done(i + 1, len(work_items), pair.stimulus_id)
    logger.finalize()

    # Write per-run column -> unit sidecar.
    # Rebuilt from current prefs; if you ever mutate prefs mid-batch, capture
    # this at batch start instead.
    try:
        from sensi.gui.planner import plan_from_prefs
        from sensi.utils.preferences import load_preferences
        from sensi.utils.session_units import build_session_units
        _plan = plan_from_prefs(load_preferences())
        _rows = build_session_units(_plan)
        if _rows:
            units_path = logger.logs_dir / f'word_session_units_{logger.run_ts}.csv'
            pd.DataFrame(_rows).to_csv(units_path, index=False)
    except Exception:
        pass  # non-fatal; don't break existing batch output


def run_doc_level_batch(args, *, on_pair_done=None, should_cancel=None) -> None:
    """Process all matched pairs and emit a master documents_summary CSV."""
    from sensi.utils.doc_summary import build_doc_summary

    transcripts_dir = Path(args.transcripts_folder)
    output_dir = Path(args.output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)
    # audio-folder optional (text-only doc batch) — see run_word_level_batch.
    audio_dir = (Path(args.audio_folder) if args.audio_folder
                 else output_dir / '__no_audio__')

    if not args.analyzer:
        raise SystemExit('An analyzer name is required (run.yaml: shared.analyzer_name, or the Inputs page).')
    _check_input_folders(transcripts_dir, args.audio_folder)

    text_exts = [e.strip() for e in args.text_extensions.split(',') if e.strip()]
    audio_exts = [e.strip() for e in args.audio_extensions.split(',') if e.strip()]

    pairing = find_pairs(transcripts_dir, audio_dir, text_exts=text_exts, audio_exts=audio_exts)
    if not pairing.pairs and not pairing.unmatched_text:
        raise InputFolderError(
            f'No transcripts ({", ".join(text_exts)}) found in {transcripts_dir}')

    doc_toggles = {
        'counts':                not args.no_doc_counts,
        'mean_zipf_freq':        not args.no_doc_mean_zipf,
        'mean_edu_zipf':         not args.no_doc_mean_edu_zipf,
        'predictability_means':  not getattr(args, 'no_doc_predictability', False),
        'ttr':                   not args.no_doc_ttr,
        'mtld':                  not args.no_doc_mtld,
        'flesch_reading_ease':   not args.no_doc_flesch_ease,
        'flesch_kincaid_grade':  not args.no_doc_fk_grade,
        'category_proportions':  not args.no_doc_category_proportions,
        'psych_authentic':       not args.no_doc_psych_authentic,
        'psych_analytic':        not args.no_doc_psych_analytic,
        'psych_tone':            not args.no_doc_psych_tone,
        'psych_clout':           not args.no_doc_psych_clout,
        'lexical_norms_agg':     not getattr(args, 'no_doc_lexical_norms', True),
        # Prosody — single master gate. compute_doc_prosody returns all 16
        # cols when this is on; per-column dropping is handled at CSV write
        # time via _doc_columns_to_drop / args.feature_toggles.
        'doc_prosody':           not getattr(args, 'no_doc_prosody', False),
    }

    config_snapshot = {'level': 'document', 'doc_toggles': doc_toggles,
                       'sem_dict': args.sem_dict,
                       'predictability': getattr(args, 'predictability', None)}

    logger = RunLogger(
        output_folder=output_dir,
        analyzer=args.analyzer,
        config=config_snapshot,
        input_transcripts_folder=transcripts_dir,
        input_audio_folder=audio_dir,
        text_extensions=text_exts,
        audio_extensions=audio_exts,
        filename_prefix='doc_run',
        run_ts=getattr(args, 'run_ts', None),
        skip_no_match=getattr(args, 'skip_no_match', False),
    )

    for warning in pairing.warnings:
        logger.add_warning(warning)
    _warn_pairing_gaps(logger, args, pairing)
    # Audio-less transcripts are processed as text-only items (see
    # _batch_work_items); only audio with no transcript stays unprocessable.
    for unmatched in pairing.unmatched_audio:
        logger.record_unmatched_audio(unmatched)

    summary_rows: list[dict] = []
    master_path = _level_dir(output_dir, 'document') / f'documents_summary_{logger.run_ts}.csv'

    # Build an extraction args copy that forces on any word-level extractor whose
    # output is needed by an enabled doc-level toggle.  Word-level toggles only
    # control what appears in the *word-level CSV* output; doc-level aggregation
    # must be independent of those flags.
    import copy as _copy
    extract_args = _copy.copy(args)
    if doc_toggles.get('mean_zipf_freq'):
        extract_args.no_freq = False
    if doc_toggles.get('mean_edu_zipf'):
        extract_args.no_edu_freq = False
    # Predictability (v18): unlike mean_zipf_freq/mean_edu_zipf above, there
    # is no no_surprisal=False-style flag to force on — predictability is a
    # models x metrics roster carried verbatim on extract_args.predictability
    # (copied from args above: either ns.predictability set by the GUI or,
    # for direct run() callers, resolved from config/features.yaml inside run()). When
    # doc_toggles['predictability_means'] is on, build_doc_summary scans
    # whatever '<slug>_<Metric>' columns that roster actually produced and
    # means them (sensi/utils/doc_summary.py's predictability_means branch) — if
    # the roster is empty, there is simply nothing to aggregate, same as any
    # other doc-agg toggle whose source columns are absent.
    if doc_toggles.get('lexical_norms_agg'):
        # Doc aggregates read the word-level norm columns; force the
        # word-level extractor on regardless of the word-CSV toggle.
        extract_args.no_lexical_norms = False
    if doc_toggles.get('category_proportions') or any(
        doc_toggles.get(k) for k in ('psych_authentic', 'psych_analytic',
                                      'psych_tone', 'psych_clout')
    ):
        extract_args.no_sem_domain = False
    # If doc-prosody is on, ensure word-level prosody runs — it populates
    # Pitch_Delta / Volume_Delta which compute_doc_prosody reads, and it
    # populates args._contours which compute_doc_prosody also needs.
    # Prosody requires MFA Onset/End; the run() prosody block silently
    # no-ops when alignment didn't run, so we don't force no_align here
    # (an explicit no_align should win, and the GUI path
    # separately ensures alignment is on when doc-prosody is wanted —
    # see sensi/gui/run_dispatch.py).
    if doc_toggles.get('doc_prosody'):
        extract_args.no_prosody = False

    work_items = _batch_work_items(pairing)
    extract_args._stage_timer = _new_stage_timer()
    _prealign_batch(extract_args, work_items, output_dir, logger)
    for i, pair in enumerate(work_items):
        if should_cancel and should_cancel():
            logger.add_warning('Run cancelled by user before processing complete')
            break
        start = time.time()
        _start_pair_timings(extract_args)
        try:
            extract_args.transcript = str(pair.transcript_path)
            extract_args.audio = str(pair.audio_path) if pair.audio_path else ""
            extract_args.doc_id = i + 1

            df = run(extract_args)
            dict_label = 'LIWC' if any(c.startswith('Sem_LIWC_') for c in df.columns) else 'Empath'
            with open(pair.transcript_path, 'r', encoding='utf-8') as f:
                text = f.read()
            liwc_path = None
            if dict_label == 'LIWC':
                _, liwc_path = _resolve_sem_backend(args.sem_dict)
            # stimulus_id is the transcript stem (original casing), the same id
            # every per-level output filename uses, so the document table joins
            # to them directly; the full filename stays in transcript_path.
            with _stage(extract_args, 'doc_summary', lambda: count_words(text)):
                summary = build_doc_summary(
                    df, stimulus_id=pair.stimulus_id, dictionary_used=dict_label,
                    text=text, toggles=doc_toggles, liwc_dictionary_path=liwc_path,
                    contours=getattr(extract_args, '_contours', None),
                    pause_threshold_s=getattr(extract_args, 'pause_threshold', 0.25),
                    pitch_units=getattr(extract_args, 'pitch_units', 'semitones'),
                    late_acquired_threshold=getattr(args, 'late_acquired_threshold', 10.0),
                    phones_df=getattr(extract_args, '_phones_df', None),
                )
            summary['transcript_path'] = str(pair.transcript_path)
            summary['audio_path'] = str(pair.audio_path) if pair.audio_path else ""
            summary_rows.append(summary)

            logger.record_pair_success(
                stimulus_id=pair.stimulus_id,
                transcript_path=pair.transcript_path,
                audio_path=pair.audio_path or "",
                runtime_seconds=time.time() - start,
                output_csv_path=master_path,
                output_summary_path=None,
            )
            _record_pair_timings(extract_args)
        except Exception as exc:
            import traceback
            tb = ''.join(traceback.format_exception_only(type(exc), exc)).strip()
            logger.record_pair_error(
                stimulus_id=pair.stimulus_id,
                transcript_path=pair.transcript_path,
                audio_path=pair.audio_path or "",
                runtime_seconds=time.time() - start,
                error_message=tb,
            )
        if on_pair_done:
            on_pair_done(i + 1, len(work_items), pair.stimulus_id)

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        # Drop per-column toggled-off doc-level columns (schema v4).
        # Note: the v4 OUTPUT TOGGLE dict lives at args.output_toggles (named so
        # when the retired CLI used args.output for its --output path).
        drop_cols = _doc_columns_to_drop(
            getattr(args, 'output_toggles', {}) or {},
            getattr(args, 'feature_toggles', {}) or {},
        )
        if drop_cols:
            summary_df = summary_df.drop(
                columns=[c for c in drop_cols if c in summary_df.columns]
            )
        summary_df.to_csv(master_path, index=False)
    logger.finalize()

    # Write per-run column -> unit sidecar.
    # Rebuilt from current prefs; if you ever mutate prefs mid-batch, capture
    # this at batch start instead.
    try:
        from sensi.gui.planner import plan_from_prefs
        from sensi.utils.preferences import load_preferences
        from sensi.utils.session_units import build_session_units
        _plan = plan_from_prefs(load_preferences())
        _rows = build_session_units(_plan)
        if _rows:
            units_path = logger.logs_dir / f'doc_session_units_{logger.run_ts}.csv'
            pd.DataFrame(_rows).to_csv(units_path, index=False)
    except Exception:
        pass  # non-fatal; don't break existing batch output


def run_acoustic_level_batch(args, *, on_pair_done=None, should_cancel=None) -> None:
    """Process every audio file in args.audio_folder independently of any
    transcript, writing acoustic/<stim>_acoustic.mat per file."""
    import scipy.io
    from sensi.extractors.acoustic_ts import extract_acoustic

    audio_dir = Path(args.audio_folder or '')
    if not args.audio_folder or not audio_dir.is_dir():
        raise InputFolderError(f'Audio folder does not exist: {audio_dir}')
    output_dir = Path(args.output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_exts = [e.strip().lower() for e in args.audio_extensions.split(',') if e.strip()]
    audio_files = sorted(
        p for p in audio_dir.rglob('*')
        if p.is_file() and p.suffix.lower().lstrip('.') in audio_exts
    )

    logger = RunLogger(
        output_folder=output_dir,
        analyzer=getattr(args, 'analyzer', '') or 'unknown',
        config={'level': 'acoustic', 'acoustic_ts': args.acoustic_ts},
        input_transcripts_folder=audio_dir,   # no transcripts; record audio dir
        input_audio_folder=audio_dir,
        text_extensions=[],
        audio_extensions=audio_exts,
        filename_prefix='acoustic_run',
        run_ts=getattr(args, 'run_ts', None),
        skip_no_match=True,
    )

    rate = float(args.acoustic_ts.get('analysis_rate', 128))
    args._stage_timer = _new_stage_timer()
    for i, audio_path in enumerate(audio_files):
        if should_cancel and should_cancel():
            logger.add_warning('Run cancelled by user before processing complete')
            break
        start = time.time()
        _start_pair_timings(args)
        try:
            with _stage(args, 'acoustic_ts', lambda: _audio_duration(audio_path)):
                out = extract_acoustic(str(audio_path), rate, args.acoustic_ts)
            # Emit the 1-D time series as (N, 1) column vectors so every feature
            # is time-down-rows and stacks directly with logmel (N, n_mels) /
            # mfcc (N, n_mfcc) as a TRF predictor matrix in MATLAB. (The in-memory
            # extractor API stays 1-D; this only shapes the saved .mat.)
            for _k in ('t', 'envelope', 'onset_envelope'):
                if _k in out and getattr(out[_k], 'ndim', 0) == 1:
                    out[_k] = out[_k].reshape(-1, 1)
            mat_dir = _level_dir(output_dir, 'acoustic')
            mat_path = mat_dir / f'{audio_path.stem}_acoustic.mat'
            scipy.io.savemat(str(mat_path), out)   # savemat renders dict 'config' as a struct
            # Audio-only: no transcript. Pass the audio path for both the
            # transcript and audio slots; the .mat is the output artifact.
            logger.record_pair_success(
                stimulus_id=audio_path.stem,
                transcript_path=audio_path,
                audio_path=audio_path,
                runtime_seconds=time.time() - start,
                output_csv_path=mat_path,
                output_summary_path=None,
            )
            _record_pair_timings(args)
        except Exception as exc:   # noqa: BLE001 — record and continue batch
            import traceback
            tb = ''.join(traceback.format_exception_only(type(exc), exc)).strip()
            # Record as a pair error (consistent with the word/doc batches) so
            # a failed stimulus shows up in the manifest's per-pair results,
            # not only as free-text run warnings.
            logger.record_pair_error(
                stimulus_id=audio_path.stem,
                transcript_path=audio_path,
                audio_path=audio_path,
                runtime_seconds=time.time() - start,
                error_message=tb,
            )
        if on_pair_done:
            on_pair_done(i + 1, len(audio_files), audio_path.stem)

    logger.finalize()


def _resolve_sem_backend(sem_dict_arg: str) -> tuple[str, 'Path | None']:
    from sensi.extractors.semantic_domain import detect_backend
    if sem_dict_arg == 'auto':
        return detect_backend()
    if sem_dict_arg == 'empath':
        return 'empath', None
    if sem_dict_arg == 'liwc':
        backend, path = detect_backend()
        if backend != 'liwc':
            raise SystemExit('Semantic backend "liwc" selected but no .dic file found in data/dictionaries/')
        return 'liwc', path
    p = Path(sem_dict_arg)
    if p.suffix.lower() == '.dic' and p.exists():
        return 'liwc', p
    raise SystemExit(f'Semantic backend: unrecognized value "{sem_dict_arg}"')
