"""
SensiExtractor — stimulus feature extraction pipeline.

Usage:
    python main.py --audio path/to/audio.wav --transcript path/to/text.txt
    python main.py --audio audio.wav --transcript text.txt --output features.csv
    python main.py --audio audio.wav --transcript text.txt \\
                   --no-align --no-freq --no-surprisal

Feature flags (all enabled by default):
    --no-align       Skip MFA forced alignment (Onset / End columns)
    --no-parse       Skip spaCy linguistic annotation
    --no-freq        Skip word frequency (Zipf_Freq)
    --no-surprisal   Skip GPT-2 surprisal

Output:
    One CSV file with one row per word token. Punctuation tokens are
    included (they carry spaCy annotations) but have NaN for numeric
    features (Zipf_Freq, Surprisal, Onset, End).

Column order in output CSV:
    --- spaCy structural ---
    doc_id, paragraph_id, sentence_id, sentence, start, end,
    term_id, token_id
    --- token-level linguistic (spaCy) ---
    token, lemma, upos, xpos, feats, dep_rel, head_token_id
    --- computed features ---
    Zipf_Freq, Surprisal
    --- MFA alignment ---
    Transcript, Onset, End

Merge strategy (alignment + parse):
    After alignment, Transcript words are merged into the token DataFrame
    on a normalized form of the token. Punctuation and numbers (which MFA
    spells out) are handled via the prepared transcript. Words MFA could
    not align will have NaN in Onset/End.
"""

import argparse
import os
import sys
from pathlib import Path

# MKL on AMD CPUs (Windows/Ryzen) raises a fatal exception (0xc06d007f) during
# BLAS matmul — notably librosa's mel-filterbank einsum in the acoustic level —
# under the default Intel threading layer. Force sequential threading BEFORE
# numpy imports and initialises MKL. Mirrors tests/conftest.py so real CLI/GUI
# runs behave like the (passing) test environment instead of hard-crashing.
os.environ.setdefault("MKL_THREADING_LAYER", "sequential")

import numpy as np
import pandas as pd

# Redirect all model/cache downloads into the project folder
_HERE = Path(__file__).parent.resolve()
os.environ.setdefault('HF_HOME',      str(_HERE / 'hf_cache'))   # GPT-2 weights
os.environ.setdefault('MFA_ROOT_DIR', str(_HERE / 'mfa_data'))   # MFA models
os.environ.setdefault('HF_HUB_DISABLE_IMPLICIT_TOKEN', '1')      # suppress unauthenticated warning


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='Extract word-level stimulus features from audio + transcript.'
    )
    p.add_argument('--audio', required=False,
                   help='Path to .wav audio file. Required unless --no-align.')
    p.add_argument('--transcript', required=False, default='',
                   help='Path to plain-text transcript file.')
    p.add_argument('--output', default='features.csv',
                   help='Output CSV path (default: features.csv).')
    p.add_argument('--doc-id', type=int, default=1,
                   help='1-indexed document identifier written to the doc_id '
                        'column (default: 1). Batch mode overrides this with '
                        'the document\'s position in the run.')

    p.add_argument('--no-align',     action='store_true', help='Skip MFA alignment.')
    p.add_argument('--no-parse',     action='store_true', help='Skip spaCy annotation.')
    p.add_argument('--no-freq',      action='store_true', help='Skip word frequency.')
    p.add_argument('--no-surprisal', action='store_true', help='Skip GPT-2 surprisal.')

    # New batch 1 feature flags
    # EWFG is opt-in: its source (Zeno et al. 1995) is print-only / not
    # digitized, so it is off by default. --no-edu-freq is kept (now redundant
    # with the default) for backward compatibility.
    p.add_argument('--edu-freq', dest='no_edu_freq', action='store_false',
                   help='Enable EWFG grade-band frequencies (off by default; '
                        'requires a user-supplied dictionaries/edu_zipf.parquet).')
    p.add_argument('--no-edu-freq', dest='no_edu_freq', action='store_true',
                   help='(default) Skip EWFG grade-band frequencies.')
    p.set_defaults(no_edu_freq=True)
    p.add_argument('--edu-freq-lookup', default=None, help='Override path to EWFG lookup parquet.')
    p.add_argument('--edu-bands',       default=None,
                   help='Comma-separated EWFG bands to emit: k2,3to5,6to8,9to12,overall. '
                        'Default (omit flag) = all five bands.')

    # Lexical norms are opt-in (no data ships; user supplies parquet files).
    p.add_argument('--lexical-norms', dest='no_lexical_norms', action='store_false',
                   default=True,
                   help='Compute lexical-norm columns (AoA / concreteness / '
                        'imageability / familiarity / valence-arousal-dominance) '
                        'from user-supplied parquets in dictionaries/.')
    p.add_argument('--doc-lexical-norms', dest='no_doc_lexical_norms',
                   action='store_false', default=True,
                   help='Emit document-level lexical-norm aggregates (means, '
                        'AoA spread, prop_late_acquired, coverage).')
    p.add_argument('--late-acquired-threshold', type=float, default=10.0,
                   help='AoA (years) above which a word counts as late-acquired '
                        'for prop_late_acquired (default 10).')

    p.add_argument('--no-sem-domain',   action='store_true', help='Skip semantic-domain columns and doc summary.')
    p.add_argument('--sem-dict',        default='auto',
                   help='Semantic dictionary backend: liwc | empath | auto | <path-to-.dic>.')
    p.add_argument('--sem-categories',  default=None,
                   help='Comma-separated category names. Default = SOP 15-category set.')
    p.add_argument('--save-prefs',      action='store_true',
                   help='Persist current --sem-categories selection to preferences.json.')

    p.add_argument('--no-llm-cloze',    action='store_true', help='Skip LLM-cloze metrics.')
    p.add_argument('--llm-model',       default=None, help='Override LLM model id (auto-detected by default).')

    # Prosody (Task 5) — group-level CLI flags only.
    # Per-column granularity is GUI-only and flows through args.feature_toggles
    # (matching the existing pattern for feat_acoustic_* / feat_psych_*).
    p.add_argument('--no-prosody',     action='store_true',
                   help='Skip all 10 word-level prosody columns.')
    p.add_argument('--no-doc-prosody', action='store_true',
                   help='Skip all 16 doc-level prosody + pause columns.')
    # Configuration
    p.add_argument('--pitch-units', choices=['hz', 'semitones'], default='semitones')
    p.add_argument('--pause-threshold', type=float, default=0.25)
    p.add_argument('--pitch-floor', type=float, default=75.0)
    p.add_argument('--pitch-ceiling', type=float, default=500.0)

    # Batch-mode flags
    p.add_argument('--transcripts-folder', default=None, help='Batch mode: transcripts folder.')
    p.add_argument('--audio-folder',       default=None, help='Batch mode: audio folder.')
    p.add_argument('--output-folder',      default=None, help='Batch mode: output folder.')
    p.add_argument('--analyzer',           default=None, help='Required for batch mode; recorded in manifest.')
    p.add_argument('--text-extensions',    default='txt,tsv,md', help='Comma-separated text extensions.')
    p.add_argument('--audio-extensions',   default='wav,flac,mp3', help='Comma-separated audio extensions.')

    p.add_argument('--level', choices=['word', 'document', 'acoustic'], default='word',
                   help='Batch mode: which level to run. Default: word.')

    # Acoustic-level (time-series) flags — audio-only, no transcript needed.
    p.add_argument('--analysis-rate', type=float, default=128.0,
                   help='Acoustic level: output sample rate in Hz (match your EEG).')
    p.add_argument('--env-method', choices=['rms', 'hilbert'], default='rms')
    p.add_argument('--env-comp', type=float, default=0.3,
                   help='Envelope power-law compression exponent.')
    p.add_argument('--n-mels', type=int, default=16)
    p.add_argument('--n-mfcc', type=int, default=13)
    p.add_argument('--mel-preset', choices=['speech', 'music', 'general'], default='speech')
    p.add_argument('--no-envelope',  action='store_true')
    p.add_argument('--no-onset-env', action='store_true')
    p.add_argument('--no-logmel',    action='store_true')
    p.add_argument('--mfcc', action='store_true', help='Acoustic level: also emit MFCCs (off by default).')

    # Doc-level feature toggles (default = on)
    p.add_argument('--no-doc-counts',                action='store_true')
    p.add_argument('--no-doc-mean-zipf',             action='store_true')
    p.add_argument('--no-doc-mean-edu-zipf',         action='store_true')
    p.add_argument('--no-doc-mean-surprisal',        action='store_true')
    p.add_argument('--no-doc-mean-llm-cloze',        action='store_true')
    p.add_argument('--no-doc-ttr',                   action='store_true')
    p.add_argument('--no-doc-mtld',                  action='store_true')
    p.add_argument('--no-doc-flesch-ease',           action='store_true')
    p.add_argument('--no-doc-fk-grade',              action='store_true')
    p.add_argument('--no-doc-category-proportions',  action='store_true')
    p.add_argument('--no-doc-psych-authentic',       action='store_true')
    p.add_argument('--no-doc-psych-analytic',        action='store_true')
    p.add_argument('--no-doc-psych-tone',            action='store_true')
    p.add_argument('--no-doc-psych-clout',           action='store_true')

    # Sub-word sidecar master toggles
    p.add_argument('--no-syllables', action='store_true',
                   help='Skip <stim>_syllables.csv emission.')
    p.add_argument('--no-phonemes',  action='store_true',
                   help='Skip <stim>_phonemes.csv emission.')

    # Sub-word per-column toggles (default = on)
    p.add_argument('--no-syl-onset',         action='store_true')
    p.add_argument('--no-syl-end',           action='store_true')
    p.add_argument('--no-syl-duration',      action='store_true')
    p.add_argument('--no-syl-label',         action='store_true')
    p.add_argument('--no-syl-arpabet',       action='store_true')
    p.add_argument('--no-syl-pitch-mean',    action='store_true')
    p.add_argument('--no-syl-volume-mean',   action='store_true')
    p.add_argument('--no-syl-pitch-delta',   action='store_true')
    p.add_argument('--no-syl-volume-delta',  action='store_true')
    p.add_argument('--no-syl-pcenter', action='store_true',
                   help='Skip P_Center column in syllables CSV.')
    p.add_argument('--pcenter-k', type=float, default=0.3,
                   help='Scott 1993 P-center coefficient (default 0.3).')

    p.add_argument('--no-phon-label',    action='store_true')
    p.add_argument('--no-phon-onset',    action='store_true')
    p.add_argument('--no-phon-end',      action='store_true')
    p.add_argument('--no-phon-duration', action='store_true')

    # Sentence-level sidecar master toggle
    p.add_argument('--no-sentences', action='store_true',
                   help='Skip <stim>_sentences.csv emission.')

    # Note: embedding .mat output (word2vec, bert_word, bert_sentence) is
    # controlled by ns.no_word2vec / ns.no_bert_word / ns.no_bert_sentence
    # set by gui/planner/emit.py. CLI does not expose these flags because the
    # models require large downloads; use the GUI to enable embeddings.

    # Sentence per-column toggles
    p.add_argument('--no-sent-text',          action='store_true')
    p.add_argument('--no-sent-onset',         action='store_true')
    p.add_argument('--no-sent-end',           action='store_true')
    p.add_argument('--no-sent-pitch-mean',    action='store_true')
    p.add_argument('--no-sent-pitch-min',     action='store_true')
    p.add_argument('--no-sent-pitch-max',     action='store_true')
    p.add_argument('--no-sent-pitch-std',     action='store_true')
    p.add_argument('--no-sent-pitch-delta',   action='store_true')
    p.add_argument('--no-sent-volume-mean',   action='store_true')
    p.add_argument('--no-sent-volume-min',    action='store_true')
    p.add_argument('--no-sent-volume-max',    action='store_true')
    p.add_argument('--no-sent-volume-std',    action='store_true')
    p.add_argument('--no-sent-volume-delta',  action='store_true')
    p.add_argument('--no-sent-hnr',           action='store_true')

    return p


# ---------------------------------------------------------------------------
# Merge helpers
# ---------------------------------------------------------------------------

def _normalize(word: str) -> str:
    """Lowercase, strip punctuation from edges for fuzzy merge."""
    import re
    return re.sub(r"[^\w'-]", '', word.lower())


def _merge_alignment(token_df: pd.DataFrame, align_df: pd.DataFrame) -> pd.DataFrame:
    """Left-join alignment (Onset/End/Transcript) onto token_df.

    Alignment is word-level and excludes punctuation; token_df includes
    punctuation. We match on normalized token form. Unmatched tokens
    (punctuation, OOV for MFA) get NaN Onset/End.
    """
    # Only try to merge non-punctuation tokens
    token_df = token_df.copy()
    token_df['_norm'] = token_df['token'].apply(_normalize)

    align_df = align_df.copy()
    align_df['_norm'] = align_df['Transcript'].apply(_normalize)

    # align_df may have repeated words; track which rows have been consumed
    used = [False] * len(align_df)
    onsets, ends, transcripts = [], [], []

    for _, row in token_df.iterrows():
        if row['upos'] == 'PUNCT':
            onsets.append(None)
            ends.append(None)
            transcripts.append(None)
            continue

        norm = row['_norm']
        matched = False
        for i, arow in align_df.iterrows():
            if used[i]:
                continue
            if arow['_norm'] == norm:
                onsets.append(arow['Onset'])
                ends.append(arow['End'])
                transcripts.append(arow['Transcript'])
                used[i] = True
                matched = True
                break

        if not matched:
            onsets.append(None)
            ends.append(None)
            transcripts.append(None)

    token_df['Transcript'] = transcripts
    token_df['Onset'] = onsets
    token_df['End'] = ends
    token_df.drop(columns=['_norm'], inplace=True)
    return token_df


# ---------------------------------------------------------------------------
# Column ordering
# ---------------------------------------------------------------------------

_COLUMN_ORDER = [
    'doc_id', 'paragraph_id', 'sentence_id', 'sentence', 'start', 'end',
    'term_id', 'token_id',
    'token', 'lemma', 'upos', 'xpos', 'feats', 'dep_rel', 'head_token_id',
    'Zipf_Freq',
    'Edu_Zipf_K2', 'Edu_Zipf_3to5', 'Edu_Zipf_6to8', 'Edu_Zipf_9to12', 'Edu_Zipf_Overall',
    # Sem_<Dict>_<cat> columns get appended dynamically by _reorder
    'Sem_Categories',
    'Surprisal',
    'LLM_Surprisal', 'LLM_Cloze', 'LLM_TopMatch',
    'Transcript', 'Onset', 'End',
    'Word_Duration', 'Word_Gap',
    # Prosody (Task 5)
    'Pitch_Mean', 'Pitch_Min', 'Pitch_Max', 'Pitch_Std',
    'Volume_Mean', 'Volume_Min', 'Volume_Max', 'Volume_Std',
    'Pitch_Delta', 'Volume_Delta',
]


def _reorder(df: pd.DataFrame) -> pd.DataFrame:
    sem_cols = sorted([c for c in df.columns if c.startswith('Sem_') and c != 'Sem_Categories'])
    canonical: list[str] = []
    for c in _COLUMN_ORDER:
        if c == 'Sem_Categories':
            canonical.extend(sem_cols)
            canonical.append('Sem_Categories')
        else:
            canonical.append(c)
    present = [c for c in canonical if c in df.columns]
    extra = [c for c in df.columns if c not in present]
    return df[present + extra]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args) -> pd.DataFrame:
    from utils.preferences import load_preferences

    prefs = load_preferences()
    if args.sem_categories:
        sem_categories = args.sem_categories.split(',')
    else:
        # v7 prefs schema unifies categories across levels into a single
        # `categories_<backend>` key. This fallback is reached for CLI
        # runs; batch callers (GUI dispatch) always pass --sem-categories
        # explicitly via ns.sem_categories.
        backend_hint = prefs.get('semantic', {}).get('backend', 'auto')
        suffix = 'liwc' if backend_hint == 'liwc' else 'empath'
        sem_categories = prefs.get('semantic', {}).get(
            f'categories_{suffix}', [])

    with open(args.transcript, 'r', encoding='utf-8') as f:
        text = f.read()

    df = None

    if not args.no_parse:
        from extractors.text_parser import parse as spacy_parse
        print('[spacy] Running spaCy annotation…')
        df = spacy_parse(text, doc_id=int(getattr(args, 'doc_id', 1) or 1))

    if not args.no_freq and df is not None:
        from extractors.word_freq import add_word_freq
        print('[freq] Computing SUBTLEX Zipf…')
        df = add_word_freq(df)

    if not args.no_edu_freq and df is not None:
        from extractors.educator_freq import add_edu_freq
        print('[edu] Computing EWFG grade-band frequencies…')
        edu_bands = (
            [b.strip() for b in args.edu_bands.split(',') if b.strip()]
            if getattr(args, 'edu_bands', None)
            else None
        )
        df = add_edu_freq(df, lookup_path=args.edu_freq_lookup, bands=edu_bands)

    if not getattr(args, 'no_lexical_norms', True) and df is not None:
        from extractors.lexical_norms import add_lexical_norms
        print('[norms] Computing lexical norms (AoA / concreteness / imageability '
              '/ familiarity / valence-arousal-dominance)…')
        df = add_lexical_norms(df, paths=getattr(args, 'lexical_norms_paths', None))

    if not args.no_sem_domain and df is not None:
        from extractors.semantic_domain import add_semantic_domain
        backend, dict_path = _resolve_sem_backend(args.sem_dict)
        print(f'[sem] Computing semantic-domain ({backend})…')
        df = add_semantic_domain(df, backend=backend, categories=sem_categories,
                                 dictionary_path=dict_path)

    if not args.no_surprisal and df is not None:
        from extractors.surprisal import add_surprisal
        print('[gpt2] Computing GPT-2 surprisal (legacy)…')
        df = add_surprisal(df, text)

    if not args.no_llm_cloze and df is not None:
        from extractors.llm_cloze import add_llm_cloze, select_model
        model_id = args.llm_model or select_model()
        print(f'[llm] Computing LLM-cloze with {model_id}…')
        df = add_llm_cloze(df, text=text, model_id=model_id)

    args._phones_df = None  # reset each run; populated below when alignment runs
    if not args.no_align and args.audio:
        from extractors.aligner import align_with_phones as mfa_align
        print('[mfa] Running MFA alignment…')
        align_df, phones_df = mfa_align(args.audio, text)
        args._phones_df = phones_df
        if df is not None:
            df = _merge_alignment(df, align_df)
        else:
            df = align_df

    # Prosody (Task 6) — needs Onset/End from alignment, so it runs here.
    args._contours = None  # initialized on every run; consumed by doc-level batch
    if (not getattr(args, 'no_prosody', False) and df is not None
            and 'Onset' in df.columns and args.audio):
        try:
            from extractors.audio_features import extract as extract_contours
            from extractors.prosody import add_word_prosody
            print('[prosody] Extracting pitch/intensity contours…')
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
        from extractors.timing import add_timing_columns
        df = add_timing_columns(df)
        df = _reorder(df)

    if df is None:
        # Don't sys.exit here — batch mode calls run() inside a per-pair
        # try/except that only catches Exception. SystemExit would bypass
        # it, kill the worker thread, and leave the GUI's Run button stuck
        # in "running" state (done_cb never fires). Raise a regular
        # RuntimeError instead so the batch handler records it as a pair
        # error and the run completes cleanly; main() below converts it to
        # an exit-1 for CLI users.
        raise RuntimeError('All extractors are disabled — nothing to extract.')

    if args.save_prefs and args.sem_categories:
        from utils.preferences import save_preferences
        # v7 unified: a single categories_<backend> key per backend. Write to
        # both backends so the saved selection survives a later backend switch.
        # --save-prefs at CLI is rare; the GUI saves via
        # category_picker._on_toggle which already writes the right key.
        save_preferences({'semantic': {
            'categories_empath': sem_categories,
            'categories_liwc':   sem_categories,
        }})

    return df


import time

from utils.file_pairing import find_pairs
from utils.run_logger import RunLogger
from utils.doc_summary import build_doc_summary


# Per-column toggle → CSV column name mappings (schema v4).
# Used by run_word_level_batch and run_doc_level_batch to drop unselected
# columns before writing the CSV. The toggles are passed via args.features
# (and args.output for doc-level toggles) when invoked from the GUI; the CLI
# leaves them unset which causes all columns to be retained.

_WORD_FEATURE_COLUMN_MAP = {
    # Structural
    "feat_struct_doc_id": "doc_id",
    "feat_struct_paragraph_id": "paragraph_id",
    "feat_struct_sentence_id": "sentence_id",
    "feat_struct_sentence_text": "sentence",
    "feat_struct_char_start": "start",
    "feat_struct_char_end": "end",
    "feat_struct_term_id": "term_id",
    "feat_struct_token_id": "token_id",
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
    "feat_syllable_pitch_mean":   "Pitch_Mean",
    "feat_syllable_volume_mean":  "Volume_Mean",
    "feat_syllable_pitch_delta":  "Pitch_Delta",
    "feat_syllable_volume_delta": "Volume_Delta",
    "feat_syllable_pcenter":      "P_Center",
}

_PHONEME_COLUMN_MAP = {
    "feat_phoneme_label":    "Label",
    "feat_phoneme_onset":    "Onset",
    "feat_phoneme_end":      "End",
    "feat_phoneme_duration": "Duration",
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


def _word_columns_to_drop(features: dict) -> set:
    """Return the set of word-level CSV columns to drop based on v4 toggles."""
    return {col for key, col in _WORD_FEATURE_COLUMN_MAP.items()
            if not features.get(key, True)}


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
        "word_id":     df["term_id"].to_numpy(dtype=np.int64),
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


def run_word_level_batch(args, *, on_pair_done=None, should_cancel=None) -> None:
    """Process all matched (transcript, audio) pairs in the configured folders."""
    transcripts_dir = Path(args.transcripts_folder)
    audio_dir = Path(args.audio_folder)
    output_dir = Path(args.output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.analyzer:
        raise SystemExit('--analyzer is required for batch mode')

    text_exts = [e.strip() for e in args.text_extensions.split(',') if e.strip()]
    audio_exts = [e.strip() for e in args.audio_extensions.split(',') if e.strip()]

    pairing = find_pairs(transcripts_dir, audio_dir, text_exts=text_exts, audio_exts=audio_exts)

    config_snapshot = {
        'no_parse': args.no_parse, 'no_freq': args.no_freq, 'no_edu_freq': args.no_edu_freq,
        'no_sem_domain': args.no_sem_domain, 'no_surprisal': args.no_surprisal,
        'no_llm_cloze': args.no_llm_cloze, 'no_align': args.no_align,
        'sem_dict': args.sem_dict, 'sem_categories': args.sem_categories,
        'llm_model': args.llm_model,
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
    for unmatched in pairing.unmatched_text:
        logger.record_unmatched_text(unmatched)
    for unmatched in pairing.unmatched_audio:
        logger.record_unmatched_audio(unmatched)

    for i, pair in enumerate(pairing.pairs):
        if should_cancel and should_cancel():
            logger.add_warning('Run cancelled by user before processing complete')
            break
        start = time.time()
        try:
            args.transcript = str(pair.transcript_path)
            args.audio = str(pair.audio_path)
            args.doc_id = i + 1
            df = run(args)

            # Sub-word sidecars (Task 7) — emit BEFORE the per-column drop.
            # The drop step below removes columns the user toggled off for the
            # features CSV (e.g. doc_id, sentence_id, token_id, Onset, End when
            # the matching feat_struct_*/feat_acoustic_* keys are False). But
            # syllabify_pairs requires exactly those columns on word_df, so
            # dropping them first silently produces empty syll/phon DataFrames.
            phones_df = getattr(args, '_phones_df', None)
            if phones_df is not None and len(phones_df) > 0 and (
                not getattr(args, 'no_syllables', False)
                or not getattr(args, 'no_phonemes', False)
            ):
                from extractors.syllabify import syllabify_pairs
                from extractors.syllable_prosody import add_syllable_prosody

                syll_df, phon_df = syllabify_pairs(df, phones_df)

                if not getattr(args, 'no_syllables', False) and len(syll_df) > 0:
                    syll_df = add_syllable_prosody(
                        syll_df,
                        getattr(args, '_contours', None),
                        pitch_units=getattr(args, 'pitch_units', 'semitones'),
                    )
                    if len(phon_df) > 0:
                        from extractors.pcenter import add_pcenter
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
                    from extractors.sentence_prosody import add_sentence_prosody
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

            # Word2Vec embedding (.mat)
            # getattr default True → skip unless emit.py explicitly sets False.
            if not getattr(args, 'no_word2vec', True):
                try:
                    import scipy.io
                    from extractors.word2vec_embed import embed_word2vec, Word2VecUnavailable
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
                    from extractors.bert_embed import embed_bert_words_and_sentences
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
                audio_path=pair.audio_path,
                runtime_seconds=time.time() - start,
                output_csv_path=features_path,
                output_summary_path=None,
            )
        except Exception as exc:
            import traceback
            tb = ''.join(traceback.format_exception_only(type(exc), exc)).strip()
            logger.record_pair_error(
                stimulus_id=pair.stimulus_id,
                transcript_path=pair.transcript_path,
                audio_path=pair.audio_path,
                runtime_seconds=time.time() - start,
                error_message=tb,
            )
        if on_pair_done:
            on_pair_done(i + 1, len(pairing.pairs), pair.stimulus_id)
    logger.finalize()

    # Write per-run column -> unit sidecar.
    # Rebuilt from current prefs; if you ever mutate prefs mid-batch, capture
    # this at batch start instead.
    try:
        from gui.planner import plan_from_prefs
        from utils.preferences import load_preferences
        from utils.session_units import build_session_units
        _plan = plan_from_prefs(load_preferences())
        _rows = build_session_units(_plan)
        if _rows:
            units_path = logger.logs_dir / f'word_session_units_{logger.run_ts}.csv'
            pd.DataFrame(_rows).to_csv(units_path, index=False)
    except Exception:
        pass  # non-fatal; don't break existing batch output


def run_doc_level_batch(args, *, on_pair_done=None, should_cancel=None) -> None:
    """Process all matched pairs and emit a master documents_summary CSV."""
    from utils.doc_summary import build_doc_summary

    transcripts_dir = Path(args.transcripts_folder)
    audio_dir = Path(args.audio_folder)
    output_dir = Path(args.output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.analyzer:
        raise SystemExit('--analyzer is required for batch mode')

    text_exts = [e.strip() for e in args.text_extensions.split(',') if e.strip()]
    audio_exts = [e.strip() for e in args.audio_extensions.split(',') if e.strip()]

    pairing = find_pairs(transcripts_dir, audio_dir, text_exts=text_exts, audio_exts=audio_exts)

    doc_toggles = {
        'counts':                not args.no_doc_counts,
        'mean_zipf_freq':        not args.no_doc_mean_zipf,
        'mean_edu_zipf':         not args.no_doc_mean_edu_zipf,
        'mean_surprisal':        not args.no_doc_mean_surprisal,
        'mean_llm_cloze':        not args.no_doc_mean_llm_cloze,
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
                       'sem_dict': args.sem_dict, 'llm_model': args.llm_model}

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
    for unmatched in pairing.unmatched_text:
        logger.record_unmatched_text(unmatched)
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
    if doc_toggles.get('mean_surprisal'):
        extract_args.no_surprisal = False
    if doc_toggles.get('mean_llm_cloze'):
        extract_args.no_llm_cloze = False
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
    # no-ops when alignment didn't run, so we don't force --no-align here
    # (an explicit user --no-align should win, and the GUI path
    # separately ensures alignment is on when doc-prosody is wanted —
    # see gui/run_dispatch.py).
    if doc_toggles.get('doc_prosody'):
        extract_args.no_prosody = False

    for i, pair in enumerate(pairing.pairs):
        if should_cancel and should_cancel():
            logger.add_warning('Run cancelled by user before processing complete')
            break
        start = time.time()
        try:
            extract_args.transcript = str(pair.transcript_path)
            extract_args.audio = str(pair.audio_path)
            extract_args.doc_id = i + 1

            df = run(extract_args)
            dict_label = 'LIWC' if any(c.startswith('Sem_LIWC_') for c in df.columns) else 'Empath'
            with open(pair.transcript_path, 'r', encoding='utf-8') as f:
                text = f.read()
            liwc_path = None
            if dict_label == 'LIWC':
                _, liwc_path = _resolve_sem_backend(args.sem_dict)
            summary = build_doc_summary(
                df, stimulus_id=pair.stimulus_id, dictionary_used=dict_label,
                text=text, toggles=doc_toggles, liwc_dictionary_path=liwc_path,
                contours=getattr(extract_args, '_contours', None),
                pause_threshold_s=getattr(extract_args, 'pause_threshold', 0.25),
                pitch_units=getattr(extract_args, 'pitch_units', 'semitones'),
                late_acquired_threshold=getattr(args, 'late_acquired_threshold', 10.0),
            )
            summary['transcript_path'] = str(pair.transcript_path)
            summary['audio_path'] = str(pair.audio_path)
            summary_rows.append(summary)

            logger.record_pair_success(
                stimulus_id=pair.stimulus_id,
                transcript_path=pair.transcript_path,
                audio_path=pair.audio_path,
                runtime_seconds=time.time() - start,
                output_csv_path=master_path,
                output_summary_path=None,
            )
        except Exception as exc:
            import traceback
            tb = ''.join(traceback.format_exception_only(type(exc), exc)).strip()
            logger.record_pair_error(
                stimulus_id=pair.stimulus_id,
                transcript_path=pair.transcript_path,
                audio_path=pair.audio_path,
                runtime_seconds=time.time() - start,
                error_message=tb,
            )
        if on_pair_done:
            on_pair_done(i + 1, len(pairing.pairs), pair.stimulus_id)

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        # Drop per-column toggled-off doc-level columns (schema v4).
        # Note: args.output is the CLI --output PATH; the v4 OUTPUT TOGGLE
        # dict lives at args.output_toggles to avoid the name collision.
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
        from gui.planner import plan_from_prefs
        from utils.preferences import load_preferences
        from utils.session_units import build_session_units
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
    from extractors.acoustic_ts import extract_acoustic

    audio_dir = Path(args.audio_folder)
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
    for i, audio_path in enumerate(audio_files):
        if should_cancel and should_cancel():
            logger.add_warning('Run cancelled by user before processing complete')
            break
        start = time.time()
        try:
            out = extract_acoustic(str(audio_path), rate, args.acoustic_ts)
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
        except Exception as exc:   # noqa: BLE001 — record and continue batch
            logger.add_warning(f'{audio_path.name}: {exc}')
        if on_pair_done:
            on_pair_done(i + 1, len(audio_files), audio_path.stem)

    logger.finalize()


def _resolve_sem_backend(sem_dict_arg: str) -> tuple[str, 'Path | None']:
    from extractors.semantic_domain import detect_backend
    if sem_dict_arg == 'auto':
        return detect_backend()
    if sem_dict_arg == 'empath':
        return 'empath', None
    if sem_dict_arg == 'liwc':
        backend, path = detect_backend()
        if backend != 'liwc':
            raise SystemExit('--sem-dict liwc but no .dic file found in dictionaries/')
        return 'liwc', path
    p = Path(sem_dict_arg)
    if p.suffix.lower() == '.dic' and p.exists():
        return 'liwc', p
    raise SystemExit(f'--sem-dict: unrecognized value "{sem_dict_arg}"')


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.level == 'acoustic':
        if not (args.audio_folder and args.output_folder):
            raise SystemExit('Acoustic level requires --audio-folder and --output-folder')
        args.acoustic_ts = {
            'envelope': not args.no_envelope, 'onset_env': not args.no_onset_env,
            'logmel': not args.no_logmel, 'mfcc': args.mfcc,
            'env_method': args.env_method, 'env_comp': args.env_comp,
            'n_mels': args.n_mels, 'n_mfcc': args.n_mfcc,
            'mel_preset': args.mel_preset, 'analysis_rate': args.analysis_rate,
        }
        run_acoustic_level_batch(args)
        return

    is_batch = bool(args.transcripts_folder or args.audio_folder or args.output_folder)
    if is_batch:
        if not (args.transcripts_folder and args.audio_folder and args.output_folder):
            raise SystemExit(
                'Batch mode requires --transcripts-folder, --audio-folder, '
                'and --output-folder'
            )
        if args.level == 'document':
            run_doc_level_batch(args)
        else:
            run_word_level_batch(args)
        return

    if not args.no_align and args.audio is None:
        print('Warning: --audio not provided; alignment will be skipped.')
    try:
        df = run(args)
    except RuntimeError as exc:
        print(f'Error: {exc}', file=sys.stderr)
        sys.exit(1)
    df.to_csv(args.output, index=False)
    print(f'\nDone. {len(df)} rows written to {args.output}')


if __name__ == '__main__':
    main()
