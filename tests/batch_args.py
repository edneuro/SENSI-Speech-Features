"""Test scaffolding: argument namespaces for driving the batch runners directly.

This is the argument parser of the retired `main.py` command line
(removed 2026-10-27; the app runs through gui.py or run.py, which build
the namespace with the planner). Tests keep it because it yields a
namespace with every feature ON unless a flag switches it off -- the
shape these batch-runner tests were written against. It is not a
user-facing interface.
"""
import argparse


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

    # New batch 1 feature flags
    # EWFG is opt-in: its source (Zeno et al. 1995) is print-only / not
    # digitized, so it is off by default. --no-edu-freq is kept (now redundant
    # with the default) for backward compatibility.
    p.add_argument('--edu-freq', dest='no_edu_freq', action='store_false',
                   help='Enable EWFG grade-band frequencies (off by default; '
                        'requires a user-supplied data/dictionaries/edu_zipf.parquet).')
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
                        'from user-supplied parquets in data/dictionaries/.')
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
                   help='Persist current --sem-categories selection to config/features.yaml.')

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
    from sensi.extractors.acoustic_ts import MTRF_COMP
    p.add_argument('--env-comp', type=float, default=MTRF_COMP,
                   help='Envelope power-law compression exponent '
                        '(default: log10(2) ~ 0.30103, exactly matching '
                        'mTRFenvelope.m — a rounded 0.3 breaks the '
                        'bit-for-bit mTRF replication).')
    p.add_argument('--env-window', type=float, default=1.0,
                   help='Envelope averaging window (mTRF window; >1 overlaps frames for smoothing).')
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
    p.add_argument('--no-doc-predictability',        action='store_true',
                   help='Skip mean_<slug>_<metric> doc-level predictability aggregates.')
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

    # Alignment options
    p.add_argument('--g2p-oov', action='store_true',
                   help='Use the MFA G2P model to generate pronunciations for '
                        'out-of-dictionary words instead of aligning them to a '
                        'single spn phone. Off by default: the pronunciation is '
                        'modeled, not observed.')

    # Per-column sub-word / sentence choices live in config/features.yaml
    # (feat_syllable_* / feat_phoneme_* / feat_sentence_*), read via the
    # planner; the CLI only carries the master skips above and below.
    p.add_argument('--pcenter-k', type=float, default=0.3,
                   help='Scott 1993 P-center coefficient (default 0.3).')

    # Sentence-level sidecar master toggle
    p.add_argument('--no-sentences', action='store_true',
                   help='Skip <stim>_sentences.csv emission.')

    # Note: embedding .mat output (word2vec, bert_word, bert_sentence) is
    # controlled by ns.no_word2vec / ns.no_bert_word / ns.no_bert_sentence
    # set by sensi/gui/planner/emit.py. CLI does not expose these flags because the
    # models require large downloads; use the GUI to enable embeddings.

    return p
