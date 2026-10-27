"""Maximum Onset Principle syllabification from MFA phoneme labels.

Algorithm and rationale: see dev/superpowers/specs/2026-05-13-phoneme-syllable-design.md §5.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import cast

import pandas as pd

from sensi.extractors.arpabet_ipa import syllable_to_ipa, to_ipa


@dataclass(frozen=True)
class Phone:
    """One phoneme with its MFA-aligned interval."""
    label: str   # ARPABET, e.g. 'B', 'EH1', 'NG' (vowels include stress digit 0/1/2)
    onset: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.onset

    @property
    def is_vowel(self) -> bool:
        return bool(self.label) and self.label[-1].isdigit()


@dataclass(frozen=True)
class Syllable:
    """A group of phonemes forming one syllable. Onset/end inherited from
    the first/last phoneme by construction (MFA's tiers are time-aligned)."""
    phonemes: tuple[Phone, ...] = field(default_factory=tuple)

    def __post_init__(self):
        # Allow callers to pass a list; freeze as tuple.
        object.__setattr__(self, 'phonemes', tuple(self.phonemes))

    @property
    def onset(self) -> float:
        return self.phonemes[0].onset if self.phonemes else float('nan')

    @property
    def end(self) -> float:
        return self.phonemes[-1].end if self.phonemes else float('nan')

    @property
    def duration(self) -> float:
        return self.end - self.onset if self.phonemes else float('nan')

    @property
    def nucleus(self) -> Phone | None:
        for p in self.phonemes:
            if p.is_vowel:
                return p
        return None


def syllabify_word(phonemes: list[Phone]) -> list[Syllable]:
    """Split a word's phoneme sequence into syllables via Maximum Onset Principle.

    Vowels (phonemes whose ARPABET label ends in a stress digit 0/1/2) are
    nuclei. Between two nuclei, all interludial consonants go to the
    FOLLOWING syllable's onset. Pre-nuclear consonants go to the first
    syllable; post-nuclear consonants go to the last syllable.

    No-vowel input (e.g., interjections like 'hmm') yields a single
    syllable containing all phonemes.
    """
    if not phonemes:
        return []

    nucleus_indices = [i for i, p in enumerate(phonemes) if p.is_vowel]

    if not nucleus_indices:
        return [Syllable(phonemes=tuple(phonemes))]

    cut_after = nucleus_indices[:-1]  # cut just after each non-final nucleus
    starts = [0, *(c + 1 for c in cut_after)]
    ends = [*(c + 1 for c in cut_after), len(phonemes)]
    return [Syllable(phonemes=tuple(phonemes[s:e])) for s, e in zip(starts, ends)]


# --- Label assignment (orthographic with pyphen, ARPABET fallback) -----------

try:
    import pyphen
    _pyphen_dict = pyphen.Pyphen(lang='en_US')
    _pyphen_unavailable = False
except Exception:
    _pyphen_dict = None
    _pyphen_unavailable = True


def _strip_stress(label: str) -> str:
    """Remove trailing stress digit from an ARPABET label, e.g. 'EH1' -> 'EH'."""
    return label[:-1] if label and label[-1].isdigit() else label


def _arpabet_for_syllable(syl: Syllable, *, stress: bool) -> str:
    """Dash-joined ARPABET phonemes; optionally with stress digits stripped."""
    return '-'.join(
        p.label if stress else _strip_stress(p.label) for p in syl.phonemes
    )


@dataclass(frozen=True)
class LabelResult:
    """Output of assign_labels: per-syllable Label + ARPABET strings,
    plus a mismatch flag the caller can use to count diagnostic events.

    Fields are tuples so the frozen contract is actually enforced —
    `frozen=True` blocks reassignment but not list mutation, and these
    results may be cached by callers building DataFrame rows.
    """
    labels: tuple[str, ...]
    arpabet: tuple[str, ...]
    mismatch: bool


def assign_labels(word: str, syllables: list[Syllable]) -> LabelResult:
    """Assign the orthographic `Label` and verbose `ARPABET` strings per syllable.

    See spec §5.4 for the fallback policy:
      - pyphen count == phoneme-syllable count → orthographic Label per syllable
      - any other case (count mismatch, or pyphen missing) → ARPABET-no-stress
    The ARPABET column is always populated regardless of pyphen status.
    """
    arpabet = tuple(_arpabet_for_syllable(s, stress=True) for s in syllables)

    if _pyphen_dict is None or not word:
        return LabelResult(
            labels=tuple(_arpabet_for_syllable(s, stress=False) for s in syllables),
            arpabet=arpabet,
            mismatch=True,
        )

    fragments = _pyphen_dict.inserted(word).split('-')
    if len(fragments) == len(syllables):
        return LabelResult(labels=tuple(fragments), arpabet=arpabet, mismatch=False)

    # Mismatch fallback
    return LabelResult(
        labels=tuple(_arpabet_for_syllable(s, stress=False) for s in syllables),
        arpabet=arpabet,
        mismatch=True,
    )


# --- Stimulus-level orchestrator ---------------------------------------------

# Empty-result schemas. When a stimulus produces zero rows we still need
# DataFrames with the correct columns so downstream consumers
# (add_syllable_prosody, the CSV writer in run_word_level_batch) don't
# trip on missing-column errors. Single source of truth for column order.
_SYLL_COLUMNS = [
    'doc_id', 'sentence_id', 'token_in_sentence_id', 'token', 'syllable_id',
    'Label', 'ARPABET', 'IPA', 'Onset', 'End', 'Duration',
]
_PHON_COLUMNS = [
    'doc_id', 'sentence_id', 'token_in_sentence_id', 'token', 'syllable_id',
    'phoneme_id', 'ARPABET', 'Stress', 'IPA', 'Pron_Source',
    'Onset', 'End', 'Duration',
]

_SPN = "spn"
_LEXICON_CACHE: set[str] | None = None
_LEXICON_EMPTY_WARNED = False


def _lexicon_words() -> set[str]:
    """Memoized set of words present in the MFA pronunciation dictionary.

    Used only to label Pron_Source: MFA does not mark G2P-derived words in the
    TextGrid, so a words-tier form absent from the dictionary is one whose
    pronunciation was generated (or absent entirely). Reuses the same .dict
    that sensi/extractors/phoneme_surprisal.py parses. Returns an empty set if the
    dictionary is unavailable, in which case every word is reported as
    'dictionary' rather than mislabelled as 'g2p'.
    """
    global _LEXICON_CACHE, _LEXICON_EMPTY_WARNED
    if _LEXICON_CACHE is None:
        from sensi.extractors.phoneme_surprisal import DEFAULT_DICT_PATH
        words: set[str] = set()
        try:
            with open(DEFAULT_DICT_PATH, "r", encoding="utf-8") as fh:
                for line in fh:
                    word = line.split("\t", 1)[0].strip()
                    if word and word[0] not in "[<":
                        words.add(word.lower())
        except OSError:
            words = set()
        _LEXICON_CACHE = words
        if not words and not _LEXICON_EMPTY_WARNED:
            # Otherwise plausible-and-wrong with no signal: _pron_source
            # falls back to 'dictionary' for every word (see its docstring)
            # when the lexicon is empty, including genuinely G2P-derived
            # ones. Warn once so a missing/unreadable dictionary doesn't
            # silently masquerade as "everything looked up cleanly".
            _LEXICON_EMPTY_WARNED = True
            print(f"[phonemes] WARNING: MFA dictionary unreadable at "
                  f"{DEFAULT_DICT_PATH} — Pron_Source will report "
                  "'dictionary' for every word (including genuinely "
                  "G2P-derived ones), since there is no lexicon to check "
                  "against.", file=sys.stderr)
    return _LEXICON_CACHE


def _split_label(label: str) -> tuple[str, int | None]:
    """'EY2' -> ('EY', 2); 'D' -> ('D', None)."""
    text = str(label)
    if text and text[-1].isdigit():
        return text[:-1], int(text[-1])
    return text, None


def _pron_source(mfa_form, phones: list[Phone]) -> str:
    """dictionary | g2p | oov_unresolved for one word's phone sequence.

    `mfa_form` is the MFA WORDS-TIER form (the word_df `Transcript` value), not
    the spaCy token. This matters: spaCy keeps digits ('48') while the aligner
    receives them spelled out, so `Transcript` is 'forty eight'. Checking the
    spaCy token would find '48' absent from the dictionary and mislabel a
    perfectly ordinary dictionary pronunciation as 'g2p' — verified to hit every
    numeric token in the real stimuli ('48', '2', '6', '8', '10', '3').

    A Transcript may hold several space-separated words (the align_merge 'expand'
    case); the pronunciation is dictionary-derived only if EVERY part is in the
    lexicon.
    """
    if any(p.label == _SPN for p in phones):
        return "oov_unresolved"
    lexicon = _lexicon_words()
    if not lexicon:
        # No dictionary available: report the neutral value rather than
        # mislabelling every word as g2p.
        return "dictionary"
    parts = [p for p in str(mfa_form or "").split() if p]
    if not parts:
        return "dictionary"
    if all(p.lower() in lexicon for p in parts):
        return "dictionary"
    return "g2p"


def _phones_for_word(phone_df: pd.DataFrame, word_onset: float, word_end: float
                     ) -> list[Phone]:
    """Select phoneme rows whose interval falls inside [word_onset, word_end].

    The 1e-6 tolerance guards against float-representation rounding in
    MFA's TextGrid output (MFA stamps to millisecond precision; 1 µs is
    far below any real boundary).
    """
    mask = (phone_df['onset'] >= word_onset - 1e-6) & (phone_df['offset'] <= word_end + 1e-6)
    sub = phone_df.loc[mask]
    return [
        Phone(label=str(row.phone), onset=float(row.onset), end=float(row.offset))
        for row in sub.itertuples(index=False)
    ]


def syllabify_pairs(word_df: pd.DataFrame, phone_df: pd.DataFrame
                    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the syllables and phonemes DataFrames for one stimulus.

    word_df must have columns: doc_id, sentence_id, token_in_sentence_id, token, Onset, End.
    phone_df must have columns: phone, onset, offset (output of parse_phones).

    Returns (syllables_df, phonemes_df) following the column inventory in
    spec §4. Words that have no phones inside their [Onset, End] interval are
    reported to stderr rather than silently dropped. Tokens that share an
    exact [Onset, End] interval with an earlier token (e.g. spaCy's "was" +
    "n't" both mapped onto MFA's single "wasn't" interval) are emitted once,
    for the first token only, so their phonemes are not duplicated.
    """
    syll_rows: list[dict] = []
    phon_rows: list[dict] = []
    seen_intervals: set[tuple] = set()
    missing: list[str] = []

    for w in word_df.itertuples(index=False):
        if pd.isna(getattr(w, 'Onset', None)) or pd.isna(getattr(w, 'End', None)):
            continue
        interval = (w.doc_id,
                    round(cast(float, w.Onset), 6),
                    round(cast(float, w.End), 6))
        if interval in seen_intervals:
            # A preceding token already claimed this exact MFA word (the
            # align_merge 'join' case: spaCy "was" + "n't" vs MFA "wasn't").
            # Emitting again would duplicate real acoustic phonemes.
            continue
        word_phones = _phones_for_word(phone_df, cast(float, w.Onset), cast(float, w.End))
        if not word_phones:
            missing.append(str(w.token))
            continue
        seen_intervals.add(interval)

        # Transcript is the MFA words-tier form; fall back to the token only
        # when a caller passes a frame without that column (see _pron_source).
        source = _pron_source(
            getattr(w, 'Transcript', None) or w.token, word_phones)
        syllables = syllabify_word(word_phones)
        labels = assign_labels(cast(str, w.token), syllables)

        for syll_idx, (syl, label_str, arpabet_str) in enumerate(
                zip(syllables, labels.labels, labels.arpabet), start=1):
            split = [_split_label(p.label) for p in syl.phonemes]
            syll_rows.append({
                'doc_id': w.doc_id, 'sentence_id': w.sentence_id,
                'token_in_sentence_id': w.token_in_sentence_id, 'token': w.token,
                'syllable_id': syll_idx,
                'Label': label_str, 'ARPABET': arpabet_str,
                'IPA': syllable_to_ipa(split),
                'Onset': syl.onset, 'End': syl.end,
                'Duration': round(syl.duration, 4),
            })
            for phon_idx, (phon, (code, stress)) in enumerate(
                    zip(syl.phonemes, split), start=1):
                phon_rows.append({
                    'doc_id': w.doc_id, 'sentence_id': w.sentence_id,
                    'token_in_sentence_id': w.token_in_sentence_id, 'token': w.token,
                    'syllable_id': syll_idx, 'phoneme_id': phon_idx,
                    'ARPABET': code,
                    'Stress': stress,
                    'IPA': to_ipa(code, stress),
                    'Pron_Source': source,
                    'Onset': phon.onset, 'End': phon.end,
                    'Duration': round(phon.duration, 4),
                })

    if missing:
        print(f'[phonemes] WARNING: {len(missing)} words had no phonemes '
              f'(likely OOV): {", ".join(missing)}', file=sys.stderr)

    syll_df = pd.DataFrame(syll_rows) if syll_rows else pd.DataFrame(columns=_SYLL_COLUMNS)
    phon_df_out = pd.DataFrame(phon_rows) if phon_rows else pd.DataFrame(columns=_PHON_COLUMNS)
    # Nullable Int64: empty on consonants (stress does not apply) as opposed to
    # 0, which means "unstressed vowel". A plain int64 column cannot hold both.
    phon_df_out['Stress'] = phon_df_out['Stress'].astype('Int64')
    return syll_df, phon_df_out
