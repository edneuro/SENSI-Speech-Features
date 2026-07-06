"""Maximum Onset Principle syllabification from MFA phoneme labels.

Algorithm and rationale: see docs/superpowers/specs/2026-05-13-phoneme-syllable-design.md §5.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


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

    if _pyphen_unavailable or not word:
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
    'Label', 'ARPABET', 'Onset', 'End', 'Duration',
]
_PHON_COLUMNS = [
    'doc_id', 'sentence_id', 'token_in_sentence_id', 'token', 'syllable_id',
    'phoneme_id', 'Label', 'Onset', 'End', 'Duration',
]


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
    spec §4. Words that have no phones inside their [Onset, End] interval
    produce no rows in either DataFrame.
    """
    syll_rows: list[dict] = []
    phon_rows: list[dict] = []

    for w in word_df.itertuples(index=False):
        if pd.isna(getattr(w, 'Onset', None)) or pd.isna(getattr(w, 'End', None)):
            continue
        word_phones = _phones_for_word(phone_df, w.Onset, w.End)
        if not word_phones:
            continue

        syllables = syllabify_word(word_phones)
        labels = assign_labels(w.token, syllables)

        for syll_idx, (syl, label_str, arpabet_str) in enumerate(
                zip(syllables, labels.labels, labels.arpabet), start=1):
            syll_rows.append({
                'doc_id': w.doc_id, 'sentence_id': w.sentence_id,
                'token_in_sentence_id': w.token_in_sentence_id, 'token': w.token,
                'syllable_id': syll_idx,
                'Label': label_str, 'ARPABET': arpabet_str,
                'Onset': syl.onset, 'End': syl.end,
                'Duration': round(syl.duration, 4),
            })
            for phon_idx, phon in enumerate(syl.phonemes, start=1):
                phon_rows.append({
                    'doc_id': w.doc_id, 'sentence_id': w.sentence_id,
                    'token_in_sentence_id': w.token_in_sentence_id, 'token': w.token,
                    'syllable_id': syll_idx, 'phoneme_id': phon_idx,
                    'Label': phon.label,
                    'Onset': phon.onset, 'End': phon.end,
                    'Duration': round(phon.duration, 4),
                })

    syll_df = pd.DataFrame(syll_rows) if syll_rows else pd.DataFrame(columns=_SYLL_COLUMNS)
    phon_df = pd.DataFrame(phon_rows) if phon_rows else pd.DataFrame(columns=_PHON_COLUMNS)
    return syll_df, phon_df
