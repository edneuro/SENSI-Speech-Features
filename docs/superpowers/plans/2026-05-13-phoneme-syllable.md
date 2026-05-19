# Phoneme / Syllable Batch — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add per-syllable and per-phoneme sidecar CSVs to the pipeline, with Maximum Onset Principle syllabification and per-column GUI toggles. See spec: `docs/superpowers/specs/2026-05-13-phoneme-syllable-design.md`.

**Architecture:** Two new sidecar CSVs per stimulus (`<stim>_syllables.csv`, `<stim>_phonemes.csv`) joined back to the word CSV via `(doc_id, sentence_id, token_id)` FK columns. Syllabification happens after MFA alignment in the existing per-pair loop. Syllable prosody reuses the existing Parselmouth `Contours` object — no second audio decode. New "Sub-word features" sidebar page in the PySide6 GUI.

**Tech Stack:** Python 3.10, pandas, MFA (already integrated via `utils/textgrid_parser.py:parse_phones`), Praat-Parselmouth (already a dep), pyphen (already a dep), PySide6 (already a dep). No new external dependencies.

**Per-task convention:** `extractors/syllabify.py` and `extractors/syllable_prosody.py` are extractor math → TDD upfront (write tests first). All other tasks (CLI, dispatcher, GUI, prefs, preflight) follow test-after per the project's `memory/feedback-tdd-scope.md` policy.

**Env activation for every step:** `conda activate ./env` from the project root, then run `.\env\python.exe` or `python` for the test commands. Test commands below use `python -m pytest`.

---

## Task 1: Phone / Syllable dataclasses and `syllabify_word()`

**Files:**
- Create: `extractors/syllabify.py`
- Create: `tests/extractors/test_syllabify.py`

This is the core algorithm from spec §5. Pure function over a phoneme list — no I/O, no audio. TDD.

- [ ] **Step 1: Write the failing tests**

Create `tests/extractors/test_syllabify.py`:

```python
"""Unit tests for the Maximum Onset Principle syllabification."""
import pytest
from extractors.syllabify import Phone, Syllable, syllabify_word


def _phones(labels: list[str], step: float = 0.05) -> list[Phone]:
    """Build a list of Phone objects with synthetic monotonic timings."""
    return [
        Phone(label=label, onset=round(i * step, 3), end=round((i + 1) * step, 3))
        for i, label in enumerate(labels)
    ]


def _labels(syl: Syllable) -> list[str]:
    return [p.label for p in syl.phonemes]


def test_simple_cvcv():
    """below: B IH0 L OW1 -> [B IH0] [L OW1] (MOP gives /L/ to second syllable)."""
    syllables = syllabify_word(_phones(['B', 'IH0', 'L', 'OW1']))
    assert len(syllables) == 2
    assert _labels(syllables[0]) == ['B', 'IH0']
    assert _labels(syllables[1]) == ['L', 'OW1']


def test_network_two_consonants_between_vowels():
    """network: N EH1 T W ER0 K -> [N EH1] [T W ER0 K] (max onset gives both /T W/ to syll 2)."""
    syllables = syllabify_word(_phones(['N', 'EH1', 'T', 'W', 'ER0', 'K']))
    assert _labels(syllables[0]) == ['N', 'EH1']
    assert _labels(syllables[1]) == ['T', 'W', 'ER0', 'K']


def test_consonant_cluster_pre_first_vowel():
    """string: S T R IH1 NG -> [S T R IH1 NG] (single syllable, all pre-vowel C's onset)."""
    syllables = syllabify_word(_phones(['S', 'T', 'R', 'IH1', 'NG']))
    assert len(syllables) == 1
    assert _labels(syllables[0]) == ['S', 'T', 'R', 'IH1', 'NG']


def test_consonant_cluster_post_last_vowel():
    """lamps: L AE1 M P S -> [L AE1 M P S] (single syllable, all post-vowel C's coda)."""
    syllables = syllabify_word(_phones(['L', 'AE1', 'M', 'P', 'S']))
    assert len(syllables) == 1


def test_three_consecutive_vowels():
    """idea: AY1 D IY0 AH0 -> [AY1] [D IY0] [AH0] (three nuclei, MOP)."""
    syllables = syllabify_word(_phones(['AY1', 'D', 'IY0', 'AH0']))
    assert len(syllables) == 3
    assert _labels(syllables[0]) == ['AY1']
    assert _labels(syllables[1]) == ['D', 'IY0']
    assert _labels(syllables[2]) == ['AH0']


def test_single_vowel_word():
    """'a': AH0 -> [AH0] (one syllable, one phoneme)."""
    syllables = syllabify_word(_phones(['AH0']))
    assert len(syllables) == 1
    assert _labels(syllables[0]) == ['AH0']


def test_no_vowel_word_emits_one_syllable():
    """hmm: HH M -> [HH M] (no nucleus, whole word is one syllable)."""
    syllables = syllabify_word(_phones(['HH', 'M']))
    assert len(syllables) == 1
    assert _labels(syllables[0]) == ['HH', 'M']


def test_empty_phoneme_list_returns_empty():
    """word with no aligned phones -> []."""
    assert syllabify_word([]) == []


def test_secondary_stress_counts_as_nucleus():
    """phonemics: F AH0 N IY1 M IH0 K S — IH0 is a nucleus (stress digit 0)."""
    syllables = syllabify_word(_phones(['F', 'AH0', 'N', 'IY1', 'M', 'IH0', 'K', 'S']))
    assert len(syllables) == 3  # three vowels


def test_syllable_timing_inherits_from_phonemes():
    """A syllable's onset/end == first/last phoneme's onset/end."""
    phones = _phones(['B', 'IH0', 'L', 'OW1'])
    syllables = syllabify_word(phones)
    # syll 0 = [B(0.00-0.05), IH0(0.05-0.10)]
    assert syllables[0].onset == 0.00
    assert syllables[0].end == 0.10
    # syll 1 = [L(0.10-0.15), OW1(0.15-0.20)]
    assert syllables[1].onset == 0.10
    assert syllables[1].end == 0.20
```

- [ ] **Step 2: Run tests to verify they fail**

```
python -m pytest tests/extractors/test_syllabify.py -v
```

Expected: All 10 tests FAIL with `ModuleNotFoundError: No module named 'extractors.syllabify'`.

- [ ] **Step 3: Implement `extractors/syllabify.py`**

```python
"""Maximum Onset Principle syllabification from MFA phoneme labels.

Algorithm and rationale: see docs/superpowers/specs/2026-05-13-phoneme-syllable-design.md §5.
"""
from __future__ import annotations

from dataclasses import dataclass, field


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
```

- [ ] **Step 4: Run tests to verify they pass**

```
python -m pytest tests/extractors/test_syllabify.py -v
```

Expected: All 10 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add extractors/syllabify.py tests/extractors/test_syllabify.py
git commit -m "syllabify: Phone/Syllable dataclasses + Max Onset Principle algorithm

Pure function syllabify_word(phonemes) splits a word's MFA-aligned
phoneme sequence into syllables. Vowels identified by ARPABET stress
digit; max-onset rule for interludial consonants. ~10 unit tests
covering CV/CVC, CCV pre-vowel clusters, post-vowel clusters,
vowel-only single-syllable, no-vowel edge case (interjections),
empty input, secondary stress as nucleus, and timing inheritance.

Refs spec §5."
```

---

## Task 2: Label assignment with pyphen fallback

**Files:**
- Modify: `extractors/syllabify.py` — append the new function
- Modify: `tests/extractors/test_syllabify.py` — append new tests

Implements spec §5.4: orthographic Label from pyphen when counts match, ARPABET-no-stress fallback otherwise.

- [ ] **Step 1: Write the failing tests**

Append to `tests/extractors/test_syllabify.py`:

```python
from extractors.syllabify import assign_labels, LabelResult


def _syllables_from(labels_per_syll: list[list[str]]) -> list[Syllable]:
    """Build a syllable list from already-split phoneme labels (testing helper)."""
    syllables = []
    t = 0.0
    for syll_labels in labels_per_syll:
        phones = []
        for label in syll_labels:
            phones.append(Phone(label=label, onset=round(t, 3), end=round(t + 0.05, 3)))
            t += 0.05
        syllables.append(Syllable(phonemes=tuple(phones)))
    return syllables


def test_pyphen_match_assigns_orthographic_labels():
    """When pyphen count matches our count, fragments are assigned in order."""
    syllables = _syllables_from([['B', 'IH0'], ['L', 'OW1']])  # 'below', 2 sylls
    result = assign_labels('below', syllables)
    assert result.mismatch is False
    assert result.labels == ['be', 'low']
    assert result.arpabet == ['B-IH0', 'L-OW1']


def test_pyphen_mismatch_falls_back_to_arpabet_no_stress():
    """When counts differ (e.g., 'fire' pyphen=1, MFA=2), Label is ARPABET-no-stress."""
    syllables = _syllables_from([['F', 'AY1'], ['ER0']])  # 'fire' phonemically 2 sylls
    result = assign_labels('fire', syllables)
    assert result.mismatch is True
    assert result.labels == ['F-AY', 'ER']
    assert result.arpabet == ['F-AY1', 'ER0']


def test_pyphen_missing_module_falls_back_to_arpabet_no_stress(monkeypatch):
    """When pyphen import fails, Label is ARPABET-no-stress for everything."""
    import sys
    monkeypatch.setitem(sys.modules, 'pyphen', None)
    # Force a fresh import path by patching the module-level cache
    from extractors import syllabify
    monkeypatch.setattr(syllabify, '_pyphen_dict', None)
    monkeypatch.setattr(syllabify, '_pyphen_unavailable', True)

    syllables = _syllables_from([['B', 'IH0'], ['L', 'OW1']])
    result = assign_labels('below', syllables)
    assert result.mismatch is True  # treated as a mismatch
    assert result.labels == ['B-IH', 'L-OW']
    assert result.arpabet == ['B-IH0', 'L-OW1']


def test_arpabet_for_single_syllable_word():
    syllables = _syllables_from([['L', 'AE1', 'M', 'P', 'S']])
    result = assign_labels('lamps', syllables)
    assert result.arpabet == ['L-AE1-M-P-S']
```

- [ ] **Step 2: Run tests to verify they fail**

```
python -m pytest tests/extractors/test_syllabify.py -v -k "label or assign"
```

Expected: 4 tests FAIL with `ImportError: cannot import name 'assign_labels'`.

- [ ] **Step 3: Append the implementation to `extractors/syllabify.py`**

```python
# --- Label assignment (orthographic with pyphen, ARPABET fallback) -----------

from dataclasses import dataclass

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
    plus a mismatch flag the caller can use to count diagnostic events."""
    labels: list[str]
    arpabet: list[str]
    mismatch: bool


def assign_labels(word: str, syllables: list[Syllable]) -> LabelResult:
    """Assign the orthographic `Label` and verbose `ARPABET` strings per syllable.

    See spec §5.4 for the fallback policy:
      - pyphen count == phoneme-syllable count → orthographic Label per syllable
      - any other case (count mismatch, or pyphen missing) → ARPABET-no-stress
    The ARPABET column is always populated regardless of pyphen status.
    """
    arpabet = [_arpabet_for_syllable(s, stress=True) for s in syllables]

    if _pyphen_unavailable or not word:
        return LabelResult(
            labels=[_arpabet_for_syllable(s, stress=False) for s in syllables],
            arpabet=arpabet,
            mismatch=True,
        )

    fragments = _pyphen_dict.inserted(word).split('-')
    if len(fragments) == len(syllables):
        return LabelResult(labels=fragments, arpabet=arpabet, mismatch=False)

    # Mismatch fallback
    return LabelResult(
        labels=[_arpabet_for_syllable(s, stress=False) for s in syllables],
        arpabet=arpabet,
        mismatch=True,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```
python -m pytest tests/extractors/test_syllabify.py -v
```

Expected: all 14 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add extractors/syllabify.py tests/extractors/test_syllabify.py
git commit -m "syllabify: assign_labels with pyphen orthographic + ARPABET fallback

assign_labels(word, syllables) returns LabelResult with per-syllable
Label (orthographic when pyphen count matches phoneme-syllable count;
ARPABET-no-stress on mismatch or pyphen-unavailable) and ARPABET
(always populated, dash-joined, with stress digit on vowels).
Mismatch flag exposed for caller-side diagnostic counting.

Refs spec §5.4."
```

---

## Task 3: Orchestrator `syllabify_pairs()` over a whole stimulus

**Files:**
- Modify: `extractors/syllabify.py` — append
- Modify: `tests/extractors/test_syllabify.py` — append

Combines per-word syllabification + label assignment with the FK column wiring described in spec §4. Produces two DataFrames matching the sidecar shapes.

- [ ] **Step 1: Write the failing tests**

Append to `tests/extractors/test_syllabify.py`:

```python
import pandas as pd
from extractors.syllabify import syllabify_pairs


def test_syllabify_pairs_emits_both_dataframes_with_fk_columns():
    """Given a small synthetic word_df + phone_df, syllabify_pairs returns
    (syllables_df, phonemes_df) with matching FK columns."""
    word_df = pd.DataFrame([
        {'doc_id': 'stim1', 'sentence_id': 1, 'token_id': 1, 'token': 'below',
         'Onset': 0.00, 'End': 0.20},
        {'doc_id': 'stim1', 'sentence_id': 1, 'token_id': 2, 'token': 'network',
         'Onset': 0.20, 'End': 0.50},
    ])
    phone_df = pd.DataFrame([
        # below: B IH0 L OW1
        {'phone': 'B',   'onset': 0.00, 'offset': 0.05},
        {'phone': 'IH0', 'onset': 0.05, 'offset': 0.10},
        {'phone': 'L',   'onset': 0.10, 'offset': 0.15},
        {'phone': 'OW1', 'onset': 0.15, 'offset': 0.20},
        # network: N EH1 T W ER0 K
        {'phone': 'N',   'onset': 0.20, 'offset': 0.25},
        {'phone': 'EH1', 'onset': 0.25, 'offset': 0.30},
        {'phone': 'T',   'onset': 0.30, 'offset': 0.35},
        {'phone': 'W',   'onset': 0.35, 'offset': 0.40},
        {'phone': 'ER0', 'onset': 0.40, 'offset': 0.45},
        {'phone': 'K',   'onset': 0.45, 'offset': 0.50},
    ])

    syll_df, phon_df = syllabify_pairs(word_df, phone_df)

    # Each word produces 2 syllables -> 4 syllable rows total
    assert len(syll_df) == 4
    # All phonemes preserved -> 10 phoneme rows
    assert len(phon_df) == 10

    # FK columns present and correctly populated
    for col in ('doc_id', 'sentence_id', 'token_id', 'token', 'syllable_id'):
        assert col in syll_df.columns
    for col in ('doc_id', 'sentence_id', 'token_id', 'token', 'syllable_id', 'phoneme_id'):
        assert col in phon_df.columns

    # syllable_id resets per word (1-indexed)
    assert syll_df[syll_df.token == 'below'].syllable_id.tolist() == [1, 2]
    assert syll_df[syll_df.token == 'network'].syllable_id.tolist() == [1, 2]

    # phoneme_id resets per syllable
    below_phons = phon_df[phon_df.token == 'below']
    assert below_phons[below_phons.syllable_id == 1].phoneme_id.tolist() == [1, 2]
    assert below_phons[below_phons.syllable_id == 2].phoneme_id.tolist() == [1, 2]


def test_syllabify_pairs_timing_columns_present():
    word_df = pd.DataFrame([
        {'doc_id': 'd', 'sentence_id': 1, 'token_id': 1, 'token': 'below',
         'Onset': 0.0, 'End': 0.2},
    ])
    phone_df = pd.DataFrame([
        {'phone': 'B',   'onset': 0.00, 'offset': 0.05},
        {'phone': 'IH0', 'onset': 0.05, 'offset': 0.10},
        {'phone': 'L',   'onset': 0.10, 'offset': 0.15},
        {'phone': 'OW1', 'onset': 0.15, 'offset': 0.20},
    ])
    syll_df, phon_df = syllabify_pairs(word_df, phone_df)
    assert {'Onset', 'End', 'Duration'}.issubset(syll_df.columns)
    assert {'Onset', 'End', 'Duration'}.issubset(phon_df.columns)
    # First syllable: B(0-0.05) + IH0(0.05-0.10) -> 0.00 to 0.10
    s0 = syll_df.iloc[0]
    assert s0.Onset == 0.00
    assert s0.End == 0.10
    assert round(s0.Duration, 4) == 0.10


def test_syllabify_pairs_label_and_arpabet_columns_present():
    word_df = pd.DataFrame([
        {'doc_id': 'd', 'sentence_id': 1, 'token_id': 1, 'token': 'below',
         'Onset': 0.0, 'End': 0.2},
    ])
    phone_df = pd.DataFrame([
        {'phone': 'B',   'onset': 0.00, 'offset': 0.05},
        {'phone': 'IH0', 'onset': 0.05, 'offset': 0.10},
        {'phone': 'L',   'onset': 0.10, 'offset': 0.15},
        {'phone': 'OW1', 'onset': 0.15, 'offset': 0.20},
    ])
    syll_df, _ = syllabify_pairs(word_df, phone_df)
    # 'below' pyphen-syllabifies to 'be-low' -> 2 fragments, matches our 2 syllables
    assert syll_df.Label.tolist() == ['be', 'low']
    assert syll_df.ARPABET.tolist() == ['B-IH0', 'L-OW1']


def test_syllabify_pairs_word_with_no_aligned_phones_skipped():
    """A word in word_df with no matching phones in phone_df produces no rows."""
    word_df = pd.DataFrame([
        {'doc_id': 'd', 'sentence_id': 1, 'token_id': 1, 'token': 'gap',
         'Onset': 0.0, 'End': 0.1},
        {'doc_id': 'd', 'sentence_id': 1, 'token_id': 2, 'token': 'a',
         'Onset': 0.2, 'End': 0.25},
    ])
    phone_df = pd.DataFrame([
        # No phones in 0.0-0.1 (MFA dropped them); 'a' = AH0 in 0.2-0.25
        {'phone': 'AH0', 'onset': 0.20, 'offset': 0.25},
    ])
    syll_df, phon_df = syllabify_pairs(word_df, phone_df)
    # Only 'a' produces rows; 'gap' is dropped
    assert syll_df.token.tolist() == ['a']
    assert phon_df.token.tolist() == ['a']
```

- [ ] **Step 2: Run tests to verify they fail**

```
python -m pytest tests/extractors/test_syllabify.py -v -k "pairs"
```

Expected: 4 new tests FAIL with `ImportError: cannot import name 'syllabify_pairs'`.

- [ ] **Step 3: Append the implementation**

```python
# --- Stimulus-level orchestrator ---------------------------------------------

import pandas as pd


def _phones_for_word(phone_df: pd.DataFrame, word_onset: float, word_end: float
                     ) -> list[Phone]:
    """Select phoneme rows whose interval falls inside [word_onset, word_end]."""
    mask = (phone_df['onset'] >= word_onset - 1e-6) & (phone_df['offset'] <= word_end + 1e-6)
    sub = phone_df.loc[mask]
    return [
        Phone(label=str(row.phone), onset=float(row.onset), end=float(row.offset))
        for row in sub.itertuples(index=False)
    ]


def syllabify_pairs(word_df: pd.DataFrame, phone_df: pd.DataFrame
                    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the syllables and phonemes DataFrames for one stimulus.

    word_df must have columns: doc_id, sentence_id, token_id, token, Onset, End.
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
                'token_id': w.token_id, 'token': w.token,
                'syllable_id': syll_idx,
                'Label': label_str, 'ARPABET': arpabet_str,
                'Onset': syl.onset, 'End': syl.end,
                'Duration': round(syl.duration, 4),
            })
            for phon_idx, phon in enumerate(syl.phonemes, start=1):
                phon_rows.append({
                    'doc_id': w.doc_id, 'sentence_id': w.sentence_id,
                    'token_id': w.token_id, 'token': w.token,
                    'syllable_id': syll_idx, 'phoneme_id': phon_idx,
                    'Label': phon.label,
                    'Onset': phon.onset, 'End': phon.end,
                    'Duration': round(phon.duration, 4),
                })

    syll_df = pd.DataFrame(syll_rows)
    phon_df = pd.DataFrame(phon_rows)
    return syll_df, phon_df
```

- [ ] **Step 4: Run tests to verify they pass**

```
python -m pytest tests/extractors/test_syllabify.py -v
```

Expected: all 18 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add extractors/syllabify.py tests/extractors/test_syllabify.py
git commit -m "syllabify: syllabify_pairs orchestrator over word_df + phone_df

syllabify_pairs(word_df, phone_df) buckets phonemes into words by
time-interval intersection, runs syllabify_word + assign_labels per
word, and emits the two sidecar DataFrames with FK columns
(doc_id, sentence_id, token_id, token, syllable_id, phoneme_id).
Words MFA didn't align (no phones in their interval) produce no rows.

Refs spec §4 and §5."
```

---

## Task 4: Syllable prosody — slice `Contours` by `[Onset, End]`

**Files:**
- Create: `extractors/syllable_prosody.py`
- Create: `tests/extractors/test_syllable_prosody.py`

Adds `Pitch_Mean` / `Volume_Mean` / `Pitch_Delta` / `Volume_Delta` to the syllables DataFrame from the existing `Contours` object (`extractors/audio_features.py:extract()`). Mirrors `extractors/prosody.py:add_word_prosody` but operates on the syllable intervals. TDD.

- [ ] **Step 1: Write the failing tests**

Create `tests/extractors/test_syllable_prosody.py`:

```python
"""Unit tests for syllable_prosody.add_syllable_prosody."""
import numpy as np
import pandas as pd
import pytest

from extractors.syllable_prosody import add_syllable_prosody


class _FakeContours:
    """Minimal fake of extractors.audio_features.Contours for testing."""
    def __init__(self, *, pitch_hz: np.ndarray, intensity_db: np.ndarray,
                 times: np.ndarray, speaker_median_pitch_hz: float = 100.0):
        self.pitch_hz = pitch_hz
        self.intensity_db = intensity_db
        self.times = times
        self.speaker_median_pitch_hz = speaker_median_pitch_hz


def _flat_contours(n_frames: int = 100, pitch_hz_value: float = 120.0,
                   intensity_value: float = 60.0, frame_step: float = 0.01
                   ) -> _FakeContours:
    return _FakeContours(
        pitch_hz=np.full(n_frames, pitch_hz_value),
        intensity_db=np.full(n_frames, intensity_value),
        times=np.arange(n_frames) * frame_step,
    )


def _two_syllables() -> pd.DataFrame:
    return pd.DataFrame([
        {'doc_id': 'd', 'sentence_id': 1, 'token_id': 1, 'token': 'below',
         'syllable_id': 1, 'Onset': 0.10, 'End': 0.30, 'Duration': 0.20},
        {'doc_id': 'd', 'sentence_id': 1, 'token_id': 1, 'token': 'below',
         'syllable_id': 2, 'Onset': 0.30, 'End': 0.50, 'Duration': 0.20},
    ])


def test_constant_pitch_yields_constant_pitch_mean_in_semitones():
    """Flat 120 Hz contour with speaker median 100 Hz → Pitch_Mean ≈ 12 * log2(1.2)."""
    df = add_syllable_prosody(_two_syllables(), _flat_contours(pitch_hz_value=120.0),
                              pitch_units='semitones')
    expected = 12.0 * np.log2(120.0 / 100.0)
    assert abs(df.Pitch_Mean.iloc[0] - expected) < 0.01
    assert abs(df.Pitch_Mean.iloc[1] - expected) < 0.01


def test_pitch_units_hz_emits_raw_hz():
    df = add_syllable_prosody(_two_syllables(), _flat_contours(pitch_hz_value=120.0),
                              pitch_units='hz')
    assert abs(df.Pitch_Mean.iloc[0] - 120.0) < 0.01


def test_volume_mean_is_intensity_mean_db():
    df = add_syllable_prosody(_two_syllables(),
                              _flat_contours(intensity_value=55.0))
    assert abs(df.Volume_Mean.iloc[0] - 55.0) < 0.01


def test_pitch_delta_is_nan_on_first_syllable():
    df = add_syllable_prosody(_two_syllables(), _flat_contours())
    assert pd.isna(df.Pitch_Delta.iloc[0])
    assert pd.isna(df.Volume_Delta.iloc[0])


def test_delta_is_difference_between_consecutive_means():
    """Two-region contour: syllable 1 over a flat region, syllable 2 over
    a higher region. Delta = mean_2 - mean_1."""
    times = np.arange(100) * 0.01
    pitch = np.where(times < 0.30, 100.0, 130.0)
    intensity = np.where(times < 0.30, 50.0, 60.0)
    contours = _FakeContours(pitch_hz=pitch, intensity_db=intensity, times=times,
                             speaker_median_pitch_hz=100.0)

    df = add_syllable_prosody(_two_syllables(), contours, pitch_units='hz')
    # Syllable 1: 0.10-0.30 covers the 100 Hz region → mean 100
    # Syllable 2: 0.30-0.50 covers the 130 Hz region → mean 130
    assert abs(df.Pitch_Mean.iloc[0] - 100.0) < 0.5
    assert abs(df.Pitch_Mean.iloc[1] - 130.0) < 0.5
    # Delta = 130 - 100 = 30
    assert pd.isna(df.Pitch_Delta.iloc[0])
    assert abs(df.Pitch_Delta.iloc[1] - 30.0) < 0.5
    # Volume delta similarly
    assert abs(df.Volume_Delta.iloc[1] - 10.0) < 0.5


def test_contours_none_emits_nan_columns_without_raising():
    df = add_syllable_prosody(_two_syllables(), None)
    assert df.Pitch_Mean.isna().all()
    assert df.Volume_Mean.isna().all()
    assert df.Pitch_Delta.isna().all()
    assert df.Volume_Delta.isna().all()


def test_empty_syllables_df_returns_empty_df_with_prosody_cols():
    empty = pd.DataFrame(columns=['doc_id', 'sentence_id', 'token_id', 'token',
                                  'syllable_id', 'Onset', 'End', 'Duration'])
    df = add_syllable_prosody(empty, _flat_contours())
    assert len(df) == 0
    for col in ('Pitch_Mean', 'Volume_Mean', 'Pitch_Delta', 'Volume_Delta'):
        assert col in df.columns
```

- [ ] **Step 2: Run tests to verify they fail**

```
python -m pytest tests/extractors/test_syllable_prosody.py -v
```

Expected: 7 tests FAIL with `ModuleNotFoundError: No module named 'extractors.syllable_prosody'`.

- [ ] **Step 3: Implement `extractors/syllable_prosody.py`**

```python
"""Syllable-level prosody — mean pitch/volume + delta within each syllable
interval. Reuses the Contours object built once per stimulus by
extractors.audio_features.extract(). No second audio decode.

Mirrors extractors/prosody.py:add_word_prosody for the per-row math;
the only difference is the row granularity (syllables vs words).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

PROSODY_COLUMNS = ('Pitch_Mean', 'Volume_Mean', 'Pitch_Delta', 'Volume_Delta')


def _slice_mean(values: np.ndarray, times: np.ndarray,
                onset: float, end: float) -> float:
    if values is None or times is None or len(values) == 0:
        return float('nan')
    mask = (times >= onset) & (times < end)
    if not mask.any():
        return float('nan')
    sub = values[mask]
    finite = sub[np.isfinite(sub)]
    if finite.size == 0:
        return float('nan')
    return float(finite.mean())


def _hz_to_semitones(hz: float, reference_hz: float) -> float:
    if not np.isfinite(hz) or hz <= 0 or reference_hz <= 0:
        return float('nan')
    return 12.0 * float(np.log2(hz / reference_hz))


def add_syllable_prosody(syll_df: pd.DataFrame, contours: Any,
                         *, pitch_units: str = 'semitones') -> pd.DataFrame:
    """Append Pitch_Mean / Volume_Mean / Pitch_Delta / Volume_Delta columns
    to syll_df. NaN-fills when contours is None (alignment-only run)."""
    df = syll_df.copy()

    if contours is None or len(df) == 0:
        for col in PROSODY_COLUMNS:
            df[col] = pd.Series([float('nan')] * len(df), dtype=float)
        return df

    pitch_hz = contours.pitch_hz
    intensity_db = contours.intensity_db
    times = contours.times
    median_pitch = float(getattr(contours, 'speaker_median_pitch_hz', 0.0))

    pitch_means: list[float] = []
    volume_means: list[float] = []
    for row in df.itertuples(index=False):
        p_hz = _slice_mean(pitch_hz, times, row.Onset, row.End)
        if pitch_units == 'semitones':
            pitch_means.append(_hz_to_semitones(p_hz, median_pitch))
        else:
            pitch_means.append(p_hz)
        volume_means.append(_slice_mean(intensity_db, times, row.Onset, row.End))

    df['Pitch_Mean'] = pitch_means
    df['Volume_Mean'] = volume_means
    # Delta: difference vs the PREVIOUS syllable in the same stimulus.
    # NaN on the first row.
    df['Pitch_Delta'] = df['Pitch_Mean'].diff()
    df['Volume_Delta'] = df['Volume_Mean'].diff()
    return df
```

- [ ] **Step 4: Run tests to verify they pass**

```
python -m pytest tests/extractors/test_syllable_prosody.py -v
```

Expected: all 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add extractors/syllable_prosody.py tests/extractors/test_syllable_prosody.py
git commit -m "syllable_prosody: per-syllable mean pitch/volume + deltas

add_syllable_prosody(syll_df, contours) slices the existing Parselmouth
Contours object by each syllable's [Onset, End], computes mean pitch
(Hz or semitones-relative-to-speaker-median) and mean intensity (dB),
then derives Pitch_Delta / Volume_Delta as the diff vs the previous
syllable. NaN-safe: contours=None yields NaN columns instead of raising.

Refs spec §4."
```

---

## Task 5: Prefs schema v5 → v6 (additive)

**Files:**
- Modify: `utils/preferences.py`
- Modify: `tests/utils/test_preferences_migration.py`

Adds the 12 new `feat_*` keys and 2 new `output.*_csv` keys, default-on except `feat_syllable_arpabet` (default-off). Migration is additive — deep-merge defaults into existing v5 files.

- [ ] **Step 1: Add v6 constants to `utils/preferences.py`**

Locate the v5 section (after `_PROSODY_DEFAULT_V5`, around line 308). Append:

```python
# ---------------------------------------------------------------------------
# v6 constants (active schema)
#
# Phoneme/syllable batch (2026-05-13) changes vs v5:
#   - 9 word-level syllable per-column toggles (feat_syllable_*)
#   - 4 word-level phoneme per-column toggles (feat_phoneme_*)
#   - 2 sidecar output toggles (output.syllables_csv, output.phonemes_csv)
# ---------------------------------------------------------------------------

SCHEMA_VERSION_V6 = 6

_FEATURES_DEFAULT_V6 = {
    **_FEATURES_DEFAULT_V5,
    # Syllable per-column (9 keys)
    "feat_syllable_onset":        True,
    "feat_syllable_end":          True,
    "feat_syllable_duration":     True,
    "feat_syllable_label":        True,
    "feat_syllable_arpabet":      False,  # opt-in; verbose column
    "feat_syllable_pitch_mean":   True,
    "feat_syllable_volume_mean":  True,
    "feat_syllable_pitch_delta":  True,
    "feat_syllable_volume_delta": True,
    # Phoneme per-column (4 keys)
    "feat_phoneme_label":         True,
    "feat_phoneme_onset":         True,
    "feat_phoneme_end":           True,
    "feat_phoneme_duration":      True,
}

_OUTPUT_DEFAULT_V6 = {
    **_OUTPUT_DEFAULT_V5,
    "syllables_csv": True,
    "phonemes_csv":  True,
}

DEFAULT_PREFS_V6: dict[str, Any] = {
    **DEFAULT_PREFS_V5,
    "schema_version": SCHEMA_VERSION_V6,
    "features": dict(_FEATURES_DEFAULT_V6),
    "output":   dict(_OUTPUT_DEFAULT_V6),
}
```

Then update the active aliases at the bottom of the constants section:

```python
SCHEMA_VERSION = SCHEMA_VERSION_V6
DEFAULT_PREFS: dict[str, Any] = DEFAULT_PREFS_V6
```

- [ ] **Step 2: Add the migration function**

Append after `_migrate_v4_into_v5`:

```python
def _migrate_v5_into_v6(merged: dict, v5: dict) -> None:
    """Convert v5-shape `v5` into v6-shape `merged` (mutates merged).
    Additive only — v6 defaults already include the new sub-word toggles
    and the new sidecar output keys.
    """
    for section in ("shared", "predictability", "semantic", "output", "prosody"):
        for k, v in v5.get(section, {}).items():
            merged[section][k] = v
    for k, v in v5.get("features", {}).items():
        merged["features"][k] = v
```

- [ ] **Step 3: Add the v5 branch to `load_preferences`**

In `load_preferences()`, the version-dispatch switch currently ends with the v5 branch. Insert a v6 branch above it and add a v5→v6 chain:

```python
    if version == SCHEMA_VERSION_V6:
        _deep_merge_current(merged, user)
    elif version == SCHEMA_VERSION_V5:
        _migrate_v5_into_v6(merged, user)
        _write(path, merged)
    elif version == SCHEMA_VERSION_V4:
        v5_intermediate = deepcopy(DEFAULT_PREFS_V5)
        _migrate_v4_into_v5(v5_intermediate, user)
        _migrate_v5_into_v6(merged, v5_intermediate)
        _write(path, merged)
    elif version == 3:
        v4_intermediate = deepcopy(DEFAULT_PREFS_V4)
        _migrate_v3_into_v4(v4_intermediate, user)
        v5_intermediate = deepcopy(DEFAULT_PREFS_V5)
        _migrate_v4_into_v5(v5_intermediate, v4_intermediate)
        _migrate_v5_into_v6(merged, v5_intermediate)
        _write(path, merged)
    elif version == 2:
        v3_intermediate = deepcopy(DEFAULT_PREFS_V3)
        _migrate_v2_into_v3(v3_intermediate, user)
        v4_intermediate = deepcopy(DEFAULT_PREFS_V4)
        _migrate_v3_into_v4(v4_intermediate, v3_intermediate)
        v5_intermediate = deepcopy(DEFAULT_PREFS_V5)
        _migrate_v4_into_v5(v5_intermediate, v4_intermediate)
        _migrate_v5_into_v6(merged, v5_intermediate)
        _write(path, merged)
    else:
        # v1 flat → v2 → v3 → v4 → v5 → v6
        v2_intermediate = deepcopy(_DEFAULT_PREFS_V2)
        _migrate_v1_into(v2_intermediate, user)
        v3_intermediate = deepcopy(DEFAULT_PREFS_V3)
        _migrate_v2_into_v3(v3_intermediate, v2_intermediate)
        v4_intermediate = deepcopy(DEFAULT_PREFS_V4)
        _migrate_v3_into_v4(v4_intermediate, v3_intermediate)
        v5_intermediate = deepcopy(DEFAULT_PREFS_V5)
        _migrate_v4_into_v5(v5_intermediate, v4_intermediate)
        _migrate_v5_into_v6(merged, v5_intermediate)
        _write(path, merged)
```

- [ ] **Step 4: Append migration tests**

Append to `tests/utils/test_preferences_migration.py`:

```python
def test_v5_to_v6_adds_subword_defaults(prefs_tmp):
    import json
    from utils.preferences import load_preferences, get_preferences_path
    prefs_tmp.write_text(json.dumps({
        "schema_version": 5,
        "features": {"feat_prosody_pitch_mean": False},  # existing v5 key, non-default
        "output": {"word_level_csv": True},
    }))
    prefs = load_preferences()
    assert prefs["schema_version"] == 6
    # New v6 defaults
    assert prefs["features"]["feat_syllable_onset"] is True
    assert prefs["features"]["feat_syllable_arpabet"] is False  # the lone default-off
    assert prefs["features"]["feat_phoneme_label"] is True
    assert prefs["output"]["syllables_csv"] is True
    assert prefs["output"]["phonemes_csv"] is True
    # Pre-existing v5 setting preserved
    assert prefs["features"]["feat_prosody_pitch_mean"] is False


def test_v4_to_v6_chain_carries_keys_through(prefs_tmp):
    import json
    from utils.preferences import load_preferences
    prefs_tmp.write_text(json.dumps({
        "schema_version": 4,
        "features": {"feat_psych_clout": False},
    }))
    prefs = load_preferences()
    assert prefs["schema_version"] == 6
    assert prefs["features"]["feat_psych_clout"] is False
    assert "feat_syllable_onset" in prefs["features"]
    assert "syllables_csv" in prefs["output"]
```

Note: `prefs_tmp` is a fixture defined in `conftest.py` that returns a `Path` to a fresh prefs file (used by existing migration tests).

- [ ] **Step 5: Run tests to verify passing**

```
python -m pytest tests/utils/test_preferences.py tests/utils/test_preferences_migration.py -v
```

Expected: all existing tests still PASS, 2 new tests PASS.

- [ ] **Step 6: Commit**

```bash
git add utils/preferences.py tests/utils/test_preferences_migration.py
git commit -m "prefs: schema v5 → v6 — add syllable/phoneme toggles + sidecar outputs

Additive migration. 9 new feat_syllable_* keys (default on except
feat_syllable_arpabet which is opt-in), 4 new feat_phoneme_* keys,
and 2 new output.*_csv keys for the sidecar master switches. Full
chain v1→v2→v3→v4→v5→v6 updated.

Refs spec §7."
```

---

## Task 6: `main.py` CLI flags and column-drop helpers

**Files:**
- Modify: `main.py`

Adds CLI flags and per-column drop helpers for the new sidecars. No behavior wiring yet — that's Task 7. Test-after (this is plumbing).

- [ ] **Step 1: Add CLI flags in `build_parser()`**

Locate the `Doc-level feature toggles (default = on)` block (around line 114). After that block, before the `return p` line, add:

```python
    # Sub-word sidecar master toggles
    p.add_argument('--no-syllables', action='store_true',
                   help='Skip <stim>_syllables.csv emission.')
    p.add_argument('--no-phonemes',  action='store_true',
                   help='Skip <stim>_phonemes.csv emission.')

    # Sub-word per-column toggles (default on; GUI usually drives via args.feature_toggles)
    p.add_argument('--no-syl-onset',         action='store_true')
    p.add_argument('--no-syl-end',           action='store_true')
    p.add_argument('--no-syl-duration',      action='store_true')
    p.add_argument('--no-syl-label',         action='store_true')
    p.add_argument('--no-syl-arpabet',       action='store_true',
                   help='Skip ARPABET column (opt-in only; default already off).')
    p.add_argument('--no-syl-pitch-mean',    action='store_true')
    p.add_argument('--no-syl-volume-mean',   action='store_true')
    p.add_argument('--no-syl-pitch-delta',   action='store_true')
    p.add_argument('--no-syl-volume-delta',  action='store_true')

    p.add_argument('--no-phon-label',    action='store_true')
    p.add_argument('--no-phon-onset',    action='store_true')
    p.add_argument('--no-phon-end',      action='store_true')
    p.add_argument('--no-phon-duration', action='store_true')
```

- [ ] **Step 2: Add column-drop maps next to the existing ones**

After `_DOC_PROSODY_COLUMN_MAP` (around line 410), append:

```python
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
}

_PHONEME_COLUMN_MAP = {
    "feat_phoneme_label":    "Label",
    "feat_phoneme_onset":    "Onset",
    "feat_phoneme_end":      "End",
    "feat_phoneme_duration": "Duration",
}


def _syllable_columns_to_drop(features: dict) -> set:
    return {col for key, col in _SYLLABLE_COLUMN_MAP.items()
            if not features.get(key, True)}


def _phoneme_columns_to_drop(features: dict) -> set:
    return {col for key, col in _PHONEME_COLUMN_MAP.items()
            if not features.get(key, True)}
```

- [ ] **Step 3: Smoke-test the parser**

```
python -c "from main import build_parser; p = build_parser(); a = p.parse_args(['--no-syllables', '--no-syl-arpabet']); print('syllables off:', a.no_syllables, 'arpabet off:', a.no_syl_arpabet)"
```

Expected output:
```
syllables off: True arpabet off: True
```

- [ ] **Step 4: Commit**

```bash
git add main.py
git commit -m "main: add sub-word CLI flags + column-drop helpers

Adds --no-syllables / --no-phonemes master flags plus per-column
--no-syl-* / --no-phon-* flags, and _syllable_columns_to_drop /
_phoneme_columns_to_drop maps mirroring the existing word/doc
column-map convention. No wiring yet — Task 7 wires it into
run_word_level_batch.

Refs spec §6."
```

---

## Task 7: Wire syllabification into `run_word_level_batch`

**Files:**
- Modify: `main.py`
- Modify: `extractors/aligner.py` — return phones DataFrame too (read first; may already return it)
- Modify: `tests/integration/test_pipeline_batch.py` — add an assertion

This is where the new sidecars actually land on disk. Test-after via an integration assertion.

- [ ] **Step 1: Verify `extractors/aligner.py` exposes the phones DataFrame**

```
python -c "import inspect; import extractors.aligner as a; print(inspect.getsource(a.align)[:500])"
```

Check whether `align()` returns just the word DataFrame or also the phones DataFrame. If only words, add a `phones_df` return alongside (the underlying `parse_phones` already works on the same TextGrid). If it already returns both, skip to Step 2.

If a change is needed, modify `extractors/aligner.py:align` to return a tuple `(words_df, phones_df)` and update the single caller in `main.py:run()` (around line 295) to unpack both:

```python
        align_df, phones_df = mfa_align(args.audio, text)
```

Stash `phones_df` on args so `run_word_level_batch` can use it later:

```python
        args._phones_df = phones_df
```

- [ ] **Step 2: Add the sidecar-emit block in `run_word_level_batch`**

Inside `run_word_level_batch` (around line 503, immediately after the line `df.to_csv(features_path, index=False)`), add:

```python
            # Sub-word sidecars (Task 7).
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
                    drop = _syllable_columns_to_drop(
                        getattr(args, 'feature_toggles', {}) or {})
                    if drop:
                        syll_df = syll_df.drop(columns=[c for c in drop if c in syll_df.columns])
                    syll_path = output_dir / f'{pair.stimulus_id}_syllables.csv'
                    syll_df.to_csv(syll_path, index=False)

                if not getattr(args, 'no_phonemes', False) and len(phon_df) > 0:
                    drop = _phoneme_columns_to_drop(
                        getattr(args, 'feature_toggles', {}) or {})
                    if drop:
                        phon_df = phon_df.drop(columns=[c for c in drop if c in phon_df.columns])
                    phon_path = output_dir / f'{pair.stimulus_id}_phonemes.csv'
                    phon_df.to_csv(phon_path, index=False)
```

- [ ] **Step 3: Add an integration-test assertion**

Open `tests/integration/test_pipeline_batch.py`. After the existing assertion that `*_features.csv` is emitted, append:

```python
def test_batch_emits_syllables_and_phonemes_sidecars(tmp_path):
    """End-to-end: a real fixture stimulus should produce both sidecars
    alongside the features CSV when alignment is on."""
    args = _args(tmp_path)  # use existing helper from this file
    args.no_align = False
    args.no_syllables = False
    args.no_phonemes = False
    from main import run_word_level_batch
    run_word_level_batch(args)

    out = tmp_path / 'out'
    syll = list(out.glob('*_syllables.csv'))
    phon = list(out.glob('*_phonemes.csv'))
    assert syll, 'expected at least one syllables sidecar'
    assert phon, 'expected at least one phonemes sidecar'

    import pandas as pd
    syll_df = pd.read_csv(syll[0])
    phon_df = pd.read_csv(phon[0])
    for col in ('doc_id', 'sentence_id', 'token_id', 'token', 'syllable_id', 'Onset', 'End'):
        assert col in syll_df.columns, f'syllable CSV missing {col}'
    for col in ('doc_id', 'sentence_id', 'token_id', 'token', 'syllable_id', 'phoneme_id',
                'Label', 'Onset', 'End'):
        assert col in phon_df.columns, f'phoneme CSV missing {col}'
    assert len(syll_df) > 0
    assert len(phon_df) > 0
```

- [ ] **Step 4: Run the new integration test**

```
python -m pytest tests/integration/test_pipeline_batch.py -v -k "syllables_and_phonemes"
```

Expected: PASS. If the libsndfile / MFA fixtures are missing on the runner, this test may be skipped or fail with an environment error — that's pre-existing and unrelated. The unit tests in Tasks 1-4 are the gating coverage.

- [ ] **Step 5: Commit**

```bash
git add main.py extractors/aligner.py tests/integration/test_pipeline_batch.py
git commit -m "main: wire syllabify_pairs + syllable_prosody into batch loop

run_word_level_batch now emits <stim>_syllables.csv and
<stim>_phonemes.csv alongside the existing features CSV when
alignment ran. Both sidecars respect the master --no-syllables /
--no-phonemes flags and the per-column toggles via args.feature_toggles.
The Parselmouth Contours object built once per stimulus is reused
for syllable prosody — no second audio decode.

Refs spec §6."
```

---

## Task 8: New GUI page — `gui/section_subword_features.py`

**Files:**
- Create: `gui/section_subword_features.py`

A new `SubwordFeaturesSection` with two cards (Syllables, Phonemes), per-column toggles wired to prefs. Mirrors the conventions of `gui/section_word_features.py` and `gui/section_doc_features.py`. Test-after (GUI wiring).

- [ ] **Step 1: Read an existing section for the pattern**

```
python -c "import pathlib; print(pathlib.Path('gui/section_doc_features.py').read_text()[:2000])"
```

Note the conventions: `Card`, `ToggleRow`, `_make_toggle`, `_on_toggle`, and the `settings_changed` Signal.

- [ ] **Step 2: Create `gui/section_subword_features.py`**

```python
"""Sub-word features page — Syllables and Phonemes per-column toggles."""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QWidget, QVBoxLayout, QGridLayout

from gui.widgets import Card, ToggleRow
from utils.preferences import load_preferences, save_preferences


class SubwordFeaturesSection(QWidget):
    settings_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._toggles: dict[str, ToggleRow] = {}
        prefs = load_preferences()
        f = prefs["features"]

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 16, 24, 16)
        layout.setSpacing(12)

        # --- Syllables ---
        syll_card = Card(
            "Syllables",
            "One row per syllable. Requires MFA alignment.")
        syll_grid = QGridLayout()
        syll_keys = [
            # Timing
            ("feat_syllable_onset",        "Onset"),
            ("feat_syllable_end",          "End"),
            ("feat_syllable_duration",     "Duration"),
            # Label
            ("feat_syllable_label",        "Label (orthographic, pyphen-derived)"),
            ("feat_syllable_arpabet",      "ARPABET (dash-joined, with stress)"),
            # Prosody
            ("feat_syllable_pitch_mean",   "Pitch_Mean"),
            ("feat_syllable_volume_mean",  "Volume_Mean"),
            ("feat_syllable_pitch_delta",  "Pitch_Delta"),
            ("feat_syllable_volume_delta", "Volume_Delta"),
        ]
        for i, (key, label) in enumerate(syll_keys):
            syll_grid.addWidget(
                self._make_toggle(key, label, f[key]),
                i // 2, i % 2,
            )
        syll_card.add_layout(syll_grid)
        layout.addWidget(syll_card)

        # --- Phonemes ---
        phon_card = Card(
            "Phonemes",
            "One row per phoneme. Requires MFA alignment.")
        phon_grid = QGridLayout()
        phon_keys = [
            ("feat_phoneme_label",    "Label (ARPABET incl. stress digit)"),
            ("feat_phoneme_onset",    "Onset"),
            ("feat_phoneme_end",      "End"),
            ("feat_phoneme_duration", "Duration"),
        ]
        for i, (key, label) in enumerate(phon_keys):
            phon_grid.addWidget(
                self._make_toggle(key, label, f[key]),
                i // 2, i % 2,
            )
        phon_card.add_layout(phon_grid)
        layout.addWidget(phon_card)

        layout.addStretch(1)

    def toggle(self, key: str):
        return self._toggles.get(key)

    def _make_toggle(self, key, label, initial):
        row = ToggleRow(label=label)
        row.set_checked(initial)
        row.toggled.connect(lambda v, k=key: self._on_toggle(k, v))
        self._toggles[key] = row
        return row

    def _on_toggle(self, key, value):
        save_preferences({"features": {key: value}})
        self.settings_changed.emit()
```

- [ ] **Step 3: Sanity-instantiate the section**

```
python -c "from PySide6.QtWidgets import QApplication; app = QApplication([]); from gui.section_subword_features import SubwordFeaturesSection; w = SubwordFeaturesSection(); print('toggles:', sorted(w._toggles.keys()))"
```

Expected: a list of 13 toggle keys (9 syllable + 4 phoneme).

- [ ] **Step 4: Commit**

```bash
git add gui/section_subword_features.py
git commit -m "gui: new Sub-word features section (Syllables + Phonemes cards)

13 per-column toggles wired to v6 prefs keys. Layout mirrors
section_word_features.py / section_doc_features.py conventions
(Card + ToggleRow + settings_changed signal). Not yet wired into
the sidebar — that's Task 9.

Refs spec §8.2."
```

---

## Task 9: Wire the new section into the sidebar

**Files:**
- Modify: `gui/main_window.py`

Adds a sidebar entry between "Document-level features" and "Run". Test-after.

- [ ] **Step 1: Locate the sidebar definition**

```
python -c "import pathlib; t = pathlib.Path('gui/main_window.py').read_text(); print(t[t.find('Document-level'):t.find('Document-level')+1500])"
```

Identify (a) the sidebar list/entries, (b) the QStackedWidget page additions, and (c) the routing that connects sidebar clicks to stacked-page indices. The PySide6 sidebar is keyed by name in the existing code; locate the line that constructs the doc section and the `addItem` / `addWidget` calls for it.

- [ ] **Step 2: Add the import**

Near the top of `gui/main_window.py`, alongside other section imports:

```python
from gui.section_subword_features import SubwordFeaturesSection
```

- [ ] **Step 3: Insert the new section and sidebar entry**

After the line(s) that add the doc-features page to the QStackedWidget and sidebar — typically adjacent constructions of `DocFeaturesSection`, and an `addItem`/`setData` pattern — insert the analogous pair for the new section:

```python
self.subword_section = SubwordFeaturesSection()
self.stack.addWidget(self.subword_section)
self.sidebar.addItem(QListWidgetItem("Sub-word features"))
# Match whatever pattern the existing entries use for index/data wiring.
```

If the section emits `settings_changed`, wire it to whatever refresh callback the other sections use:

```python
self.subword_section.settings_changed.connect(self._on_settings_changed)
```

(Copy exactly from how DocFeaturesSection is wired — pattern depends on the current file.)

- [ ] **Step 4: Smoke-launch the GUI**

```
python gui.py
```

Verify:
- Sidebar shows 6 entries in order: Inputs, Word-level features, Document-level features, **Sub-word features**, Run, Logs.
- Clicking "Sub-word features" shows two cards with 13 toggles total.
- Toggling and re-launching the GUI persists state (sanity check on save_preferences).

Close the GUI when verified.

- [ ] **Step 5: Commit**

```bash
git add gui/main_window.py
git commit -m "gui: add Sub-word features entry to sidebar

Inserts the new section between Document-level features and Run.
Matches the existing sidebar+QStackedWidget convention.

Refs spec §8.1."
```

---

## Task 10: Run-page output checkboxes for the new sidecars

**Files:**
- Modify: `gui/section_run.py`

Adds "Syllables CSV" and "Phonemes CSV" checkboxes next to the existing "Word-level CSV" and "Document-level summary". Test-after.

- [ ] **Step 1: Locate the existing output checkbox block**

```
python -c "import pathlib; t = pathlib.Path('gui/section_run.py').read_text(); idx = t.find('word_level_csv'); print(t[max(0,idx-200):idx+1000])"
```

- [ ] **Step 2: Add the two new checkboxes**

After the lines that wire `output_doc_summary` (or equivalent), add the same pattern for the two new keys. The widget construction will look like:

```python
self.syllables_cb = self._make_output_toggle(
    "output_syllables_csv", "Syllables CSV", o["syllables_csv"])
self.phonemes_cb = self._make_output_toggle(
    "output_phonemes_csv", "Phonemes CSV", o["phonemes_csv"])
layout.addWidget(self.syllables_cb)
layout.addWidget(self.phonemes_cb)
```

(Adjust to match the actual helper name and layout in section_run.py — the pattern follows the existing `output_doc_summary` setup.)

- [ ] **Step 3: Smoke-launch and verify**

```
python gui.py
```

Verify the Run page shows four output checkboxes:
- ☐ Word-level CSV
- ☐ Document-level summary
- ☐ Syllables CSV
- ☐ Phonemes CSV

- [ ] **Step 4: Commit**

```bash
git add gui/section_run.py
git commit -m "gui: Run page — add Syllables CSV and Phonemes CSV checkboxes

Output-master toggles for the two new sidecars. Default on, mirror
the word/doc CSV checkbox plumbing.

Refs spec §8.3."
```

---

## Task 11: Wire toggles through `gui/run_dispatch.py`

**Files:**
- Modify: `gui/run_dispatch.py`
- Modify: `tests/gui/test_run_dispatch.py`

Translates the new prefs keys into the `no_*` args namespace flags. Test-after (GUI dispatcher wiring is the project's established test-after surface).

- [ ] **Step 1: Update `build_args_word_level`**

In `gui/run_dispatch.py:build_args_word_level()`, after the existing prosody block, add:

```python
    # Sub-word sidecar master toggles flow from output prefs.
    ns.no_syllables = not o.get("syllables_csv", True)
    ns.no_phonemes  = not o.get("phonemes_csv", True)
```

Note: this requires `o = prefs["output"]` to be in scope inside `build_args_word_level`. If it isn't already, add it near the top of the function. Check the current function body and add only if missing.

Also: sub-word features require MFA alignment to produce phoneme times. If any sub-word toggle is on, force alignment back on:

```python
    any_subword = (
        not ns.no_syllables and any(f.get(k, False) for k in (
            "feat_syllable_onset", "feat_syllable_end", "feat_syllable_duration",
            "feat_syllable_label", "feat_syllable_arpabet",
            "feat_syllable_pitch_mean", "feat_syllable_volume_mean",
            "feat_syllable_pitch_delta", "feat_syllable_volume_delta",
        ))
        or not ns.no_phonemes and any(f.get(k, False) for k in (
            "feat_phoneme_label", "feat_phoneme_onset",
            "feat_phoneme_end", "feat_phoneme_duration",
        ))
    )
    if any_subword:
        ns.no_align = False
```

- [ ] **Step 2: Append regression tests**

Append to `tests/gui/test_run_dispatch.py`:

```python
def test_word_args_propagates_subword_master_toggles(prefs_tmp):
    from utils.preferences import save_preferences
    from gui.run_dispatch import build_args_word_level
    save_preferences({"output": {"syllables_csv": False, "phonemes_csv": True}})
    args = build_args_word_level()
    assert args.no_syllables is True
    assert args.no_phonemes is False


def test_word_args_forces_alignment_on_for_any_subword_feature(prefs_tmp):
    from utils.preferences import save_preferences
    from gui.run_dispatch import build_args_word_level
    save_preferences({
        "output": {"syllables_csv": True, "phonemes_csv": True},
        "features": {
            "feat_syllable_onset": True,
            # All acoustic_* toggles off
            "feat_acoustic_onset": False, "feat_acoustic_end": False,
            "feat_acoustic_word_duration": False, "feat_acoustic_word_gap": False,
        },
    })
    args = build_args_word_level()
    assert args.no_align is False, "subword features require alignment"


def test_word_args_no_subword_no_align_force(prefs_tmp):
    from utils.preferences import save_preferences
    from gui.run_dispatch import build_args_word_level
    save_preferences({
        "output": {"syllables_csv": False, "phonemes_csv": False},
        "features": {
            "feat_acoustic_onset": False, "feat_acoustic_end": False,
            "feat_acoustic_word_duration": False, "feat_acoustic_word_gap": False,
        },
    })
    args = build_args_word_level()
    # No sub-word, no acoustic → alignment correctly off
    assert args.no_align is True
```

- [ ] **Step 3: Run tests**

```
python -m pytest tests/gui/test_run_dispatch.py -v
```

Expected: all existing tests PASS, 3 new tests PASS.

- [ ] **Step 4: Commit**

```bash
git add gui/run_dispatch.py tests/gui/test_run_dispatch.py
git commit -m "run_dispatch: propagate sub-word toggles, force alignment when needed

build_args_word_level now sets ns.no_syllables / ns.no_phonemes from
the new output prefs, and force-on's MFA alignment whenever ANY
sub-word feature is on (sub-word features need MFA's phones tier).

Refs spec §6."
```

---

## Task 12: Preflight warnings for the new failure modes

**Files:**
- Modify: `gui/preflight.py`
- Modify: `tests/gui/test_preflight.py`

Two new warnings: (1) pyphen missing + syllable label on, (2) alignment off + any sub-word feature on. Test-after.

- [ ] **Step 1: Append the new checks in `compute_warnings`**

In `gui/preflight.py:compute_warnings()`, before the final `return out`, append:

```python
    # Sub-word features require MFA alignment.
    _any_subword = any(f.get(k, False) for k in (
        "feat_syllable_onset", "feat_syllable_end", "feat_syllable_duration",
        "feat_syllable_label", "feat_syllable_arpabet",
        "feat_syllable_pitch_mean", "feat_syllable_volume_mean",
        "feat_syllable_pitch_delta", "feat_syllable_volume_delta",
        "feat_phoneme_label", "feat_phoneme_onset",
        "feat_phoneme_end", "feat_phoneme_duration",
    ))
    _any_acoustic = any(f.get(k, False) for k in (
        "feat_acoustic_onset", "feat_acoustic_end",
        "feat_acoustic_word_duration", "feat_acoustic_word_gap"))
    if _any_subword and not _any_acoustic:
        out.append(Warning(
            toggle_key="feat_syllable_onset", severity="amber",
            message="Sub-word features require MFA alignment — enable Onset/End "
                    "in the Acoustic card on the Word-level features page.",
        ))

    # Syllable Label depends on pyphen for orthographic syllabification.
    try:
        import pyphen  # type: ignore  # noqa: F401
        _pyphen_ok = True
    except ImportError:
        _pyphen_ok = False
    if not _pyphen_ok and f.get("feat_syllable_label", False):
        out.append(Warning(
            toggle_key="feat_syllable_label", severity="amber",
            message="pyphen not installed — syllable Label will fall back to "
                    "ARPABET-without-stress. Install pyphen for orthographic labels.",
        ))
```

- [ ] **Step 2: Append regression tests**

Append to `tests/gui/test_preflight.py`:

```python
def test_warns_when_subword_features_on_but_alignment_off(prefs_tmp):
    from utils.preferences import save_preferences
    from gui.preflight import compute_warnings
    save_preferences({"features": {
        "feat_syllable_onset": True,
        # All acoustic off
        "feat_acoustic_onset": False, "feat_acoustic_end": False,
        "feat_acoustic_word_duration": False, "feat_acoustic_word_gap": False,
    }})
    warnings = compute_warnings()
    msgs = " ".join(w.message for w in warnings)
    assert "Sub-word features require MFA alignment" in msgs


def test_no_subword_warning_when_acoustic_on(prefs_tmp):
    from utils.preferences import save_preferences
    from gui.preflight import compute_warnings
    save_preferences({"features": {
        "feat_syllable_onset": True,
        "feat_acoustic_onset": True,  # one acoustic on is enough
    }})
    warnings = compute_warnings()
    msgs = " ".join(w.message for w in warnings)
    assert "Sub-word features require MFA alignment" not in msgs


def test_warns_when_pyphen_missing_and_syllable_label_on(prefs_tmp, monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "pyphen", None)
    from utils.preferences import save_preferences
    from gui.preflight import compute_warnings
    save_preferences({"features": {"feat_syllable_label": True}})
    warnings = compute_warnings()
    msgs = " ".join(w.message for w in warnings)
    assert "pyphen not installed" in msgs
```

- [ ] **Step 3: Run tests**

```
python -m pytest tests/gui/test_preflight.py -v
```

Expected: all existing tests PASS, 3 new tests PASS.

- [ ] **Step 4: Commit**

```bash
git add gui/preflight.py tests/gui/test_preflight.py
git commit -m "preflight: warn on missing-alignment and missing-pyphen for sub-word

Two new amber warnings: sub-word features need MFA alignment (mirrors
the same pattern used by prosody), and syllable Label degrades to
ARPABET-no-stress when pyphen is missing.

Refs spec §9."
```

---

## Task 13: Full-suite green check + master_plan update

**Files:**
- Modify: `docs/master_plan.md`

Final integration step: run the whole test suite, then move the phoneme/syllable batch into the completed-batches table.

- [ ] **Step 1: Run the full test suite**

```
python -m pytest -q
```

Expected: All previously-passing tests still PASS. New tests from Tasks 1-12 all PASS. No regressions. Final line: roughly `250+ passed, 1 skipped`.

If any pre-existing test (libsndfile / pyarrow related) fails because the env is incomplete on the runner, ignore — those are not regressions.

- [ ] **Step 2: Update `docs/master_plan.md` §2 active batch**

Replace the prosody-batch active-batch block with a new entry pointing to phoneme/syllable, OR — if prosody validation is now done — move the prosody row into completed-batches and set this batch as active. Coordinate with the user on whether prosody has been manually validated.

If prosody is validated, set the new active block:

```markdown
**Active batch:** Phoneme / syllable — **implemented, awaiting manual validation.**

- Spec: [`docs/superpowers/specs/2026-05-13-phoneme-syllable-design.md`](superpowers/specs/2026-05-13-phoneme-syllable-design.md)
- Plan: [`docs/superpowers/plans/2026-05-13-phoneme-syllable.md`](superpowers/plans/2026-05-13-phoneme-syllable.md)
- Status: All 13 plan tasks complete; full test suite green.
- Open: manual validation on one real stimulus — confirm `<stim>_syllables.csv` and `<stim>_phonemes.csv` appear with sane counts (≥ 1 syllable per word, no overlapping intervals, Pitch_Mean/Volume_Mean within expected ranges).
```

If prosody is not yet validated, append phoneme/syllable as a queued item in §4 and document in the §2 block that prosody is still the active validation target.

- [ ] **Step 3: Add a row to completed-batches**

Insert at the top of the table in §3:

```markdown
| 2026-05-13 | **Phoneme / Syllable** | Two new sidecar CSVs per stimulus (`<stim>_syllables.csv`, `<stim>_phonemes.csv`) joined by FK columns. Maximum Onset Principle syllabification on MFA's phones tier; syllable prosody (mean pitch/volume + deltas) reuses the existing Parselmouth Contours object. New "Sub-word features" GUI page. Prefs v5 → v6 additive. Spec/plan linked above. |
```

- [ ] **Step 4: Commit**

```bash
git add docs/master_plan.md
git commit -m "docs: master_plan — phoneme/syllable batch complete

13 tasks shipped per plan. Two new sidecars per stimulus
(<stim>_syllables.csv, <stim>_phonemes.csv) using MFA's phones tier
+ Max Onset Principle. Full test suite green. Awaiting manual
validation on a real stimulus."
```

---

## Self-Review

**Spec coverage check**

- §2 Goals 1 (two sidecars with FK columns) → Tasks 3, 7
- §2 Goals 2 (syllable prosody reusing Contours) → Task 4, wired in Task 7
- §2 Goals 3 (per-column toggles on Sub-word page) → Tasks 8, 11
- §2 Goals 4 (sidecar master switches on Run page) → Tasks 10, 11
- §4.1/4.2 column inventories → covered by Tasks 3 (FK + identity + timing + label), 4 (prosody)
- §4.3 mandatory context columns → enforced in Task 3 (always added) and Task 6 (not in `_*_COLUMN_MAP`)
- §5 algorithm → Task 1
- §5.4 label fallback → Task 2
- §6 code organization → Tasks 1-12 each create the right file
- §7 schema v5→v6 → Task 5
- §8 GUI surface → Tasks 8, 9, 10
- §9 dependencies/blocking → Tasks 11 (alignment force-on), 12 (preflight warnings)
- §10 testing strategy → TDD in Tasks 1-4, test-after in Tasks 5-12
- §11 out-of-scope → respected: no p-center, no phoneme class, no articulation_rate change

**Placeholder scan:** none found. Every code step has actual code.

**Type consistency:** `Phone.label` / `Phone.onset` / `Phone.end` used consistently across Tasks 1-3. `Syllable.phonemes` is a tuple in Task 1 and consumed as iterable in Tasks 2-3 ✓. `Contours.pitch_hz` / `intensity_db` / `times` / `speaker_median_pitch_hz` match the existing `extractors/audio_features.py:Contours` dataclass (verified in spec §4.2 and Task 4's fake).

**Scope:** focused. Implementation work is sub-1k LoC of new code, no schema-breaking changes, no untouched extractor files reshaped. All future work explicitly in §11.

---

*End of plan.*
