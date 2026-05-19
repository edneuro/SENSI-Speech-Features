"""Unit tests for the Maximum Onset Principle syllabification."""
import pandas as pd
import pytest
from extractors.syllabify import (
    LabelResult, Phone, Syllable, assign_labels, syllabify_word, syllabify_pairs,
)


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
    assert _labels(syllables[0]) == ['L', 'AE1', 'M', 'P', 'S']


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
    """phonemics: F AH0 N IY1 M IH0 K S — IH0 is a nucleus (stress digit 0).
    Also verifies the partition is correct, not just the count."""
    syllables = syllabify_word(_phones(['F', 'AH0', 'N', 'IY1', 'M', 'IH0', 'K', 'S']))
    assert len(syllables) == 3  # three vowels
    assert _labels(syllables[0]) == ['F', 'AH0']
    assert _labels(syllables[1]) == ['N', 'IY1']
    assert _labels(syllables[2]) == ['M', 'IH0', 'K', 'S']


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


# ---------------------------------------------------------------------------
# Task 2: assign_labels tests
# ---------------------------------------------------------------------------

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
    assert result.labels == ('be', 'low')
    assert result.arpabet == ('B-IH0', 'L-OW1')


def test_pyphen_mismatch_falls_back_to_arpabet_no_stress():
    """When counts differ (e.g., 'fire' pyphen=1, MFA=2), Label is ARPABET-no-stress."""
    syllables = _syllables_from([['F', 'AY1'], ['ER0']])  # 'fire' phonemically 2 sylls
    result = assign_labels('fire', syllables)
    assert result.mismatch is True
    assert result.labels == ('F-AY', 'ER')
    assert result.arpabet == ('F-AY1', 'ER0')


def test_pyphen_missing_module_falls_back_to_arpabet_no_stress(monkeypatch):
    """When pyphen is unavailable, Label is ARPABET-no-stress for everything.

    assign_labels reads the module-level _pyphen_unavailable flag at call
    time and never re-imports pyphen, so patching the flag is sufficient —
    no sys.modules manipulation needed.
    """
    from extractors import syllabify
    monkeypatch.setattr(syllabify, '_pyphen_dict', None)
    monkeypatch.setattr(syllabify, '_pyphen_unavailable', True)

    syllables = _syllables_from([['B', 'IH0'], ['L', 'OW1']])
    result = assign_labels('below', syllables)
    assert result.mismatch is True  # treated as a mismatch
    assert result.labels == ('B-IH', 'L-OW')
    assert result.arpabet == ('B-IH0', 'L-OW1')


def test_arpabet_for_single_syllable_word():
    """Single-syllable word: pyphen matches (1 == 1), so labels also tested here."""
    syllables = _syllables_from([['L', 'AE1', 'M', 'P', 'S']])
    result = assign_labels('lamps', syllables)
    assert result.arpabet == ('L-AE1-M-P-S',)
    assert result.labels == ('lamps',)
    assert result.mismatch is False


# ---------------------------------------------------------------------------
# Task 3: syllabify_pairs orchestrator tests
# ---------------------------------------------------------------------------

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
    assert s0.Onset == pytest.approx(0.00)
    assert s0.End == pytest.approx(0.10)
    assert s0.Duration == pytest.approx(0.10, abs=1e-4)


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
