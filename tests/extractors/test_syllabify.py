"""Unit tests for the Maximum Onset Principle syllabification."""
import pandas as pd
import pytest
from sensi.extractors import syllabify
from sensi.extractors.syllabify import (
    LabelResult, Phone, Syllable, assign_labels, syllabify_word, syllabify_pairs,
)

# Explicit fixture lexicon for Pron_Source tests, so they never depend on the
# real (git-ignored) MFA dictionary at tests/../mfa_data/. On a fresh clone
# without data/mfa_data/, _lexicon_words() would return an empty set and
# _pron_source's no-lexicon branch always returns "dictionary" — which would
# make these tests pass vacuously, including the '48'/'forty eight' regression
# guard. Must contain the dictionary-side fixtures and exclude 'zorblax'/'48'.
_FIXTURE_LEXICON = {"cup", "forty", "eight", "debtor", "was", "wasn't"}


def _patch_lexicon(monkeypatch):
    """Force _pron_source to consult _FIXTURE_LEXICON instead of the real
    MFA dictionary, regardless of test order (clears the memoization cache
    too, since _lexicon_words() only recomputes when the cache is None)."""
    monkeypatch.setattr(syllabify, "_LEXICON_CACHE", None)
    monkeypatch.setattr(syllabify, "_lexicon_words", lambda: _FIXTURE_LEXICON)


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
    from sensi.extractors import syllabify
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
        {'doc_id': 'stim1', 'sentence_id': 1, 'token_in_sentence_id': 1, 'token': 'below',
         'Onset': 0.00, 'End': 0.20},
        {'doc_id': 'stim1', 'sentence_id': 1, 'token_in_sentence_id': 2, 'token': 'network',
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
    for col in ('doc_id', 'sentence_id', 'token_in_sentence_id', 'token', 'syllable_id'):
        assert col in syll_df.columns
    for col in ('doc_id', 'sentence_id', 'token_in_sentence_id', 'token', 'syllable_id', 'phoneme_id'):
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
        {'doc_id': 'd', 'sentence_id': 1, 'token_in_sentence_id': 1, 'token': 'below',
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
        {'doc_id': 'd', 'sentence_id': 1, 'token_in_sentence_id': 1, 'token': 'below',
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
        {'doc_id': 'd', 'sentence_id': 1, 'token_in_sentence_id': 1, 'token': 'gap',
         'Onset': 0.0, 'End': 0.1},
        {'doc_id': 'd', 'sentence_id': 1, 'token_in_sentence_id': 2, 'token': 'a',
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


# ---------------------------------------------------------------------------
# Task 3: split ARPABET/Stress/IPA, dedup shared alignments, Pron_Source
# ---------------------------------------------------------------------------

def _word_row(token, onset, end, tis_id, transcript=None):
    """One word_df row. `transcript` is the MFA words-tier form; it defaults to
    the token, and differs from it for digits ('48' -> 'forty eight')."""
    return {"doc_id": 1, "sentence_id": 1, "token_in_sentence_id": tis_id,
            "token": token, "Transcript": token if transcript is None else transcript,
            "Onset": onset, "End": end}


def test_split_columns_replace_label():
    word_df = pd.DataFrame([_word_row("cup", 0.0, 0.30, 1)])
    phone_df = pd.DataFrame([
        {"phone": "K",   "onset": 0.00, "offset": 0.10},
        {"phone": "AH1", "onset": 0.10, "offset": 0.20},
        {"phone": "P",   "onset": 0.20, "offset": 0.30},
    ])
    _, phon = syllabify_pairs(word_df, phone_df)

    assert "Label" not in phon.columns
    assert list(phon["ARPABET"]) == ["K", "AH", "P"]
    assert list(phon["IPA"]) == ["k", "ʌ", "p"]


def test_stress_is_empty_on_consonants_and_zero_on_unstressed_vowels():
    """Empty != 0. Empty means 'does not apply'; 0 means 'unstressed vowel'."""
    word_df = pd.DataFrame([_word_row("about", 0.0, 0.40, 1)])
    phone_df = pd.DataFrame([
        {"phone": "AH0", "onset": 0.00, "offset": 0.10},
        {"phone": "B",   "onset": 0.10, "offset": 0.20},
        {"phone": "AW1", "onset": 0.20, "offset": 0.30},
        {"phone": "T",   "onset": 0.30, "offset": 0.40},
    ])
    _, phon = syllabify_pairs(word_df, phone_df)

    assert str(phon["Stress"].dtype) == "Int64"
    stress = list(phon["Stress"])
    assert stress[0] == 0            # unstressed vowel -> 0, NOT empty
    assert pd.isna(stress[1])        # consonant -> empty, NOT 0
    assert stress[2] == 1
    assert pd.isna(stress[3])
    # And the schwa/wedge distinction follows from it:
    assert list(phon["IPA"]) == ["ə", "b", "aʊ", "t"]


def test_contraction_phonemes_are_emitted_once_for_the_first_token():
    """spaCy 'was' + "n't" both carry MFA "wasn't"'s interval."""
    word_df = pd.DataFrame([
        _word_row("was", 4.50, 4.87, 1),
        _word_row("n't", 4.50, 4.87, 2),
    ])
    phone_df = pd.DataFrame([
        {"phone": "W",   "onset": 4.50, "offset": 4.60},
        {"phone": "AH1", "onset": 4.60, "offset": 4.68},
        {"phone": "Z",   "onset": 4.68, "offset": 4.75},
        {"phone": "AH0", "onset": 4.75, "offset": 4.81},
        {"phone": "N",   "onset": 4.81, "offset": 4.84},
        {"phone": "T",   "onset": 4.84, "offset": 4.87},
    ])
    syll, phon = syllabify_pairs(word_df, phone_df)

    assert len(phon) == 6, "phonemes must not be emitted twice"
    assert set(phon["token"]) == {"was"}
    assert len(syll) == 2
    assert set(syll["token"]) == {"was"}


def test_distinct_words_with_distinct_intervals_are_both_emitted():
    """Dedup must key on the interval, not merely on adjacency."""
    word_df = pd.DataFrame([
        _word_row("a", 0.00, 0.10, 1),
        _word_row("bee", 0.10, 0.30, 2),
    ])
    phone_df = pd.DataFrame([
        {"phone": "AH0", "onset": 0.00, "offset": 0.10},
        {"phone": "B",   "onset": 0.10, "offset": 0.20},
        {"phone": "IY1", "onset": 0.20, "offset": 0.30},
    ])
    _, phon = syllabify_pairs(word_df, phone_df)
    assert set(phon["token"]) == {"a", "bee"}
    assert len(phon) == 3


def test_oov_word_yields_one_row_flagged_unresolved():
    word_df = pd.DataFrame([_word_row("zorblax", 0.0, 0.33, 1)])
    phone_df = pd.DataFrame([
        {"phone": "spn", "onset": 0.0, "offset": 0.33},
    ])
    syll, phon = syllabify_pairs(word_df, phone_df)

    assert len(phon) == 1
    assert phon["ARPABET"].iloc[0] == "spn"
    assert pd.isna(phon["Stress"].iloc[0])
    assert phon["IPA"].iloc[0] == ""
    assert phon["Pron_Source"].iloc[0] == "oov_unresolved"
    assert len(syll) == 1, "an spn word still occupies one syllable"


def test_in_dictionary_word_is_flagged_dictionary(monkeypatch):
    _patch_lexicon(monkeypatch)
    word_df = pd.DataFrame([_word_row("cup", 0.0, 0.30, 1)])
    phone_df = pd.DataFrame([
        {"phone": "K",   "onset": 0.00, "offset": 0.10},
        {"phone": "AH1", "onset": 0.10, "offset": 0.20},
        {"phone": "P",   "onset": 0.20, "offset": 0.30},
    ])
    _, phon = syllabify_pairs(word_df, phone_df)
    assert set(phon["Pron_Source"]) == {"dictionary"}


def test_numeric_token_is_dictionary_not_g2p(monkeypatch):
    """Regression: spaCy keeps '48' but the aligner got 'forty eight'.

    Judging provenance from the spaCy token would mark every number in the
    corpus as a modeled pronunciation. The MFA words-tier form is the truth.
    Uses the fixture lexicon (not the real dictionary) so this guard can't
    pass vacuously via the no-lexicon fallback on a fresh clone/CI.
    """
    _patch_lexicon(monkeypatch)
    word_df = pd.DataFrame([
        _word_row("48", 4.27, 4.94, 1, transcript="forty eight"),
    ])
    phone_df = pd.DataFrame([
        {"phone": "F",   "onset": 4.27, "offset": 4.40},
        {"phone": "AO1", "onset": 4.40, "offset": 4.55},
        {"phone": "R",   "onset": 4.55, "offset": 4.65},
        {"phone": "T",   "onset": 4.65, "offset": 4.72},
        {"phone": "IY0", "onset": 4.72, "offset": 4.80},
        {"phone": "EY1", "onset": 4.80, "offset": 4.90},
        {"phone": "T",   "onset": 4.90, "offset": 4.94},
    ])
    _, phon = syllabify_pairs(word_df, phone_df)
    assert set(phon["Pron_Source"]) == {"dictionary"}


def test_pron_source_falls_back_to_the_token_without_a_transcript_column(monkeypatch):
    """A caller may pass a frame that never went through merge_alignment."""
    _patch_lexicon(monkeypatch)
    word_df = pd.DataFrame([{
        "doc_id": 1, "sentence_id": 1, "token_in_sentence_id": 1,
        "token": "cup", "Onset": 0.0, "End": 0.30,
    }])
    phone_df = pd.DataFrame([
        {"phone": "K",   "onset": 0.00, "offset": 0.10},
        {"phone": "AH1", "onset": 0.10, "offset": 0.20},
        {"phone": "P",   "onset": 0.20, "offset": 0.30},
    ])
    _, phon = syllabify_pairs(word_df, phone_df)
    assert set(phon["Pron_Source"]) == {"dictionary"}


def test_oov_word_with_real_phones_is_flagged_g2p(monkeypatch):
    """G2P recovered a pronunciation for a word absent from the dictionary."""
    _patch_lexicon(monkeypatch)
    word_df = pd.DataFrame([_word_row("zorblax", 0.0, 0.20, 1)])
    phone_df = pd.DataFrame([
        {"phone": "Z",   "onset": 0.00, "offset": 0.10},
        {"phone": "AO1", "onset": 0.10, "offset": 0.20},
    ])
    _, phon = syllabify_pairs(word_df, phone_df)
    assert set(phon["Pron_Source"]) == {"g2p"}


def test_syllable_gains_ipa_and_keeps_label_and_arpabet():
    word_df = pd.DataFrame([_word_row("cup", 0.0, 0.30, 1)])
    phone_df = pd.DataFrame([
        {"phone": "K",   "onset": 0.00, "offset": 0.10},
        {"phone": "AH1", "onset": 0.10, "offset": 0.20},
        {"phone": "P",   "onset": 0.20, "offset": 0.30},
    ])
    syll, _ = syllabify_pairs(word_df, phone_df)
    # 'cup' is one syllable (onset K, nucleus AH1, coda P); unchanged: ARPABET
    # keeps stress digits.
    assert syll["ARPABET"].iloc[0] == "K-AH1-P", "unchanged: keeps stress digits"
    assert syll["IPA"].iloc[0] == "k-ʌ-p"


def test_zero_phoneme_words_are_reported(capsys):
    """A word with no phones at all must not vanish silently."""
    word_df = pd.DataFrame([
        _word_row("cup", 0.0, 0.10, 1),
        _word_row("ghost", 5.0, 5.20, 2),   # no phones in its interval
    ])
    phone_df = pd.DataFrame([
        {"phone": "K", "onset": 0.00, "offset": 0.10},
    ])
    syllabify_pairs(word_df, phone_df)
    err = capsys.readouterr().err
    assert "ghost" in err
    assert "no phonemes" in err


def test_deduped_second_token_is_not_reported_as_missing(capsys):
    """n't legitimately has no rows; that is not an anomaly."""
    word_df = pd.DataFrame([
        _word_row("was", 4.50, 4.60, 1),
        _word_row("n't", 4.50, 4.60, 2),
    ])
    phone_df = pd.DataFrame([{"phone": "W", "onset": 4.50, "offset": 4.60}])
    _, phon = syllabify_pairs(word_df, phone_df)
    assert "n't" not in capsys.readouterr().err
    # The real dedup pin: "n't" shares was's exact [Onset, End] interval, so
    # it must not get its own emitted phoneme row at all — merely being
    # absent from stderr is also true if dedup were removed entirely (n't
    # would still get the W phone and simply not be "missing").
    assert set(phon["token"]) == {"was"}


def test_lexicon_empty_warns_once_to_stderr(monkeypatch, capsys):
    """Final-review Fix 5: an unreadable MFA dictionary makes _pron_source
    report 'dictionary' for EVERY word (including genuinely G2P-derived
    ones) with no signal that anything went wrong. Warn once to stderr."""
    import sensi.extractors.phoneme_surprisal as ps
    monkeypatch.setattr(syllabify, "_LEXICON_CACHE", None)
    monkeypatch.setattr(syllabify, "_LEXICON_EMPTY_WARNED", False)
    monkeypatch.setattr(ps, "DEFAULT_DICT_PATH", "Z:/does/not/exist_at_all.dict")

    lex1 = syllabify._lexicon_words()
    assert lex1 == set()
    err = capsys.readouterr().err
    assert "unreadable" in err.lower()

    # Memoized + warned-once: a second call must not warn again.
    lex2 = syllabify._lexicon_words()
    assert lex2 == set()
    assert capsys.readouterr().err == ""


def test_empty_frames_still_carry_the_new_schema():
    syll, phon = syllabify_pairs(
        pd.DataFrame(columns=["doc_id", "sentence_id", "token_in_sentence_id",
                              "token", "Onset", "End"]),
        pd.DataFrame(columns=["phone", "onset", "offset"]),
    )
    for col in ("ARPABET", "Stress", "IPA", "Pron_Source"):
        assert col in phon.columns
    assert "Label" not in phon.columns
    assert "IPA" in syll.columns
