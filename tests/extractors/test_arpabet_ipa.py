"""ARPABET -> IPA mapping (extractors/arpabet_ipa.py)."""
import pytest

from sensi.extractors.arpabet_ipa import ARPABET_TO_IPA, to_ipa, syllable_to_ipa


def test_stress_dependent_ah_is_schwa_when_unstressed():
    assert to_ipa("AH", 0) == "ə"      # schwa


@pytest.mark.parametrize("stress", [1, 2])
def test_stress_dependent_ah_is_wedge_when_stressed(stress):
    assert to_ipa("AH", stress) == "ʌ"  # turned v


def test_stress_dependent_er():
    assert to_ipa("ER", 0) == "ɚ"       # schwa with hook
    assert to_ipa("ER", 1) == "ɝ"       # reversed epsilon with hook
    assert to_ipa("ER", 2) == "ɝ"


def test_ah_and_er_with_no_stress_digit_fall_back_to_stressed_form():
    # A vowel should always carry a stress digit from MFA; if one is missing
    # we must still return a vowel, not "".
    assert to_ipa("AH", None) == "ʌ"
    assert to_ipa("ER", None) == "ɝ"


@pytest.mark.parametrize("code,expected", [
    ("AA", "ɑ"), ("AE", "æ"), ("AO", "ɔ"), ("AW", "aʊ"),
    ("AY", "aɪ"), ("EH", "ɛ"), ("EY", "eɪ"), ("IH", "ɪ"),
    ("IY", "i"), ("OW", "oʊ"), ("OY", "ɔɪ"), ("UH", "ʊ"),
    ("UW", "u"),
])
def test_vowels_are_stress_independent(code, expected):
    assert to_ipa(code, 1) == expected
    assert to_ipa(code, 0) == expected


@pytest.mark.parametrize("code,expected", [
    ("B", "b"), ("CH", "tʃ"), ("D", "d"), ("DH", "ð"), ("F", "f"),
    ("G", "ɡ"), ("HH", "h"), ("JH", "dʒ"), ("K", "k"), ("L", "l"),
    ("M", "m"), ("N", "n"), ("NG", "ŋ"), ("P", "p"), ("R", "ɹ"),
    ("S", "s"), ("SH", "ʃ"), ("T", "t"), ("TH", "θ"), ("V", "v"),
    ("W", "w"), ("Y", "j"), ("Z", "z"), ("ZH", "ʒ"),
])
def test_consonants(code, expected):
    assert to_ipa(code, None) == expected


def test_all_39_arpabet_phones_are_covered():
    # 15 vowels (AH and ER counted once each) + 24 consonants = 39.
    assert len(ARPABET_TO_IPA) == 39
    assert all(v for v in ARPABET_TO_IPA.values())


def test_unknown_and_silence_map_to_empty_string_not_a_guess():
    assert to_ipa("spn", None) == ""
    assert to_ipa("sil", None) == ""
    assert to_ipa("", None) == ""
    assert to_ipa("QQ", None) == ""


def test_stress_digit_on_input_is_tolerated():
    # Callers pass a bare code, but a fused label must not silently yield "".
    assert to_ipa("AH0", None) == "ə"
    assert to_ipa("EY2", None) == "eɪ"


def test_syllable_to_ipa_dash_joins():
    assert syllable_to_ipa([("D", None), ("R", None), ("IY", 0)]) == "d-ɹ-i"


def test_syllable_to_ipa_skips_unmappable_segments():
    assert syllable_to_ipa([("D", None), ("spn", None), ("IY", 0)]) == "d-i"


def test_syllable_to_ipa_empty_input():
    assert syllable_to_ipa([]) == ""
