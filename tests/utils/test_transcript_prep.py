"""Tests for utils/transcript_prep.py — number spelling for the MFA .lab.

MFA's dictionary contains spoken English words only, so every digit sequence
must become the words a narrator would actually say. Wrong spellings don't
just misalign one word — they desynchronize the token↔alignment merge.
"""
import re

import pytest

from utils.transcript_prep import (
    normalize_whitespace,
    prepare_for_aligner,
    spell_out_numbers,
)


def _no_digits(s: str) -> bool:
    return re.search(r"\d", s) is None


# --- existing behavior (regression guard) ----------------------------------

def test_bare_integer():
    assert spell_out_numbers("Bella has 2 apples.") == "Bella has two apples."


def test_two_digit_integer_hyphenated():
    assert spell_out_numbers("There were 24 scenarios.") == \
        "There were twenty-four scenarios."


def test_text_without_numbers_unchanged():
    assert spell_out_numbers("No numbers here.") == "No numbers here."


# --- thousands separators ---------------------------------------------------

def test_thousands_separator_is_one_number():
    # The old \b\d+\b pattern matched "1" and "000" separately, producing
    # the unspeakable "one,zero" — MFA then aligned two wrong words.
    assert spell_out_numbers("over 1,000 people") == "over one thousand people"


def test_millions_with_two_separators():
    out = spell_out_numbers("about 1,234,567 grains")
    assert _no_digits(out)
    assert "million" in out


def test_comma_between_numbers_is_not_a_separator():
    # "3, 4" is a list, not a grouped number.
    assert spell_out_numbers("items 3, 4 and 5") == "items three, four and five"


# --- decimals ----------------------------------------------------------------

def test_decimal_spoken_with_point():
    out = spell_out_numbers("It weighs 3.5 kilograms.")
    assert _no_digits(out)
    assert out.startswith("It weighs three point")


# --- ordinals ----------------------------------------------------------------

@pytest.mark.parametrize("src,expected", [
    ("the 1st time", "the first time"),
    ("her 2nd try", "her second try"),
    ("the 3rd door", "the third door"),
    ("his 24th birthday", "his twenty-fourth birthday"),
])
def test_ordinals(src, expected):
    assert spell_out_numbers(src) == expected


# --- pipeline ----------------------------------------------------------------

def test_prepare_for_aligner_composes_whitespace_and_numbers():
    out = prepare_for_aligner("He   won\n\n1st place  with 1,500 points")
    assert out == "He won first place with one thousand, five hundred points"


def test_normalize_whitespace_collapses_runs():
    assert normalize_whitespace("a  b\n\nc\t d") == "a b c d"
