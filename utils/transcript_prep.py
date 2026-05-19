"""
Transcript preprocessing utilities.

Prepares raw transcript text before it is sent to the forced aligner.
The key requirement is that numbers must be spelled out — MFA's pronunciation
dictionary contains English words, not digits.
"""

import re
from num2words import num2words


def spell_out_numbers(text: str) -> str:
    """Replace digit sequences with their spoken English form.

    Examples:
        "Bella has 2 apples."      -> "Bella has two apples."
        "Joseph gives 6 more."     -> "Joseph gives six more."
        "There were 24 scenarios." -> "There were twenty-four scenarios."

    Edge cases handled:
        Ordinals ("1st", "2nd") are not matched by this pattern — only
        bare integers. If ordinals appear in transcripts, extend this
        function using num2words(n, to='ordinal').
    """
    def _replace(match):
        return num2words(int(match.group()))

    return re.sub(r'\b\d+\b', _replace, text)


def normalize_whitespace(text: str) -> str:
    """Collapse multiple spaces/newlines into single spaces."""
    return re.sub(r'\s+', ' ', text).strip()


def prepare_for_aligner(text: str) -> str:
    """Full preprocessing pipeline: clean whitespace, spell out numbers.

    This is the function to call before passing a transcript to MFA.
    The result is also stored in the 'Transcript' column of the output CSV.
    """
    text = normalize_whitespace(text)
    text = spell_out_numbers(text)
    return text
