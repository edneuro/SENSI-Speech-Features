"""
Transcript preprocessing utilities.

Prepares raw transcript text before it is sent to the forced aligner.
The key requirement is that numbers must be spelled out — MFA's pronunciation
dictionary contains English words, not digits. The same spelling function is
used by the alignment merge (extractors/align_merge.py) so a digit token in
the spaCy parse can be matched back to the spelled-out words MFA aligned.
"""

import re
from num2words import num2words

# One pattern, alternation ordered most-specific-first so e.g. "1,000" is
# consumed as one grouped number before \b\d+\b can grab the "1".
_NUMBER_RE = re.compile(
    r"""
    \b\d{1,3}(?:,\d{3})+\b      # grouped thousands: 1,000 / 1,234,567
    | \b\d+\.\d+\b              # decimals: 3.5
    | \b(\d+)(st|nd|rd|th)\b    # ordinals: 1st, 2nd, 3rd, 24th
    | \b\d+\b                   # bare integers
    """,
    re.VERBOSE | re.IGNORECASE,
)


def spell_out_numbers(text: str) -> str:
    """Replace digit sequences with their spoken English form.

    Examples:
        "Bella has 2 apples."      -> "Bella has two apples."
        "There were 24 scenarios." -> "There were twenty-four scenarios."
        "over 1,000 people"        -> "over one thousand people"
        "It weighs 3.5 kg"         -> "It weighs three point five kg"
        "the 1st time"             -> "the first time"

    Known limitation: four-digit years are read as cardinals ("1997" ->
    "one thousand, nine hundred and ninety-seven"), not as a speaker would
    ("nineteen ninety-seven") — digits vs. years are not distinguishable
    from text alone. If a narrator reads years aloud, spell them out in the
    transcript file itself.
    """
    def _replace(match: re.Match) -> str:
        s = match.group()
        if match.group(1) is not None:              # ordinal: digits + suffix
            return num2words(int(match.group(1)), to='ordinal')
        if ',' in s:                                # grouped thousands
            return num2words(int(s.replace(',', '')))
        if '.' in s:                                # decimal
            return num2words(float(s))
        return num2words(int(s))

    return _NUMBER_RE.sub(_replace, text)


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
