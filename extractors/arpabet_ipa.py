"""ARPABET -> IPA mapping for the phoneme and syllable sidecars.

Pure lookup, no I/O and no pandas. The MFA `english_us_arpa` dictionary uses
the 39-phone ARPABET set; vowels carry a stress digit (0/1/2) which MFA appends
to the label and which this pipeline stores in a separate `Stress` column.

Two ARPABET vowels are two *different* IPA vowels depending on stress, so
`to_ipa` takes the stress digit as an argument:

    AH0 -> schwa  /ə/        AH1, AH2 -> /ʌ/
    ER0 -> /ɚ/               ER1, ER2 -> /ɝ/

No stress marks (ˈ/ˌ) are emitted — the `Stress` column carries that
information, and the phoneme CSV's IPA column holds exactly one segment.

Reference: the standard CMU/ARPABET-to-IPA correspondence used by the CMU
Pronouncing Dictionary documentation.
"""
from __future__ import annotations

_STRESS_DIGITS = ("0", "1", "2")

# Stress-independent segments. AH and ER are listed at their STRESSED value;
# `to_ipa` overrides them for stress 0 via _UNSTRESSED below.
ARPABET_TO_IPA: dict[str, str] = {
    # Vowels (15)
    "AA": "ɑ",   # ɑ   odd
    "AE": "æ",   # æ   at
    "AH": "ʌ",   # ʌ   hut   (stress 0 -> ə, see _UNSTRESSED)
    "AO": "ɔ",   # ɔ   ought
    "AW": "aʊ",  # aʊ  cow
    "AY": "aɪ",  # aɪ  hide
    "EH": "ɛ",   # ɛ   Ed
    "ER": "ɝ",   # ɝ   hurt  (stress 0 -> ɚ, see _UNSTRESSED)
    "EY": "eɪ",  # eɪ  ate
    "IH": "ɪ",   # ɪ   it
    "IY": "i",        # i   eat
    "OW": "oʊ",  # oʊ  oat
    "OY": "ɔɪ",  # ɔɪ toy
    "UH": "ʊ",   # ʊ   hood
    "UW": "u",        # u   two
    # Consonants (24)
    "B": "b", "CH": "tʃ", "D": "d", "DH": "ð", "F": "f",
    "G": "ɡ", "HH": "h", "JH": "dʒ", "K": "k", "L": "l",
    "M": "m", "N": "n", "NG": "ŋ", "P": "p", "R": "ɹ",
    "S": "s", "SH": "ʃ", "T": "t", "TH": "θ", "V": "v",
    "W": "w", "Y": "j", "Z": "z", "ZH": "ʒ",
}

# Overrides applied only when stress == 0.
_UNSTRESSED: dict[str, str] = {
    "AH": "ə",   # ə  schwa
    "ER": "ɚ",   # ɚ  r-coloured schwa
}


def _strip_stress(code: str) -> str:
    """'EY2' -> 'EY'. Tolerated so a fused label never silently yields ''."""
    return code[:-1] if code and code[-1] in _STRESS_DIGITS else code


def to_ipa(arpabet: str, stress: int | None) -> str:
    """IPA for one ARPABET segment.

    Args:
        arpabet: Bare ARPABET code ('EY'). A fused label ('EY2') is tolerated.
        stress:  0/1/2, or None when stress does not apply (consonants). When a
                 stress-dependent vowel arrives with stress None, the stressed
                 form is returned — never ''.

    Returns:
        A single IPA segment, or '' for silence/noise/unknown codes. Never a
        guess: an unrecognised code maps to '' so a wrong symbol can't enter
        the data silently.
    """
    if not arpabet:
        return ""
    code = _strip_stress(str(arpabet).strip())
    if stress is None and code != str(arpabet).strip():
        # Stress came fused into the label; recover it.
        digit = str(arpabet).strip()[-1]
        stress = int(digit)
    if stress == 0 and code in _UNSTRESSED:
        return _UNSTRESSED[code]
    return ARPABET_TO_IPA.get(code, "")


def syllable_to_ipa(pairs) -> str:
    """Dash-joined IPA over one syllable's (arpabet, stress) pairs.

    Unmappable segments are dropped rather than emitted as empty fields, so a
    syllable containing noise yields 'd-i', not 'd--i'.
    """
    segments = [to_ipa(code, stress) for code, stress in pairs]
    return "-".join(s for s in segments if s)
