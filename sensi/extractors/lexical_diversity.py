"""Lexical diversity measures (TTR, MTLD).

Operates on a list of lowercased content-word tokens.

TTR (Type-Token Ratio):
    Unique tokens / total tokens. Simple but length-biased.

MTLD (Measure of Textual Lexical Diversity, McCarthy & Jarvis 2010,
doi:10.3758/BRM.42.2.381):
    Walks tokens; whenever running TTR drops at-or-below `threshold`
    (default 0.72), counts a factor and resets. Leftover tokens at the
    end contribute a proportional partial factor,
    (1 - TTR_remaining) / (1 - threshold). Final score is the mean of
    forward and reverse passes. Length-corrected.

    Implements the published procedure (equivalent, not verified
    bit-for-bit against the authors' original tool). Edge convention:
    a text that never completes any factor — fully unique in both
    directions, e.g. very short texts — returns NaN (MTLD undefined),
    never a fabricated finite value.
"""

import math

_DEFAULT_THRESHOLD = 0.72


def compute_ttr(tokens: list[str]) -> float:
    """Return type-token ratio. NaN for empty input. Case-insensitive."""
    if not tokens:
        return float('nan')
    lowered = [t.lower() for t in tokens]
    return len(set(lowered)) / len(lowered)


def compute_mtld(tokens: list[str], threshold: float = _DEFAULT_THRESHOLD) -> float:
    """Return MTLD (forward+reverse mean). NaN for empty or unscoreable input."""
    if not tokens:
        return float('nan')
    lowered = [t.lower() for t in tokens]
    fwd = _mtld_one_direction(lowered, threshold)
    rev = _mtld_one_direction(list(reversed(lowered)), threshold)
    if math.isnan(fwd) or math.isnan(rev):
        return float('nan')
    return (fwd + rev) / 2.0


def _mtld_one_direction(tokens: list[str], threshold: float) -> float:
    if not tokens:
        return float('nan')
    factor_count = 0
    seen: set[str] = set()
    running_total = 0
    for tok in tokens:
        running_total += 1
        seen.add(tok)
        ttr = len(seen) / running_total
        if ttr <= threshold:
            factor_count += 1
            seen.clear()
            running_total = 0
    # Proportional partial factor for the trailing segment (McCarthy &
    # Jarvis 2010): how far the leftover TTR has descended toward the
    # threshold, as a fraction of a full factor. An all-unique tail
    # (TTR 1.0) contributes 0 — not a fabricated full factor.
    if running_total > 0:
        ttr = len(seen) / running_total
        factor_count += (1.0 - ttr) / (1.0 - threshold)
    if factor_count == 0:
        return float('nan')
    return len(tokens) / factor_count
