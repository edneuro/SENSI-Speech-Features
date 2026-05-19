"""Lexical diversity measures (TTR, MTLD).

Operates on a list of lowercased content-word tokens.

TTR (Type-Token Ratio):
    Unique tokens / total tokens. Simple but length-biased.

MTLD (Measure of Textual Lexical Diversity, McCarthy & Jarvis 2010):
    Walks tokens; whenever running TTR drops at-or-below `threshold`
    (default 0.72), counts a factor and resets. Final score is the mean
    of forward and reverse passes. Length-corrected.
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
    # Partial factor for trailing tokens that didn't cross the threshold
    if running_total > 0:
        ttr = len(seen) / running_total
        if ttr < 1.0:
            partial = (1.0 - ttr) / (1.0 - threshold)
        elif running_total > 1:
            # All-unique trailing segment with >1 token counts as 1 partial factor
            partial = 1.0
        else:
            partial = 0.0
        factor_count += partial
    if factor_count == 0:
        return float('nan')
    return len(tokens) / factor_count
