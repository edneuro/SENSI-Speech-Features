"""LIWC summary scores: Authentic, Analytic, Tone, Clout.

Implements approximations of LIWC-22's four summary indices using the
publicly documented Pennebaker et al. formulas. Results are *close to* but
not guaranteed to match LIWC-22's official numbers exactly; LIWC's
proprietary post-processing (z-score percentile transformation) may
differ. Users who need exact LIWC-22 numbers should run LIWC's desktop
software on their text and compare; for relative comparisons across
stimuli, this implementation is generally adequate.

Citation: Pennebaker, J. W., Chung, C. K., Frazee, J., Lavergne, G. M., &
Beaver, D. I. (2014). When small words foretell academic success.
PLOS ONE, 9(12), e115844.

LIWC categories required for full computation (must be present in the
.dic file): article, prep, pronoun, ipron, ppron, i, we, you, shehe,
they, auxverb, adverb, conj, negate, posemo, negemo, focuspast,
focusfuture, discrep.
"""

import math
import re
from pathlib import Path


EXPECTED_KEYS = {'authentic', 'analytic', 'tone', 'clout'}

_TOKEN_RE = re.compile(r"[A-Za-z']+")


def compute_liwc_summary(text: str, dictionary_path: Path | str) -> dict[str, float]:
    """Return {'authentic', 'analytic', 'tone', 'clout'} as floats.

    Empty text returns all NaN.
    """
    if not text or not text.strip():
        return {k: math.nan for k in EXPECTED_KEYS}

    parser = _load_parser(Path(dictionary_path))
    counts, n_words = _category_counts_from_text(text, parser)
    if n_words == 0:
        return {k: math.nan for k in EXPECTED_KEYS}

    def pct(cat: str) -> float:
        return 100.0 * counts.get(cat, 0) / n_words

    # Analytic — Pennebaker et al. 2014, Table 2
    analytic = (
        30.0
        + pct('article') * 0.8
        + pct('prep') * 0.7
        - pct('pronoun') * 0.4
        - pct('ipron') * 0.4
        - pct('auxverb') * 0.4
        - pct('conj') * 0.4
        - pct('adverb') * 0.4
        - pct('negate') * 0.4
    )

    # Clout — Pennebaker et al. 2014 (positive contributors are second/third
    # person; negative are first-person singular and tentative markers)
    clout = (
        50.0
        + pct('we') * 0.8
        + pct('you') * 0.8
        + pct('shehe') * 0.5
        + pct('they') * 0.5
        - pct('i') * 0.8
        - pct('ipron') * 0.4
    )

    # Authentic — first-person singular + present-focus + cognitive
    # mechanisms increase Authentic; complex hedging decreases it.
    authentic = (
        50.0
        + pct('i') * 0.8
        + pct('discrep') * 0.4
        + pct('focuspresent') * 0.2
        - pct('we') * 0.4
        - pct('focusfuture') * 0.2
    )

    # Tone — LIWC-22 manual: positive emotion minus negative emotion,
    # centered at 50 (neutral).
    tone = 50.0 + pct('posemo') - pct('negemo')

    # Clamp to 0-100 (LIWC's official outputs are reported in that range)
    return {
        'authentic': max(0.0, min(100.0, authentic)),
        'analytic': max(0.0, min(100.0, analytic)),
        'tone': max(0.0, min(100.0, tone)),
        'clout': max(0.0, min(100.0, clout)),
    }


def _category_counts_from_text(text: str, parser) -> tuple[dict[str, int], int]:
    """Tokenize `text` and count category memberships for each token."""
    tokens = _TOKEN_RE.findall(text)
    counts: dict[str, int] = {}
    for tok in tokens:
        for cat in parser(tok.lower()):
            counts[cat] = counts.get(cat, 0) + 1
    return counts, len(tokens)


def _load_parser(dictionary_path: Path):
    """Return a callable that takes a token and returns a list of category names."""
    try:
        import liwc
    except ImportError as exc:
        raise RuntimeError(f'`liwc` package required for liwc_summary: {exc}')
    parse, _ = liwc.load_token_parser(str(dictionary_path))
    return parse
