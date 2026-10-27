"""License-clean composite psychological scores: Authentic, Analytic, Tone, Clout.

This is the replacement for the retired `extractors/liwc_summary.py`. It
implements the same four publicly-documented Pennebaker et al. (2014)
summary indices, but sources its category proportions from the
spaCy-parsed token DataFrame (POS + morphology, already produced by the
pipeline) and from a free sentiment lexicon (VADER) — **no LIWC
dictionary material is required or used.**

Why the switch: the LIWC-22 EULA (https://www.liwc.app/help/eula) forbids
using LIWC dictionary material outside LIWC's own software, so
SensiExtractor can no longer ship the `.dic`-based implementation.

Provenance and scale (important):
- `analytic` follows the **Categorical-Dynamic Index (CDI)** of Pennebaker
  et al. (2014) — same eight function-word classes and signs — but as a
  *simplified variant*: the published CDI standardizes each category
  (z-scores) with equal weights, while this uses raw percentages with the
  fractional weights below.
- `clout`, `authentic`, `tone` mirror the LIWC2015 / LIWC-22 summary
  variables, whose official algorithms are **proprietary** and output
  corpus-standardized percentiles (1–99). The linear forms here are an
  in-house approximation that reproduces only the *direction* of each
  variable; they are NOT the LIWC algorithm.
- Therefore these 0–100 scores are NOT comparable to official LIWC
  numbers — use them only as relative descriptors within a dataset.

Citations:
- Pennebaker, Chung, Frazee, Lavergne, Beaver (2014). When small words
  foretell academic success. PLOS ONE, 9(12), e115844 — the CDI.
- Pennebaker, Boyd, Jordan, Blackburn (2015). The Development and
  Psychometric Properties of LIWC2015 — the summary variables.
- Hutto & Gilbert (2014). VADER. ICWSM — the valence lexicon for Tone.
"""

import math

import pandas as pd


EXPECTED_KEYS = {'authentic', 'analytic', 'tone', 'clout'}

# Tokens that don't count toward the word total (LIWC counts words only).
_NON_WORD_UPOS = {'PUNCT', 'SYM', 'SPACE', 'X'}

# Closed-class English pronoun sets (object/possessive/contraction forms
# included). These are just English pronouns — not LIWC material.
_I_SET = {"i", "me", "my", "mine", "myself", "i'm", "i'll", "i'd", "i've"}
_WE_SET = {"we", "us", "our", "ours", "ourselves", "we're", "we'll", "we'd", "we've"}
_YOU_SET = {"you", "your", "yours", "yourself", "yourselves", "u",
            "you're", "you'll", "you'd", "you've", "thou", "thee", "thy"}
_SHEHE_SET = {"he", "him", "his", "himself", "she", "her", "hers", "herself",
              "he's", "she's", "he'll", "she'll", "he'd", "she'd"}
_THEY_SET = {"they", "them", "their", "theirs", "themselves",
             "they're", "they'll", "they'd", "they've"}
_PERSONAL = _I_SET | _WE_SET | _YOU_SET | _SHEHE_SET | _THEY_SET

_ARTICLES = {"a", "an", "the"}
_NEGATE = {"no", "not", "never", "none", "nobody", "nothing", "neither",
           "nor", "nowhere", "cannot", "n't", "without"}
_DISCREP = {"should", "would", "could", "ought", "must", "need",
            "should've", "would've", "could've", "shoulda", "woulda", "coulda",
            "hopefully"}
_FUTURE = {"will", "shall", "gonna", "'ll"}

_VALENCE_POS_THRESHOLD = 0.0
_VALENCE_NEG_THRESHOLD = 0.0


def compute_psych_composite(
    token_df: pd.DataFrame,
    *,
    valence_lexicon: dict[str, float] | None = None,
) -> dict[str, float]:
    """Return {'authentic', 'analytic', 'tone', 'clout'} as 0–100 floats.

    Args:
        token_df: spaCy-parsed token DataFrame. Requires at least `token`
            and `upos`; uses `lemma`, `feats`, `dep_rel` when present.
        valence_lexicon: word → valence float (>0 positive, <0 negative).
            Defaults to VADER's lexicon, loaded lazily. If VADER is not
            installed, Tone degrades to neutral (50) rather than raising.

    Empty / word-less input returns all four as NaN.
    """
    if token_df is None or len(token_df) == 0:
        return {k: math.nan for k in EXPECTED_KEYS}

    if valence_lexicon is None:
        valence_lexicon = _load_vader_lexicon()
    # No usable valence lexicon (e.g. vaderSentiment not installed) → Tone
    # cannot be computed. Emit NaN rather than a misleading neutral 50.
    valence_available = bool(valence_lexicon)

    counts, n_words = _category_counts(token_df, valence_lexicon)
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

    # Clout — positive: 2nd/3rd person; negative: 1st-person singular + tentative
    clout = (
        50.0
        + pct('we') * 0.8
        + pct('you') * 0.8
        + pct('shehe') * 0.5
        + pct('they') * 0.5
        - pct('i') * 0.8
        - pct('ipron') * 0.4
    )

    # Authentic — first-person singular + present-focus + cognitive mechanisms up
    authentic = (
        50.0
        + pct('i') * 0.8
        + pct('discrep') * 0.4
        + pct('focuspresent') * 0.2
        - pct('we') * 0.4
        - pct('focusfuture') * 0.2
    )

    # Tone — positive emotion minus negative emotion, centred at 50 (neutral).
    # NaN when no valence lexicon is available (not a fake-neutral 50).
    tone = _clamp(50.0 + pct('posemo') - pct('negemo')) if valence_available else math.nan

    return {
        'authentic': _clamp(authentic),
        'analytic': _clamp(analytic),
        'tone': tone,
        'clout': _clamp(clout),
    }


def _clamp(x: float) -> float:
    return max(0.0, min(100.0, x))


def _category_counts(
    df: pd.DataFrame, valence: dict[str, float]
) -> tuple[dict[str, int], int]:
    """Tally Pennebaker-category memberships across word tokens."""
    counts: dict[str, int] = {}
    n_words = 0

    has_feats = 'feats' in df.columns
    has_lemma = 'lemma' in df.columns
    has_dep = 'dep_rel' in df.columns

    def bump(cat: str) -> None:
        counts[cat] = counts.get(cat, 0) + 1

    for row in df.itertuples(index=False):
        upos = str(getattr(row, 'upos', '') or '')
        if upos in _NON_WORD_UPOS:
            continue
        tok_raw = getattr(row, 'token', '')
        if not isinstance(tok_raw, str) or not tok_raw.strip():
            continue
        n_words += 1

        t = tok_raw.lower()
        lemma = str(getattr(row, 'lemma', '') or '').lower() if has_lemma else ''
        feats = str(getattr(row, 'feats', '') or '') if has_feats else ''
        dep = str(getattr(row, 'dep_rel', '') or '') if has_dep else ''

        # Grammatical categories from POS / morphology
        if t in _ARTICLES:
            bump('article')
        if upos == 'ADP':
            bump('prep')
        if upos == 'PRON':
            bump('pronoun')
            if t in _PERSONAL:
                if t in _I_SET:
                    bump('i')
                elif t in _WE_SET:
                    bump('we')
                elif t in _YOU_SET:
                    bump('you')
                elif t in _SHEHE_SET:
                    bump('shehe')
                elif t in _THEY_SET:
                    bump('they')
            else:
                bump('ipron')
        if upos == 'AUX':
            bump('auxverb')
        if upos == 'ADV':
            bump('adverb')
        if upos in ('CCONJ', 'SCONJ'):
            bump('conj')

        # Negation: lexical, dependency, or morphological
        if t in _NEGATE or dep == 'neg' or 'Polarity=Neg' in feats:
            bump('negate')

        # Discrepancy / tentative modals
        if t in _DISCREP or lemma in _DISCREP:
            bump('discrep')

        # Temporal focus from verb tense / future markers
        if 'Tense=Pres' in feats:
            bump('focuspresent')
        if t in _FUTURE or lemma in {'will', 'shall'}:
            bump('focusfuture')

        # Emotional valence from the sentiment lexicon
        val = valence.get(t)
        if val is None and lemma:
            val = valence.get(lemma)
        if val is not None:
            if val > _VALENCE_POS_THRESHOLD:
                bump('posemo')
            elif val < _VALENCE_NEG_THRESHOLD:
                bump('negemo')

    return counts, n_words


def _load_vader_lexicon() -> dict[str, float]:
    """Return VADER's word→valence lexicon, or {} if vaderSentiment is absent.

    Imported lazily so the package is only needed at run time (not for the
    test suite, which injects a fake lexicon).
    """
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    except ImportError:
        return {}
    try:
        return dict(SentimentIntensityAnalyzer().lexicon)
    except Exception:  # pragma: no cover - defensive
        return {}
