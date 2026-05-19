"""Readability metrics — Flesch Reading Ease and Flesch-Kincaid Grade Level.

Operates on raw text (preserves the sentence structure that the formulas need).
Uses the `textstat` package, which implements both metrics with the standard
published coefficients.

Empty or whitespace-only text returns NaN for both values, since the formulas
divide by sentence count and word count.
"""

import math


def compute_readability(text: str) -> dict[str, float]:
    """Return {'flesch_reading_ease': float, 'flesch_kincaid_grade': float}."""
    if not text or not text.strip():
        return {
            'flesch_reading_ease': math.nan,
            'flesch_kincaid_grade': math.nan,
        }

    import textstat  # lazy import — heavy dependency
    try:
        ease = float(textstat.flesch_reading_ease(text))
        grade = float(textstat.flesch_kincaid_grade(text))
    except Exception:
        ease = math.nan
        grade = math.nan
    return {
        'flesch_reading_ease': ease,
        'flesch_kincaid_grade': grade,
    }
