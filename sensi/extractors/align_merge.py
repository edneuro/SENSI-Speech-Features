"""Merge MFA word alignment onto the spaCy token DataFrame.

Adds Transcript / Onset / End columns to a token-level DataFrame by pairing
each non-punctuation token with the MFA word interval it was spoken as.

Why this is not a simple join: spaCy and MFA tokenize differently.
    - contractions: spaCy "do" + "n't"  vs  MFA "don't"
    - numbers:      spaCy "1,500"       vs  MFA "one thousand five hundred"
                    (numbers are spelled out by sensi.utils.transcript_prep before
                    alignment, so the audio-side words are the spelled forms)
    - hyphens:      spaCy "mother-in-law" vs MFA "mother" "in" "law"

The pairing is MONOTONIC: both sequences are consumed strictly left to
right (difflib anchors on exact matches; mismatched regions are repaired
locally). A token that cannot be paired gets NaN — it can never take the
timing of a same-form word elsewhere in the document, which is the failure
mode of a global "first unused match" scan.

Repair rules inside a mismatched region, in order:
    1. join    — k consecutive tokens whose concatenation equals one MFA
                 word (apostrophes/hyphens ignored): each of the k token
                 rows gets that word's full [Onset, End] interval.
    2. expand  — one token (digits are spelled out first) equals the
                 concatenation of m consecutive MFA words: the token gets
                 Onset of the first and End of the last.
    3. pair    — the region has equally many tokens and MFA words: pair
                 them positionally (word-form variants, e.g. "&" vs "and").
    4. drain   — otherwise the longer side advances (token → NaN,
                 or MFA word left unconsumed).
"""
from __future__ import annotations

import difflib
import re
import sys

import numpy as np
import pandas as pd

from sensi.utils.transcript_prep import spell_out_numbers

# Fraction of words that must merge before we stop warning the user.
_WARN_BELOW = 0.95
# Only warn for documents long enough that the fraction is meaningful.
_WARN_MIN_WORDS = 20


def _norm(word: str) -> str:
    """Lowercase; strip everything but word chars, apostrophes, hyphens."""
    return re.sub(r"[^\w'-]", '', str(word).lower())


def _squash(norm: str) -> str:
    """Collapse a normalized form for cross-tokenization comparison."""
    return norm.replace("'", '').replace('-', '')


def _spelled_target(raw_token: str, norm: str) -> str:
    """Squashed comparison target for a token; digits are spelled out."""
    if any(ch.isdigit() for ch in raw_token):
        words = spell_out_numbers(str(raw_token))
        return ''.join(_squash(_norm(w)) for w in words.split())
    return _squash(norm)


def merge_alignment(token_df: pd.DataFrame, align_df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of token_df with Transcript / Onset / End appended.

    Args:
        token_df: token-level DataFrame with 'token' and 'upos' columns
                  (punctuation rows get NaN and consume nothing).
        align_df: MFA word tier with 'Transcript' / 'Onset' / 'End', in
                  temporal order (sensi.extractors.aligner output).

    Onset/End come back as float64 (NaN where unmatched) so downstream
    arithmetic never meets object-dtype None columns.
    """
    out = token_df.copy()
    n = len(out)
    onsets = np.full(n, np.nan, dtype=np.float64)
    ends = np.full(n, np.nan, dtype=np.float64)
    transcripts = np.full(n, None, dtype=object)

    # Word tokens that can participate in matching: every non-PUNCT row.
    # Symbols whose normalized form is empty (e.g. "&", "—") keep their raw
    # form as the matching key: an identical MFA word still pairs exactly,
    # a substituted word (MFA "&" spoken as "and") pairs positionally, and
    # a symbol MFA dropped drains to NaN.
    tok_items: list[tuple[int, str, str]] = []   # (row position, raw, norm)
    upos = out['upos'].tolist() if 'upos' in out.columns else [''] * n
    for pos, (raw, pos_tag) in enumerate(zip(out['token'].tolist(), upos)):
        if pos_tag == 'PUNCT':
            continue
        norm = _norm(raw) or str(raw).lower().strip()
        if norm:
            tok_items.append((pos, str(raw), norm))

    ali_norms = [_norm(w) or str(w).lower().strip()
                 for w in align_df['Transcript'].tolist()]
    ali_onsets = align_df['Onset'].tolist()
    ali_ends = align_df['End'].tolist()
    ali_words = align_df['Transcript'].tolist()

    matched = 0

    def _assign(row_pos: int, onset, end, transcript) -> None:
        nonlocal matched
        onsets[row_pos] = onset
        ends[row_pos] = end
        transcripts[row_pos] = transcript
        matched += 1

    tok_norms = [t[2] for t in tok_items]
    sm = difflib.SequenceMatcher(None, tok_norms, ali_norms, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'equal':
            for off in range(i2 - i1):
                row_pos = tok_items[i1 + off][0]
                j = j1 + off
                _assign(row_pos, ali_onsets[j], ali_ends[j], ali_words[j])
            continue
        _repair(tok_items[i1:i2], list(range(j1, j2)),
                ali_norms, ali_onsets, ali_ends, ali_words, _assign)

    out['Transcript'] = transcripts
    out['Onset'] = onsets
    out['End'] = ends

    total = len(tok_items)
    print(f'[merge] aligned {matched}/{total} words to MFA timing')
    if total >= _WARN_MIN_WORDS and matched < _WARN_BELOW * total:
        print(f'[merge] WARNING: {total - matched} of {total} words have no '
              'acoustic timing — check that the transcript matches the audio.',
              file=sys.stderr)
    return out


def _repair(tokens: list[tuple[int, str, str]], ali_idx: list[int],
            ali_norms, ali_onsets, ali_ends, ali_words, assign) -> None:
    """Locally pair a mismatched region; unpaired tokens stay NaN."""
    ti, aj = 0, 0
    while ti < len(tokens) and aj < len(ali_idx):
        j = ali_idx[aj]

        # 1. join: k tokens -> one MFA word (contractions).
        ali_sq = _squash(ali_norms[j])
        k = _match_join(tokens, ti, ali_sq)
        if k:
            for off in range(k):
                assign(tokens[ti + off][0], ali_onsets[j], ali_ends[j], ali_words[j])
            ti += k
            aj += 1
            continue

        # 2. expand: one token -> m MFA words (numbers, hyphenated words).
        _, raw, norm = tokens[ti]
        target = _spelled_target(raw, norm)
        m = _match_expand(target, ali_idx, aj, ali_norms)
        if m:
            j_last = ali_idx[aj + m - 1]
            assign(tokens[ti][0], ali_onsets[j], ali_ends[j_last],
                   ' '.join(str(ali_words[ali_idx[aj + off]]) for off in range(m)))
            ti += 1
            aj += m
            continue

        # 3. pair: equally-sized remainder -> positional 1:1.
        if len(tokens) - ti == len(ali_idx) - aj:
            assign(tokens[ti][0], ali_onsets[j], ali_ends[j], ali_words[j])
            ti += 1
            aj += 1
            continue

        # 4. drain the longer side; a skipped token stays NaN.
        if len(tokens) - ti >= len(ali_idx) - aj:
            ti += 1
        else:
            aj += 1


def _match_join(tokens, ti: int, ali_squashed: str) -> int:
    """Smallest k >= 2 whose concatenated token forms equal one MFA word."""
    accum = ''
    for k in range(0, min(4, len(tokens) - ti)):
        accum += _squash(tokens[ti + k][2])
        if len(accum) > len(ali_squashed) or not ali_squashed.startswith(accum):
            return 0
        if accum == ali_squashed and k >= 1:
            return k + 1
    return 0


def _match_expand(target: str, ali_idx, aj: int, ali_norms) -> int:
    """Number of consecutive MFA words whose concatenation equals target."""
    if not target:
        return 0
    accum = ''
    m = 0
    while aj + m < len(ali_idx) and len(accum) < len(target):
        nxt = _squash(ali_norms[ali_idx[aj + m]])
        if not nxt or not target.startswith(accum + nxt):
            return 0
        accum += nxt
        m += 1
    return m if accum == target else 0
