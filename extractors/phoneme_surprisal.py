"""Context-independent phoneme surprisal + cohort entropy.

Frequency-weighted cohort model over a phoneme-prefix trie (Marslen-Wilson
cohort; surprisal/entropy per Brodbeck et al. 2018/2022, Gillis et al. 2021).
Each trie node caches Z = Σf(w) and SFL = Σf(w)·log2 f(w) over the words whose
pronunciation passes through it, so per-phoneme scoring is O(1):

    surprisal(φk) = log2(Z_before) − log2(Z_after)   # Z_before(φ1) = Z_root
    entropy(φk)   = log2(Z_after) − SFL_after/Z_after # entropy of the AFTER cohort

See docs/superpowers/specs/2026-07-06-phoneme-surprisal-design.md.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field


@dataclass
class _Node:
    z: float = 0.0            # Σ f(w) over words through this node
    sfl: float = 0.0          # Σ f(w)·log2 f(w)
    children: dict[str, "_Node"] = field(default_factory=dict)


class CohortTrie:
    """Frequency-weighted phoneme-prefix trie with O(1) per-phoneme scoring."""

    def __init__(self, root: _Node, f_min: float):
        self._root = root
        self.f_min = f_min
        self.z_root = root.z
        # Surprisal of the single rarest lexicon word (max possible surprisal).
        self.cap = -math.log2(f_min / root.z) if root.z > 0 and f_min > 0 else 0.0

    @staticmethod
    def _entropy(node: _Node) -> float:
        if node.z <= 0:
            return 0.0
        h = math.log2(node.z) - node.sfl / node.z
        return 0.0 if h < 0 else h    # clamp -0.0 / tiny negatives from fp error

    def walk(self, phones: list[str]) -> tuple[list[float], list[float]]:
        surprisals: list[float] = []
        entropies: list[float] = []
        node = self._root
        off_lexicon = False
        for ph in phones:
            if off_lexicon or ph not in node.children:
                off_lexicon = True
                surprisals.append(self.cap)
                entropies.append(0.0)
                continue
            nxt = node.children[ph]
            surprisals.append(math.log2(node.z) - math.log2(nxt.z))
            entropies.append(self._entropy(nxt))
            node = nxt
        return surprisals, entropies


def build_cohort_trie(entries: list[tuple[tuple[str, ...], float]]) -> CohortTrie:
    """Build a CohortTrie from (phone_tuple, freq) entries.

    freq == 0 is floored to the minimum nonzero freq so every dictionary word
    stays in the cohort (the OOV cap is reserved for out-of-dictionary paths).
    """
    nonzero = [f for _, f in entries if f > 0]
    f_min = min(nonzero) if nonzero else 1.0

    root = _Node()
    for phones, freq in entries:
        f = freq if freq > 0 else f_min
        flogf = f * math.log2(f)
        node = root
        node.z += f
        node.sfl += flogf
        for ph in phones:
            node = node.children.setdefault(ph, _Node())
            node.z += f
            node.sfl += flogf
    return CohortTrie(root, f_min)


def phoneme_scores(trie: CohortTrie, phones: list[str], *,
                   context_independent: bool = True,
                   first_phoneme_empty_context: bool = True
                   ) -> tuple[list[float], list[float]]:
    """Per-phoneme (surprisal, entropy) for one word's aligned phoneme sequence.

    `context_independent` / `first_phoneme_empty_context` are guard flags: only
    their default (True) is implemented. LM-reweighted / non-empty-context modes
    are a future batch — a non-default value raises so callers can't silently get
    the wrong model. Keeping surprisal context-independent is what keeps this
    feature decorrelated from the GPT-2 word-level Surprisal feature.
    """
    if not context_independent:
        raise NotImplementedError(
            "Only context-independent (unigram-prior) cohort surprisal is implemented.")
    if not first_phoneme_empty_context:
        raise NotImplementedError(
            "Only empty-context (word-initial prior) first-phoneme scoring is implemented.")
    return trie.walk(list(phones))


_SILENCE_PHONES = {"sil", "sp", "spn", ""}
_STRESS_DIGITS = ("0", "1", "2")

DEFAULT_DICT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "mfa_data", "pretrained_models", "dictionary", "english_us_arpa.dict",
)

_TRIE_CACHE: dict[tuple, CohortTrie] = {}


def _strip_stress(ph: str) -> str:
    return ph[:-1] if ph and ph[-1] in _STRESS_DIGITS else ph


def load_mfa_lexicon(dict_path: str, *, strip_stress: bool = True,
                     primary_only: bool = True
                     ) -> list[tuple[tuple[str, ...], float]]:
    """Parse the MFA .dict into (phone_tuple, frequency) entries.

    The phone string is always the LAST tab-separated field; MFA special tokens
    ([...]/<...>) and silence/noise phones are skipped. Frequencies come from
    wordfreq (SUBTLEX-US); f==0 words are kept and floored later by the trie.
    """
    from wordfreq import word_frequency

    seen: set[str] = set()
    entries: list[tuple[tuple[str, ...], float]] = []
    with open(dict_path, "r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            word = parts[0]
            if not word or word[0] in "[<":
                continue
            if primary_only and word in seen:
                continue
            phones = [p for p in parts[-1].split() if p not in _SILENCE_PHONES]
            if strip_stress:
                phones = [_strip_stress(p) for p in phones]
            if not phones:
                continue
            seen.add(word)
            entries.append((tuple(phones), word_frequency(word, "en")))
    return entries


def get_cohort_trie(dict_path: str = DEFAULT_DICT_PATH, *,
                    strip_stress: bool = True, primary_only: bool = True) -> CohortTrie:
    """Memoized trie build (parsing 200k dict entries once per process)."""
    key = (dict_path, strip_stress, primary_only)
    trie = _TRIE_CACHE.get(key)
    if trie is None:
        trie = build_cohort_trie(
            load_mfa_lexicon(dict_path, strip_stress=strip_stress, primary_only=primary_only))
        _TRIE_CACHE[key] = trie
    return trie
