# Phoneme Surprisal + Cohort Entropy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add context-independent phoneme surprisal and cohort entropy as phoneme-onset-aligned TRF regressors — emitted both as per-phoneme CSV columns and as an impulse-train `.mat` sidecar (with `phoneme_onset` + `word_onset` controls) — reusing `wordfreq` and the MFA dictionary, with no new dependencies and no change to the GPT-2 word feature.

**Architecture:** A pure-function extractor (`extractors/phoneme_surprisal.py`) builds a frequency-weighted cohort trie from the MFA pronunciation dictionary once per run, then scores each aligned phoneme in O(1). `main.run_word_level_batch` augments the existing `phonemes_df` with two columns and rasterizes it to a `.mat`. Wiring mirrors the GPT-2 surprisal feature (planner Stage + Cluster) and the acoustic level (`.mat` gated by an opt-in output key).

**Tech Stack:** Python 3.10, pandas, numpy, `wordfreq` (SUBTLEX-US), `scipy.io.savemat`, `soundfile` (duration), PySide6 (GUI), pytest / pytest-qt.

## Global Constraints

- **Design spec:** `docs/superpowers/specs/2026-07-06-phoneme-surprisal-design.md` — authoritative; read §2/§2b/§3 before Task 1.
- **No new runtime dependency.** Lexicon = `mfa_data/pretrained_models/dictionary/english_us_arpa.dict`; frequency = `wordfreq.word_frequency(w, "en")`.
- **Do NOT modify** `extractors/surprisal.py` (GPT-2 word feature) in any way.
- **Verified cohort math (spec §2):** `surprisal(φk) = log2(Z_before) − log2(Z_after)` (first phoneme `Z_before = Z_root`); `entropy(φk) = log2(Z_after) − SFL_after/Z_after` (evaluated at the **after** node). `Z = Σf(w)`, `SFL = Σf(w)·log2 f(w)`.
- **OOV path:** stepping off the trie → `surprisal = cap = −log2(f_min/Z_root)`, `entropy = 0`.
- **Defaults:** `strip_stress=True`, `primary_only=True`, `analysis_sr=128`; guard flags `context_independent=True`, `first_phoneme_empty_context=True` (non-default → `NotImplementedError`).
- **Env:** `conda activate ./env` before running anything. If pytest dies on `ModuleNotFoundError`, run `pip install -r requirements.txt` (env out of sync — see memory `project-num2words-env-gap`).
- **Prefs are at v15** (`utils/preferences.py`); this batch adds **v16** additively.
- **Commit after every task.** Commit directly to `main`. End messages with the `Co-Authored-By` trailer.
- **Run the full suite** (`python -m pytest`) at the end of wiring tasks; baseline was 503 passed, 2 skipped.

---

### Task 1: Cohort trie + phoneme scoring (the verified math)

**Files:**
- Create: `extractors/phoneme_surprisal.py`
- Test: `tests/extractors/test_phoneme_surprisal.py`

**Interfaces:**
- Produces:
  - `build_cohort_trie(entries: list[tuple[tuple[str, ...], float]]) -> CohortTrie` — `entries` are `(phone_tuple, freq)`; `freq==0` is floored to the min nonzero freq. `CohortTrie` exposes `.z_root: float`, `.f_min: float`, `.cap: float`, and `.walk(phones) -> tuple[list[float], list[float]]` (surprisals, entropies).
  - `phoneme_scores(trie, phones, *, context_independent=True, first_phoneme_empty_context=True) -> tuple[list[float], list[float]]`.

- [ ] **Step 1: Write the failing test — the spec's hand-computable check + OOV + guard flags**

```python
# tests/extractors/test_phoneme_surprisal.py
import math
import pytest
from extractors.phoneme_surprisal import build_cohort_trie, phoneme_scores

# Lexicon {cat:3, cab:1, dog:2} with stress-stripped ARPABET pronunciations.
_ENTRIES = [
    (("K", "AE", "T"), 3.0),
    (("K", "AE", "B"), 1.0),
    (("D", "AO", "G"), 2.0),
]


def test_cat_surprisal_matches_hand_computation():
    trie = build_cohort_trie(_ENTRIES)
    surp, _ = phoneme_scores(trie, ["K", "AE", "T"])
    assert surp == pytest.approx([0.585, 0.0, 0.415], abs=1e-3)


def test_cat_entropy_matches_hand_computation():
    trie = build_cohort_trie(_ENTRIES)
    _, ent = phoneme_scores(trie, ["K", "AE", "T"])
    # after-node entropy: {cat3,cab1}=0.811, {cat3,cab1}=0.811, {cat3}=0.0
    assert ent == pytest.approx([0.811, 0.811, 0.0], abs=1e-3)


def test_first_phoneme_uses_empty_context_root():
    # surprisal(K) = log2(6) - log2(4): Z_root=6 (all words), after-K Z=4.
    trie = build_cohort_trie(_ENTRIES)
    surp, _ = phoneme_scores(trie, ["K", "AE", "B"])
    assert surp[0] == pytest.approx(math.log2(6) - math.log2(4), abs=1e-6)


def test_oov_path_caps_surprisal_and_zeros_entropy():
    trie = build_cohort_trie(_ENTRIES)
    # 'K AE Z' leaves the trie at Z (only T/B follow AE).
    surp, ent = phoneme_scores(trie, ["K", "AE", "Z"])
    assert surp[2] == pytest.approx(trie.cap, abs=1e-9)
    assert ent[2] == 0.0
    # cap == -log2(f_min / z_root) == -log2(1/6)
    assert trie.cap == pytest.approx(-math.log2(1.0 / 6.0), abs=1e-9)


def test_guard_flags_raise_on_non_default():
    trie = build_cohort_trie(_ENTRIES)
    with pytest.raises(NotImplementedError):
        phoneme_scores(trie, ["K"], context_independent=False)
    with pytest.raises(NotImplementedError):
        phoneme_scores(trie, ["K"], first_phoneme_empty_context=False)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/extractors/test_phoneme_surprisal.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'extractors.phoneme_surprisal'`.

- [ ] **Step 3: Implement the trie + scorer**

```python
# extractors/phoneme_surprisal.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/extractors/test_phoneme_surprisal.py -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add extractors/phoneme_surprisal.py tests/extractors/test_phoneme_surprisal.py
git commit -m "feat(phoneme-surprisal): frequency-weighted cohort trie + scorer

Verified against the hand-computable check (cat -> surprisal [0.585,0,0.415],
entropy [0.811,0.811,0]). O(1) per phoneme via cached Z/SFL. OOV cap + entropy=0;
guard flags raise on non-default.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: MFA-dictionary lexicon loader

**Files:**
- Modify: `extractors/phoneme_surprisal.py`
- Test: `tests/extractors/test_phoneme_surprisal.py`

**Interfaces:**
- Consumes: `build_cohort_trie` (Task 1).
- Produces:
  - `load_mfa_lexicon(dict_path: str, *, strip_stress=True, primary_only=True) -> list[tuple[tuple[str, ...], float]]`.
  - `get_cohort_trie(dict_path: str, *, strip_stress=True, primary_only=True) -> CohortTrie` — memoized (module cache keyed by `(dict_path, strip_stress, primary_only)`).
  - `DEFAULT_DICT_PATH: str` — resolves `mfa_data/pretrained_models/dictionary/english_us_arpa.dict` relative to repo root.

**Context — MFA `.dict` format (verified):** TAB-separated, variable columns. The word is the FIRST field; the phone string is ALWAYS the LAST field (space-separated ARPABET). Some lines carry 4 pronunciation-probability columns between them, some don't. Multiple lines per word = variants (first-listed = primary). Skip MFA special tokens (`word` starting with `[` or `<`) and phones that are silence/noise (`sil`, `sp`, `spn`).

- [ ] **Step 1: Write the failing test with a tiny fixture dict**

```python
def test_load_mfa_lexicon_parses_variable_columns(tmp_path):
    from extractors.phoneme_surprisal import load_mfa_lexicon
    d = tmp_path / "mini.dict"
    d.write_text(
        "cat\t0.99\t0.62\t0.58\t1.07\tK AE1 T\n"   # 4 prob cols then phones
        "dog\t0.09\t0.48\t1.01\t1.0\tD AO1 G\n"      # first-listed dog variant
        "dog\t0.99\t0.41\t0.49\t1.06\tD AA1 G\n"     # second variant (dropped when primary_only)
        "'ve\tV\n"                                     # no prob columns
        "[bracketed]\t0.99\t0.18\t1.0\t1.0\tspn\n",  # special token -> skipped
        encoding="utf-8",
    )
    entries = dict((p, f) for p, f in
                   load_mfa_lexicon(str(d), strip_stress=True, primary_only=True))
    assert ("K", "AE", "T") in entries          # stress stripped
    assert ("D", "AO", "G") in entries           # first dog variant kept
    assert ("D", "AA", "G") not in entries       # second variant dropped
    assert ("V",) in entries                     # no-prob-column line parsed
    assert not any("spn" in p for p in entries)  # special token skipped
    # frequencies came from wordfreq (all > 0 for these common words)
    assert entries[("K", "AE", "T")] > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/extractors/test_phoneme_surprisal.py::test_load_mfa_lexicon_parses_variable_columns -q`
Expected: FAIL — `ImportError: cannot import name 'load_mfa_lexicon'`.

- [ ] **Step 3: Implement the loader (append to `extractors/phoneme_surprisal.py`)**

```python
import os

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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/extractors/test_phoneme_surprisal.py -q`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add extractors/phoneme_surprisal.py tests/extractors/test_phoneme_surprisal.py
git commit -m "feat(phoneme-surprisal): MFA-dictionary lexicon loader + memoized trie

Parses the variable-column MFA .dict (phones = last tab field), joins wordfreq
(SUBTLEX-US) frequencies, strips stress, keeps primary pronunciation. No new dep.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: DataFrame augmentation + `.mat` rasterization

**Files:**
- Modify: `extractors/phoneme_surprisal.py`
- Test: `tests/extractors/test_phoneme_surprisal.py`

**Interfaces:**
- Consumes: `get_cohort_trie`, `phoneme_scores` (Tasks 1–2).
- Produces:
  - `add_phoneme_surprisal(phon_df, *, dict_path=DEFAULT_DICT_PATH, strip_stress=True, primary_only=True) -> pd.DataFrame` — appends `phoneme_surprisal` + `cohort_entropy` columns. Groups rows by `(doc_id, sentence_id, token_in_sentence_id)`; strips stress from each row's `Label` to match the trie; assigns scores back in row order.
  - `phoneme_surprisal_mat_vars(phon_df, duration_s, analysis_sr, config) -> dict` — builds the four impulse trains + `t`/`fs`/`config` for `scipy.io.savemat`. Word-initial = first row of each word group (`syllable_id==1 & phoneme_id==1`).

- [ ] **Step 1: Write the failing tests (augmentation + rasterization)**

```python
import numpy as np
import pandas as pd


def _cat_phon_df():
    # One word "cat", one syllable, three phonemes at 0.10/0.15/0.20 s.
    return pd.DataFrame([
        dict(doc_id=1, sentence_id=1, token_in_sentence_id=1, token="cat",
             syllable_id=1, phoneme_id=1, Label="K", Onset=0.10, End=0.15, Duration=0.05),
        dict(doc_id=1, sentence_id=1, token_in_sentence_id=1, token="cat",
             syllable_id=1, phoneme_id=2, Label="AE1", Onset=0.15, End=0.18, Duration=0.03),
        dict(doc_id=1, sentence_id=1, token_in_sentence_id=1, token="cat",
             syllable_id=1, phoneme_id=3, Label="T", Onset=0.18, End=0.20, Duration=0.02),
    ])


def test_add_phoneme_surprisal_appends_two_columns(monkeypatch):
    from extractors import phoneme_surprisal as ps
    trie = ps.build_cohort_trie([
        (("K", "AE", "T"), 3.0), (("K", "AE", "B"), 1.0), (("D", "AO", "G"), 2.0)])
    monkeypatch.setattr(ps, "get_cohort_trie", lambda *a, **k: trie)
    out = ps.add_phoneme_surprisal(_cat_phon_df())
    assert list(out["phoneme_surprisal"]) == pytest.approx([0.585, 0.0, 0.415], abs=1e-3)
    assert list(out["cohort_entropy"]) == pytest.approx([0.811, 0.811, 0.0], abs=1e-3)


def test_phoneme_surprisal_mat_vars_impulses_and_controls():
    from extractors import phoneme_surprisal as ps
    df = _cat_phon_df()
    df["phoneme_surprisal"] = [0.585, 0.0, 0.415]
    df["cohort_entropy"] = [0.811, 0.811, 0.0]
    out = ps.phoneme_surprisal_mat_vars(df, duration_s=1.0, analysis_sr=100,
                                        config={"analysis_sr": 100})
    N = 100  # round(1.0 * 100)
    for k in ("phoneme_surprisal", "cohort_entropy", "phoneme_onset", "word_onset", "t"):
        assert out[k].shape[0] == N
    assert out["fs"] == 100
    # impulses land at round(onset*sr): 0.10->10, 0.15->15, 0.18->18
    assert out["phoneme_surprisal"][10, 0] == pytest.approx(0.585, abs=1e-3)
    assert out["phoneme_surprisal"][18, 0] == pytest.approx(0.415, abs=1e-3)
    assert out["phoneme_onset"][10, 0] == 1 and out["phoneme_onset"][18, 0] == 1
    # only the first phoneme of the word is a word onset
    assert out["word_onset"][10, 0] == 1
    assert out["word_onset"][15, 0] == 0 and out["word_onset"][18, 0] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/extractors/test_phoneme_surprisal.py -q`
Expected: FAIL — `AttributeError: module ... has no attribute 'add_phoneme_surprisal'`.

- [ ] **Step 3: Implement (append to `extractors/phoneme_surprisal.py`)**

```python
_GROUP_KEYS = ["doc_id", "sentence_id", "token_in_sentence_id"]


def add_phoneme_surprisal(phon_df, *, dict_path: str = DEFAULT_DICT_PATH,
                          strip_stress: bool = True, primary_only: bool = True):
    """Append phoneme_surprisal + cohort_entropy columns to a phonemes DataFrame.

    Scores each word's aligned phoneme sequence (rows grouped by doc/sentence/
    token, in phoneme order). Row `Label`s carry ARPABET stress digits; they are
    stripped here to match the (stress-stripped) trie.
    """
    import pandas as pd  # local import keeps module import light

    df = phon_df.copy()
    if len(df) == 0:
        df["phoneme_surprisal"] = pd.Series(dtype=float)
        df["cohort_entropy"] = pd.Series(dtype=float)
        return df

    trie = get_cohort_trie(dict_path, strip_stress=strip_stress, primary_only=primary_only)
    surp_col = [None] * len(df)
    ent_col = [None] * len(df)
    positions = df.reset_index().rename(columns={"index": "_pos"})
    for _, grp in positions.groupby(_GROUP_KEYS, sort=False):
        phones = [_strip_stress(str(lbl)) if strip_stress else str(lbl)
                  for lbl in grp["Label"]]
        surp, ent = phoneme_scores(trie, phones)
        for pos, s, e in zip(grp["_pos"], surp, ent):
            surp_col[pos] = round(s, 4)
            ent_col[pos] = round(e, 4)
    df["phoneme_surprisal"] = surp_col
    df["cohort_entropy"] = ent_col
    return df


def _mround(v: float) -> int:
    """MATLAB-style round (half away from zero), matching acoustic_ts._mround."""
    import numpy as np
    return int(np.floor(np.abs(v) + 0.5) * np.sign(v))


def phoneme_surprisal_mat_vars(phon_df, duration_s: float, analysis_sr: float,
                               config: dict) -> dict:
    """Rasterize per-phoneme scores into impulse trains for scipy.io.savemat.

    Four length-N (N = round(duration*sr)) column vectors: value-modulated
    phoneme_surprisal / cohort_entropy, and unit-height phoneme_onset (every
    phoneme) / word_onset (first phoneme of each word). Impulses at
    round(onset*sr); same-sample collisions sum for value trains, stay 1 for
    controls. Plus t, fs, and a config-provenance struct.
    """
    import numpy as np

    n = int(round(duration_s * analysis_sr))
    n = max(n, 0)
    surp = np.zeros((n, 1), dtype=np.float64)
    ent = np.zeros((n, 1), dtype=np.float64)
    phon_on = np.zeros((n, 1), dtype=np.float64)
    word_on = np.zeros((n, 1), dtype=np.float64)

    for row in phon_df.itertuples(index=False):
        onset = getattr(row, "Onset", None)
        if onset is None or (isinstance(onset, float) and np.isnan(onset)):
            continue
        idx = _mround(onset * analysis_sr)
        if idx < 0 or idx >= n:
            continue
        s = getattr(row, "phoneme_surprisal", None)
        e = getattr(row, "cohort_entropy", None)
        if s is not None and not (isinstance(s, float) and np.isnan(s)):
            surp[idx, 0] += float(s)
        if e is not None and not (isinstance(e, float) and np.isnan(e)):
            ent[idx, 0] += float(e)
        phon_on[idx, 0] = 1.0
        if getattr(row, "syllable_id", None) == 1 and getattr(row, "phoneme_id", None) == 1:
            word_on[idx, 0] = 1.0

    return {
        "phoneme_surprisal": surp,
        "cohort_entropy": ent,
        "phoneme_onset": phon_on,
        "word_onset": word_on,
        "t": (np.arange(n) / analysis_sr).reshape(-1, 1).astype(np.float64),
        "fs": float(analysis_sr),
        "config": dict(config),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/extractors/test_phoneme_surprisal.py -q`
Expected: PASS (8 passed).

- [ ] **Step 5: Commit**

```bash
git add extractors/phoneme_surprisal.py tests/extractors/test_phoneme_surprisal.py
git commit -m "feat(phoneme-surprisal): DataFrame augmentation + .mat rasterization

add_phoneme_surprisal appends per-phoneme columns; phoneme_surprisal_mat_vars
builds 4 impulse trains (surprisal, entropy, phoneme_onset, word_onset) + t/fs.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Planner registry — Stage + two Clusters

**Files:**
- Modify: `gui/planner/registry.py`
- Modify: `gui/planner/plan.py` (add to `_OUTPUT_CSV_MAP`)
- Test: `tests/planner/test_plan.py`

**Interfaces:**
- Consumes: nothing new at runtime (planning reads metadata only).
- Produces: a `phoneme_surprisal` stage and two clusters that make `plan_from_prefs` include the stage when the CSV cluster or the `.mat` output key is active.

- [ ] **Step 1: Write failing planner tests**

```python
# tests/planner/test_plan.py  (add these)
import copy
from gui.planner.plan import plan_from_prefs
from utils.preferences import DEFAULT_PREFS


def test_phoneme_surprisal_csv_cluster_activates_stage():
    prefs = copy.deepcopy(DEFAULT_PREFS)
    prefs["output"]["phonemes_csv"] = True
    prefs["features"]["master_phoneme_surprisal"] = True
    prefs["features"]["feat_phoneme_surprisal"] = True
    plan = plan_from_prefs(prefs)
    assert "phoneme_surprisal" in plan.stages_to_run
    assert "phonemes" in plan.output_csvs


def test_phoneme_surprisal_mat_output_activates_stage():
    prefs = copy.deepcopy(DEFAULT_PREFS)
    prefs["output"]["phoneme_surprisal_mat"] = True
    plan = plan_from_prefs(prefs)
    assert "phoneme_surprisal_mat" in plan.output_csvs
    assert "phoneme_surprisal" in plan.stages_to_run


def test_phoneme_surprisal_off_by_default():
    plan = plan_from_prefs(DEFAULT_PREFS)
    assert "phoneme_surprisal" not in plan.stages_to_run
    assert "phoneme_surprisal_mat" not in plan.output_csvs
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/planner/test_plan.py -q -k phoneme_surprisal`
Expected: FAIL (stage/cluster not registered).

- [ ] **Step 3a: Add the lazy binding + Stage in `gui/planner/registry.py`**

Add the lazy binding next to the others (after line 68, `_extract_acoustic = ...`):

```python
_add_phoneme_surprisal = _lazy("extractors.phoneme_surprisal", "add_phoneme_surprisal")
```

Add the Stage after the `syllabify_pairs`/`pcenter` sub-word stages (after line 142):

```python
    Stage(name="phoneme_surprisal",
          consumes=("phonemes_df",),
          produces=("phonemes_df.phoneme_surprisal", "phonemes_df.cohort_entropy"),
          run=_add_phoneme_surprisal),
```

- [ ] **Step 3b: Add the two Clusters** (after the P-center cluster, ~line 403)

```python
    # §2.5 Phoneme surprisal (CSV columns) — Brodbeck/Gillis cohort model
    FeatureCluster(
        name="phoneme surprisal",
        toggle_keys=("feat_phoneme_surprisal", "feat_phoneme_cohort_entropy"),
        output_csv="phonemes",
        requires_stages=("spacy_parse", "mfa_align", "syllabify_pairs", "phoneme_surprisal"),
        emits_columns=("phoneme_surprisal", "cohort_entropy"),
        master_toggle="master_phoneme_surprisal",
    ),

    # §2.5b Phoneme surprisal (.mat impulse trains) — gated by the output key
    # phoneme_surprisal_mat (Run tab), like the acoustic level. No master.
    FeatureCluster(
        name="phoneme surprisal mat",
        toggle_keys=(),
        output_csv="phoneme_surprisal_mat",
        requires_stages=("spacy_parse", "mfa_align", "syllabify_pairs", "phoneme_surprisal"),
        emits_columns=(),
    ),
```

- [ ] **Step 3c: Route the output key in `gui/planner/plan.py`**

Add to `_OUTPUT_CSV_MAP` (after line 52, `"acoustic_ts_mat": "acoustic",`):

```python
    "phoneme_surprisal_mat": "phoneme_surprisal_mat",
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/planner/test_plan.py -q -k phoneme_surprisal`
Expected: PASS (3 passed). (Schema-drift guard will still fail until Task 5 — that's expected; run only these tests here.)

- [ ] **Step 5: Commit**

```bash
git add gui/planner/registry.py gui/planner/plan.py tests/planner/test_plan.py
git commit -m "feat(phoneme-surprisal): planner stage + CSV/.mat clusters

Stage consumes phonemes_df; two clusters (phonemes CSV columns; phoneme_surprisal_mat
output-keyed like acoustic). Output key routed via _OUTPUT_CSV_MAP.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: Prefs v16 (additive) + schema-drift guard

**Files:**
- Modify: `utils/preferences.py`
- Modify: `tests/planner/test_schema_drift.py`
- Test: `tests/utils/test_preferences_migration.py`

**Interfaces:**
- Consumes: cluster keys from Task 4.
- Produces: prefs keys `master_phoneme_surprisal` (features, default True), `feat_phoneme_surprisal` / `feat_phoneme_cohort_entropy` (features, default False), `phoneme_surprisal_mat` (output, default False), and a `phoneme_surprisal` config section `{analysis_sr:128, strip_stress:True, primary_only:True}`; `SCHEMA_VERSION == 16`.

- [ ] **Step 1: Write failing migration test**

```python
# tests/utils/test_preferences_migration.py  (add)
def test_v15_prefs_migrate_to_v16_additively(tmp_path, monkeypatch):
    import json
    from utils import preferences as P
    p = tmp_path / "prefs.json"
    # A minimal v15 file: additive migration must fill the new v16 keys.
    p.write_text(json.dumps({"schema_version": 15, "features": {}, "output": {}}),
                 encoding="utf-8")
    monkeypatch.setenv("SENSI_PREFS_PATH", str(p))
    prefs = P.load_preferences()
    assert prefs["schema_version"] == 16
    assert prefs["features"]["master_phoneme_surprisal"] is True
    assert prefs["features"]["feat_phoneme_surprisal"] is False
    assert prefs["features"]["feat_phoneme_cohort_entropy"] is False
    assert prefs["output"]["phoneme_surprisal_mat"] is False
    assert prefs["phoneme_surprisal"]["analysis_sr"] == 128
    assert prefs["phoneme_surprisal"]["strip_stress"] is True
    assert prefs["phoneme_surprisal"]["primary_only"] is True
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/utils/test_preferences_migration.py -q -k v16`
Expected: FAIL (`schema_version` is 15 / `KeyError`).

- [ ] **Step 3a: Add v16 constants in `utils/preferences.py`** (after the `DEFAULT_PREFS_V15` block, before the "Active schema aliases" section ~line 733)

```python
# ---------------------------------------------------------------------------
# v16 constants
#
# Phoneme surprisal + cohort entropy (2026-07-06). Additive. Card master
# defaults ON (v15 policy: card enabled, children off); feature toggles + the
# .mat output key default OFF (opt-in, like acoustic_ts_mat). New
# phoneme_surprisal config section holds analysis_sr + the functional flags.
# ---------------------------------------------------------------------------

SCHEMA_VERSION_V16 = 16

_FEATURES_DEFAULT_V16 = {
    **_FEATURES_DEFAULT_V15,
    "master_phoneme_surprisal": True,
    "feat_phoneme_surprisal": False,
    "feat_phoneme_cohort_entropy": False,
}
_OUTPUT_DEFAULT_V16 = {**DEFAULT_PREFS_V15["output"], "phoneme_surprisal_mat": False}

DEFAULT_PREFS_V16: dict[str, Any] = {
    **DEFAULT_PREFS_V15,
    "schema_version": SCHEMA_VERSION_V16,
    "features": dict(_FEATURES_DEFAULT_V16),
    "output":   dict(_OUTPUT_DEFAULT_V16),
    "phoneme_surprisal": {
        "analysis_sr": 128,
        "strip_stress": True,
        "primary_only": True,
    },
}
```

- [ ] **Step 3b: Flip the active aliases** (lines 737–738)

```python
SCHEMA_VERSION = SCHEMA_VERSION_V16  # active version
DEFAULT_PREFS: dict[str, Any] = DEFAULT_PREFS_V16  # active defaults
```

- [ ] **Step 3c: Recognize v16 as additive** — add `SCHEMA_VERSION_V16` to the tuple at line 784:

```python
    if version in (SCHEMA_VERSION_V10, SCHEMA_VERSION_V11,
                   SCHEMA_VERSION_V12, SCHEMA_VERSION_V13, SCHEMA_VERSION_V14,
                   SCHEMA_VERSION_V15, SCHEMA_VERSION_V16):
```

- [ ] **Step 3d: Register the new section** — add `'phoneme_surprisal'` to `_CURRENT_SECTIONS` (line ~977):

```python
_CURRENT_SECTIONS = ('shared', 'features', 'predictability',
                     'semantic', 'output', 'prosody', 'lexical_norms',
                     'acoustic_ts', 'phoneme_surprisal')
```

- [ ] **Step 3e: Update the schema-drift guard** (`tests/planner/test_schema_drift.py`)

Add `"master_phoneme_surprisal"` to `_INTENTIONALLY_UNROUTED_FEATURES` (it routes via `master_toggle`), and `"phoneme_surprisal_mat"` to `_OUTPUT_KEYS_VIA_CSV_MAP` (routed via `_OUTPUT_CSV_MAP`):

```python
    # v16 phoneme surprisal — routes via FeatureCluster.master_toggle
    "master_phoneme_surprisal",
```
```python
    "phoneme_surprisal_mat",  # routed via _OUTPUT_CSV_MAP
```

- [ ] **Step 4: Run migration + drift + full prefs suites**

Run: `python -m pytest tests/utils/test_preferences_migration.py tests/utils/test_preferences.py tests/planner/test_schema_drift.py -q`
Expected: PASS (all green).

- [ ] **Step 5: Commit**

```bash
git add utils/preferences.py tests/planner/test_schema_drift.py tests/utils/test_preferences_migration.py
git commit -m "feat(phoneme-surprisal): prefs v16 (additive) + schema-drift routing

master_phoneme_surprisal (on), feat_* (off), phoneme_surprisal_mat output (off),
phoneme_surprisal config section (analysis_sr/strip_stress/primary_only).

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: emit adapter — namespace flags

**Files:**
- Modify: `gui/planner/emit.py`
- Test: `tests/planner/test_emit.py` (or wherever `to_namespace` is tested — check `tests/planner/`)

**Interfaces:**
- Consumes: `Plan` from Task 4, prefs from Task 5.
- Produces on the word-level namespace: `no_phoneme_surprisal: bool`, `no_phoneme_surprisal_mat: bool`, `phoneme_surprisal: dict`.

- [ ] **Step 1: Write failing emit test**

```python
# tests/planner/test_emit.py  (add; mirror existing to_namespace tests)
import copy
from gui.planner.plan import plan_from_prefs
from gui.planner.emit import to_namespace
from utils.preferences import DEFAULT_PREFS


def test_namespace_enables_phoneme_surprisal_when_active():
    prefs = copy.deepcopy(DEFAULT_PREFS)
    prefs["output"]["phonemes_csv"] = True
    prefs["features"]["master_phoneme_surprisal"] = True
    prefs["features"]["feat_phoneme_surprisal"] = True
    prefs["output"]["phoneme_surprisal_mat"] = True
    ns = to_namespace(plan_from_prefs(prefs), prefs, "word")
    assert ns.no_phoneme_surprisal is False
    assert ns.no_phoneme_surprisal_mat is False
    assert ns.phoneme_surprisal["analysis_sr"] == 128


def test_namespace_disables_phoneme_surprisal_by_default():
    prefs = DEFAULT_PREFS
    ns = to_namespace(plan_from_prefs(prefs), prefs, "word")
    assert ns.no_phoneme_surprisal is True
    assert ns.no_phoneme_surprisal_mat is True
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/planner/test_emit.py -q -k phoneme_surprisal`
Expected: FAIL (`AttributeError: 'Namespace' object has no attribute 'no_phoneme_surprisal'`).

- [ ] **Step 3a: Add `"phoneme_surprisal_mat"` to `_WORD_LEVEL_CSVS`** (`gui/planner/emit.py` line 18):

```python
_WORD_LEVEL_CSVS = frozenset({
    "features", "syllables", "phonemes", "sentences",
    "word2vec", "bert_word", "bert_sentence", "phoneme_surprisal_mat",
})
```

- [ ] **Step 3b: Set the flags in `to_namespace`** — after the sidecar/output flag block (after line 210, `ns.no_bert_sentence = ...`):

```python
    ns.no_phoneme_surprisal     = "phoneme_surprisal" not in word_stages
    ns.no_phoneme_surprisal_mat = "phoneme_surprisal_mat" not in plan.output_csvs
    _psr = prefs.get("phoneme_surprisal", {})
    ns.phoneme_surprisal = {
        "analysis_sr":  float(_psr.get("analysis_sr", 128)),
        "strip_stress": bool(_psr.get("strip_stress", True)),
        "primary_only": bool(_psr.get("primary_only", True)),
    }
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/planner/test_emit.py -q -k phoneme_surprisal`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add gui/planner/emit.py tests/planner/test_emit.py
git commit -m "feat(phoneme-surprisal): emit no_* flags + config to word namespace

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 7: main.py wiring — compute columns, write .mat

**Files:**
- Modify: `main.py`
- Test: `tests/integration/test_phoneme_surprisal_integration.py` (new)

**Interfaces:**
- Consumes: `ns.no_phoneme_surprisal`, `ns.no_phoneme_surprisal_mat`, `ns.phoneme_surprisal` (Task 6); `add_phoneme_surprisal`, `phoneme_surprisal_mat_vars` (Task 3).
- Produces: `phoneme_surprisal` / `cohort_entropy` columns in `<stim>_phonemes.csv`; `<stim>_phoneme_surprisal.mat`.

- [ ] **Step 1: Write failing integration test** (drives the augmentation + rasterization path with a synthetic `phon_df`, no MFA/audio)

```python
# tests/integration/test_phoneme_surprisal_integration.py
import numpy as np
import pandas as pd
import scipy.io
from extractors import phoneme_surprisal as ps


def test_mat_roundtrip_has_four_trains(tmp_path, monkeypatch):
    trie = ps.build_cohort_trie([
        (("K", "AE", "T"), 3.0), (("K", "AE", "B"), 1.0), (("D", "AO", "G"), 2.0)])
    monkeypatch.setattr(ps, "get_cohort_trie", lambda *a, **k: trie)
    df = pd.DataFrame([
        dict(doc_id=1, sentence_id=1, token_in_sentence_id=1, token="cat",
             syllable_id=1, phoneme_id=i + 1, Label=lbl, Onset=on, End=on + 0.03, Duration=0.03)
        for i, (lbl, on) in enumerate([("K", 0.10), ("AE1", 0.15), ("T", 0.18)])
    ])
    df = ps.add_phoneme_surprisal(df)
    out = ps.phoneme_surprisal_mat_vars(df, 1.0, 100, {"analysis_sr": 100})
    p = tmp_path / "s_phoneme_surprisal.mat"
    scipy.io.savemat(str(p), out)
    loaded = scipy.io.loadmat(str(p))
    for k in ("phoneme_surprisal", "cohort_entropy", "phoneme_onset", "word_onset"):
        assert loaded[k].shape == (100, 1)
    assert loaded["word_onset"][10, 0] == 1
```

- [ ] **Step 2: Run to verify pass of the extractor path** (this test exercises Task 3 code and should already pass — it guards the savemat roundtrip)

Run: `python -m pytest tests/integration/test_phoneme_surprisal_integration.py -q`
Expected: PASS. (If it fails on import of scipy, the env is out of sync — `pip install -r requirements.txt`.)

- [ ] **Step 3a: Add the `_PHONEME_COLUMN_MAP` entries** (`main.py` ~line 585) so per-column drop gates the new columns:

```python
_PHONEME_COLUMN_MAP = {
    "feat_phoneme_label":    "Label",
    "feat_phoneme_onset":    "Onset",
    "feat_phoneme_end":      "End",
    "feat_phoneme_duration": "Duration",
    "feat_phoneme_surprisal":      "phoneme_surprisal",
    "feat_phoneme_cohort_entropy": "cohort_entropy",
}
```

- [ ] **Step 3b: Add an audio-duration helper** near `_level_dir` (`main.py` ~line 671):

```python
def _audio_duration(audio_path) -> float:
    """Duration in seconds via soundfile (no full decode)."""
    import soundfile as sf
    info = sf.info(str(audio_path))
    return float(info.frames) / float(info.samplerate)
```

- [ ] **Step 3c: Widen the sub-word gate and compute/emit** in `run_word_level_batch`.

Replace the outer condition (lines 789–792) so the block also runs when phoneme surprisal (CSV or .mat) is on:

```python
            phones_df = getattr(args, '_phones_df', None)
            _want_psr = (not getattr(args, 'no_phoneme_surprisal', True)
                         or not getattr(args, 'no_phoneme_surprisal_mat', True))
            if phones_df is not None and len(phones_df) > 0 and (
                not getattr(args, 'no_syllables', False)
                or not getattr(args, 'no_phonemes', False)
                or _want_psr
            ):
```

Then, immediately AFTER `syll_df, phon_df = syllabify_pairs(df, phones_df)` (line 796) and BEFORE the syllable block, insert:

```python
                if _want_psr and len(phon_df) > 0:
                    from extractors.phoneme_surprisal import (
                        add_phoneme_surprisal, phoneme_surprisal_mat_vars)
                    _pcfg = getattr(args, 'phoneme_surprisal', {}) or {}
                    phon_df = add_phoneme_surprisal(
                        phon_df,
                        strip_stress=_pcfg.get('strip_stress', True),
                        primary_only=_pcfg.get('primary_only', True),
                    )
                    if not getattr(args, 'no_phoneme_surprisal_mat', True) and pair.audio_path:
                        import scipy.io
                        sr = float(_pcfg.get('analysis_sr', 128))
                        mat_vars = phoneme_surprisal_mat_vars(
                            phon_df, _audio_duration(pair.audio_path), sr,
                            {'analysis_sr': sr,
                             'strip_stress': _pcfg.get('strip_stress', True),
                             'primary_only': _pcfg.get('primary_only', True),
                             'context_independent': True,
                             'first_phoneme_empty_context': True,
                             'cap': None, 'n_phonemes': int(len(phon_df)),
                             'duration_s': _audio_duration(pair.audio_path),
                             'lexicon': 'english_us_arpa'})
                        mat_path = (_level_dir(output_dir, 'phoneme')
                                    / f'{pair.stimulus_id}_phoneme_surprisal.mat')
                        scipy.io.savemat(str(mat_path), mat_vars)
```

(The `.mat` write is BEFORE the per-column CSV drop at line 818, so the columns still exist. The CSV block at line 817 emits the two columns subject to `_phoneme_columns_to_drop`.)

- [ ] **Step 4: Run the extractor + integration + planner suites**

Run: `python -m pytest tests/extractors/test_phoneme_surprisal.py tests/integration/test_phoneme_surprisal_integration.py tests/planner -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add main.py tests/integration/test_phoneme_surprisal_integration.py
git commit -m "feat(phoneme-surprisal): wire compute + .mat write into word batch

Widen the sub-word gate; append columns; rasterize to phoneme/<stim>_phoneme_surprisal.mat
before the per-column drop; _PHONEME_COLUMN_MAP gates the two new columns.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 8: GUI — Sub-word card + Run-page output toggle

**Files:**
- Modify: `gui/section_subword_features.py`
- Modify: `gui/section_run.py`
- Test: `tests/gui/test_subword_features.py` (or the existing sub-word GUI test)

**Interfaces:**
- Consumes: prefs keys from Task 5.
- Produces: a `ClusterCard` (master `master_phoneme_surprisal`, two toggles) + `analysis_sr`/`strip_stress`/`primary_only` controls saved to the `phoneme_surprisal` section; a `phoneme_surprisal_mat` output toggle on the Run page.

- [ ] **Step 1: Write a failing GUI test** (mirror the existing sub-word section test — confirm the card + toggles exist)

```python
# tests/gui/test_subword_features.py  (add; needs the qtbot fixture)
def test_phoneme_surprisal_card_present(qtbot):
    from gui.section_subword_features import SubwordFeaturesSection
    w = SubwordFeaturesSection()
    qtbot.addWidget(w)
    assert w.toggle("feat_phoneme_surprisal") is not None
    assert w.toggle("feat_phoneme_cohort_entropy") is not None
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/gui/test_subword_features.py -q -k phoneme_surprisal`
Expected: FAIL (toggle returns None).

- [ ] **Step 3a: Add the card in `gui/section_subword_features.py`** — after the Phonemes card (after line 141), before `layout.addStretch(1)`:

```python
        # --- Phoneme surprisal (Brodbeck/Gillis cohort model) ---
        psr_card = ClusterCard(
            "Phoneme surprisal",
            "Context-independent cohort surprisal + entropy per phoneme. "
            "Requires MFA alignment. Also emits a .mat impulse-train sidecar "
            "when enabled on the Run page.",
            master_key="master_phoneme_surprisal",
            master_label="Emit phoneme surprisal columns",
        )
        psr_grid = QGridLayout()
        psr_keys = [
            ("feat_phoneme_surprisal",      "phoneme_surprisal (bits)"),
            ("feat_phoneme_cohort_entropy", "cohort_entropy (bits)"),
        ]
        for i, (key, label) in enumerate(psr_keys):
            psr_grid.addWidget(self._make_toggle(key, label, f[key]), i // 2, i % 2)
        psr_card.add_child_layout(psr_grid)

        # Config params saved directly to the phoneme_surprisal section
        # (like pcenter_k -> prosody).
        _psr = prefs.get("phoneme_surprisal", {})
        sr_row = QHBoxLayout()
        sr_row.addWidget(QLabel("Analysis rate (Hz):"))
        self._psr_sr_spin = QDoubleSpinBox()
        self._psr_sr_spin.setRange(1.0, 1000.0)
        self._psr_sr_spin.setDecimals(0)
        self._psr_sr_spin.setValue(_psr.get("analysis_sr", 128))
        self._psr_sr_spin.valueChanged.connect(
            lambda v: save_preferences({"phoneme_surprisal": {"analysis_sr": v}}))
        sr_row.addWidget(self._psr_sr_spin)
        sr_row.addStretch(1)
        sr_w = QWidget(); sr_w.setLayout(sr_row)
        psr_card.add_child_layout(sr_row) if hasattr(psr_card, "add_child_layout") else None
        psr_card.add_widget(sr_w) if hasattr(psr_card, "add_widget") else None
        psr_card.settings_changed.connect(self.settings_changed)
        layout.addWidget(psr_card)
```

Note: check `ClusterCard`'s API in `gui/widgets.py` — use `add_child_layout` for the grid (as the other cards do) and add the spinbox row via the same method the P-center card uses (`add_widget` on a plain `Card`; `ClusterCard` may differ). Match the sibling cards exactly; drop the `strip_stress`/`primary_only` checkboxes if a simple toggle row is cleaner — analysis_sr is the one param researchers change most. Keep `strip_stress`/`primary_only` at their defaults via prefs if not surfaced (they remain functional through emit).

- [ ] **Step 3b: Add the `phoneme_surprisal_mat` output toggle to `gui/section_run.py`** — locate where `acoustic_ts_mat` (or the other `*_mat` sidecar output toggles) are listed and add an entry mirroring it, label e.g. `"Phoneme surprisal .mat (impulse trains)"`, output key `phoneme_surprisal_mat`.

Run first to find the site: `grep -n "acoustic_ts_mat" gui/section_run.py`

- [ ] **Step 4: Run GUI tests**

Run: `python -m pytest tests/gui/test_subword_features.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add gui/section_subword_features.py gui/section_run.py tests/gui/test_subword_features.py
git commit -m "feat(phoneme-surprisal): GUI card + Run-page .mat output toggle

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 9: Feature docs + full-suite verification + master_plan update

**Files:**
- Create: `docs/features/feat_phoneme_surprisal.md`
- Create: `docs/features/feat_phoneme_cohort_entropy.md` (or one shared file via `_SHARED_DOC_MAP` in `gui/feature_docs.py`)
- Modify: `docs/master_plan.md`

- [ ] **Step 1: Write the two feature docs** following the authoring rules (memory `feedback-feature-docs-authorship` / `feedback-feature-docs-citations`): frontmatter (key, levels, libraries, status, depth) + body (meaning, EEG rationale, math, **Implementation in SensiExtractor** sourced from `extractors/phoneme_surprisal.py`, library snippet, **Papers** with inline `[Author Year]` citations). Use the spec §8b verified citation set: Marslen-Wilson & Welsh 1978; Brodbeck, Hong & Simon 2018 (*Current Biology*); Brodbeck et al. 2022 (*eLife* e72056); Gillis et al. 2021 (*J Neurosci* 41(50):10316); Donhauser & Baillet 2020 (*Neuron*). Mark any interpolated *why* with ⚠️. If sharing one file, add the mapping in `gui/feature_docs.py::_SHARED_DOC_MAP` (route `feat_phoneme_cohort_entropy` → the surprisal doc).

- [ ] **Step 2: Verify the ⓘ icon renders** (a doc file existing makes `ToggleRow` show the icon automatically — no GUI wiring). Confirm via:

Run: `python -m pytest tests/gui/test_feature_docs.py -q`
Expected: PASS.

- [ ] **Step 3: Run the FULL suite**

Run: `python -m pytest -q`
Expected: all green (baseline 503 passed, 2 skipped, plus the new tests). Investigate any failure before proceeding — do not claim done on red.

- [ ] **Step 4: Real-run smoke check (verification-before-completion)**

With MFA + a real stimulus available, enable phoneme surprisal (GUI or CLI), run one stimulus, and confirm: `phoneme/<stim>_phonemes.csv` has `phoneme_surprisal` + `cohort_entropy` columns with sane values (first phoneme > 0, entropy 0 on final phoneme of monosyllables), and `phoneme/<stim>_phoneme_surprisal.mat` loads with four length-N trains + `t`/`fs`/`config`. If MFA is unavailable in this environment, note that explicitly rather than claiming the real run passed. (See memory `project-mkl-threading-entrypoints`: tests passing ≠ feature runs.)

- [ ] **Step 5: Update `docs/master_plan.md`** — add a §3 completed-batches row (date 2026-07-06, "Phoneme surprisal + cohort entropy", prefs v15→v16, spec + plan links), and fix the stale "prefs at v13" note in §2/§6 to reflect v16. Commit:

```bash
git add docs/features/feat_phoneme_surprisal.md docs/features/feat_phoneme_cohort_entropy.md gui/feature_docs.py docs/master_plan.md
git commit -m "docs(phoneme-surprisal): feature docs + master_plan v16 batch row

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:**
- §2 cohort math → Task 1 (hand-check test). §2 OOV cap → Task 1. §2b onset controls → Task 3 (`phoneme_onset` + `word_onset`). §3 lexicon/frequency/floor → Task 2 (loader) + Task 1 (floor in `build_cohort_trie`). §4 extractor API + guard flags → Tasks 1–3. §5 CSV columns + `.mat` → Tasks 3, 7. §6 wiring (Stage, 2 clusters, output-key routing, emit, main.py) → Tasks 4, 6, 7. §7 prefs v16 + drift guard → Task 5. §8 tests → each task's TDD + Task 9 full suite. §8b citations → Task 9 docs.
- Coverage complete; no spec section unaddressed.

**Placeholder scan:** No TBD/TODO. One soft spot: Task 8 Step 3a hedges on `ClusterCard`'s exact param API (`add_child_layout` vs `add_widget`) and Task 8 Step 3b points at a grep rather than exact lines — deliberate, because the GUI widget API and Run-page layout weren't read in full during planning; the implementer must match the sibling cards/toggles verbatim. Every code-bearing step elsewhere shows complete code.

**Type consistency:** `add_phoneme_surprisal(phon_df, *, dict_path, strip_stress, primary_only)`, `phoneme_surprisal_mat_vars(phon_df, duration_s, analysis_sr, config)`, `build_cohort_trie(entries)`, `phoneme_scores(trie, phones, *, context_independent, first_phoneme_empty_context)`, `get_cohort_trie(dict_path, *, strip_stress, primary_only)` — names and signatures match across Tasks 1–3, 6, 7. Prefs keys (`master_phoneme_surprisal`, `feat_phoneme_surprisal`, `feat_phoneme_cohort_entropy`, `phoneme_surprisal_mat`, `phoneme_surprisal.{analysis_sr,strip_stress,primary_only}`), stage name `phoneme_surprisal`, and output_csv `phoneme_surprisal_mat` are consistent across Tasks 4–8.
