# Phoneme Label Split, IPA, and Phoneme-Coverage Correctness — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split the phoneme CSV's fused `Label` column into `ARPABET` / `Stress` / `IPA`, add a dash-joined `IPA` to the syllable CSV, and fix two confirmed silent defects in phoneme coverage (OOV words deleted; contractions duplicated).

**Architecture:** A new pure `extractors/arpabet_ipa.py` owns the ARPABET→IPA table. `extractors/syllabify.py` emits the new columns, deduplicates shared alignments, and reports zero-phoneme words. `utils/textgrid_parser.py` stops filtering `spn` so OOV words survive. `extractors/aligner.py` gains an opt-in `--g2p_model_path` pass-through. Two existing consumers (`pcenter.py`, `phoneme_surprisal.py`) are rewired off the fused label. Prefs go v19→v20 with a removal-migration.

**Tech Stack:** Python 3, pandas (nullable `Int64`), pytest, spaCy, MFA 3.4.0 (Kaldi), PySide6.

**Spec:** [`docs/superpowers/specs/2026-07-27-phoneme-labels-ipa-oov-design.md`](../specs/2026-07-27-phoneme-labels-ipa-oov-design.md)

## Global Constraints

- **Environment:** `conda activate ./env` from the repo root. It is a path-based conda *prefix* env, not a venv, and there is no env named `sensi`. If invoking `./env/python.exe` directly without activation, prepend `env/Library/bin` to `PATH` or ~26 tests fail on `libsndfile.dll`.
- **Nothing that works today may change its output values.** The only intended value changes are: duplicate phoneme/syllable rows disappearing, `phoneme_surprisal.mat` no longer double-counting, and OOV words appearing where they previously vanished. `P_Center` in particular must come out **bit-identical** — see Task 4.
- **`MKL_THREADING_LAYER` must stay set at every entry point** (`main.py`, `gui.py`, `run.py`, `tests/conftest.py`). Never remove those lines.
- **A green suite does not prove a real run works.** Repo fixtures under `tests/fixtures/audio/` are empty placeholders. Task 12 is the real-stimulus gate and is not optional.
- **MFA must run at most once per document.** Never add a code path that re-aligns per pass; alignment is cached via `_prealign_batch` / `ensure_alignments`.
- **`Stress` is nullable `Int64`**, never float and never `0`-for-consonant. Empty means "stress does not apply"; `0` means "unstressed vowel".
- **Prefs target version: 20.** `SCHEMA_VERSION_V20 = 20`.
- **TDD applies to `extractors/` only** (Tasks 1–5). Tasks 6–11 are test-after.
- **Commit after each task**, directly to the current branch (`main`), per project convention.
- Run the full suite with `python -m pytest`; it must be green before each commit.

---

## File Structure

| File | Responsibility | Task |
|---|---|---|
| `extractors/arpabet_ipa.py` | **New.** ARPABET→IPA table + `to_ipa` / `syllable_to_ipa`. Pure, no I/O. | 1 |
| `utils/textgrid_parser.py` | Stop filtering `spn` from the phones tier. | 2 |
| `extractors/syllabify.py` | Emit `ARPABET`/`Stress`/`IPA`/`Pron_Source` + syllable `IPA`; dedup shared alignments; collect zero-phoneme words. | 3 |
| `extractors/pcenter.py` | Vowel detection off `Stress` instead of the fused label. | 4 |
| `extractors/phoneme_surprisal.py` | Read `ARPABET` (+ `Stress` when recombining); `NaN` for `spn`. | 5 |
| `extractors/aligner.py` | Optional `--g2p_model_path`; `mfa_g2p` sidecar key; `SIDECAR_VERSION` 1→2. | 6 |
| `utils/download_models.py` | Fetch the g2p model. | 6 |
| `utils/preferences.py` | v19→v20 removal-migration. | 7 |
| `main.py` | Column maps; `--g2p-oov`; thread the pref to alignment. | 8 |
| `gui/planner/registry.py`, `gui/planner/emit.py` | Cluster toggle keys + `emits_columns`; `g2p_oov` into the Namespace. | 9 |
| `gui/section_subword_features.py` | Four phoneme-label toggles; syllable `IPA`; G2P toggle. | 9 |
| `extractors/text_parser.py` | Skip SPACE tokens; renumber ids; remap `head_token_id`. | 10 |
| `docs/features/*.md`, `gui/feature_docs.py` | Per-feature docs + toggle→doc routing. | 11 |
| `docs/master_plan.md`, `docs/changelog.md` | Batch bookkeeping. | 11 |
| `validation/` re-run | Real-stimulus gate. | 12 |

**Dependency order:** 1 → 2 → 3 → (4, 5 in parallel) → 6 → 7 → 8 → 9 → 10 → 11 → 12.
Task 3 is the hinge: Tasks 4 and 5 fix consumers that Task 3 breaks, so 3–5 must land before the suite is green. If executing task-by-task with a green-suite gate, treat **3+4+5 as one commit boundary**; the per-task tests below still run individually.

---

### Task 1: ARPABET→IPA mapping module

**Files:**
- Create: `extractors/arpabet_ipa.py`
- Test: `tests/extractors/test_arpabet_ipa.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `ARPABET_TO_IPA: dict[str, str]` — bare-code → IPA, stress-independent entries only.
  - `to_ipa(arpabet: str, stress: int | None) -> str` — one segment; `""` for unknown/`spn`/empty.
  - `syllable_to_ipa(pairs: list[tuple[str, int | None]]) -> str` — dash-joined, `""` for empty input.

- [ ] **Step 1: Write the failing test**

Create `tests/extractors/test_arpabet_ipa.py`:

```python
"""ARPABET -> IPA mapping (extractors/arpabet_ipa.py)."""
import pytest

from extractors.arpabet_ipa import ARPABET_TO_IPA, to_ipa, syllable_to_ipa


def test_stress_dependent_ah_is_schwa_when_unstressed():
    assert to_ipa("AH", 0) == "ə"      # schwa


@pytest.mark.parametrize("stress", [1, 2])
def test_stress_dependent_ah_is_wedge_when_stressed(stress):
    assert to_ipa("AH", stress) == "ʌ"  # turned v


def test_stress_dependent_er():
    assert to_ipa("ER", 0) == "ɚ"       # schwa with hook
    assert to_ipa("ER", 1) == "ɝ"       # reversed epsilon with hook
    assert to_ipa("ER", 2) == "ɝ"


def test_ah_and_er_with_no_stress_digit_fall_back_to_stressed_form():
    # A vowel should always carry a stress digit from MFA; if one is missing
    # we must still return a vowel, not "".
    assert to_ipa("AH", None) == "ʌ"
    assert to_ipa("ER", None) == "ɝ"


@pytest.mark.parametrize("code,expected", [
    ("AA", "ɑ"), ("AE", "æ"), ("AO", "ɔ"), ("AW", "aʊ"),
    ("AY", "aɪ"), ("EH", "ɛ"), ("EY", "eɪ"), ("IH", "ɪ"),
    ("IY", "i"), ("OW", "oʊ"), ("OY", "ɔɪ"), ("UH", "ʊ"),
    ("UW", "u"),
])
def test_vowels_are_stress_independent(code, expected):
    assert to_ipa(code, 1) == expected
    assert to_ipa(code, 0) == expected


@pytest.mark.parametrize("code,expected", [
    ("B", "b"), ("CH", "tʃ"), ("D", "d"), ("DH", "ð"), ("F", "f"),
    ("G", "ɡ"), ("HH", "h"), ("JH", "dʒ"), ("K", "k"), ("L", "l"),
    ("M", "m"), ("N", "n"), ("NG", "ŋ"), ("P", "p"), ("R", "ɹ"),
    ("S", "s"), ("SH", "ʃ"), ("T", "t"), ("TH", "θ"), ("V", "v"),
    ("W", "w"), ("Y", "j"), ("Z", "z"), ("ZH", "ʒ"),
])
def test_consonants(code, expected):
    assert to_ipa(code, None) == expected


def test_all_39_arpabet_phones_are_covered():
    # 15 vowels (AH and ER counted once each) + 24 consonants = 39.
    assert len(ARPABET_TO_IPA) == 39
    assert all(v for v in ARPABET_TO_IPA.values())


def test_unknown_and_silence_map_to_empty_string_not_a_guess():
    assert to_ipa("spn", None) == ""
    assert to_ipa("sil", None) == ""
    assert to_ipa("", None) == ""
    assert to_ipa("QQ", None) == ""


def test_stress_digit_on_input_is_tolerated():
    # Callers pass a bare code, but a fused label must not silently yield "".
    assert to_ipa("AH0", None) == "ə"
    assert to_ipa("EY2", None) == "eɪ"


def test_syllable_to_ipa_dash_joins():
    assert syllable_to_ipa([("D", None), ("R", None), ("IY", 0)]) == "d-ɹ-i"


def test_syllable_to_ipa_skips_unmappable_segments():
    assert syllable_to_ipa([("D", None), ("spn", None), ("IY", 0)]) == "d-i"


def test_syllable_to_ipa_empty_input():
    assert syllable_to_ipa([]) == ""
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/extractors/test_arpabet_ipa.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'extractors.arpabet_ipa'`

- [ ] **Step 3: Write the implementation**

Create `extractors/arpabet_ipa.py`:

```python
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
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/extractors/test_arpabet_ipa.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
git add extractors/arpabet_ipa.py tests/extractors/test_arpabet_ipa.py
git commit -m "feat(phonemes): ARPABET->IPA mapping module (stress-aware)"
```

---

### Task 2: Stop filtering `spn` so OOV words survive

**Files:**
- Modify: `utils/textgrid_parser.py:51-68` (`parse_phones`)
- Test: `tests/utils/test_textgrid_parser.py` (create if absent)

**Interfaces:**
- Consumes: nothing.
- Produces: `parse_phones` now returns rows whose `phone` may be `'spn'`. Task 3 consumes this.

**Context:** MFA aligns any out-of-dictionary word with its `<unk>` model, whose single phone is `spn`. Filtering `spn` as silence made those words reach `syllabify_pairs` with an empty phone list, so they produced **zero** phoneme and syllable rows with no warning. `sil`/`sp`/`''` are genuine silence and stay filtered.

- [ ] **Step 1: Write the failing test**

Add to `tests/utils/test_textgrid_parser.py`:

```python
"""TextGrid parsing (utils/textgrid_parser.py)."""
import textwrap

from utils.textgrid_parser import parse_phones, parse_words


def _write_textgrid(tmp_path, intervals_words, intervals_phones):
    """Minimal long-form TextGrid with a words tier and a phones tier."""
    def _tier(name, intervals):
        body = "".join(
            f'        intervals [{i}]:\n'
            f'            xmin = {s}\n'
            f'            xmax = {e}\n'
            f'            text = "{t}"\n'
            for i, (s, e, t) in enumerate(intervals, start=1)
        )
        return (
            f'    item [{{n}}]:\n'
            f'        class = "IntervalTier"\n'
            f'        name = "{name}"\n'
            f'        xmin = 0\n'
            f'        xmax = 1\n'
            f'        intervals: size = {len(intervals)}\n'
        ) + body

    header = textwrap.dedent('''\
        File type = "ooTextFile"
        Object class = "TextGrid"

        xmin = 0
        xmax = 1
        tiers? <exists>
        size = 2
        item []:
        ''')
    text = (header
            + _tier("words", intervals_words).format(n=1)
            + _tier("phones", intervals_phones).format(n=2))
    path = tmp_path / "oov.TextGrid"
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_spn_is_kept_so_oov_words_are_not_silently_dropped(tmp_path):
    tg = _write_textgrid(
        tmp_path,
        [(0.0, 0.33, "zorblax"), (0.33, 0.5, "has")],
        [(0.0, 0.33, "spn"), (0.33, 0.5, "HH")],
    )
    phones = parse_phones(tg)
    assert list(phones["phone"]) == ["spn", "HH"]


def test_real_silence_labels_are_still_filtered(tmp_path):
    tg = _write_textgrid(
        tmp_path,
        [(0.0, 0.5, "has")],
        [(0.0, 0.1, "sil"), (0.1, 0.2, "sp"), (0.2, 0.3, ""),
         (0.3, 0.5, "HH")],
    )
    phones = parse_phones(tg)
    assert list(phones["phone"]) == ["HH"]


def test_words_tier_unchanged(tmp_path):
    tg = _write_textgrid(
        tmp_path,
        [(0.0, 0.33, "zorblax"), (0.33, 0.5, "has")],
        [(0.0, 0.33, "spn"), (0.33, 0.5, "HH")],
    )
    words = parse_words(tg)
    assert list(words["word"]) == ["zorblax", "has"]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/utils/test_textgrid_parser.py -v`
Expected: `test_spn_is_kept_so_oov_words_are_not_silently_dropped` FAILS with `assert [] == ['spn', 'HH']`. The other two pass.

- [ ] **Step 3: Make the change**

In `utils/textgrid_parser.py::parse_phones`, change the silence set and the docstring:

```python
    Returns a DataFrame with columns:
        phone  — ARPABET phoneme label (e.g. 'B', 'EH1', 'L', 'AH0'), or
                 'spn' for a word MFA had no dictionary pronunciation for
        onset  — phoneme start time in seconds
        offset — phoneme end time in seconds

    Silence intervals (SIL, sp, '') are excluded. 'spn' is NOT excluded: MFA
    emits it as the single phone of an out-of-dictionary word, so dropping it
    deleted that word from every sub-word output with no warning. Callers
    distinguish it via Pron_Source (see extractors/syllabify.py).
    """
    tg = tgt.read_textgrid(textgrid_path)
    phone_tier = tg.get_tier_by_name('phones')

    silence_labels = {'', 'SIL', 'sil', 'sp'}
```

Lowercase `'sil'` is added purely as defensive future-proofing. Verified against a real
`english_us_arpa` TextGrid: **MFA 3.4 writes silence as empty text (`text = ""`)**, not `sil`
or `sp`, so the existing set already caught every silence interval and nothing was leaking.
Do not describe this addition as a bug fix.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/utils/test_textgrid_parser.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add utils/textgrid_parser.py tests/utils/test_textgrid_parser.py
git commit -m "fix(phonemes): keep spn so OOV words are not silently deleted"
```

---

### Task 3: Emit split columns, dedup shared alignments, report zero-phoneme words

**Files:**
- Modify: `extractors/syllabify.py` (`_SYLL_COLUMNS`, `_PHON_COLUMNS`, `syllabify_pairs`; add `_oov_lexicon` helper)
- Test: `tests/extractors/test_syllabify.py`

**Interfaces:**
- Consumes: `extractors.arpabet_ipa.to_ipa` / `syllable_to_ipa` (Task 1); `parse_phones` output that may contain `'spn'` (Task 2).
- Produces:
  - Phoneme frame columns: `doc_id, sentence_id, token_in_sentence_id, token, syllable_id, phoneme_id, ARPABET, Stress, IPA, Pron_Source, Onset, End, Duration`. `Label` is **gone**.
  - Syllable frame gains `IPA` after `ARPABET`; `Label`/`ARPABET` unchanged.
  - `syllabify_pairs(word_df, phone_df)` keeps its two-value return; it additionally prints a `[phonemes] WARNING: …` line when words end with no phonemes.
  - `Stress` dtype is `Int64` (nullable).

**Context:** Three changes in one function, because they all touch the same row-emission loop.

1. **Split columns.** `Phone.label` stays the raw MFA label internally (`is_vowel` depends on the trailing digit); the split happens at row-emission time.
2. **Dedup.** `align_merge`'s *join* rule gives both spaCy tokens of a contraction (`was` + `n't`) the same `[Onset, End]`, and `_phones_for_word` selects by containment, so each token claimed the full phone set. Emit for the **first** token only.
3. **`Pron_Source`.** A words-tier form absent from `english_us_arpa.dict` had its pronunciation generated by G2P (or none at all). Reuse the lexicon parse already in `phoneme_surprisal`.

- [ ] **Step 1: Write the failing test**

Add to `tests/extractors/test_syllabify.py`:

```python
import pandas as pd
import pytest

from extractors.syllabify import syllabify_pairs


def _word_row(token, onset, end, tis_id):
    return {"doc_id": 1, "sentence_id": 1, "token_in_sentence_id": tis_id,
            "token": token, "Onset": onset, "End": end}


def test_split_columns_replace_label():
    word_df = pd.DataFrame([_word_row("cup", 0.0, 0.30, 1)])
    phone_df = pd.DataFrame([
        {"phone": "K",   "onset": 0.00, "offset": 0.10},
        {"phone": "AH1", "onset": 0.10, "offset": 0.20},
        {"phone": "P",   "onset": 0.20, "offset": 0.30},
    ])
    _, phon = syllabify_pairs(word_df, phone_df)

    assert "Label" not in phon.columns
    assert list(phon["ARPABET"]) == ["K", "AH", "P"]
    assert list(phon["IPA"]) == ["k", "ʌ", "p"]


def test_stress_is_empty_on_consonants_and_zero_on_unstressed_vowels():
    """Empty != 0. Empty means 'does not apply'; 0 means 'unstressed vowel'."""
    word_df = pd.DataFrame([_word_row("about", 0.0, 0.40, 1)])
    phone_df = pd.DataFrame([
        {"phone": "AH0", "onset": 0.00, "offset": 0.10},
        {"phone": "B",   "onset": 0.10, "offset": 0.20},
        {"phone": "AW1", "onset": 0.20, "offset": 0.30},
        {"phone": "T",   "onset": 0.30, "offset": 0.40},
    ])
    _, phon = syllabify_pairs(word_df, phone_df)

    assert str(phon["Stress"].dtype) == "Int64"
    stress = list(phon["Stress"])
    assert stress[0] == 0            # unstressed vowel -> 0, NOT empty
    assert pd.isna(stress[1])        # consonant -> empty, NOT 0
    assert stress[2] == 1
    assert pd.isna(stress[3])
    # And the schwa/wedge distinction follows from it:
    assert list(phon["IPA"]) == ["ə", "b", "aʊ", "t"]


def test_contraction_phonemes_are_emitted_once_for_the_first_token():
    """spaCy 'was' + \"n't\" both carry MFA "wasn't"'s interval."""
    word_df = pd.DataFrame([
        _word_row("was", 4.50, 4.87, 1),
        _word_row("n't", 4.50, 4.87, 2),
    ])
    phone_df = pd.DataFrame([
        {"phone": "W",   "onset": 4.50, "offset": 4.60},
        {"phone": "AH1", "onset": 4.60, "offset": 4.68},
        {"phone": "Z",   "onset": 4.68, "offset": 4.75},
        {"phone": "AH0", "onset": 4.75, "offset": 4.81},
        {"phone": "N",   "onset": 4.81, "offset": 4.84},
        {"phone": "T",   "onset": 4.84, "offset": 4.87},
    ])
    syll, phon = syllabify_pairs(word_df, phone_df)

    assert len(phon) == 6, "phonemes must not be emitted twice"
    assert set(phon["token"]) == {"was"}
    assert len(syll) == 2
    assert set(syll["token"]) == {"was"}


def test_distinct_words_with_distinct_intervals_are_both_emitted():
    """Dedup must key on the interval, not merely on adjacency."""
    word_df = pd.DataFrame([
        _word_row("a", 0.00, 0.10, 1),
        _word_row("bee", 0.10, 0.30, 2),
    ])
    phone_df = pd.DataFrame([
        {"phone": "AH0", "onset": 0.00, "offset": 0.10},
        {"phone": "B",   "onset": 0.10, "offset": 0.20},
        {"phone": "IY1", "onset": 0.20, "offset": 0.30},
    ])
    _, phon = syllabify_pairs(word_df, phone_df)
    assert set(phon["token"]) == {"a", "bee"}
    assert len(phon) == 3


def test_oov_word_yields_one_row_flagged_unresolved():
    word_df = pd.DataFrame([_word_row("zorblax", 0.0, 0.33, 1)])
    phone_df = pd.DataFrame([
        {"phone": "spn", "onset": 0.0, "offset": 0.33},
    ])
    syll, phon = syllabify_pairs(word_df, phone_df)

    assert len(phon) == 1
    assert phon["ARPABET"].iloc[0] == "spn"
    assert pd.isna(phon["Stress"].iloc[0])
    assert phon["IPA"].iloc[0] == ""
    assert phon["Pron_Source"].iloc[0] == "oov_unresolved"
    assert len(syll) == 1, "an spn word still occupies one syllable"


def test_in_dictionary_word_is_flagged_dictionary():
    word_df = pd.DataFrame([_word_row("cup", 0.0, 0.30, 1)])
    phone_df = pd.DataFrame([
        {"phone": "K",   "onset": 0.00, "offset": 0.10},
        {"phone": "AH1", "onset": 0.10, "offset": 0.20},
        {"phone": "P",   "onset": 0.20, "offset": 0.30},
    ])
    _, phon = syllabify_pairs(word_df, phone_df)
    assert set(phon["Pron_Source"]) == {"dictionary"}


def test_oov_word_with_real_phones_is_flagged_g2p():
    """G2P recovered a pronunciation for a word absent from the dictionary."""
    word_df = pd.DataFrame([_word_row("zorblax", 0.0, 0.20, 1)])
    phone_df = pd.DataFrame([
        {"phone": "Z",   "onset": 0.00, "offset": 0.10},
        {"phone": "AO1", "onset": 0.10, "offset": 0.20},
    ])
    _, phon = syllabify_pairs(word_df, phone_df)
    assert set(phon["Pron_Source"]) == {"g2p"}


def test_syllable_gains_ipa_and_keeps_label_and_arpabet():
    word_df = pd.DataFrame([_word_row("cup", 0.0, 0.30, 1)])
    phone_df = pd.DataFrame([
        {"phone": "K",   "onset": 0.00, "offset": 0.10},
        {"phone": "AH1", "onset": 0.10, "offset": 0.20},
        {"phone": "P",   "onset": 0.20, "offset": 0.30},
    ])
    syll, _ = syllabify_pairs(word_df, phone_df)
    assert syll["ARPABET"].iloc[0] == "K-AH1", "unchanged: keeps stress digits"
    assert syll["IPA"].iloc[0] == "k-ʌ-p"


def test_zero_phoneme_words_are_reported(capsys):
    """A word with no phones at all must not vanish silently."""
    word_df = pd.DataFrame([
        _word_row("cup", 0.0, 0.10, 1),
        _word_row("ghost", 5.0, 5.20, 2),   # no phones in its interval
    ])
    phone_df = pd.DataFrame([
        {"phone": "K", "onset": 0.00, "offset": 0.10},
    ])
    syllabify_pairs(word_df, phone_df)
    err = capsys.readouterr().err
    assert "ghost" in err
    assert "no phonemes" in err


def test_deduped_second_token_is_not_reported_as_missing(capsys):
    """n't legitimately has no rows; that is not an anomaly."""
    word_df = pd.DataFrame([
        _word_row("was", 4.50, 4.60, 1),
        _word_row("n't", 4.50, 4.60, 2),
    ])
    phone_df = pd.DataFrame([{"phone": "W", "onset": 4.50, "offset": 4.60}])
    syllabify_pairs(word_df, phone_df)
    assert "n't" not in capsys.readouterr().err


def test_empty_frames_still_carry_the_new_schema():
    syll, phon = syllabify_pairs(
        pd.DataFrame(columns=["doc_id", "sentence_id", "token_in_sentence_id",
                              "token", "Onset", "End"]),
        pd.DataFrame(columns=["phone", "onset", "offset"]),
    )
    for col in ("ARPABET", "Stress", "IPA", "Pron_Source"):
        assert col in phon.columns
    assert "Label" not in phon.columns
    assert "IPA" in syll.columns
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/extractors/test_syllabify.py -v`
Expected: the new tests FAIL (`KeyError: 'ARPABET'` / duplicate counts / missing `Pron_Source`). Pre-existing tests in this file that assert on `Label` will also fail — that is expected and Step 3 updates them.

- [ ] **Step 3: Implement**

In `extractors/syllabify.py`:

**3a.** Add imports and the lexicon helper near the top of the "Stimulus-level orchestrator" section:

```python
import sys

from extractors.arpabet_ipa import syllable_to_ipa, to_ipa

_SPN = "spn"
_LEXICON_CACHE: set[str] | None = None


def _lexicon_words() -> set[str]:
    """Memoized set of words present in the MFA pronunciation dictionary.

    Used only to label Pron_Source: MFA does not mark G2P-derived words in the
    TextGrid, so a words-tier form absent from the dictionary is one whose
    pronunciation was generated (or absent entirely). Reuses the same .dict
    that extractors/phoneme_surprisal.py parses. Returns an empty set if the
    dictionary is unavailable, in which case every word is reported as
    'dictionary' rather than mislabelled as 'g2p'.
    """
    global _LEXICON_CACHE
    if _LEXICON_CACHE is None:
        from extractors.phoneme_surprisal import DEFAULT_DICT_PATH
        words: set[str] = set()
        try:
            with open(DEFAULT_DICT_PATH, "r", encoding="utf-8") as fh:
                for line in fh:
                    word = line.split("\t", 1)[0].strip()
                    if word and word[0] not in "[<":
                        words.add(word.lower())
        except OSError:
            words = set()
        _LEXICON_CACHE = words
    return _LEXICON_CACHE


def _split_label(label: str) -> tuple[str, int | None]:
    """'EY2' -> ('EY', 2); 'D' -> ('D', None)."""
    text = str(label)
    if text and text[-1].isdigit():
        return text[:-1], int(text[-1])
    return text, None


def _pron_source(token: str, phones: list[Phone]) -> str:
    """dictionary | g2p | oov_unresolved for one word's phone sequence."""
    if any(p.label == _SPN for p in phones):
        return "oov_unresolved"
    lexicon = _lexicon_words()
    if lexicon and str(token).lower() not in lexicon:
        return "g2p"
    return "dictionary"
```

**3b.** Update the column schemas:

```python
_SYLL_COLUMNS = [
    'doc_id', 'sentence_id', 'token_in_sentence_id', 'token', 'syllable_id',
    'Label', 'ARPABET', 'IPA', 'Onset', 'End', 'Duration',
]
_PHON_COLUMNS = [
    'doc_id', 'sentence_id', 'token_in_sentence_id', 'token', 'syllable_id',
    'phoneme_id', 'ARPABET', 'Stress', 'IPA', 'Pron_Source',
    'Onset', 'End', 'Duration',
]
```

**3c.** Rewrite the body of `syllabify_pairs` (replacing the loop and the return):

```python
    syll_rows: list[dict] = []
    phon_rows: list[dict] = []
    seen_intervals: set[tuple] = set()
    missing: list[str] = []

    for w in word_df.itertuples(index=False):
        if pd.isna(getattr(w, 'Onset', None)) or pd.isna(getattr(w, 'End', None)):
            continue
        interval = (w.doc_id, round(float(w.Onset), 6), round(float(w.End), 6))
        if interval in seen_intervals:
            # A preceding token already claimed this exact MFA word (the
            # align_merge 'join' case: spaCy "was" + "n't" vs MFA "wasn't").
            # Emitting again would duplicate real acoustic phonemes.
            continue
        word_phones = _phones_for_word(phone_df, cast(float, w.Onset), cast(float, w.End))
        if not word_phones:
            missing.append(str(w.token))
            continue
        seen_intervals.add(interval)

        source = _pron_source(cast(str, w.token), word_phones)
        syllables = syllabify_word(word_phones)
        labels = assign_labels(cast(str, w.token), syllables)

        for syll_idx, (syl, label_str, arpabet_str) in enumerate(
                zip(syllables, labels.labels, labels.arpabet), start=1):
            split = [_split_label(p.label) for p in syl.phonemes]
            syll_rows.append({
                'doc_id': w.doc_id, 'sentence_id': w.sentence_id,
                'token_in_sentence_id': w.token_in_sentence_id, 'token': w.token,
                'syllable_id': syll_idx,
                'Label': label_str, 'ARPABET': arpabet_str,
                'IPA': syllable_to_ipa(split),
                'Onset': syl.onset, 'End': syl.end,
                'Duration': round(syl.duration, 4),
            })
            for phon_idx, (phon, (code, stress)) in enumerate(
                    zip(syl.phonemes, split), start=1):
                phon_rows.append({
                    'doc_id': w.doc_id, 'sentence_id': w.sentence_id,
                    'token_in_sentence_id': w.token_in_sentence_id, 'token': w.token,
                    'syllable_id': syll_idx, 'phoneme_id': phon_idx,
                    'ARPABET': code,
                    'Stress': stress,
                    'IPA': to_ipa(code, stress),
                    'Pron_Source': source,
                    'Onset': phon.onset, 'End': phon.end,
                    'Duration': round(phon.duration, 4),
                })

    if missing:
        print(f'[phonemes] WARNING: {len(missing)} words had no phonemes '
              f'(likely OOV): {", ".join(missing)}', file=sys.stderr)

    syll_df = pd.DataFrame(syll_rows) if syll_rows else pd.DataFrame(columns=_SYLL_COLUMNS)
    phon_df_out = pd.DataFrame(phon_rows) if phon_rows else pd.DataFrame(columns=_PHON_COLUMNS)
    # Nullable Int64: empty on consonants (stress does not apply) as opposed to
    # 0, which means "unstressed vowel". A plain int64 column cannot hold both.
    phon_df_out['Stress'] = phon_df_out['Stress'].astype('Int64')
    return syll_df, phon_df_out
```

Note the dedup check runs **before** `_phones_for_word`, so a deduped token is never added to `missing` (that is what `test_deduped_second_token_is_not_reported_as_missing` pins).

**3d.** Update the `syllabify_pairs` docstring: replace "Words that have no phones inside their `[Onset, End]` interval produce no rows in either DataFrame." with a note that such words are reported to stderr, and that tokens sharing an interval with an earlier token are emitted once.

**3e.** Update any pre-existing test in `tests/extractors/test_syllabify.py` that asserts on the phoneme `Label` column to use `ARPABET` + `Stress`.

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/extractors/test_syllabify.py -v`
Expected: PASS. Then `python -m pytest tests/extractors/test_pcenter.py tests/extractors/test_phoneme_surprisal.py -v` — expected to FAIL; Tasks 4 and 5 fix them.

- [ ] **Step 5: Commit (with Tasks 4 and 5 — see the dependency note)**

```bash
git add extractors/syllabify.py tests/extractors/test_syllabify.py
git commit -m "feat(phonemes): split ARPABET/Stress/IPA, dedup shared alignments, flag Pron_Source"
```

---

### Task 4: Rewire P-center vowel detection off the fused label

**Files:**
- Modify: `extractors/pcenter.py:21`
- Test: `tests/extractors/test_pcenter.py`

**Interfaces:**
- Consumes: the Task 3 phoneme frame (`ARPABET`, `Stress`).
- Produces: no signature change. `add_pcenter(syllables_df, phonemes_df, *, k=0.3)` is unchanged.

**Context — this is the silent breakage.** `_pcenter_raw` finds the syllable nucleus with `group['Label'].str[-1:].str.isdigit()`. With `Label` gone that raises `KeyError`; and had we merely made `Label` bare, **no vowel would ever be found** and every `P_Center` would quietly fall back to the syllable onset with no error. The replacement is `Stress.notna()` — a vowel is exactly a phone that carries a stress digit.

- [ ] **Step 1: Write the failing test**

Add to `tests/extractors/test_pcenter.py`:

```python
import pandas as pd

from extractors.pcenter import add_pcenter


def _phon(arpabet, stress, onset, end, phoneme_id):
    return {"doc_id": 1, "sentence_id": 1, "token_in_sentence_id": 1,
            "syllable_id": 1, "phoneme_id": phoneme_id,
            "ARPABET": arpabet, "Stress": stress,
            "Onset": onset, "End": end, "Duration": round(end - onset, 4)}


def test_nucleus_is_found_via_stress_column():
    """Regression: vowel detection must not depend on a fused Label string."""
    syll = pd.DataFrame([{
        "doc_id": 1, "sentence_id": 1, "token_in_sentence_id": 1,
        "syllable_id": 1, "Onset": 0.0, "End": 0.30,
    }])
    phon = pd.DataFrame([
        _phon("K",  None, 0.00, 0.10, 1),
        _phon("AH", 1,    0.10, 0.20, 2),
        _phon("P",  None, 0.20, 0.30, 3),
    ]).astype({"Stress": "Int64"})

    out = add_pcenter(syll, phon, k=0.3)
    # nucleus_onset 0.10 - 0.3 * onset_dur 0.10 = 0.07
    assert out["P_Center"].iloc[0] == 0.07
    assert out["P_Center"].iloc[0] != 0.0, "must not degrade to syllable onset"


def test_unstressed_vowel_still_counts_as_a_nucleus():
    """Stress == 0 is a vowel. Only <NA> means 'not a vowel'."""
    syll = pd.DataFrame([{
        "doc_id": 1, "sentence_id": 1, "token_in_sentence_id": 1,
        "syllable_id": 1, "Onset": 0.0, "End": 0.20,
    }])
    phon = pd.DataFrame([
        _phon("B",  None, 0.00, 0.10, 1),
        _phon("AH", 0,    0.10, 0.20, 2),
    ]).astype({"Stress": "Int64"})

    out = add_pcenter(syll, phon, k=0.3)
    assert out["P_Center"].iloc[0] == 0.07


def test_vowelless_syllable_falls_back_to_syllable_onset():
    syll = pd.DataFrame([{
        "doc_id": 1, "sentence_id": 1, "token_in_sentence_id": 1,
        "syllable_id": 1, "Onset": 1.5, "End": 1.70,
    }])
    phon = pd.DataFrame([
        _phon("HH", None, 1.50, 1.60, 1),
        _phon("M",  None, 1.60, 1.70, 2),
    ]).astype({"Stress": "Int64"})

    out = add_pcenter(syll, phon, k=0.3)
    assert out["P_Center"].iloc[0] == 1.5
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/extractors/test_pcenter.py -v`
Expected: FAIL with `KeyError: 'Label'`

- [ ] **Step 3: Implement**

In `extractors/pcenter.py`, replace line 21 and update the module docstring line 4:

```python
"""Perceptual center (P-center) for syllables.

Formula: P_Center = nucleus_onset - k * onset_consonant_duration
where nucleus = first vowel phoneme (a phone carrying a Stress digit; note
Stress == 0 is an unstressed VOWEL, only <NA> means 'not a vowel')
and onset_consonant_duration = sum of durations of preceding consonants.
```

```python
    is_vowel = group['Stress'].notna().tolist()
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/extractors/test_pcenter.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add extractors/pcenter.py tests/extractors/test_pcenter.py
git commit -m "fix(pcenter): detect nucleus via Stress column, not fused Label"
```

---

### Task 5: Rewire phoneme surprisal off the fused label

**Files:**
- Modify: `extractors/phoneme_surprisal.py:170-203` (`add_phoneme_surprisal`)
- Test: `tests/extractors/test_phoneme_surprisal.py`

**Interfaces:**
- Consumes: the Task 3 phoneme frame (`ARPABET`, `Stress`).
- Produces: `add_phoneme_surprisal(phon_df, *, dict_path=..., strip_stress=True, primary_only=True)` — signature unchanged; still appends `phoneme_surprisal` + `cohort_entropy`.

**Context:** The function read `Label` and stripped the stress digit to match the (stress-stripped) trie. `ARPABET` is *already* stress-free, so the `strip_stress=True` path just reads it. The `strip_stress=False` path needs the fused form back, so recombine `ARPABET + Stress`. An `spn` row has no phoneme identity, so it must score `NaN` — not the OOV cap, which means "a real phoneme sequence left the lexicon".

- [ ] **Step 1: Write the failing test**

Add to `tests/extractors/test_phoneme_surprisal.py`:

```python
import pandas as pd

from extractors.phoneme_surprisal import add_phoneme_surprisal


def _row(arpabet, stress, phoneme_id, tis_id=1):
    return {"doc_id": 1, "sentence_id": 1, "token_in_sentence_id": tis_id,
            "token": "cup", "syllable_id": 1, "phoneme_id": phoneme_id,
            "ARPABET": arpabet, "Stress": stress,
            "Onset": 0.1 * phoneme_id, "End": 0.1 * phoneme_id + 0.1,
            "Duration": 0.1}


def test_reads_arpabet_column():
    df = pd.DataFrame([
        _row("K", None, 1), _row("AH", 1, 2), _row("P", None, 3),
    ]).astype({"Stress": "Int64"})
    out = add_phoneme_surprisal(df)
    assert "phoneme_surprisal" in out.columns
    assert out["phoneme_surprisal"].iloc[0] > 0
    assert out["phoneme_surprisal"].notna().all()


def test_spn_row_scores_nan_not_the_oov_cap():
    df = pd.DataFrame([
        {"doc_id": 1, "sentence_id": 1, "token_in_sentence_id": 1,
         "token": "zorblax", "syllable_id": 1, "phoneme_id": 1,
         "ARPABET": "spn", "Stress": None, "Onset": 0.0, "End": 0.33,
         "Duration": 0.33},
    ]).astype({"Stress": "Int64"})
    out = add_phoneme_surprisal(df)
    assert pd.isna(out["phoneme_surprisal"].iloc[0])
    assert pd.isna(out["cohort_entropy"].iloc[0])


def test_strip_stress_false_recombines_arpabet_and_stress():
    """The stress-sensitive trie needs the fused form back."""
    df = pd.DataFrame([
        _row("K", None, 1), _row("AH", 1, 2), _row("P", None, 3),
    ]).astype({"Stress": "Int64"})
    out = add_phoneme_surprisal(df, strip_stress=False, primary_only=True)
    # A stress-sensitive trie still finds 'K AH1 P', so the first phoneme is
    # scored from the real cohort rather than pinned at the OOV cap.
    assert out["phoneme_surprisal"].iloc[0] > 0
    assert out["phoneme_surprisal"].notna().all()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/extractors/test_phoneme_surprisal.py -v`
Expected: FAIL with `KeyError: 'Label'`

- [ ] **Step 3: Implement**

In `extractors/phoneme_surprisal.py`, replace the grouping loop inside `add_phoneme_surprisal` (currently lines 194–200) and update the docstring:

```python
    """Append phoneme_surprisal + cohort_entropy columns to a phonemes DataFrame.

    Scores each word's aligned phoneme sequence (rows grouped by doc/sentence/
    token, in phoneme order). Reads the stress-free `ARPABET` column directly;
    when `strip_stress=False` the fused form is rebuilt from `ARPABET` + `Stress`
    to match a stress-sensitive trie.

    Rows whose ARPABET is `spn` (an out-of-dictionary word MFA had no
    pronunciation for) score NaN: there is no phoneme identity to evaluate.
    That is deliberately distinct from the OOV *cap*, which means a real
    phoneme sequence walked off the lexicon.
    """
```

```python
    positions = df.reset_index(drop=True)
    for _, grp in positions.groupby(_GROUP_KEYS, sort=False):
        codes = [str(c) for c in grp["ARPABET"]]
        if any(c in _SILENCE_PHONES for c in codes):
            # spn / silence: no phoneme identity to score. Leave NaN.
            continue
        if strip_stress:
            phones = codes
        else:
            phones = [
                c if pd.isna(s) else f"{c}{int(s)}"
                for c, s in zip(codes, grp["Stress"])
            ]
        surp, ent = phoneme_scores(trie, phones)
        for pos, s, e in zip(grp.index, surp, ent):
            surp_col[pos] = round(s, 4)
            ent_col[pos] = round(e, 4)
```

`_SILENCE_PHONES` already exists at module level and contains `"spn"`.

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/extractors/ -v`
Expected: PASS (Tasks 1–5 all green)

- [ ] **Step 5: Commit**

```bash
git add extractors/phoneme_surprisal.py tests/extractors/test_phoneme_surprisal.py
git commit -m "fix(phoneme-surprisal): read ARPABET/Stress; NaN for spn rows"
```

---

### Task 6: Optional MFA G2P for OOV words

**Files:**
- Modify: `extractors/aligner.py` (`SIDECAR_VERSION`, `_freshness_dict`, `_is_fresh`, `_run_mfa`, `_align_uncached`, `align_with_phones`, `ensure_alignments`)
- Modify: `utils/download_models.py`
- Test: `tests/extractors/test_aligner.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `G2P_MODEL_PATH: Path` — module constant, `mfa_data/pretrained_models/g2p/english_us_arpa.zip`.
  - `g2p_model_available() -> bool`
  - `_run_mfa(corpus_dir, output_dir, *, g2p: bool = False) -> None`
  - `ensure_alignments(items, persist_dir, *, g2p: bool = False) -> dict`
  - `align_with_phones(..., g2p: bool = False)` — new keyword-only arg, default `False`.
  - `SIDECAR_VERSION == 2`; sidecar dicts gain an `mfa_g2p: bool` key.

**Context:** MFA 3.4.0's `align` accepts `--g2p_model_path TEXT` ("Path to G2P model to use for OOV items"). Verified on a real run: an OOV word received a real boundary-aligned pronunciation, and in-dictionary phone intervals were bit-identical except the OOV word's own shared boundary. Because the flag changes alignment output, the freshness sidecar must record it — otherwise a cached TextGrid from a run with the opposite setting would be reused.

- [ ] **Step 1: Write the failing test**

Add to `tests/extractors/test_aligner.py`:

```python
import json

import extractors.aligner as aligner


def test_sidecar_records_g2p_and_version_2(tmp_path, monkeypatch):
    audio = tmp_path / "s.wav"
    audio.write_bytes(b"\0" * 16)
    monkeypatch.setattr(aligner, "_sha256_file", lambda p, _chunk=1: "deadbeef")

    meta_off = aligner._freshness_dict(str(audio), "hello", g2p=False)
    meta_on = aligner._freshness_dict(str(audio), "hello", g2p=True)

    assert meta_off["version"] == 2
    assert meta_off["mfa_g2p"] is False
    assert meta_on["mfa_g2p"] is True


def test_cached_alignment_is_stale_when_g2p_setting_differs(tmp_path, monkeypatch):
    audio = tmp_path / "s.wav"
    audio.write_bytes(b"\0" * 16)
    monkeypatch.setattr(aligner, "_sha256_file", lambda p, _chunk=1: "deadbeef")
    (tmp_path / "s.TextGrid").write_text("x", encoding="utf-8")
    (tmp_path / "s.json").write_text(
        json.dumps(aligner._freshness_dict(str(audio), "hello", g2p=False)),
        encoding="utf-8")

    assert aligner._is_fresh(tmp_path, "s", str(audio), "hello", g2p=False)
    assert not aligner._is_fresh(tmp_path, "s", str(audio), "hello", g2p=True)


def test_run_mfa_passes_g2p_flag_only_when_requested(monkeypatch):
    calls = []

    class _Result:
        returncode = 0
        stdout = ""
        stderr = ""

    monkeypatch.setattr(aligner.subprocess, "run",
                        lambda cmd, **kw: calls.append(cmd) or _Result())
    monkeypatch.setattr(aligner, "g2p_model_available", lambda: True)

    aligner._run_mfa("corpus", "out", g2p=False)
    assert "--g2p_model_path" not in calls[0]

    aligner._run_mfa("corpus", "out", g2p=True)
    assert "--g2p_model_path" in calls[1]
    assert calls[1][calls[1].index("--g2p_model_path") + 1].endswith(".zip")


def test_g2p_requested_but_model_missing_is_a_warning_not_a_crash(monkeypatch, capsys):
    class _Result:
        returncode = 0
        stdout = ""
        stderr = ""

    calls = []
    monkeypatch.setattr(aligner.subprocess, "run",
                        lambda cmd, **kw: calls.append(cmd) or _Result())
    monkeypatch.setattr(aligner, "g2p_model_available", lambda: False)

    aligner._run_mfa("corpus", "out", g2p=True)
    assert "--g2p_model_path" not in calls[0]
    assert "g2p" in capsys.readouterr().err.lower()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/extractors/test_aligner.py -v`
Expected: FAIL — `_freshness_dict() got an unexpected keyword argument 'g2p'`

- [ ] **Step 3: Implement**

In `extractors/aligner.py`:

```python
SIDECAR_VERSION = 2

G2P_MODEL_PATH = (Path(__file__).resolve().parent.parent
                  / 'mfa_data' / 'pretrained_models' / 'g2p'
                  / 'english_us_arpa.zip')


def g2p_model_available() -> bool:
    """True when the vendored MFA G2P model is present."""
    return G2P_MODEL_PATH.exists()
```

Thread a keyword-only `g2p: bool = False` through `_freshness_dict`, `_write_sidecar`,
`_is_fresh`, `_align_uncached`, `align_with_phones`, and `ensure_alignments`, and:

- `_freshness_dict`: add `'mfa_g2p': bool(g2p)` to the returned dict.
- `_is_fresh`: add `if bool(meta.get('mfa_g2p', False)) != bool(g2p): return False`.
- `align_with_phones`: include `g2p` in the memo `key` tuple so one process can't serve a
  cached alignment built with the other setting:
  `key = (os.path.abspath(audio_path), transcript, bool(g2p))`.
- `_run_mfa`: append the flag when requested and available; warn when requested and missing:

```python
def _run_mfa(corpus_dir: str, output_dir: str, *, g2p: bool = False) -> None:
    """Invoke MFA as a subprocess. Raises RuntimeError on failure.

    When `g2p` is set and the vendored G2P model is present, MFA generates a
    pronunciation for out-of-dictionary words instead of aligning them to a
    single `spn` phone. This changes alignment output, so callers must record it
    in the freshness sidecar (see _freshness_dict).
    """
    cmd = [
        'mfa', 'align',
        corpus_dir,
        MFA_DICTIONARY,
        MFA_ACOUSTIC_MODEL,
        output_dir,
        '--clean',
        '--quiet',
    ]
    if g2p:
        if g2p_model_available():
            cmd += ['--g2p_model_path', str(G2P_MODEL_PATH)]
        else:
            print(f'[mfa] WARNING: g2p requested but no G2P model at '
                  f'{G2P_MODEL_PATH}; out-of-dictionary words will align to '
                  f'spn. Run: python utils/download_models.py',
                  file=sys.stderr)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"MFA alignment failed.\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        )
```

Add `import sys` at the top of the module.

In `utils/download_models.py`, add a third entry to the existing
`(model_type, model_name)` loop at line 25 (there is no `_download` helper — the loop calls
`subprocess.run(['mfa', 'model', 'download', model_type, model_name])` directly):

```python
    for model_type, model_name in [
        ('acoustic',    'english_us_arpa'),
        ('dictionary',  'english_us_arpa'),
        # g2p generates pronunciations for out-of-dictionary words (~11 MB).
        # Only used when the alignment.g2p_oov pref is on, but vendored here so
        # the toggle works offline for collaborators.
        ('g2p',         'english_us_arpa'),
    ]:

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/extractors/test_aligner.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add extractors/aligner.py utils/download_models.py tests/extractors/test_aligner.py
git commit -m "feat(align): opt-in MFA G2P for OOV words; sidecar v2 records the setting"
```

---

### Task 7: Prefs v19 → v20 removal-migration

**Files:**
- Modify: `utils/preferences.py`
- Test: `tests/utils/test_preferences_migration.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `SCHEMA_VERSION_V20 = 20`, `DEFAULT_PREFS_V20`, `SCHEMA_VERSION == 20`, `DEFAULT_PREFS == DEFAULT_PREFS_V20`.
  - Features: `feat_phoneme_label` **removed**; `feat_phoneme_arpabet`, `feat_phoneme_stress`, `feat_phoneme_ipa`, `feat_phoneme_pron_source`, `feat_syllable_ipa` added.
  - New top-level section `alignment: {"g2p_oov": False}`.

**Context:** Removal-migration precedent is v18 (see the `version < SCHEMA_VERSION_V18` blocks around `preferences.py:1095-1114`) and v14's struct-id rename. The four replacements inherit the old `feat_phoneme_label` value so a user who had it off keeps all four off. `feat_syllable_ipa` defaults `False`, matching `feat_syllable_arpabet`'s verbose-column precedent. **No paragraph key is added** — `feat_struct_paragraph_id` already exists.

- [ ] **Step 1: Write the failing test**

Add to `tests/utils/test_preferences_migration.py`:

There is **no** public `_migrate(dict) -> dict`. Migration runs inside
`load_preferences()`, which reads the legacy JSON path (or the YAML pair) from the directory
named by the `SENSI_PREFS_PATH` env var. Follow the existing pattern in this file exactly:
write an old-schema JSON file to `tmp_path`, `monkeypatch.setenv('SENSI_PREFS_PATH', ...)`,
then call `load_preferences()`.

```python
import json

from utils.preferences import (
    DEFAULT_PREFS,
    SCHEMA_VERSION,
    SCHEMA_VERSION_V20,
    load_preferences,
)

_NEW_PHONEME_KEYS = ("feat_phoneme_arpabet", "feat_phoneme_stress",
                     "feat_phoneme_ipa", "feat_phoneme_pron_source")


def _load_from(tmp_path, monkeypatch, stored: dict) -> dict:
    """Write `stored` as the legacy json and return the migrated prefs."""
    monkeypatch.setenv('SENSI_PREFS_PATH', str(tmp_path / 'p.json'))
    (tmp_path / 'p.json').write_text(json.dumps(stored), encoding='utf-8')
    return load_preferences()


def test_active_version_is_20():
    assert SCHEMA_VERSION == SCHEMA_VERSION_V20 == 20
    assert DEFAULT_PREFS['schema_version'] == 20


def test_defaults_drop_label_and_add_the_split_keys():
    f = DEFAULT_PREFS['features']
    assert 'feat_phoneme_label' not in f
    for key in _NEW_PHONEME_KEYS:
        assert key in f
    assert f['feat_syllable_ipa'] is False
    assert DEFAULT_PREFS['alignment'] == {'g2p_oov': False}


def test_paragraph_toggle_is_untouched():
    """It already existed; the migration must not add or reset it."""
    assert DEFAULT_PREFS['features']['feat_struct_paragraph_id'] is True


def test_v19_label_on_migrates_all_four_replacements_on(tmp_path, monkeypatch):
    prefs = _load_from(tmp_path, monkeypatch, {
        'schema_version': 19,
        'features': {'feat_phoneme_label': True},
    })
    assert prefs['schema_version'] == 20
    assert 'feat_phoneme_label' not in prefs['features']
    for key in _NEW_PHONEME_KEYS:
        assert prefs['features'][key] is True, key


def test_v19_label_off_migrates_all_four_replacements_off(tmp_path, monkeypatch):
    prefs = _load_from(tmp_path, monkeypatch, {
        'schema_version': 19,
        'features': {'feat_phoneme_label': False},
    })
    for key in _NEW_PHONEME_KEYS:
        assert prefs['features'][key] is False, key


def test_g2p_oov_defaults_off_after_migration(tmp_path, monkeypatch):
    prefs = _load_from(tmp_path, monkeypatch,
                       {'schema_version': 19, 'features': {}})
    assert prefs['alignment']['g2p_oov'] is False


def test_post_v20_user_values_survive_a_reload(tmp_path, monkeypatch):
    """A v20 source is not re-migrated; the user's own values win."""
    prefs = _load_from(tmp_path, monkeypatch, {
        'schema_version': 20,
        'features': {'feat_phoneme_ipa': False, 'feat_phoneme_arpabet': True},
        'alignment': {'g2p_oov': True},
    })
    assert prefs['features']['feat_phoneme_ipa'] is False
    assert prefs['features']['feat_phoneme_arpabet'] is True
    assert prefs['alignment']['g2p_oov'] is True
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/utils/test_preferences_migration.py -v`
Expected: FAIL — `ImportError: cannot import name 'SCHEMA_VERSION_V20'`

- [ ] **Step 3: Implement**

Add after the v19 block (`preferences.py:875`):

```python
# ---------------------------------------------------------------------------
# v20 constants
#
# Phoneme label split + IPA + OOV coverage. REMOVAL-migration (precedent: v14's
# struct-id rename, v18's predictability unification): the fused
# `feat_phoneme_label` toggle is replaced by four per-column toggles, because
# the phoneme CSV's `Label` column (ARPABET + stress digit fused, e.g. 'EY2')
# becomes ARPABET / Stress / IPA / Pron_Source.
#
# Additive alongside it: `feat_syllable_ipa` (opt-in verbose column, matching
# feat_syllable_arpabet) and a new `alignment` section holding `g2p_oov`
# (opt-in MFA G2P for out-of-dictionary words — off by default so no run
# silently gains modeled pronunciations).
# ---------------------------------------------------------------------------

SCHEMA_VERSION_V20 = 20

_FEATURES_DEFAULT_V20 = {
    k: v for k, v in DEFAULT_PREFS_V19["features"].items()
    if k != "feat_phoneme_label"
}
_FEATURES_DEFAULT_V20.update({
    "feat_phoneme_arpabet":     False,
    "feat_phoneme_stress":      False,
    "feat_phoneme_ipa":         False,
    "feat_phoneme_pron_source": False,
    "feat_syllable_ipa":        False,
})

DEFAULT_PREFS_V20: dict[str, Any] = {
    **DEFAULT_PREFS_V19,
    "schema_version": SCHEMA_VERSION_V20,
    "features": dict(_FEATURES_DEFAULT_V20),
    "alignment": {"g2p_oov": False},
}
```

Update the active aliases:

```python
SCHEMA_VERSION = SCHEMA_VERSION_V20  # active version
DEFAULT_PREFS: dict[str, Any] = DEFAULT_PREFS_V20  # active defaults
```

Add `SCHEMA_VERSION_V20` to the known-versions tuple at `preferences.py:948-949`.

Add the finalize block alongside the existing v18 one:

```python
    # v20 finalize (pre-v20 sources only — post-v20 values are the user's):
    # the fused phoneme `Label` column is split, so its single toggle becomes
    # four. Carry the old on/off state to all four so a user who had the
    # column off keeps every replacement off. Not additive — the key is
    # removed, not merely defaulted (precedent: v14, v18).
    if isinstance(version, int) and version < SCHEMA_VERSION_V20:
        f = merged.get("features", {})
        label_on = f.pop("feat_phoneme_label", None)
        if label_on is not None:
            for key in ("feat_phoneme_arpabet", "feat_phoneme_stress",
                        "feat_phoneme_ipa", "feat_phoneme_pron_source"):
                f[key] = bool(label_on)
```

- [ ] **Step 4: Run the tests and regenerate the templates**

```bash
python -m pytest tests/utils/test_preferences_migration.py -v
python scripts/export_default_config.py
python -m pytest tests/ -v -k "drift or schema"
```
Expected: PASS. `config/features_defaults.yaml` and `config/run_defaults.yaml` are rewritten; the drift-guard test enforces this.

- [ ] **Step 5: Commit**

```bash
git add utils/preferences.py tests/utils/test_preferences_migration.py \
        config/features_defaults.yaml config/run_defaults.yaml
git commit -m "feat(prefs): v20 — phoneme label split, syllable IPA, alignment.g2p_oov"
```

---

### Task 8: Column-drop maps and CLI wiring

**Files:**
- Modify: `main.py:561-568` (`_PHONEME_COLUMN_MAP`), `main.py:548-559` (`_SYLLABLE_COLUMN_MAP`), the arg parser near `main.py:192`, `_prealign_batch` (`main.py:709`), and the `align_with_phones` call site in `run()`
- Test: `tests/integration/test_pipeline_batch.py`

**Interfaces:**
- Consumes: `alignment.g2p_oov` (Task 7), `aligner` g2p args (Task 6).
- Produces: `args.g2p_oov: bool` on the Namespace, consumed by `_prealign_batch` and `run()`.

- [ ] **Step 1: Write the failing test**

Add to `tests/integration/test_pipeline_batch.py`:

```python
from main import _phoneme_columns_to_drop, _syllable_columns_to_drop, build_parser


def test_phoneme_drop_map_covers_the_split_columns():
    all_off = {
        "feat_phoneme_arpabet": False, "feat_phoneme_stress": False,
        "feat_phoneme_ipa": False, "feat_phoneme_pron_source": False,
        "feat_phoneme_onset": False, "feat_phoneme_end": False,
        "feat_phoneme_duration": False,
        "feat_phoneme_surprisal": False, "feat_phoneme_cohort_entropy": False,
    }
    drop = _phoneme_columns_to_drop(all_off)
    assert {"ARPABET", "Stress", "IPA", "Pron_Source"} <= drop
    assert "Label" not in drop, "the Label column no longer exists"


def test_phoneme_drop_map_is_per_column():
    drop = _phoneme_columns_to_drop({"feat_phoneme_ipa": False})
    assert drop == {"IPA"}


def test_syllable_drop_map_covers_ipa():
    assert _syllable_columns_to_drop({"feat_syllable_ipa": False}) == {"IPA"}


def test_cli_exposes_g2p_oov_defaulting_off():
    args = build_parser().parse_args(["--transcript", "x.txt"])
    assert args.g2p_oov is False
    args = build_parser().parse_args(["--transcript", "x.txt", "--g2p-oov"])
    assert args.g2p_oov is True
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/integration/test_pipeline_batch.py -v -k "drop_map or g2p_oov"`
Expected: FAIL — `AttributeError: 'Namespace' object has no attribute 'g2p_oov'` and the drop-map assertions fail.

- [ ] **Step 3: Implement**

Replace `_PHONEME_COLUMN_MAP` (`main.py:561`):

```python
_PHONEME_COLUMN_MAP = {
    "feat_phoneme_arpabet":     "ARPABET",
    "feat_phoneme_stress":      "Stress",
    "feat_phoneme_ipa":         "IPA",
    "feat_phoneme_pron_source": "Pron_Source",
    "feat_phoneme_onset":    "Onset",
    "feat_phoneme_end":      "End",
    "feat_phoneme_duration": "Duration",
    "feat_phoneme_surprisal":      "phoneme_surprisal",
    "feat_phoneme_cohort_entropy": "cohort_entropy",
}
```

Add to `_SYLLABLE_COLUMN_MAP` after the `feat_syllable_arpabet` entry:

```python
    "feat_syllable_ipa":          "IPA",
```

Add the CLI flag beside `--no-phonemes` (`main.py:192`):

```python
    p.add_argument('--g2p-oov', action='store_true',
                   help='Use the MFA G2P model to generate pronunciations for '
                        'out-of-dictionary words instead of aligning them to a '
                        'single spn phone. Off by default: the pronunciation is '
                        'modeled, not observed.')
```

Thread it to both alignment call sites:

- In `_prealign_batch`, pass `g2p=bool(getattr(run_args, 'g2p_oov', False))` to
  `ensure_alignments(items, alignment_dir(output_dir), g2p=...)`.
- In `run()`, pass the same to `align_with_phones(...)`.

> Do not add any new alignment call. MFA must still run at most once per document; both
> sites already share the cache and the persisted TextGrid.

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/integration/ -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add main.py tests/integration/test_pipeline_batch.py
git commit -m "feat(cli): phoneme/syllable drop maps for split columns; --g2p-oov"
```

---

### Task 9: Planner and GUI

**Files:**
- Modify: `gui/planner/registry.py:371-400`
- Modify: `gui/planner/emit.py` (near line 212, where `phoneme_surprisal` prefs are read)
- Modify: `gui/section_subword_features.py:45-64` and `:125-135`
- Test: `tests/planner/test_registry.py`, `tests/planner/test_emit.py`, `tests/gui/test_subword_features.py`

**Interfaces:**
- Consumes: prefs v20 keys (Task 7); `args.g2p_oov` shape (Task 8).
- Produces: `to_namespace` sets `ns.g2p_oov`. Planner clusters reference the new toggle keys.

- [ ] **Step 1: Write the failing test**

Add to `tests/planner/test_registry.py`:

```python
from gui.planner.registry import CLUSTERS


def _cluster(name):
    return next(c for c in CLUSTERS if c.name == name)


def test_phoneme_cluster_uses_the_split_toggles():
    c = _cluster("phoneme intervals")
    assert "feat_phoneme_label" not in c.toggle_keys
    assert {"feat_phoneme_arpabet", "feat_phoneme_stress",
            "feat_phoneme_ipa", "feat_phoneme_pron_source"} <= set(c.toggle_keys)
    assert {"ARPABET", "Stress", "IPA", "Pron_Source"} <= set(c.emits_columns)
    assert "Label" not in c.emits_columns


def test_syllable_labels_cluster_gains_ipa():
    c = _cluster("syllable labels")
    assert "feat_syllable_ipa" in c.toggle_keys
    assert "IPA" in c.emits_columns
    assert "ARPABET" in c.emits_columns   # unchanged
```

Add to `tests/planner/test_emit.py`:

```python
def test_g2p_oov_reaches_the_namespace(monkeypatch):
    from utils.preferences import DEFAULT_PREFS
    from gui.planner.plan import plan_from_prefs
    from gui.planner.emit import to_namespace
    import copy

    prefs = copy.deepcopy(DEFAULT_PREFS)
    prefs["alignment"]["g2p_oov"] = True
    ns = to_namespace(plan_from_prefs(prefs, "word"), prefs, "word")
    assert ns.g2p_oov is True

    prefs["alignment"]["g2p_oov"] = False
    ns = to_namespace(plan_from_prefs(prefs, "word"), prefs, "word")
    assert ns.g2p_oov is False
```

Add to `tests/gui/test_subword_features.py`:

```python
def test_phoneme_card_shows_four_label_toggles(qtbot):
    """Follow this file's existing page-construction fixture pattern."""
    from gui.section_subword_features import SubwordFeaturesSection
    from utils.preferences import DEFAULT_PREFS
    import copy

    page = SubwordFeaturesSection(copy.deepcopy(DEFAULT_PREFS))
    qtbot.addWidget(page)
    keys = set(page._toggles.keys())     # match the real attribute name
    assert {"feat_phoneme_arpabet", "feat_phoneme_stress",
            "feat_phoneme_ipa", "feat_phoneme_pron_source",
            "feat_syllable_ipa"} <= keys
    assert "feat_phoneme_label" not in keys
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/planner/ tests/gui/test_subword_features.py -v`
Expected: FAIL on the new assertions.

- [ ] **Step 3: Implement**

`gui/planner/registry.py` — phoneme cluster:

```python
    # §2.1 Phoneme intervals — phoneme half of syllabify_pairs
    FeatureCluster(
        name="phoneme intervals",
        toggle_keys=("feat_phoneme_arpabet", "feat_phoneme_stress",
                     "feat_phoneme_ipa", "feat_phoneme_pron_source",
                     "feat_phoneme_onset", "feat_phoneme_end",
                     "feat_phoneme_duration"),
        output_csv="phonemes",
        requires_stages=("spacy_parse", "mfa_align", "syllabify_pairs"),
        emits_columns=("ARPABET", "Stress", "IPA", "Pron_Source",
                       "Onset", "End", "Duration"),
        master_toggle="master_phoneme_intervals",
    ),
```

Syllable labels cluster:

```python
    # §2.2b Syllable labels (Label + ARPABET + IPA)
    FeatureCluster(
        name="syllable labels",
        toggle_keys=("feat_syllable_label", "feat_syllable_arpabet",
                     "feat_syllable_ipa"),
        output_csv="syllables",
        requires_stages=("spacy_parse", "mfa_align", "syllabify_pairs"),
        emits_columns=("Label", "ARPABET", "IPA"),
        master_toggle="master_syllable_labels",
    ),
```

`gui/planner/emit.py` — add beside the other alignment/namespace assignments in
`to_namespace` (and in the acoustic-level namespace builder if it also aligns):

```python
    ns.g2p_oov = bool(prefs.get("alignment", {}).get("g2p_oov", False))
```

`gui/section_subword_features.py` — syllable labels card:

```python
        syll_labels_keys = [
            ("feat_syllable_label",   "Label (orthographic, pyphen-derived)"),
            ("feat_syllable_arpabet", "ARPABET (dash-joined, with stress)"),
            ("feat_syllable_ipa",     "IPA (dash-joined)"),
        ]
```

Phoneme card (replacing the single `feat_phoneme_label` row at line 129):

```python
            ("feat_phoneme_arpabet",     "ARPABET (bare code, no stress digit)"),
            ("feat_phoneme_stress",      "Stress (0/1/2; empty on consonants)"),
            ("feat_phoneme_ipa",         "IPA"),
            ("feat_phoneme_pron_source", "Pron_Source (dictionary/g2p/oov_unresolved)"),
            ("feat_phoneme_onset",    "Onset"),
            ("feat_phoneme_end",      "End"),
            ("feat_phoneme_duration", "Duration"),
```

Add a G2P toggle. Put it on the Sub-word page as its own `Card` beneath the phoneme card,
because it changes alignment rather than a single column:

```python
        g2p_card = Card(
            "Out-of-dictionary words",
            "MFA aligns words missing from its dictionary (unusual names) to a "
            "single 'spn' phone with no phoneme identity. Enable this to have "
            "MFA generate a pronunciation for them instead. The pronunciation "
            "is modeled from spelling, not observed — Pron_Source marks every "
            "affected word as 'g2p'. Changing this re-runs alignment once.",
        )
        g2p_card.add_widget(
            self._make_toggle(
                "g2p_oov",
                "Generate pronunciations for out-of-dictionary words (G2P)",
                prefs.get("alignment", {}).get("g2p_oov", False),
            )
        )
        g2p_card.settings_changed.connect(self.settings_changed)
        layout.addWidget(g2p_card)
```

`g2p_oov` lives in `prefs["alignment"]`, **not** `prefs["features"]`. Follow this page's
existing pattern for non-`features` keys (see how the P-center `k` spinbox writes to
`prefs["prosody"]`) so the save path targets the right section.

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/planner/ tests/gui/ -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add gui/planner/registry.py gui/planner/emit.py gui/section_subword_features.py \
        tests/planner/ tests/gui/test_subword_features.py
git commit -m "feat(gui): split phoneme label toggles, syllable IPA, G2P OOV toggle"
```

---

### Task 10: Drop SPACE rows and remap ids

**Files:**
- Modify: `extractors/text_parser.py:99-142` (`parse`)
- Test: `tests/extractors/test_text_parser.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `parse(text, doc_id=1)` — unchanged signature; no `upos == 'SPACE'` rows in the output.

**Context:** spaCy emits a literal newline as a token, which became an all-NaN word-CSV row consuming both a `token_id` and a `token_in_sentence_id` — so every word after a line break was off by one, and `token_in_sentence_id` was not the sentence position the docs claim.

> ⚠️ `head_token_id = token.head.i + 1` uses spaCy's doc index, which counts SPACE tokens.
> Today `token_id` also counts them, so the two agree **by accident**. Skipping SPACE rows
> without remapping makes every head past the first newline point at the wrong word, silently.
> The two-pass structure below is mandatory.

**Do not touch the paragraph logic** (`_PARAGRAPH_SEP_RE`, `_paragraph_spans`,
`_paragraph_id_for_char`). It already implements the blank-line rule and already ships.

- [ ] **Step 1: Write the failing test**

Add to `tests/extractors/test_text_parser.py`:

```python
import pytest

from extractors.text_parser import parse

_WRAPPED = ("Others who arrived just after the time of distribution\n"
            "were often 48 hours without food.\n")


@pytest.mark.parametrize("text", [_WRAPPED])
def test_no_space_rows_are_emitted(text):
    df = parse(text)
    assert (df["upos"] != "SPACE").all()
    assert not df["token"].str.strip().eq("").any()


def test_token_ids_are_contiguous_after_dropping_newlines():
    df = parse(_WRAPPED)
    assert list(df["token_id"]) == list(range(1, len(df) + 1))


def test_token_in_sentence_id_is_a_true_sentence_position():
    df = parse(_WRAPPED)
    for _, sent in df.groupby("sentence_id"):
        assert list(sent["token_in_sentence_id"]) == list(range(1, len(sent) + 1))


def test_head_token_id_resolves_to_a_real_token_id():
    """The remap regression: heads must not point past the newline shift."""
    df = parse(_WRAPPED)
    valid = set(df["token_id"])
    assert set(df["head_token_id"]) <= valid


def test_head_token_id_still_names_the_correct_word():
    df = parse(_WRAPPED)
    by_id = dict(zip(df["token_id"], df["token"]))
    row = df[df["token"] == "distribution"].iloc[0]
    # 'distribution' is the object of 'of'; its head must be a real word,
    # never a whitespace token or an off-by-one neighbour.
    assert by_id[row["head_token_id"]].strip() != ""


def test_a_root_is_its_own_head():
    df = parse("The cat sat.")
    roots = df[df["dep_rel"] == "ROOT"]
    assert len(roots) >= 1
    for _, r in roots.iterrows():
        assert r["head_token_id"] == r["token_id"]


def test_mid_sentence_newline_does_not_start_a_paragraph():
    """A hard line wrap is not a paragraph break; only blank lines are."""
    df = parse(_WRAPPED)
    assert set(df["paragraph_id"]) == {1}


def test_blank_line_still_starts_a_paragraph():
    df = parse("First para here.\n\nSecond para here.\n")
    assert set(df["paragraph_id"]) == {1, 2}


def test_punctuation_rows_are_still_emitted():
    df = parse("Hello, world.")
    assert (df["upos"] == "PUNCT").sum() == 2
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/extractors/test_text_parser.py -v`
Expected: `test_no_space_rows_are_emitted`, `test_token_ids_are_contiguous_after_dropping_newlines`, and `test_token_in_sentence_id_is_a_true_sentence_position` FAIL. The paragraph tests already pass (that logic ships).

- [ ] **Step 3: Implement**

Replace the row-building loop in `extractors/text_parser.py::parse` with a two-pass version:

```python
    nlp = _get_nlp()
    doc = nlp(text)
    paragraphs = _paragraph_spans(text)

    # Pass 1: assign token_id over EMITTED tokens only, and record the
    # spaCy-doc-index -> token_id mapping.
    #
    # Whitespace tokens (a literal newline in the transcript) are not words:
    # they used to occupy a row with NaN for every feature and to consume both
    # a token_id and a token_in_sentence_id, shifting every following word by
    # one. They are dropped here.
    #
    # This mapping is what keeps head_token_id correct. spaCy's token.head.i is
    # a doc-level index that still counts the skipped whitespace, so
    # `token.head.i + 1` is only equal to the head's token_id while nothing is
    # skipped. Resolving through id_by_doc_index is mandatory, not defensive.
    id_by_doc_index: dict[int, int] = {}
    next_id = 1
    for token in doc:
        if token.is_space:
            continue
        id_by_doc_index[token.i] = next_id
        next_id += 1

    rows = []

    for sent_idx, sent in enumerate(doc.sents, start=1):
        sent_start_char = sent.start_char
        paragraph_id = _paragraph_id_for_char(sent_start_char, paragraphs)

        tok_idx = 0
        for token in sent:
            if token.is_space:
                continue
            tok_idx += 1

            # Character offsets relative to the sentence string (1-indexed)
            char_start = token.idx - sent_start_char + 1
            char_end = char_start + len(token.text) - 1

            # spaCy's MorphAnalysis already stringifies to '' when a token has
            # no morphology; writing a literal 'NA' sentinel instead collides
            # with pandas' default na_values on any standard pd.read_csv().
            feats = str(token.morph)

            # head_token_id is the head's token_id (global, whole-document,
            # 1-based over emitted tokens), so it is self-contained against the
            # token_id column and unambiguous across sentence boundaries. A
            # root token's head is itself. If a head is itself whitespace
            # (spaCy can attach to one in malformed input) the token is treated
            # as its own head rather than dangling.
            head_token_id = id_by_doc_index.get(
                token.head.i, id_by_doc_index[token.i])

            rows.append({
                'doc_id': doc_id,
                'paragraph_id': paragraph_id,
                'sentence_id': sent_idx,
                'sentence': sent.text,
                'start': char_start,
                'end': char_end,
                'token_id': id_by_doc_index[token.i],
                'token_in_sentence_id': tok_idx,
                'token': token.text,
                'lemma': token.lemma_,
                'upos': token.pos_,
                'xpos': token.tag_,
                'feats': feats,
                'dep_rel': token.dep_,
                'head_token_id': head_token_id,
            })

    return pd.DataFrame(rows)
```

Update the module docstring (lines 13-18) to state that whitespace tokens are excluded and
that `token_id` / `token_in_sentence_id` count emitted tokens only.

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/extractors/test_text_parser.py -v && python -m pytest -q`
Expected: PASS, full suite green.

- [ ] **Step 5: Commit**

```bash
git add extractors/text_parser.py tests/extractors/test_text_parser.py
git commit -m "fix(parser): drop whitespace tokens; remap token_id and head_token_id"
```

---

### Task 11: Documentation

**Files:**
- Rewrite: `docs/features/phoneme.md`
- Modify: `docs/features/syllable_labels.md`, `docs/features/structural_ids.md`, `docs/features/phoneme_surprisal.md`
- Create: `docs/features/feat_phoneme_ipa.md`, `docs/features/g2p_oov.md`
- Modify: `gui/feature_docs.py:120-126` (`_SHARED_DOC_MAP`)
- Modify: `docs/master_plan.md`, `docs/changelog.md`
- Test: `tests/gui/test_feature_docs.py`

**Authoring rules:** write every section yourself, including "Implementation in SensiExtractor". Mark any interpolated *why* with ⚠️ rather than a TODO stub. Every non-trivial empirical claim gets an inline `[Author Year]` link; never fabricate a reference. The ARPABET↔IPA correspondence should cite the CMU Pronouncing Dictionary documentation.

- [ ] **Step 1: Write the failing test**

Add to `tests/gui/test_feature_docs.py`:

The module's public predicate is `has_feature_doc(toggle_key)` (there is no `doc_for_toggle`);
`load_feature_doc(toggle_key)` returns the parsed doc or `None`.

```python
from gui.feature_docs import _SHARED_DOC_MAP, has_feature_doc, load_feature_doc


def test_every_new_toggle_has_a_doc():
    for key in ("feat_phoneme_arpabet", "feat_phoneme_stress",
                "feat_phoneme_ipa", "feat_phoneme_pron_source",
                "feat_syllable_ipa", "g2p_oov"):
        assert has_feature_doc(key), key
        assert load_feature_doc(key) is not None, key


def test_retired_toggle_is_gone():
    assert "feat_phoneme_label" not in _SHARED_DOC_MAP
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/gui/test_feature_docs.py -v`
Expected: FAIL — the new keys resolve to `None`.

- [ ] **Step 3: Write the docs and the routing**

`gui/feature_docs.py::_SHARED_DOC_MAP` — replace the `feat_phoneme_label` entry:

```python
    "feat_phoneme_arpabet":     "phoneme.md",
    "feat_phoneme_stress":      "phoneme.md",
    "feat_phoneme_ipa":         "feat_phoneme_ipa.md",
    "feat_phoneme_pron_source": "phoneme.md",
    "feat_phoneme_onset":    "phoneme.md",
    "feat_phoneme_end":      "phoneme.md",
    "feat_phoneme_duration": "phoneme.md",
    "feat_syllable_ipa":     "syllable_labels.md",
    "g2p_oov":               "g2p_oov.md",
```

Content requirements, following the existing front-matter shape in `docs/features/*.md`
(`gui_toggles:` / `output_columns:` keys):

- **`phoneme.md`** — rewrite the column table for `ARPABET` / `Stress` / `IPA` /
  `Pron_Source`. State explicitly that `Stress` is empty on consonants and `0` on unstressed
  vowels, and that these are different claims. Document the three `Pron_Source` values, and
  that `oov_unresolved` means MFA had no pronunciation — **not** that the word was unspoken.
  Note that a word sharing an MFA word with a preceding token (contractions, possessives)
  has its phonemes emitted once, under the first token, and that the word CSV's `Transcript`
  column identifies the shared MFA word.
- **`feat_phoneme_ipa.md`** — the mapping table, the stress-dependent `AH`/`ER` rows, that no
  `ˈ`/`ˌ` marks are emitted because `Stress` carries that, and that `IPA` depends on both
  `ARPABET` and `Stress`.
- **`syllable_labels.md`** — add the `IPA` row; state that `ARPABET` still carries stress
  digits at the syllable level (deliberately unchanged).
- **`g2p_oov.md`** — what G2P does, that the pronunciation is modeled from spelling rather
  than observed, that it defaults off, that enabling it re-runs alignment once, and — plainly
  — that it does **not** detect whether a word was actually spoken. Cross-reference
  `phoneme.md`'s `Pron_Source`.
- **`phoneme_surprisal.md`** — note that `spn` rows score `NaN`, distinct from the OOV cap.
- **`structural_ids.md`** — note that whitespace tokens are excluded, so `token_id` and
  `token_in_sentence_id` count emitted tokens only and `token_in_sentence_id` is a true
  sentence position.

`docs/master_plan.md`:
- Add the §3 completed-batch row.
- Update the §6 prefs-version note to v20 with the removal-migration description.
- Add a §2 validation row for this batch.
- **Correct the §4 Brain Treebank parity claim**: it stated `token_in_sentence_id` gives
  `Index_in_Sentence`, which was false for newline-containing transcripts before Task 10.

`docs/changelog.md`: full batch write-up including the investigation evidence (the `spn`
reproduction and the duplicate-row counts).

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/gui/test_feature_docs.py -v && python -m pytest -q`
Expected: PASS, full suite green.

- [ ] **Step 5: Commit**

```bash
git add docs/ gui/feature_docs.py tests/gui/test_feature_docs.py
git commit -m "docs: phoneme split/IPA/G2P feature docs; batch bookkeeping"
```

---

### Task 12: Real-stimulus validation gate

**Files:**
- Read: `validation/data/`, `validation/output/` (existing baseline)
- Create: `validation/verify_phoneme_batch.py`

**Interfaces:**
- Consumes: everything above.
- Produces: a pass/fail report. **No production code changes.**

**Context:** This gate is not optional. `tests/fixtures/audio/*.wav` are empty placeholders, so no test above exercises a real MFA-dependent path. The current `validation/output/` tree is the pre-change baseline — **copy it aside before re-running**, or the comparison target is destroyed.

- [ ] **Step 1: Preserve the baseline**

```bash
cp -r validation/output validation/output_baseline_pre_v20
```

- [ ] **Step 2: Write the verification script**

Create `validation/verify_phoneme_batch.py`:

```python
"""Post-change validation against the pre-v20 baseline.

Run AFTER re-running the pipeline over validation/data/. Compares the new
validation/output/ against validation/output_baseline_pre_v20/ and asserts the
spec's expected deltas -- and, crucially, that nothing else moved.

Usage:  python validation/verify_phoneme_batch.py
Exit 0 = all checks passed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
NEW = ROOT / "output"
OLD = ROOT / "output_baseline_pre_v20"

# Measured on the baseline; see spec section 10.
EXPECTED_PHONEME_DELTA = {"3_M": -6, "LJ003-0007": -5}
EXPECTED_SYLLABLE_DELTA = {"3_M": -2, "LJ003-0007": -2}

failures: list[str] = []


def check(cond: bool, msg: str) -> None:
    if not cond:
        failures.append(msg)


def _stems() -> list[str]:
    return sorted(p.name.replace("_features.csv", "")
                  for p in (OLD / "word").glob("*_features.csv"))


for stem in _stems():
    old_p = pd.read_csv(OLD / "phoneme" / f"{stem}_phonemes.csv")
    new_p = pd.read_csv(NEW / "phoneme" / f"{stem}_phonemes.csv")
    old_s = pd.read_csv(OLD / "syllable" / f"{stem}_syllables.csv")
    new_s = pd.read_csv(NEW / "syllable" / f"{stem}_syllables.csv")

    # 1. P_Center must be BIT-IDENTICAL. This is the silent regression the
    #    pcenter rewire could have introduced.
    if "P_Center" in old_s.columns:
        check("P_Center" in new_s.columns, f"{stem}: P_Center column vanished")
        if "P_Center" in new_s.columns:
            merged = new_s.merge(
                old_s[["doc_id", "sentence_id", "token_in_sentence_id",
                       "syllable_id", "P_Center"]],
                on=["doc_id", "sentence_id", "token_in_sentence_id", "syllable_id"],
                how="inner", suffixes=("_new", "_old"))
            bad = merged[merged["P_Center_new"] != merged["P_Center_old"]]
            check(bad.empty,
                  f"{stem}: {len(bad)} P_Center values changed (must be identical)")

    # 2. Row-count deltas are exactly the deduplicated copies.
    check(len(new_p) - len(old_p) == EXPECTED_PHONEME_DELTA.get(stem, 0),
          f"{stem}: phoneme rows {len(old_p)} -> {len(new_p)}, expected delta "
          f"{EXPECTED_PHONEME_DELTA.get(stem, 0)}")
    check(len(new_s) - len(old_s) == EXPECTED_SYLLABLE_DELTA.get(stem, 0),
          f"{stem}: syllable rows {len(old_s)} -> {len(new_s)}, expected delta "
          f"{EXPECTED_SYLLABLE_DELTA.get(stem, 0)}")

    # 3. No duplicate acoustic intervals remain.
    dup = new_p.duplicated(subset=["Onset", "End", "ARPABET"], keep="first").sum()
    check(dup == 0, f"{stem}: {dup} duplicate phoneme intervals remain")

    # 4. Stress is empty on consonants, populated on vowels.
    vowels = new_p["IPA"].isin(list("ɑæʌɔɛ iɪ"
                                    "uʊəɚɝ")) | \
        new_p["ARPABET"].isin(["AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER",
                               "EY", "IH", "IY", "OW", "OY", "UH", "UW"])
    check(new_p.loc[vowels, "Stress"].notna().all(),
          f"{stem}: some vowel rows have empty Stress")
    check(new_p.loc[~vowels & (new_p["ARPABET"] != "spn"), "Stress"].isna().all(),
          f"{stem}: some consonant rows have a Stress value")

    # 5. IPA populated for every real phone; Label column gone.
    real = new_p["ARPABET"] != "spn"
    check(new_p.loc[real, "IPA"].astype(str).str.len().gt(0).all(),
          f"{stem}: some non-spn rows have empty IPA")
    check("Label" not in new_p.columns, f"{stem}: Label column still present")

    # 6. Word CSV: no whitespace rows, contiguous ids, heads resolve.
    new_w = pd.read_csv(NEW / "word" / f"{stem}_features.csv")
    check((new_w["upos"] != "SPACE").all(), f"{stem}: SPACE rows still emitted")
    check(list(new_w["token_id"]) == list(range(1, len(new_w) + 1)),
          f"{stem}: token_id is not contiguous")
    check(set(new_w["head_token_id"]) <= set(new_w["token_id"]),
          f"{stem}: head_token_id points outside token_id")
    check(new_w["paragraph_id"].notna().all(),
          f"{stem}: paragraph_id has gaps")

    # 7. Columns that should be untouched are untouched. Compare against the
    #    baseline with its SPACE rows removed, since those rows are the only
    #    intended difference in the word CSV's row set.
    old_w = pd.read_csv(OLD / "word" / f"{stem}_features.csv")
    for col in ("Onset", "End", "Word_Duration", "Zipf_Freq"):
        if col in old_w.columns and col in new_w.columns:
            o = old_w.loc[old_w["upos"] != "SPACE", col].reset_index(drop=True)
            n = new_w[col].reset_index(drop=True)
            if len(o) == len(n):
                check(o.equals(n), f"{stem}: {col} values changed in the word CSV")

print(f"checked {len(_stems())} stimuli")
if failures:
    print(f"\n{len(failures)} FAILURES:")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("all checks passed")
```

- [ ] **Step 3: Re-run the pipeline over the real stimuli**

Enable the sub-word outputs (defaults are opt-in/off), then run headlessly:

```bash
conda activate ./env
python main.py \
  --transcripts-folder validation/data/transcripts \
  --audio-folder validation/data/audio \
  --output-folder validation/output \
  --analyzer validation
```

Watch the log for `[mfa] 10 stimuli: 0 cached, 10 aligned in 1 run` — the sidecar version
bump invalidates the cache exactly once. Confirm it reads `N cached, 0 aligned` on an
immediate second run, proving MFA still runs at most once per document.

- [ ] **Step 4: Run the verification and the OOV check**

```bash
python validation/verify_phoneme_batch.py
```
Expected: `all checks passed`.

Then the OOV behaviour, both ways. Build a one-stimulus corpus whose transcript contains a
word absent from `english_us_arpa.dict` (e.g. copy `validation/data/audio/1_M.wav` and write
a transcript with `Bella` replaced by `Zorblax`):

- With `--g2p-oov` **omitted**: the phoneme CSV has exactly one row for that word, with
  `ARPABET == 'spn'`, empty `Stress`/`IPA`, and `Pron_Source == 'oov_unresolved'`; stderr
  carries the `[phonemes] WARNING: … no phonemes` line.
- With `--g2p-oov`: that word has real phones with `Pron_Source == 'g2p'`, and no
  `oov_unresolved` rows remain.

- [ ] **Step 5: Verify the .mat sidecar and commit**

```bash
python - <<'PY'
import scipy.io, numpy as np
m = scipy.io.loadmat('validation/output/phoneme/3_M_phoneme_surprisal.mat')
b = scipy.io.loadmat('validation/output_baseline_pre_v20/phoneme/3_M_phoneme_surprisal.mat')
print('word_onset impulses  new:', int(m['word_onset'].sum()),
      ' baseline:', int(b['word_onset'].sum()))
print('max surprisal        new:', float(m['phoneme_surprisal'].max()),
      ' baseline:', float(b['phoneme_surprisal'].max()))
print('same length:', m['phoneme_surprisal'].shape == b['phoneme_surprisal'].shape)
PY
```

Expected: the new `word_onset` impulse count is **lower** (the duplicated contraction no
longer fires twice), the new max surprisal is **not higher** than baseline (no summed
collisions), and the lengths match.

```bash
git add validation/verify_phoneme_batch.py
git commit -m "test(validation): real-stimulus gate for the phoneme split batch"
```

Leave `validation/output_baseline_pre_v20/` uncommitted — it is a local comparison artifact.
Delete it once the gate passes, or add it to `.gitignore` if kept.

---

## Self-Review

**Spec coverage:**

| Spec § | Requirement | Task |
|---|---|---|
| 3 | Phoneme CSV `ARPABET`/`Stress`/`IPA`/`Pron_Source`, `Label` removed | 3 |
| 3 | `Stress` nullable `Int64`, empty on consonants | 3 |
| 4 | Syllable `IPA`, `Label`/`ARPABET` unchanged | 3 |
| 5 | `extractors/arpabet_ipa.py`, stress-aware `AH`/`ER` | 1 |
| 6.1 | Stop filtering `spn`; `spn` word yields one syllable + one phoneme row | 2, 3 |
| 6.1 | `add_phoneme_surprisal` returns `NaN` for `spn` | 5 |
| 6.2 | `--g2p_model_path`; sidecar `mfa_g2p` + `SIDECAR_VERSION` 2; g2p model download | 6 |
| 6.3 | `Pron_Source` from the memoized lexicon | 3 |
| 6.4 | Zero-phoneme warning, excluding punctuation and deduped tokens | 3 |
| 6.5 | Dedup shared alignments | 3 |
| 7 | SPACE skip, id renumber, head remap; paragraph logic untouched | 10 |
| 8 | Prefs v20 removal-migration; `alignment.g2p_oov` | 7 |
| 9 | Column maps, CLI, planner, GUI, docs; `session_units` unchanged | 8, 9, 11 |
| 10 | Unit, integration, schema-drift, real-stimulus validation | 1–12 |
| 11 | Non-goals respected: no confidence scoring, no clitic splitting, no syllable `Stress`, no sentence `paragraph_id` | — |

**Placeholder scan:** no TBD/TODO. Two places intentionally instruct the implementer to match
an existing local name rather than assume one (`doc_for_toggle` in `gui/feature_docs.py`,
`_migrate` in `preferences.py`, the download helper in `utils/download_models.py`, the toggle
dict in `section_subword_features.py`) — each says so explicitly and names the file to check.

**Type consistency:** `to_ipa(arpabet, stress)` / `syllable_to_ipa(pairs)` are used with those
exact signatures in Task 3. `Stress` is `Int64` everywhere it is asserted (Tasks 3, 4, 5).
`_run_mfa(..., *, g2p=False)`, `ensure_alignments(..., *, g2p=False)`, and
`align_with_phones(..., g2p=False)` match between Tasks 6 and 8. `ns.g2p_oov` / `args.g2p_oov`
name the same attribute in Tasks 8 and 9. `Pron_Source` values are the same three strings in
Tasks 3, 11, and 12.
