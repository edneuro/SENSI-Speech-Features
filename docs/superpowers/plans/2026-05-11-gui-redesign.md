# SensiExtractor GUI Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the tkinter GUI with a PySide6 sidebar/content interface organized around feature taxonomy, with continuous inline validation, an inline category picker, in-session API-key paste, and per-band EWFG toggles — while migrating preferences from schema v2 to v3 and renaming `liwc_*` doc-level columns to `psych_*`.

**Architecture:** New `gui/` Python package replaces the four flat `gui_*.py` modules. Each section (Inputs, Features, Run, Logs) is its own module loaded into a `QStackedWidget`. A reactive `preflight` engine listens to setting changes and surfaces inline warnings next to the offending toggle. A `run_dispatch` module calls existing pipeline functions (`run_word_level_batch`, `run_doc_level_batch`) in a background thread.

**Tech Stack:** PySide6 (Qt 6), tiktoken, pandas, pytest, pytest-qt. Existing extractors and CLI in `main.py` are unchanged.

**Spec:** `docs/superpowers/specs/2026-05-11-gui-redesign-design.md`

---

## Phase 1: Foundations (deps, schema, migration)

### Task 1: Add PySide6, tiktoken, pytest-qt to requirements

**Files:**
- Modify: `requirements.txt`
- Modify: `environment.yml` (conda env spec — add the same pip-installed packages)

- [ ] **Step 1: Inspect current requirements and env**

Read `requirements.txt` and `environment.yml` to understand existing structure.

- [ ] **Step 2: Add the three new dependencies**

Append to `requirements.txt` (after the `textstat` line):

```text

# GUI (PySide6)
PySide6>=6.6

# Token counting for OpenAI cost estimate
tiktoken>=0.6

# Testing GUI widgets
pytest-qt>=4.4
```

Append to `environment.yml` under the `pip:` section (mirror the same package list).

- [ ] **Step 3: Install into the active `sensi` conda env**

Run: `pip install PySide6>=6.6 tiktoken>=0.6 pytest-qt>=4.4`
Expected: clean install, no conflict with existing `torch`/`transformers`/`spacy`.

- [ ] **Step 4: Smoke test the import**

Run: `python -c "import PySide6.QtWidgets; import tiktoken; print('ok')"`
Expected: prints `ok` with no traceback.

- [ ] **Step 5: Commit**

```bash
git add requirements.txt environment.yml
git commit -m "deps: add PySide6, tiktoken, pytest-qt for GUI redesign"
```

---

### Task 2: Define preferences schema v3 (data shape only, no migration yet)

**Files:**
- Modify: `utils/preferences.py` (add v3 constants alongside the existing v2 constants)
- Test: `tests/utils/test_preferences.py` (extend existing test file)

- [ ] **Step 1: Write the failing test**

Add to `tests/utils/test_preferences.py`:

```python
def test_default_prefs_v3_has_expected_top_level_keys():
    from utils.preferences import DEFAULT_PREFS_V3
    assert DEFAULT_PREFS_V3["schema_version"] == 3
    assert set(DEFAULT_PREFS_V3.keys()) == {
        "schema_version", "shared", "features",
        "predictability", "semantic", "output",
    }


def test_default_prefs_v3_features_keys():
    from utils.preferences import DEFAULT_PREFS_V3
    expected = {
        "structural",
        "lexling_tokens", "lexling_pos", "lexling_morph", "lexling_dep",
        "freq_subtlex",
        "freq_ewfg_k2", "freq_ewfg_3to5", "freq_ewfg_6to8",
        "freq_ewfg_9to12", "freq_ewfg_overall",
        "pred_surprisal_gpt2", "pred_surprisal_llm",
        "pred_cloze", "pred_topmatch",
        "sem_indicators", "sem_composite_psych",
        "acoustic_boundaries",
    }
    assert set(DEFAULT_PREFS_V3["features"].keys()) == expected
    assert all(isinstance(v, bool) for v in DEFAULT_PREFS_V3["features"].values())


def test_default_prefs_v3_has_two_category_lists():
    from utils.preferences import DEFAULT_PREFS_V3
    assert "categories_liwc" in DEFAULT_PREFS_V3["semantic"]
    assert "categories_empath" in DEFAULT_PREFS_V3["semantic"]
    assert isinstance(DEFAULT_PREFS_V3["semantic"]["categories_liwc"], list)
    assert isinstance(DEFAULT_PREFS_V3["semantic"]["categories_empath"], list)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/utils/test_preferences.py::test_default_prefs_v3_has_expected_top_level_keys tests/utils/test_preferences.py::test_default_prefs_v3_features_keys tests/utils/test_preferences.py::test_default_prefs_v3_has_two_category_lists -v`
Expected: ImportError / AttributeError on `DEFAULT_PREFS_V3`.

- [ ] **Step 3: Add `DEFAULT_PREFS_V3` and `_SOP_CATEGORIES_LIWC` constants**

In `utils/preferences.py`, after the existing `_SOP_CATEGORIES` constant, add:

```python
SCHEMA_VERSION_V3 = 3

# Empath-style names (current 15 — kept as-is for backward compat / Empath backend)
_SOP_CATEGORIES_EMPATH = list(_SOP_CATEGORIES)

# LIWC-22 native category names (placeholder — to be confirmed against the .dic file).
# Names below are best-effort matches to the Empath SOP list.
_SOP_CATEGORIES_LIWC = [
    "affect", "cogproc", "emo_pos", "emo_neg",
    "body", "motion", "percept", "time", "space",
    "work", "family", "money", "relig", "focuspresent", "social",
]

_FEATURES_DEFAULT = {
    "structural": True,
    "lexling_tokens": True,
    "lexling_pos": True,
    "lexling_morph": True,
    "lexling_dep": True,
    "freq_subtlex": True,
    "freq_ewfg_k2": True,
    "freq_ewfg_3to5": True,
    "freq_ewfg_6to8": True,
    "freq_ewfg_9to12": True,
    "freq_ewfg_overall": True,
    "pred_surprisal_gpt2": True,
    "pred_surprisal_llm": True,
    "pred_cloze": True,
    "pred_topmatch": True,
    "sem_indicators": True,
    "sem_composite_psych": True,
    "acoustic_boundaries": True,
}

_OUTPUT_DEFAULT = {
    "word_level_csv": True,
    "document_summary": True,
    "doc_counts": True,
    "doc_aggregates": True,
    "doc_lex_diversity": True,
    "doc_readability": True,
    "doc_category_proportions": True,
}

DEFAULT_PREFS_V3: dict[str, Any] = {
    "schema_version": SCHEMA_VERSION_V3,
    "shared": {
        "analyzer_name": "",
        "last_transcripts_folder": "",
        "last_audio_folder": "",
        "last_output_folder": "",
        "theme": "system",
    },
    "features": dict(_FEATURES_DEFAULT),
    "predictability": {
        "llm_model": "",
    },
    "semantic": {
        "backend": "auto",
        "categories_liwc": list(_SOP_CATEGORIES_LIWC),
        "categories_empath": list(_SOP_CATEGORIES_EMPATH),
    },
    "output": dict(_OUTPUT_DEFAULT),
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/utils/test_preferences.py -v -k "v3"`
Expected: 3 passing tests.

- [ ] **Step 5: Commit**

```bash
git add utils/preferences.py tests/utils/test_preferences.py
git commit -m "prefs: add DEFAULT_PREFS_V3 constants (no migration yet)"
```

---

### Task 3: Implement v2 → v3 migration

**Files:**
- Modify: `utils/preferences.py` (add `_migrate_v2_into_v3`, update `load_preferences` and `save_preferences` to use v3)
- Test: `tests/utils/test_preferences_migration.py` (extend)

- [ ] **Step 1: Write the failing tests**

Append to `tests/utils/test_preferences_migration.py`:

```python
import json
import os
from pathlib import Path

import pytest


@pytest.fixture
def v2_prefs_file(tmp_path, monkeypatch):
    """A realistic v2 prefs file on disk, with SENSI_PREFS_PATH pointing at it."""
    p = tmp_path / "preferences.json"
    p.write_text(json.dumps({
        "schema_version": 2,
        "shared": {
            "analyzer_name": "Amilcar",
            "last_transcripts_folder": "/tmp/t",
            "last_audio_folder": "/tmp/a",
            "last_output_folder": "/tmp/o",
            "dictionary_backend": "liwc",
        },
        "word_level": {
            "feature_toggles": {
                "spacy_parse": True,
                "zipf_freq": True,
                "edu_freq": False,
                "sem_domain": True,
                "gpt2_surprisal": False,
                "llm_cloze": True,
                "alignment": True,
            },
            "semantic_categories": ["social", "cognition"],
            "llm_model": "gpt-4o-mini",
        },
        "document_level": {
            "feature_toggles": {
                "counts": True,
                "mean_zipf_freq": True,
                "mean_edu_zipf": False,
                "mean_surprisal": True,
                "mean_llm_cloze": True,
                "ttr": True,
                "mtld": False,
                "flesch_reading_ease": True,
                "flesch_kincaid_grade": True,
                "category_proportions": True,
                "liwc_authentic": True,
                "liwc_analytic": False,
                "liwc_tone": True,
                "liwc_clout": True,
            },
            "semantic_categories": ["social"],
            "llm_model": "gpt-4o-mini",
        },
    }), encoding="utf-8")
    monkeypatch.setenv("SENSI_PREFS_PATH", str(p))
    return p


def test_v2_file_loads_as_v3(v2_prefs_file):
    from utils.preferences import load_preferences
    prefs = load_preferences()
    assert prefs["schema_version"] == 3


def test_v2_migration_carries_shared_fields(v2_prefs_file):
    from utils.preferences import load_preferences
    prefs = load_preferences()
    assert prefs["shared"]["analyzer_name"] == "Amilcar"
    assert prefs["shared"]["last_transcripts_folder"] == "/tmp/t"


def test_v2_migration_splits_spacy_parse(v2_prefs_file):
    from utils.preferences import load_preferences
    prefs = load_preferences()
    f = prefs["features"]
    assert f["structural"] is True
    assert f["lexling_tokens"] is True
    assert f["lexling_pos"] is True
    assert f["lexling_morph"] is True
    assert f["lexling_dep"] is True


def test_v2_migration_splits_edu_freq_into_five_bands(v2_prefs_file):
    from utils.preferences import load_preferences
    prefs = load_preferences()
    f = prefs["features"]
    for band in ("freq_ewfg_k2", "freq_ewfg_3to5", "freq_ewfg_6to8",
                 "freq_ewfg_9to12", "freq_ewfg_overall"):
        assert f[band] is False, band


def test_v2_migration_splits_llm_cloze_into_three(v2_prefs_file):
    from utils.preferences import load_preferences
    prefs = load_preferences()
    f = prefs["features"]
    assert f["pred_surprisal_llm"] is True
    assert f["pred_cloze"] is True
    assert f["pred_topmatch"] is True


def test_v2_migration_maps_doc_toggles(v2_prefs_file):
    from utils.preferences import load_preferences
    prefs = load_preferences()
    o = prefs["output"]
    assert o["doc_counts"] is True
    assert o["doc_aggregates"] is True   # mean_* aggregated to one switch
    assert o["doc_lex_diversity"] is True  # ttr OR mtld OR'd → True
    assert o["doc_readability"] is True
    assert o["doc_category_proportions"] is True


def test_v2_migration_maps_liwc_to_psych(v2_prefs_file):
    from utils.preferences import load_preferences
    prefs = load_preferences()
    # at least one liwc_* was True in v2, so sem_composite_psych is True
    assert prefs["features"]["sem_composite_psych"] is True


def test_v2_migration_carries_semantic_categories_to_active_backend(v2_prefs_file):
    from utils.preferences import load_preferences
    prefs = load_preferences()
    # v2 dictionary_backend was liwc → categories go to categories_liwc
    assert prefs["semantic"]["categories_liwc"] == ["social", "cognition"]
    # categories_empath defaults to SOP
    from utils.preferences import _SOP_CATEGORIES_EMPATH
    assert prefs["semantic"]["categories_empath"] == _SOP_CATEGORIES_EMPATH


def test_v2_migration_carries_llm_model(v2_prefs_file):
    from utils.preferences import load_preferences
    prefs = load_preferences()
    assert prefs["predictability"]["llm_model"] == "gpt-4o-mini"


def test_v2_migration_carries_dictionary_backend(v2_prefs_file):
    from utils.preferences import load_preferences
    prefs = load_preferences()
    assert prefs["semantic"]["backend"] == "liwc"


def test_v2_migration_rewrites_file_in_v3(v2_prefs_file):
    from utils.preferences import load_preferences
    load_preferences()
    # File on disk should now be v3
    on_disk = json.loads(Path(v2_prefs_file).read_text(encoding="utf-8"))
    assert on_disk["schema_version"] == 3
    assert "features" in on_disk
    assert "word_level" not in on_disk
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/utils/test_preferences_migration.py -v`
Expected: all the new tests fail (load_preferences returns v2 / doesn't migrate).

- [ ] **Step 3: Implement migration**

In `utils/preferences.py`, replace the `SCHEMA_VERSION = 2` constant with v3 as the active schema, and update `load_preferences` + `_deep_merge_sections` to handle v3. Add a `_migrate_v2_into_v3` function:

```python
SCHEMA_VERSION = SCHEMA_VERSION_V3  # active version
DEFAULT_PREFS: dict[str, Any] = DEFAULT_PREFS_V3  # active defaults


def _migrate_v2_into_v3(merged: dict, v2: dict) -> None:
    """Convert v2-shape `v2` into the v3-shape `merged` (mutates merged)."""
    v2_shared = v2.get("shared", {})
    v2_word = v2.get("word_level", {})
    v2_doc = v2.get("document_level", {})
    v2_word_toggles = v2_word.get("feature_toggles", {})
    v2_doc_toggles = v2_doc.get("feature_toggles", {})

    # shared
    for key in ("analyzer_name", "last_transcripts_folder",
                "last_audio_folder", "last_output_folder"):
        if key in v2_shared:
            merged["shared"][key] = v2_shared[key]

    # backend
    if "dictionary_backend" in v2_shared:
        merged["semantic"]["backend"] = v2_shared["dictionary_backend"]

    # features
    f = merged["features"]
    spacy_on = bool(v2_word_toggles.get("spacy_parse", True))
    for sub in ("structural", "lexling_tokens", "lexling_pos",
                "lexling_morph", "lexling_dep"):
        f[sub] = spacy_on
    f["freq_subtlex"] = bool(v2_word_toggles.get("zipf_freq", True))
    edu_on = bool(v2_word_toggles.get("edu_freq", True))
    for band in ("freq_ewfg_k2", "freq_ewfg_3to5", "freq_ewfg_6to8",
                 "freq_ewfg_9to12", "freq_ewfg_overall"):
        f[band] = edu_on
    f["pred_surprisal_gpt2"] = bool(v2_word_toggles.get("gpt2_surprisal", True))
    llm_on = bool(v2_word_toggles.get("llm_cloze", True))
    f["pred_surprisal_llm"] = llm_on
    f["pred_cloze"] = llm_on
    f["pred_topmatch"] = llm_on
    f["sem_indicators"] = bool(v2_word_toggles.get("sem_domain", True))
    f["acoustic_boundaries"] = bool(v2_word_toggles.get("alignment", True))
    # composite psych = any of the four liwc_* was on in v2
    f["sem_composite_psych"] = any(
        bool(v2_doc_toggles.get(k, False))
        for k in ("liwc_authentic", "liwc_analytic", "liwc_tone", "liwc_clout")
    )

    # predictability
    if "llm_model" in v2_word:
        merged["predictability"]["llm_model"] = v2_word["llm_model"]

    # semantic categories — route to whichever backend was active
    backend = merged["semantic"]["backend"]
    cats = v2_word.get("semantic_categories")
    if cats is not None:
        if backend == "liwc":
            merged["semantic"]["categories_liwc"] = list(cats)
        elif backend == "empath":
            merged["semantic"]["categories_empath"] = list(cats)
        else:  # auto — write to both, user can change later
            merged["semantic"]["categories_liwc"] = list(cats)
            merged["semantic"]["categories_empath"] = list(cats)

    # output
    o = merged["output"]
    o["word_level_csv"] = True   # v2 always emitted word-level when word run
    o["document_summary"] = True # v2 always emitted doc summary when doc run
    o["doc_counts"] = bool(v2_doc_toggles.get("counts", True))
    o["doc_aggregates"] = any(
        bool(v2_doc_toggles.get(k, False))
        for k in ("mean_zipf_freq", "mean_edu_zipf",
                  "mean_surprisal", "mean_llm_cloze")
    )
    o["doc_lex_diversity"] = (
        bool(v2_doc_toggles.get("ttr", False))
        or bool(v2_doc_toggles.get("mtld", False))
    )
    o["doc_readability"] = (
        bool(v2_doc_toggles.get("flesch_reading_ease", False))
        or bool(v2_doc_toggles.get("flesch_kincaid_grade", False))
    )
    o["doc_category_proportions"] = bool(
        v2_doc_toggles.get("category_proportions", True)
    )
```

Then update `load_preferences`:

```python
def load_preferences() -> dict[str, Any]:
    path = get_preferences_path()
    merged: dict[str, Any] = deepcopy(DEFAULT_PREFS)

    if not path.exists():
        return merged
    try:
        with path.open("r", encoding="utf-8") as f:
            user = json.load(f)
    except (json.JSONDecodeError, OSError):
        return merged
    if not isinstance(user, dict):
        return merged

    version = user.get("schema_version", 1)
    if version == SCHEMA_VERSION_V3:
        _deep_merge_v3(merged, user)
    elif version == 2:
        _migrate_v2_into_v3(merged, user)
        _write(path, merged)
    else:
        # v1 flat → first do v1→v2 (existing logic), then v2→v3
        v2_intermediate = deepcopy(_DEFAULT_PREFS_V2)
        _migrate_v1_into(v2_intermediate, user)
        _migrate_v2_into_v3(merged, v2_intermediate)
        _write(path, merged)

    return merged
```

Add a `_deep_merge_v3` helper that merges a v3-shape dict on top of defaults (similar pattern to existing `_deep_merge_sections`, but for v3 keys: `shared`, `features`, `predictability`, `semantic`, `output`).

Preserve the existing v1 → v2 migration helper (`_migrate_v1_into`) so v1 files still work; chain through to v3. Keep `_DEFAULT_PREFS_V2` as a constant for that intermediate step.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/utils/test_preferences_migration.py -v`
Expected: all new tests pass.

- [ ] **Step 5: Run the existing prefs test file to verify no regressions**

Run: `pytest tests/utils/test_preferences.py tests/utils/test_preferences_migration.py -v`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add utils/preferences.py tests/utils/test_preferences_migration.py
git commit -m "prefs: migrate v2 → v3 with split toggles and dual category lists"
```

---

## Phase 2: Backend behavior changes (column renames, EWFG band granularity)

### Task 4: Per-band EWFG column filtering

**Files:**
- Modify: `extractors/word_freq.py` (or whichever file emits `Edu_Zipf_*` — locate first)
- Modify: `main.py` (extend `run_word_level_batch` signature to accept per-band toggles, defaulting to all-on for backward compat)
- Test: `tests/extractors/test_educator_freq.py` (extend)

- [ ] **Step 1: Locate the EWFG emit code**

Run: `grep -rn "Edu_Zipf" extractors/ main.py utils/`
Expected: identifies the file that creates the five columns.

- [ ] **Step 2: Write failing tests**

Add to `tests/extractors/test_educator_freq.py`:

```python
def test_band_filter_emits_only_selected_columns(monkeypatch, tmp_path):
    """When only k2 and overall are requested, other band columns must be absent."""
    from extractors.word_freq import add_edu_zipf  # adjust import to actual API
    import pandas as pd

    df = pd.DataFrame({"lemma": ["cat", "feline", "the"]})
    out = add_edu_zipf(df, bands=["k2", "overall"])
    assert "Edu_Zipf_K2" in out.columns
    assert "Edu_Zipf_Overall" in out.columns
    assert "Edu_Zipf_3to5" not in out.columns
    assert "Edu_Zipf_6to8" not in out.columns
    assert "Edu_Zipf_9to12" not in out.columns


def test_band_filter_all_bands_matches_previous_behavior():
    from extractors.word_freq import add_edu_zipf
    import pandas as pd

    df = pd.DataFrame({"lemma": ["cat"]})
    out = add_edu_zipf(df, bands=["k2", "3to5", "6to8", "9to12", "overall"])
    for col in ("Edu_Zipf_K2", "Edu_Zipf_3to5", "Edu_Zipf_6to8",
                "Edu_Zipf_9to12", "Edu_Zipf_Overall"):
        assert col in out.columns
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/extractors/test_educator_freq.py -v -k "band_filter"`
Expected: TypeError on `bands=` argument.

- [ ] **Step 4: Implement band filtering**

In the EWFG extractor:
- Add `bands: list[str] | None = None` parameter (None means "all bands" for backward compat).
- After computing all five columns, drop any column whose band suffix is not in `bands`.
- Map: `"k2" → "Edu_Zipf_K2", "3to5" → "Edu_Zipf_3to5", "6to8" → "Edu_Zipf_6to8", "9to12" → "Edu_Zipf_9to12", "overall" → "Edu_Zipf_Overall"`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/extractors/test_educator_freq.py -v`
Expected: green.

- [ ] **Step 6: Thread the `bands` parameter through `main.py`**

In `main.run_word_level_batch` (and the CLI argparse setup if it has `--no-edu-freq`):
- Accept a `edu_bands: list[str] | None = None` keyword that gets passed to `add_edu_zipf`.
- Default `None` preserves current CLI behavior.

- [ ] **Step 7: Commit**

```bash
git add extractors/ main.py tests/extractors/test_educator_freq.py
git commit -m "edu-freq: per-band column filtering"
```

---

### Task 5: Rename `liwc_*` doc-level CSV columns to `psych_*`

**Files:**
- Modify: `extractors/liwc_summary.py` (or whichever file emits the four columns — locate)
- Modify: `main.py` if it references the old column names anywhere
- Modify: `Feature_Extraction_Guide.md` (update column-name documentation)
- Test: `tests/extractors/test_liwc_summary.py`

- [ ] **Step 1: Locate the existing column emission**

Run: `grep -rn "liwc_authentic\|liwc_analytic\|liwc_tone\|liwc_clout" extractors/ main.py tests/`
Expected: pinpoints the column names in code and docs.

- [ ] **Step 2: Update tests first**

Replace every occurrence of `liwc_authentic / liwc_analytic / liwc_tone / liwc_clout` in `tests/extractors/test_liwc_summary.py` with `psych_authentic / psych_analytic / psych_tone / psych_clout`. Add an assertion that the old names are NOT present:

```python
def test_old_liwc_column_names_are_gone():
    # Build a result DF using the public function and assert
    # only the new psych_ names appear, never the old liwc_ ones.
    from extractors.liwc_summary import compute_summary
    import pandas as pd
    df = pd.DataFrame({"token": ["I", "think"], "upos": ["PRON", "VERB"]})
    out = compute_summary(df)  # adjust signature to actual API
    for old in ("liwc_authentic", "liwc_analytic", "liwc_tone", "liwc_clout"):
        assert old not in out.columns, f"old column {old} still present"
    for new in ("psych_authentic", "psych_analytic", "psych_tone", "psych_clout"):
        assert new in out.columns, f"new column {new} missing"
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/extractors/test_liwc_summary.py -v`
Expected: failing tests.

- [ ] **Step 4: Rename columns in the extractor**

In `extractors/liwc_summary.py`:
- Replace every `liwc_authentic` → `psych_authentic`, and the same for the other three.
- Ensure no `liwc_*` column names remain.

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/extractors/test_liwc_summary.py -v`
Expected: green.

- [ ] **Step 6: Update documentation**

In `Feature_Extraction_Guide.md`:
- In the "LIWC summary scores" subsection of Batch 1.5: rename to "Composite psychological scores (Pennebaker-formula approximations)."
- Update the column names listed there.
- Add a 1-paragraph note: "These are approximations computed from publicly documented Pennebaker et al. (2014) formulas applied to the loaded LIWC-style dictionary. For LIWC-22's official Authentic/Analytic/Tone/Clout numbers, use the LIWC desktop software (liwc.app). The dictionary used in this run is recorded in the run manifest."

- [ ] **Step 7: Run the full doc-summary test file**

Run: `pytest tests/utils/test_doc_summary.py tests/extractors/test_liwc_summary.py -v`
Expected: green.

- [ ] **Step 8: Commit**

```bash
git add extractors/ main.py tests/extractors/test_liwc_summary.py Feature_Extraction_Guide.md
git commit -m "doc-summary: rename liwc_* columns to psych_* (honest naming)"
```

---

### Task 6: Verify document-level pipeline independence

The spec requires that `run_doc_level_batch` works regardless of which word-level toggles are set — it must compute its own intermediates.

**Files:**
- Read: `main.py` (`run_word_level_batch`, `run_doc_level_batch`)
- Possibly modify: `main.py` if doc-level currently depends on word-level toggles
- Test: `tests/utils/test_doc_summary.py` (extend)

- [ ] **Step 1: Audit doc-level pipeline for word-level dependencies**

Run: `grep -n "feature_toggles\|toggles" main.py`
Read the relevant functions and identify: does `run_doc_level_batch` skip a word-level computation if its corresponding word-level toggle is off?

- [ ] **Step 2: Write failing test (if dependency exists)**

Add to `tests/utils/test_doc_summary.py`:

```python
def test_doc_summary_computes_mean_zipf_even_when_word_zipf_disabled(tmp_path):
    """Document-level mean_zipf must compute regardless of word-level freq_subtlex toggle."""
    from main import run_doc_level_batch  # adjust import / signature to actual API
    # Build minimal CLI args namespace where word-level freq_subtlex is off but
    # doc-level aggregate is on. Run with one tiny stimulus and confirm the
    # output summary CSV contains mean_zipf_freq with a non-NaN value.
    # (Implementation detail: pass a Namespace with --no-edu-freq set and verify
    #  the mean_zipf_freq column exists.)
    ...
```

(Sketch this fully against the actual function signature once it's read in step 1.)

- [ ] **Step 3: Run tests to verify they fail (or pass if no dependency exists)**

Run: `pytest tests/utils/test_doc_summary.py -v -k "independence"`
- If failing: doc-level has dependencies that need fixing.
- If passing: the pipeline already meets the spec; no code change needed for this task.

- [ ] **Step 4: Fix dependencies if needed**

If doc-level skipped a computation when the word-level toggle was off:
- Decouple: doc-level should call the extractor directly with its own (always-on) requirements, ignoring word-level toggles entirely.
- Word-level toggles only filter the *output CSV columns* for the word-level CSV, never gate computation.

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/utils/test_doc_summary.py -v`
Expected: green.

- [ ] **Step 6: Commit**

```bash
git add main.py tests/utils/test_doc_summary.py
git commit -m "doc-pipeline: decouple from word-level toggles (independence)"
```

If no change was needed, skip the commit — just record in the plan that this task was an audit only.

---

## Phase 3: GUI package skeleton

### Task 7: Create `gui/` package and theme tokens

**Files:**
- Create: `gui/__init__.py` (empty)
- Create: `gui/theme.py`
- Test: `tests/gui/test_theme.py`
- Create: `tests/gui/__init__.py` (empty)

- [ ] **Step 1: Create the empty package skeleton**

Create empty `gui/__init__.py` and `tests/gui/__init__.py`.

- [ ] **Step 2: Write failing test**

Create `tests/gui/test_theme.py`:

```python
def test_accent_color_is_hex():
    from gui.theme import ACCENT_COLOR
    assert ACCENT_COLOR.startswith("#")
    assert len(ACCENT_COLOR) == 7


def test_grid_unit_is_8():
    from gui.theme import GRID_UNIT
    assert GRID_UNIT == 8


def test_qss_for_appearance_returns_valid_css(qtbot):
    from gui.theme import qss_for_appearance
    light = qss_for_appearance("light")
    dark = qss_for_appearance("dark")
    assert "background" in light and "background" in dark
    assert light != dark
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/gui/test_theme.py -v`
Expected: ImportError on `gui.theme`.

- [ ] **Step 4: Implement `gui/theme.py`**

```python
"""Theme tokens and QSS stylesheets for the SensiExtractor GUI."""

from pathlib import Path


ACCENT_COLOR = "#5B5BD6"   # indigo — works on light and dark backgrounds
GRID_UNIT = 8              # 8px spacing grid
CARD_RADIUS = 8
BUTTON_RADIUS = 6
INPUT_RADIUS = 4

FONT_BODY = ("Segoe UI", 10)
FONT_HEADER = ("Segoe UI", 12)
FONT_MONO = ("JetBrains Mono", 10)


_LIGHT_QSS = """
QMainWindow, QWidget { background: #FAFAFA; color: #1A1A1A; }
QFrame#Sidebar { background: #F2F2F4; border-right: 1px solid #E3E3E6; }
QFrame#Card { background: #FFFFFF; border: 1px solid #E3E3E6; border-radius: 8px; }
QPushButton#Primary {
    background: #5B5BD6; color: white;
    padding: 8px 16px; border-radius: 6px; border: none;
    font-weight: 600;
}
QPushButton#Primary:hover { background: #4848C0; }
QPushButton#Primary:disabled { background: #B8B8E0; }
QLabel#SectionHeader { font-size: 13px; font-weight: 600; padding-bottom: 4px; }
QLabel#Subtitle { color: #6B6B70; font-size: 11px; }
QLineEdit { padding: 6px 8px; border: 1px solid #D1D1D6; border-radius: 4px; }
QLineEdit:focus { border-color: #5B5BD6; }
QLineEdit[error="true"] { border-color: #D14A3F; }
"""

_DARK_QSS = """
QMainWindow, QWidget { background: #1C1C1E; color: #ECECEE; }
QFrame#Sidebar { background: #2C2C2E; border-right: 1px solid #3A3A3C; }
QFrame#Card { background: #2C2C2E; border: 1px solid #3A3A3C; border-radius: 8px; }
QPushButton#Primary {
    background: #5B5BD6; color: white;
    padding: 8px 16px; border-radius: 6px; border: none;
    font-weight: 600;
}
QPushButton#Primary:hover { background: #6F6FE4; }
QPushButton#Primary:disabled { background: #3A3A6A; }
QLabel#SectionHeader { font-size: 13px; font-weight: 600; padding-bottom: 4px; }
QLabel#Subtitle { color: #9B9BA0; font-size: 11px; }
QLineEdit { padding: 6px 8px; background: #1C1C1E; color: #ECECEE; border: 1px solid #3A3A3C; border-radius: 4px; }
QLineEdit:focus { border-color: #5B5BD6; }
QLineEdit[error="true"] { border-color: #FF6B5E; }
"""


def qss_for_appearance(mode: str) -> str:
    if mode == "dark":
        return _DARK_QSS
    return _LIGHT_QSS


def detect_system_appearance() -> str:
    """Return 'light' or 'dark' based on OS palette."""
    from PySide6.QtGui import QGuiApplication
    scheme = QGuiApplication.styleHints().colorScheme()
    return "dark" if scheme.name == "Dark" else "light"
```

- [ ] **Step 5: Configure pytest-qt fixture**

If `tests/conftest.py` doesn't already provide `qtbot`, install it transitively via `pytest-qt` (already a dependency from Task 1). The `qtbot` fixture is provided automatically by `pytest-qt`.

- [ ] **Step 6: Run tests**

Run: `pytest tests/gui/test_theme.py -v`
Expected: green.

- [ ] **Step 7: Commit**

```bash
git add gui/__init__.py gui/theme.py tests/gui/__init__.py tests/gui/test_theme.py
git commit -m "gui: theme tokens and QSS stylesheets"
```

---

### Task 8: Reusable widgets

**Files:**
- Create: `gui/widgets.py`
- Test: `tests/gui/test_widgets.py`

- [ ] **Step 1: Write failing tests**

Create `tests/gui/test_widgets.py`:

```python
def test_section_header_renders_text(qtbot):
    from gui.widgets import SectionHeader
    w = SectionHeader("Frequency", subtitle="Lexical-corpus frequency lookups.")
    qtbot.addWidget(w)
    assert w.title_label.text() == "Frequency"
    assert w.subtitle_label.text() == "Lexical-corpus frequency lookups."


def test_folder_row_emits_signal_on_text_change(qtbot):
    from gui.widgets import FolderRow
    w = FolderRow(label="Transcripts")
    qtbot.addWidget(w)
    captured = []
    w.path_changed.connect(lambda p: captured.append(p))
    w.set_path("/tmp/abc")
    assert captured == ["/tmp/abc"]


def test_toggle_row_emits_signal_on_toggle(qtbot):
    from gui.widgets import ToggleRow
    w = ToggleRow(label="Surprisal from GPT-2")
    qtbot.addWidget(w)
    captured = []
    w.toggled.connect(lambda v: captured.append(v))
    w.set_checked(True)
    assert captured == [True]


def test_amber_dot_indicator_shows_tooltip(qtbot):
    from gui.widgets import AmberDotIndicator
    w = AmberDotIndicator("EWFG parquet not found")
    qtbot.addWidget(w)
    assert w.toolTip() == "EWFG parquet not found"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/gui/test_widgets.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `gui/widgets.py`**

```python
"""Reusable Qt widgets used across the GUI."""

from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QVBoxLayout,
    QLabel, QLineEdit, QPushButton, QCheckBox,
    QFileDialog, QWidget,
)


class SectionHeader(QWidget):
    def __init__(self, title: str, subtitle: str = "", parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        self.title_label = QLabel(title)
        self.title_label.setObjectName("SectionHeader")
        layout.addWidget(self.title_label)
        self.subtitle_label = QLabel(subtitle)
        self.subtitle_label.setObjectName("Subtitle")
        if not subtitle:
            self.subtitle_label.hide()
        layout.addWidget(self.subtitle_label)
        divider = QFrame()
        divider.setFrameShape(QFrame.HLine)
        divider.setStyleSheet("color: #E3E3E6;")
        layout.addWidget(divider)


class FolderRow(QWidget):
    path_changed = Signal(str)

    def __init__(self, label: str, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        lbl = QLabel(label)
        lbl.setMinimumWidth(120)
        self.line = QLineEdit()
        self.line.textChanged.connect(self._on_text)
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._on_browse)
        layout.addWidget(lbl)
        layout.addWidget(self.line, 1)
        layout.addWidget(browse)

    def set_path(self, p: str) -> None:
        self.line.setText(p)

    def path(self) -> str:
        return self.line.text()

    def set_error(self, is_error: bool) -> None:
        self.line.setProperty("error", "true" if is_error else "false")
        self.line.style().unpolish(self.line)
        self.line.style().polish(self.line)

    def _on_text(self, text: str) -> None:
        self.path_changed.emit(text)

    def _on_browse(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select folder", self.line.text())
        if path:
            self.line.setText(path)


class ToggleRow(QWidget):
    toggled = Signal(bool)

    def __init__(self, label: str, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.check = QCheckBox(label)
        self.check.toggled.connect(self.toggled)
        layout.addWidget(self.check)
        layout.addStretch(1)
        self.warning = None  # populated by set_warning

    def set_checked(self, v: bool) -> None:
        self.check.setChecked(v)

    def is_checked(self) -> bool:
        return self.check.isChecked()

    def set_warning(self, message: str | None) -> None:
        if self.warning is not None:
            self.layout().removeWidget(self.warning)
            self.warning.deleteLater()
            self.warning = None
        if message:
            self.warning = AmberDotIndicator(message)
            self.layout().insertWidget(1, self.warning)


class AmberDotIndicator(QLabel):
    def __init__(self, tooltip: str, parent=None):
        super().__init__("🟡", parent)
        self.setToolTip(tooltip)
        self.setFixedWidth(20)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/gui/test_widgets.py -v`
Expected: green.

- [ ] **Step 5: Commit**

```bash
git add gui/widgets.py tests/gui/test_widgets.py
git commit -m "gui: reusable SectionHeader, FolderRow, ToggleRow, AmberDotIndicator"
```

---

### Task 9: Main window shell with sidebar + QStackedWidget

**Files:**
- Create: `gui/main_window.py`
- Create: `gui/status_strip.py`
- Modify: `gui.py` (rewrite as a thin entry point)
- Test: `tests/gui/test_main_window.py`

- [ ] **Step 1: Write failing tests**

Create `tests/gui/test_main_window.py`:

```python
def test_main_window_has_sidebar_with_four_items(qtbot):
    from gui.main_window import MainWindow
    w = MainWindow()
    qtbot.addWidget(w)
    labels = [w.sidebar.item(i).text() for i in range(w.sidebar.count())]
    assert labels == ["Inputs", "Features", "Run", "Logs"]


def test_main_window_default_section_is_inputs(qtbot):
    from gui.main_window import MainWindow
    w = MainWindow()
    qtbot.addWidget(w)
    assert w.stack.currentIndex() == 0


def test_main_window_clicking_sidebar_switches_stack(qtbot):
    from gui.main_window import MainWindow
    w = MainWindow()
    qtbot.addWidget(w)
    w.sidebar.setCurrentRow(2)  # Run
    assert w.stack.currentIndex() == 2


def test_main_window_min_size(qtbot):
    from gui.main_window import MainWindow
    w = MainWindow()
    qtbot.addWidget(w)
    assert w.minimumWidth() >= 1000
    assert w.minimumHeight() >= 700
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/gui/test_main_window.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `gui/status_strip.py`**

```python
"""Persistent bottom status strip."""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QProgressBar, QPushButton,
)


class StatusStrip(QFrame):
    cancel_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("StatusStrip")
        self.setFixedHeight(32)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 4, 12, 4)
        self.state_label = QLabel("○  Idle")
        self.progress = QProgressBar()
        self.progress.setMaximum(100)
        self.progress.setValue(0)
        self.progress.setFixedWidth(280)
        self.detail_label = QLabel("Ready to run.")
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel_requested)
        layout.addWidget(self.state_label)
        layout.addWidget(self.progress)
        layout.addWidget(self.detail_label, 1)
        layout.addWidget(self.cancel_btn)

    def set_idle(self) -> None:
        self.state_label.setText("○  Idle")
        self.detail_label.setText("Ready to run.")
        self.progress.setValue(0)
        self.cancel_btn.setEnabled(False)

    def set_running(self, current: int, total: int, file_name: str) -> None:
        self.state_label.setText("●  Running")
        pct = int(100 * current / total) if total else 0
        self.progress.setValue(pct)
        self.detail_label.setText(f"{current}/{total}   {file_name}")
        self.cancel_btn.setEnabled(True)

    def set_done(self, total: int, elapsed_s: float) -> None:
        self.state_label.setText("✓  Done")
        self.progress.setValue(100)
        self.detail_label.setText(f"{total}/{total} files in {elapsed_s:.0f}s")
        self.cancel_btn.setEnabled(False)

    def set_errored(self, n_errors: int) -> None:
        self.state_label.setText("⚠  Errored")
        self.detail_label.setText(f"{n_errors} files failed — see Logs.")
        self.cancel_btn.setEnabled(False)
```

- [ ] **Step 4: Implement `gui/main_window.py`**

```python
"""Main window shell: sidebar + content stack + status strip."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QListWidget, QListWidgetItem, QStackedWidget,
    QFrame, QLabel, QScrollArea,
)

from gui.theme import qss_for_appearance, detect_system_appearance
from gui.status_strip import StatusStrip
from utils.preferences import load_preferences


SECTIONS = ["Inputs", "Features", "Run", "Logs"]


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("SensiExtractor")
        self.setMinimumSize(1000, 700)
        self.resize(1200, 800)

        self._prefs = load_preferences()
        self._apply_theme()

        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        body = QWidget()
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(0)

        # Sidebar
        sidebar_frame = QFrame()
        sidebar_frame.setObjectName("Sidebar")
        sidebar_frame.setFixedWidth(220)
        sb_layout = QVBoxLayout(sidebar_frame)
        sb_layout.setContentsMargins(8, 8, 8, 8)
        self.sidebar = QListWidget()
        self.sidebar.setFrameShape(QFrame.NoFrame)
        for name in SECTIONS:
            self.sidebar.addItem(QListWidgetItem(name))
        self.sidebar.setCurrentRow(0)
        sb_layout.addWidget(self.sidebar)
        sb_layout.addStretch(1)
        self.sidebar_footer = QLabel(self._footer_text())
        self.sidebar_footer.setObjectName("Subtitle")
        self.sidebar_footer.setWordWrap(True)
        sb_layout.addWidget(self.sidebar_footer)

        # Content stack (each entry is a placeholder QWidget for now;
        # later tasks replace these with the section widgets).
        self.stack = QStackedWidget()
        for name in SECTIONS:
            placeholder = QScrollArea()
            placeholder.setWidgetResizable(True)
            inner = QWidget()
            inner_layout = QVBoxLayout(inner)
            inner_layout.addWidget(QLabel(f"{name} section (placeholder)"))
            inner_layout.addStretch(1)
            placeholder.setWidget(inner)
            self.stack.addWidget(placeholder)

        self.sidebar.currentRowChanged.connect(self.stack.setCurrentIndex)

        body_layout.addWidget(sidebar_frame)
        body_layout.addWidget(self.stack, 1)

        # Status strip
        self.status_strip = StatusStrip()

        root.addWidget(body, 1)
        root.addWidget(self.status_strip)

        self.setCentralWidget(central)

    def _footer_text(self) -> str:
        analyzer = self._prefs["shared"].get("analyzer_name") or "(no analyzer)"
        backend = self._prefs["semantic"].get("backend", "auto")
        return f"Analyzer: {analyzer}\nBackend: {backend}"

    def _apply_theme(self) -> None:
        mode = self._prefs["shared"].get("theme", "system")
        if mode == "system":
            mode = detect_system_appearance()
        self.setStyleSheet(qss_for_appearance(mode))
```

- [ ] **Step 5: Rewrite `gui.py` as entry point**

Replace the current contents of `gui.py` with:

```python
"""SensiExtractor — PySide6 entry point.

Run: python gui.py
"""

import os
import sys
from pathlib import Path

_HERE = Path(__file__).parent.resolve()
os.environ.setdefault("HF_HOME",      str(_HERE / "hf_cache"))
os.environ.setdefault("MFA_ROOT_DIR", str(_HERE / "mfa_data"))
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")

from PySide6.QtWidgets import QApplication

from gui.main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("SensiExtractor")
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/gui/test_main_window.py -v`
Expected: green.

- [ ] **Step 7: Smoke test the window opens**

Run: `python gui.py`
Expected: window opens with sidebar and four placeholder pages. Close the window.

- [ ] **Step 8: Commit**

```bash
git add gui/ gui.py tests/gui/test_main_window.py
git commit -m "gui: main window shell with sidebar, stack, and status strip"
```

---

## Phase 4: Section pages

### Task 10: Inputs section with continuous validation

**Files:**
- Create: `gui/section_inputs.py`
- Test: `tests/gui/test_section_inputs.py`
- Modify: `gui/main_window.py` (swap placeholder for the real Inputs widget)

- [ ] **Step 1: Write failing tests**

Create `tests/gui/test_section_inputs.py`:

```python
def test_empty_analyzer_shows_error(qtbot):
    from gui.section_inputs import InputsSection
    w = InputsSection()
    qtbot.addWidget(w)
    w.analyzer_line.setText("")
    assert w.analyzer_line.property("error") == "true"


def test_filled_analyzer_clears_error(qtbot):
    from gui.section_inputs import InputsSection
    w = InputsSection()
    qtbot.addWidget(w)
    w.analyzer_line.setText("Amilcar")
    assert w.analyzer_line.property("error") != "true"


def test_nonexistent_folder_shows_error(qtbot, tmp_path):
    from gui.section_inputs import InputsSection
    w = InputsSection()
    qtbot.addWidget(w)
    w.transcripts_row.set_path("/does/not/exist")
    assert w.transcripts_row.line.property("error") == "true"


def test_valid_folder_shows_file_count(qtbot, tmp_path):
    from gui.section_inputs import InputsSection
    (tmp_path / "a.txt").write_text("hello", encoding="utf-8")
    (tmp_path / "b.txt").write_text("world", encoding="utf-8")
    w = InputsSection()
    qtbot.addWidget(w)
    w.transcripts_row.set_path(str(tmp_path))
    assert "2 files" in w.transcripts_count_label.text()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/gui/test_section_inputs.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `gui/section_inputs.py`**

```python
"""Inputs section: analyzer name, folders, pairing preview."""

from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QTableView,
    QHeaderView,
)
from PySide6.QtGui import QStandardItemModel, QStandardItem

from gui.widgets import SectionHeader, FolderRow
from utils.preferences import load_preferences, save_preferences


TRANSCRIPT_EXTS = {".txt", ".tsv", ".md"}
AUDIO_EXTS = {".wav", ".flac", ".mp3"}


class InputsSection(QWidget):
    settings_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 16, 24, 16)
        layout.setSpacing(16)

        prefs = load_preferences()["shared"]

        # Analyzer
        layout.addWidget(SectionHeader(
            "Analyzer",
            "Recorded in every run manifest and log file.",
        ))
        self.analyzer_line = QLineEdit(prefs.get("analyzer_name", ""))
        self.analyzer_line.textChanged.connect(self._on_analyzer_change)
        layout.addWidget(self.analyzer_line)
        self._on_analyzer_change(self.analyzer_line.text())

        # Folders
        layout.addWidget(SectionHeader("Folders"))
        self.transcripts_row, self.transcripts_count_label = self._folder_block(
            layout, "Transcripts folder:", prefs.get("last_transcripts_folder", ""),
            self._on_transcripts_change)
        self.audio_row, self.audio_count_label = self._folder_block(
            layout, "Audio folder:", prefs.get("last_audio_folder", ""),
            self._on_audio_change)
        self.output_row, self.output_count_label = self._folder_block(
            layout, "Output folder:", prefs.get("last_output_folder", ""),
            self._on_output_change)

        # Pairing preview
        layout.addWidget(SectionHeader("Pairing preview"))
        self.pairing_summary = QLabel("No transcripts/audio selected yet.")
        layout.addWidget(self.pairing_summary)
        self.pairing_table = QTableView()
        self.pairing_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.pairing_table, 1)

        # Initial validation pass
        self._on_transcripts_change(self.transcripts_row.path())
        self._on_audio_change(self.audio_row.path())
        self._on_output_change(self.output_row.path())

    def _folder_block(self, parent_layout, label, initial_path, callback):
        wrapper = QWidget()
        v = QVBoxLayout(wrapper)
        v.setContentsMargins(0, 0, 0, 0)
        row = FolderRow(label=label)
        row.set_path(initial_path)
        count_label = QLabel("")
        count_label.setObjectName("Subtitle")
        v.addWidget(row)
        v.addWidget(count_label)
        parent_layout.addWidget(wrapper)
        row.path_changed.connect(callback)
        return row, count_label

    def _on_analyzer_change(self, text: str) -> None:
        is_empty = not text.strip()
        self.analyzer_line.setProperty("error", "true" if is_empty else "false")
        self.analyzer_line.style().unpolish(self.analyzer_line)
        self.analyzer_line.style().polish(self.analyzer_line)
        save_preferences({"shared": {"analyzer_name": text.strip()}})
        self.settings_changed.emit()

    def _validate_and_count(self, path_str: str, exts: set[str], count_label: QLabel,
                            row: FolderRow) -> int:
        if not path_str.strip():
            row.set_error(False)
            count_label.setText("")
            return -1
        p = Path(path_str)
        if not p.is_dir():
            row.set_error(True)
            count_label.setText("Folder does not exist.")
            return -1
        row.set_error(False)
        if not exts:
            count_label.setText("(created if missing)")
            return 0
        n = sum(1 for f in p.iterdir() if f.is_file() and f.suffix.lower() in exts)
        count_label.setText(f"✓ {n} files")
        return n

    def _on_transcripts_change(self, path: str) -> None:
        self._validate_and_count(path, TRANSCRIPT_EXTS,
                                  self.transcripts_count_label, self.transcripts_row)
        save_preferences({"shared": {"last_transcripts_folder": path.strip()}})
        self._refresh_pairing()
        self.settings_changed.emit()

    def _on_audio_change(self, path: str) -> None:
        self._validate_and_count(path, AUDIO_EXTS,
                                  self.audio_count_label, self.audio_row)
        save_preferences({"shared": {"last_audio_folder": path.strip()}})
        self._refresh_pairing()
        self.settings_changed.emit()

    def _on_output_change(self, path: str) -> None:
        self._validate_and_count(path, set(),  # any folder OK
                                  self.output_count_label, self.output_row)
        save_preferences({"shared": {"last_output_folder": path.strip()}})
        self.settings_changed.emit()

    def _refresh_pairing(self) -> None:
        t = Path(self.transcripts_row.path() or "")
        a = Path(self.audio_row.path() or "")
        if not (t.is_dir() and a.is_dir()):
            self.pairing_summary.setText("Pick both folders to see pairing.")
            self.pairing_table.setModel(QStandardItemModel(self))
            return
        ts = {p.stem.lower(): p.name for p in t.iterdir()
              if p.is_file() and p.suffix.lower() in TRANSCRIPT_EXTS}
        au = {p.stem.lower(): p.name for p in a.iterdir()
              if p.is_file() and p.suffix.lower() in AUDIO_EXTS}
        matched = sorted(set(ts) & set(au))
        unmatched_t = sorted(set(ts) - set(au))
        unmatched_a = sorted(set(au) - set(ts))
        self.pairing_summary.setText(
            f"{len(ts)} transcripts × {len(au)} audio  →  {len(matched)} matched pairs, "
            f"{len(unmatched_t) + len(unmatched_a)} unmatched"
        )
        model = QStandardItemModel(0, 3, self)
        model.setHorizontalHeaderLabels(["Transcript", "Audio", "Status"])
        for stem in matched:
            model.appendRow([QStandardItem(ts[stem]),
                             QStandardItem(au[stem]),
                             QStandardItem("paired")])
        for stem in unmatched_t:
            model.appendRow([QStandardItem(ts[stem]),
                             QStandardItem(""),
                             QStandardItem("no audio match")])
        for stem in unmatched_a:
            model.appendRow([QStandardItem(""),
                             QStandardItem(au[stem]),
                             QStandardItem("no transcript match")])
        self.pairing_table.setModel(model)
```

- [ ] **Step 4: Wire into `main_window.py`**

In `gui/main_window.py`, replace the Inputs placeholder by importing and instantiating `InputsSection`:

```python
from gui.section_inputs import InputsSection
# ...
# Replace the loop that creates 4 placeholders with explicit instantiation:
sections = [
    InputsSection(),
    QLabel("Features (placeholder)"),
    QLabel("Run (placeholder)"),
    QLabel("Logs (placeholder)"),
]
for s in sections:
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setWidget(s)
    self.stack.addWidget(area)
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/gui/test_section_inputs.py -v`
Expected: green.

- [ ] **Step 6: Smoke test**

Run: `python gui.py`
Expected: Inputs page now shows analyzer + folder rows + pairing preview. Pick a real folder; see file count update. Type a non-path; see red border.

- [ ] **Step 7: Commit**

```bash
git add gui/section_inputs.py gui/main_window.py tests/gui/test_section_inputs.py
git commit -m "gui: Inputs section with continuous validation and pairing preview"
```

---

### Task 11: Features section — Output + Structural + Lex-ling subsections

**Files:**
- Create: `gui/section_features.py`
- Test: `tests/gui/test_section_features.py`
- Modify: `gui/main_window.py`

- [ ] **Step 1: Write failing tests**

Create `tests/gui/test_section_features.py`:

```python
def test_features_section_has_output_checkboxes(qtbot):
    from gui.section_features import FeaturesSection
    w = FeaturesSection()
    qtbot.addWidget(w)
    assert w.toggle("output_word_level_csv").is_checked()
    assert w.toggle("output_document_summary").is_checked()


def test_structural_toggle_present(qtbot):
    from gui.section_features import FeaturesSection
    w = FeaturesSection()
    qtbot.addWidget(w)
    assert w.toggle("structural") is not None


def test_lexling_has_four_subtoggles(qtbot):
    from gui.section_features import FeaturesSection
    w = FeaturesSection()
    qtbot.addWidget(w)
    for k in ("lexling_tokens", "lexling_pos", "lexling_morph", "lexling_dep"):
        assert w.toggle(k) is not None, k


def test_toggle_change_persists_to_prefs(qtbot, tmp_path, monkeypatch):
    monkeypatch.setenv("SENSI_PREFS_PATH", str(tmp_path / "prefs.json"))
    from gui.section_features import FeaturesSection
    w = FeaturesSection()
    qtbot.addWidget(w)
    w.toggle("lexling_pos").set_checked(False)
    from utils.preferences import load_preferences
    assert load_preferences()["features"]["lexling_pos"] is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/gui/test_section_features.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement the Output + Structural + Lex-ling subsections**

Create `gui/section_features.py`:

```python
"""Features section: feature taxonomy with toggles."""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QWidget, QVBoxLayout, QGridLayout, QFrame

from gui.widgets import SectionHeader, ToggleRow
from utils.preferences import load_preferences, save_preferences


class FeaturesSection(QWidget):
    settings_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._toggles: dict[str, ToggleRow] = {}
        prefs = load_preferences()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 16, 24, 16)
        layout.setSpacing(20)

        # --- Output ---
        layout.addWidget(SectionHeader("Output"))
        out_row = QGridLayout()
        out_row.addWidget(self._make_toggle(
            "output_word_level_csv", "Word-level CSV",
            prefs["output"]["word_level_csv"], "output"), 0, 0)
        out_row.addWidget(self._make_toggle(
            "output_document_summary", "Document-level summary",
            prefs["output"]["document_summary"], "output"), 0, 1)
        layout.addLayout(out_row)

        # --- Structural ---
        layout.addWidget(SectionHeader(
            "Structural",
            "Document position and offsets — from spaCy."))
        layout.addWidget(self._make_toggle(
            "structural", "Structural columns (doc_id, paragraph_id, …)",
            prefs["features"]["structural"], "features"))

        # --- Lexical-linguistic ---
        layout.addWidget(SectionHeader(
            "Lexical–linguistic",
            "Linguistic annotation from spaCy (en_core_web_sm)."))
        for key, label in [
            ("lexling_tokens", "Tokenization & lemmatization (token, lemma)"),
            ("lexling_pos",    "Part-of-speech tags (upos, xpos)"),
            ("lexling_morph",  "Morphological features (feats)"),
            ("lexling_dep",    "Dependency parse (dep_rel, head_token_id, deps, misc)"),
        ]:
            layout.addWidget(self._make_toggle(
                key, label, prefs["features"][key], "features"))

        layout.addStretch(1)

    def _make_toggle(self, key: str, label: str, initial: bool,
                     section: str) -> ToggleRow:
        row = ToggleRow(label=label)
        row.set_checked(initial)
        row.toggled.connect(lambda v, k=key, s=section: self._on_toggle(s, k, v))
        self._toggles[key] = row
        return row

    def _on_toggle(self, section: str, key: str, value: bool) -> None:
        prefix = "output_"
        if key.startswith(prefix):
            save_preferences({"output": {key[len(prefix):]: value}})
        else:
            save_preferences({section: {key: value}})
        self.settings_changed.emit()

    def toggle(self, key: str) -> ToggleRow | None:
        return self._toggles.get(key)
```

- [ ] **Step 4: Wire `FeaturesSection` into `main_window.py`**

Replace the second placeholder in `gui/main_window.py` with `FeaturesSection()`.

- [ ] **Step 5: Run tests**

Run: `pytest tests/gui/test_section_features.py -v`
Expected: green.

- [ ] **Step 6: Commit**

```bash
git add gui/section_features.py gui/main_window.py tests/gui/test_section_features.py
git commit -m "gui: Features section — Output, Structural, Lex-ling subsections"
```

---

### Task 12: Features section — Frequency subsection (with EWFG band checklist)

**Files:**
- Modify: `gui/section_features.py` (add Frequency block)
- Test: `tests/gui/test_section_features.py` (extend)

- [ ] **Step 1: Write failing test**

Append to `tests/gui/test_section_features.py`:

```python
def test_frequency_has_subtlex_and_five_ewfg_bands(qtbot):
    from gui.section_features import FeaturesSection
    w = FeaturesSection()
    qtbot.addWidget(w)
    assert w.toggle("freq_subtlex") is not None
    for band in ("freq_ewfg_k2", "freq_ewfg_3to5", "freq_ewfg_6to8",
                 "freq_ewfg_9to12", "freq_ewfg_overall"):
        assert w.toggle(band) is not None, band
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/gui/test_section_features.py::test_frequency_has_subtlex_and_five_ewfg_bands -v`
Expected: fails — no Frequency block yet.

- [ ] **Step 3: Add the Frequency block to `FeaturesSection.__init__`**

After the Lex-ling block, before `layout.addStretch(1)`, add:

```python
# --- Frequency ---
layout.addWidget(SectionHeader(
    "Frequency",
    "Lexical-corpus frequency lookups."))
layout.addWidget(self._make_toggle(
    "freq_subtlex",
    "General-corpus frequency (SUBTLEX-US, Zipf scale) → Zipf_Freq",
    prefs["features"]["freq_subtlex"], "features"))

# EWFG bands as an indented sub-block
ewfg_frame = QFrame()
ewfg_layout = QVBoxLayout(ewfg_frame)
ewfg_layout.setContentsMargins(24, 0, 0, 0)
for key, label in [
    ("freq_ewfg_k2",     "K–2      → Edu_Zipf_K2"),
    ("freq_ewfg_3to5",   "3–5      → Edu_Zipf_3to5"),
    ("freq_ewfg_6to8",   "6–8      → Edu_Zipf_6to8"),
    ("freq_ewfg_9to12",  "9–12     → Edu_Zipf_9to12"),
    ("freq_ewfg_overall","Overall  → Edu_Zipf_Overall"),
]:
    ewfg_layout.addWidget(self._make_toggle(
        key, label, prefs["features"][key], "features"))
layout.addWidget(ewfg_frame)
```

- [ ] **Step 4: Run test**

Run: `pytest tests/gui/test_section_features.py::test_frequency_has_subtlex_and_five_ewfg_bands -v`
Expected: green.

- [ ] **Step 5: Smoke test**

Run: `python gui.py`, navigate to Features, scroll to Frequency. Confirm the five EWFG bands appear indented under "Grade-banded frequency".

- [ ] **Step 6: Commit**

```bash
git add gui/section_features.py tests/gui/test_section_features.py
git commit -m "gui: Features — Frequency block with per-band EWFG toggles"
```

---

### Task 13: Features section — Predictability subsection with in-session API key

**Files:**
- Modify: `gui/section_features.py`
- Test: `tests/gui/test_section_features.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/gui/test_section_features.py`:

```python
def test_predictability_has_four_toggles_and_model_picker(qtbot):
    from gui.section_features import FeaturesSection
    w = FeaturesSection()
    qtbot.addWidget(w)
    for k in ("pred_surprisal_gpt2", "pred_surprisal_llm",
              "pred_cloze", "pred_topmatch"):
        assert w.toggle(k) is not None, k
    assert w.llm_model_combo is not None


def test_api_key_row_hidden_for_local_model(qtbot):
    from gui.section_features import FeaturesSection
    from extractors.llm_cloze import LLM_OPTIONS
    # Find any non-openai entry; if all are openai, this test is a no-op.
    local = next((mid for mid, _ in LLM_OPTIONS if not mid.startswith("gpt-")), None)
    if local is None:
        return  # no local models registered; skip
    w = FeaturesSection()
    qtbot.addWidget(w)
    w.llm_model_combo.setCurrentText(
        next(lab for mid, lab in LLM_OPTIONS if mid == local))
    assert not w.api_key_row.isVisible()


def test_api_key_row_visible_for_openai_model(qtbot):
    from gui.section_features import FeaturesSection
    from extractors.llm_cloze import LLM_OPTIONS
    openai_mid = next((mid for mid, _ in LLM_OPTIONS if mid.startswith("gpt-")), None)
    assert openai_mid is not None, "no openai model registered"
    w = FeaturesSection()
    qtbot.addWidget(w)
    w.llm_model_combo.setCurrentText(
        next(lab for mid, lab in LLM_OPTIONS if mid == openai_mid))
    assert w.api_key_row.isVisible()


def test_api_key_never_persisted(qtbot, tmp_path, monkeypatch):
    monkeypatch.setenv("SENSI_PREFS_PATH", str(tmp_path / "prefs.json"))
    from gui.section_features import FeaturesSection
    w = FeaturesSection()
    qtbot.addWidget(w)
    w.api_key_line.setText("sk-test123")
    from utils.preferences import load_preferences
    prefs = load_preferences()
    serialized = str(prefs)
    assert "sk-test123" not in serialized
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/gui/test_section_features.py -v -k "predict or api_key"`
Expected: failures.

- [ ] **Step 3: Implement the Predictability block**

In `gui/section_features.py`, add imports at the top:

```python
from PySide6.QtWidgets import QComboBox, QLineEdit, QHBoxLayout, QLabel, QPushButton
from extractors.llm_cloze import LLM_OPTIONS
```

In `__init__`, after the Frequency block, add:

```python
# --- Predictability ---
layout.addWidget(SectionHeader(
    "Predictability",
    "Context-conditional probability features."))

# Surprisal
layout.addWidget(QLabel(
    "Surprisal — -log₂ P(word | preceding context), in bits."))
layout.addWidget(self._make_toggle(
    "pred_surprisal_gpt2",
    "From GPT-2 → column: Surprisal (legacy baseline)",
    prefs["features"]["pred_surprisal_gpt2"], "features"))
layout.addWidget(self._make_toggle(
    "pred_surprisal_llm",
    "From LLM → column: LLM_Surprisal",
    prefs["features"]["pred_surprisal_llm"], "features"))

# Cloze features
layout.addWidget(QLabel("Cloze features (LLM forward pass)"))
layout.addWidget(self._make_toggle(
    "pred_cloze", "Cloze probability → LLM_Cloze",
    prefs["features"]["pred_cloze"], "features"))
layout.addWidget(self._make_toggle(
    "pred_topmatch", "Top-1 match → LLM_TopMatch",
    prefs["features"]["pred_topmatch"], "features"))

# Model picker
mp = QHBoxLayout()
mp.addWidget(QLabel("LLM model:"))
self.llm_model_combo = QComboBox()
self.llm_model_combo.addItems([lab for _, lab in LLM_OPTIONS])
saved_model = prefs["predictability"].get("llm_model", "")
saved_label = next((lab for mid, lab in LLM_OPTIONS if mid == saved_model),
                   LLM_OPTIONS[0][1])
self.llm_model_combo.setCurrentText(saved_label)
self.llm_model_combo.currentTextChanged.connect(self._on_model_change)
mp.addWidget(self.llm_model_combo, 1)
mp_w = QWidget(); mp_w.setLayout(mp)
layout.addWidget(mp_w)

# API key row (only visible for OpenAI models)
self.api_key_row = QWidget()
ak = QHBoxLayout(self.api_key_row); ak.setContentsMargins(0, 0, 0, 0)
ak.addWidget(QLabel("API key:"))
self.api_key_line = QLineEdit()
self.api_key_line.setEchoMode(QLineEdit.Password)
self.api_key_line.setPlaceholderText("Paste OpenAI API key — never saved to disk")
ak.addWidget(self.api_key_line, 1)
show_btn = QPushButton("Show")
show_btn.setCheckable(True)
show_btn.toggled.connect(lambda v: self.api_key_line.setEchoMode(
    QLineEdit.Normal if v else QLineEdit.Password))
ak.addWidget(show_btn)
layout.addWidget(self.api_key_row)
self._refresh_api_key_visibility()
```

Add helper methods:

```python
def _on_model_change(self, label: str) -> None:
    model_id = next((mid for mid, lab in LLM_OPTIONS if lab == label), label)
    save_preferences({"predictability": {"llm_model": model_id}})
    self._refresh_api_key_visibility()
    self.settings_changed.emit()

def _refresh_api_key_visibility(self) -> None:
    label = self.llm_model_combo.currentText()
    model_id = next((mid for mid, lab in LLM_OPTIONS if lab == label), label)
    is_openai = model_id.startswith("gpt-")
    self.api_key_row.setVisible(is_openai)

def session_api_key(self) -> str:
    """Return the in-session API key (never persisted)."""
    return self.api_key_line.text() if self.api_key_row.isVisible() else ""
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/gui/test_section_features.py -v -k "predict or api_key"`
Expected: green.

- [ ] **Step 5: Commit**

```bash
git add gui/section_features.py tests/gui/test_section_features.py
git commit -m "gui: Features — Predictability with in-session-only API key paste"
```

---

### Task 14: Inline category picker widget

**Files:**
- Create: `gui/category_picker.py`
- Test: `tests/gui/test_category_picker.py`

- [ ] **Step 1: Write failing tests**

Create `tests/gui/test_category_picker.py`:

```python
def test_picker_lists_all_backend_categories(qtbot):
    from gui.category_picker import CategoryPicker
    w = CategoryPicker(backend="empath")
    qtbot.addWidget(w)
    assert w.total_count() > 100   # Empath ships ~200 categories


def test_picker_shows_selected_count(qtbot):
    from gui.category_picker import CategoryPicker
    w = CategoryPicker(backend="empath")
    qtbot.addWidget(w)
    w.set_selection(["social", "cognition"])
    assert "2 of" in w.summary_label.text()


def test_picker_search_filters_categories(qtbot):
    from gui.category_picker import CategoryPicker
    w = CategoryPicker(backend="empath")
    qtbot.addWidget(w)
    w.search_field.setText("emoti")
    visible = [name for name, cb in w._checks.items() if cb.isVisible()]
    assert all("emoti" in n.lower() or w._checks[n].isChecked() for n in visible)


def test_picker_sop_default_button_resets_selection(qtbot, tmp_path, monkeypatch):
    monkeypatch.setenv("SENSI_PREFS_PATH", str(tmp_path / "prefs.json"))
    from gui.category_picker import CategoryPicker
    from utils.preferences import _SOP_CATEGORIES_EMPATH
    w = CategoryPicker(backend="empath")
    qtbot.addWidget(w)
    w.set_selection([])
    w.sop_default_button.click()
    assert sorted(w.current_selection()) == sorted(_SOP_CATEGORIES_EMPATH)


def test_picker_auto_saves_on_toggle(qtbot, tmp_path, monkeypatch):
    monkeypatch.setenv("SENSI_PREFS_PATH", str(tmp_path / "prefs.json"))
    from gui.category_picker import CategoryPicker
    from utils.preferences import load_preferences
    w = CategoryPicker(backend="empath")
    qtbot.addWidget(w)
    w.set_selection([])
    w._checks["social"].setChecked(True)
    prefs = load_preferences()
    assert "social" in prefs["semantic"]["categories_empath"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/gui/test_category_picker.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `gui/category_picker.py`**

```python
"""Inline category picker — replaces the modal CategoriesDialog."""

from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton,
    QLineEdit, QLabel, QCheckBox, QScrollArea, QFrame,
)

from utils.preferences import (
    load_preferences, save_preferences,
    _SOP_CATEGORIES_LIWC, _SOP_CATEGORIES_EMPATH,
)
from extractors.semantic_domain import detect_backend


def _list_backend_categories(backend: str) -> list[str]:
    if backend == "empath":
        try:
            from empath import Empath
            return sorted(Empath().cats.keys())
        except Exception:
            return list(_SOP_CATEGORIES_EMPATH)
    if backend == "liwc":
        _, dict_path = detect_backend()
        if dict_path is None:
            return list(_SOP_CATEGORIES_LIWC)
        try:
            import liwc
            _, names = liwc.load_token_parser(str(dict_path))
            return sorted(names)
        except Exception:
            return list(_SOP_CATEGORIES_LIWC)
    # auto: prefer LIWC if dict available
    b, _ = detect_backend()
    return _list_backend_categories(b)


class CategoryPicker(QWidget):
    selection_changed = Signal(list)

    def __init__(self, backend: str, parent=None):
        super().__init__(parent)
        self._backend = backend
        self._categories = _list_backend_categories(backend)
        self._checks: dict[str, QCheckBox] = {}
        self._sop = (_SOP_CATEGORIES_LIWC if backend == "liwc"
                     else _SOP_CATEGORIES_EMPATH)

        prefs = load_preferences()
        key = "categories_liwc" if backend == "liwc" else "categories_empath"
        initial = set(prefs["semantic"].get(key, self._sop))

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Header row with count + buttons
        header = QHBoxLayout()
        self.summary_label = QLabel()
        header.addWidget(self.summary_label, 1)
        self.sop_default_button = QPushButton("SOP default")
        self.sop_default_button.clicked.connect(self._reset_to_sop)
        self.all_button = QPushButton("All")
        self.all_button.clicked.connect(self._select_all)
        self.none_button = QPushButton("None")
        self.none_button.clicked.connect(self._select_none)
        for b in (self.sop_default_button, self.all_button, self.none_button):
            header.addWidget(b)
        layout.addLayout(header)

        # Search
        self.search_field = QLineEdit()
        self.search_field.setPlaceholderText("🔍 Search categories…")
        self.search_field.textChanged.connect(self._refilter)
        layout.addWidget(self.search_field)

        # Scroll area with grid of checkboxes
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        inner = QWidget()
        grid = QGridLayout(inner)
        grid.setContentsMargins(8, 8, 8, 8)
        col_count = 2
        for i, cat in enumerate(self._categories):
            cb = QCheckBox(cat)
            cb.setChecked(cat in initial)
            cb.toggled.connect(self._on_toggle)
            self._checks[cat] = cb
            row = i // col_count
            col = i % col_count
            container = QHBoxLayout()
            container.addWidget(cb)
            if cat in self._sop:
                badge = QLabel("★ SOP")
                badge.setStyleSheet("color: #5B5BD6; font-size: 10px;")
                container.addWidget(badge)
            container.addStretch(1)
            w = QWidget(); w.setLayout(container)
            grid.addWidget(w, row, col)
        scroll.setWidget(inner)
        layout.addWidget(scroll, 1)

        self._refresh_summary()

    def total_count(self) -> int:
        return len(self._categories)

    def current_selection(self) -> list[str]:
        return [c for c, cb in self._checks.items() if cb.isChecked()]

    def set_selection(self, cats: list[str]) -> None:
        cats_set = set(cats)
        for c, cb in self._checks.items():
            cb.setChecked(c in cats_set)

    def _refresh_summary(self) -> None:
        sel = self.current_selection()
        self.summary_label.setText(
            f"{len(sel)} of {len(self._categories)} selected")

    def _on_toggle(self, _checked: bool) -> None:
        sel = self.current_selection()
        key = "categories_liwc" if self._backend == "liwc" else "categories_empath"
        save_preferences({"semantic": {key: sel}})
        self._refresh_summary()
        self.selection_changed.emit(sel)

    def _reset_to_sop(self) -> None:
        self.set_selection(list(self._sop))

    def _select_all(self) -> None:
        self.set_selection(list(self._categories))

    def _select_none(self) -> None:
        self.set_selection([])

    def _refilter(self, q: str) -> None:
        ql = q.lower().strip()
        for cat, cb in self._checks.items():
            container = cb.parent()
            visible = (not ql) or (ql in cat.lower())
            container.setVisible(visible)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/gui/test_category_picker.py -v`
Expected: green.

- [ ] **Step 5: Commit**

```bash
git add gui/category_picker.py tests/gui/test_category_picker.py
git commit -m "gui: inline CategoryPicker (auto-save, SOP-default badge, search)"
```

---

### Task 15: Features section — Semantic domain + Acoustic + Doc-level subsections

**Files:**
- Modify: `gui/section_features.py`
- Test: `tests/gui/test_section_features.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/gui/test_section_features.py`:

```python
def test_semantic_subsection_present(qtbot):
    from gui.section_features import FeaturesSection
    w = FeaturesSection()
    qtbot.addWidget(w)
    assert w.toggle("sem_indicators") is not None
    assert w.toggle("sem_composite_psych") is not None
    assert w.category_picker is not None


def test_acoustic_subsection_present(qtbot):
    from gui.section_features import FeaturesSection
    w = FeaturesSection()
    qtbot.addWidget(w)
    assert w.toggle("acoustic_boundaries") is not None


def test_doc_subsection_hidden_when_output_unchecked(qtbot):
    from gui.section_features import FeaturesSection
    w = FeaturesSection()
    qtbot.addWidget(w)
    w.toggle("output_document_summary").set_checked(False)
    assert not w.doc_block.isVisible()


def test_doc_subsection_visible_when_output_checked(qtbot):
    from gui.section_features import FeaturesSection
    w = FeaturesSection()
    qtbot.addWidget(w)
    w.toggle("output_document_summary").set_checked(True)
    assert w.doc_block.isVisible()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/gui/test_section_features.py -v -k "semantic or acoustic or doc_subsection"`
Expected: failures.

- [ ] **Step 3: Add the three remaining subsections to `FeaturesSection`**

In `gui/section_features.py`, after the Predictability block, add:

```python
from PySide6.QtWidgets import QButtonGroup, QRadioButton
from gui.category_picker import CategoryPicker

# --- Semantic domain ---
layout.addWidget(SectionHeader(
    "Semantic domain",
    "Per-word category indicators from a dictionary backend."))

backend_row = QHBoxLayout()
backend_row.addWidget(QLabel("Backend:"))
self._backend_group = QButtonGroup(self)
saved_backend = prefs["semantic"].get("backend", "auto")
for value, label in [("auto", "Auto"), ("liwc", "LIWC"), ("empath", "Empath")]:
    rb = QRadioButton(label)
    rb.setChecked(value == saved_backend)
    rb.toggled.connect(lambda checked, v=value: checked and self._on_backend_change(v))
    self._backend_group.addButton(rb)
    backend_row.addWidget(rb)
backend_row.addStretch(1)
br_w = QWidget(); br_w.setLayout(backend_row)
layout.addWidget(br_w)

layout.addWidget(self._make_toggle(
    "sem_indicators",
    "Category indicators (per-word, content POS only)",
    prefs["features"]["sem_indicators"], "features"))

# Inline category picker
self.category_picker = CategoryPicker(backend=self._resolve_backend(saved_backend))
layout.addWidget(self.category_picker)

layout.addWidget(self._make_toggle(
    "sem_composite_psych",
    "Composite psychological scores (doc-level, Pennebaker formulas)\n"
    "  → psych_authentic, psych_analytic, psych_tone, psych_clout",
    prefs["features"]["sem_composite_psych"], "features"))

# --- Acoustic ---
layout.addWidget(SectionHeader(
    "Acoustic",
    "True acoustic word boundaries from audio (forced alignment, MFA)."))
layout.addWidget(self._make_toggle(
    "acoustic_boundaries",
    "Word boundaries → Onset, End, Word_Duration, Word_Gap",
    prefs["features"]["acoustic_boundaries"], "features"))

# --- Document-level summaries (conditional) ---
self.doc_block = QWidget()
db = QVBoxLayout(self.doc_block); db.setContentsMargins(0, 0, 0, 0)
db.addWidget(SectionHeader(
    "Document-level summaries",
    "Only emitted when 'Document-level summary' output is checked."))
for key, label in [
    ("output_doc_counts",             "Counts (n_tokens, n_content_tokens, n_sentences, n_paragraphs)"),
    ("output_doc_aggregates",         "Aggregates of word-level metrics"),
    ("output_doc_lex_diversity",      "Lexical diversity (TTR, MTLD)"),
    ("output_doc_readability",        "Readability (Flesch reading ease, FK grade)"),
    ("output_doc_category_proportions","Category proportions (count_<cat>, prop_<cat>)"),
]:
    db.addWidget(self._make_toggle(
        key, label, prefs["output"][key[len("output_"):]], "output"))
layout.addWidget(self.doc_block)
self.doc_block.setVisible(prefs["output"]["document_summary"])

# Wire output_document_summary toggle to show/hide doc_block
self.toggle("output_document_summary").toggled.connect(self.doc_block.setVisible)
```

Add the helpers:

```python
def _resolve_backend(self, backend: str) -> str:
    if backend == "auto":
        from extractors.semantic_domain import detect_backend
        b, _ = detect_backend()
        return b
    return backend

def _on_backend_change(self, backend: str) -> None:
    save_preferences({"semantic": {"backend": backend}})
    # Rebuild the category picker for the new backend
    parent_layout = self.category_picker.parent().layout()
    idx = parent_layout.indexOf(self.category_picker)
    self.category_picker.deleteLater()
    self.category_picker = CategoryPicker(backend=self._resolve_backend(backend))
    parent_layout.insertWidget(idx, self.category_picker)
    self.settings_changed.emit()
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/gui/test_section_features.py -v`
Expected: green.

- [ ] **Step 5: Smoke test**

Run: `python gui.py`. Navigate to Features. Confirm all six subsections render, category picker is inline below the "Category indicators" toggle, doc-level block toggles visibility when the Output checkbox changes.

- [ ] **Step 6: Commit**

```bash
git add gui/section_features.py tests/gui/test_section_features.py
git commit -m "gui: Features — Semantic, Acoustic, and conditional Doc-level subsections"
```

---

## Phase 5: Continuous pre-flight validation

### Task 16: Pre-flight engine — produce warnings from current settings

**Files:**
- Create: `gui/preflight.py`
- Test: `tests/gui/test_preflight.py`

- [ ] **Step 1: Write failing tests**

Create `tests/gui/test_preflight.py`:

```python
def test_preflight_warns_on_missing_ewfg(monkeypatch, tmp_path):
    monkeypatch.setenv("SENSI_PREFS_PATH", str(tmp_path / "prefs.json"))
    from gui.preflight import compute_warnings
    warnings = compute_warnings()
    # EWFG parquet is not bundled by default
    keys = {w.toggle_key for w in warnings}
    assert any(k.startswith("freq_ewfg_") for k in keys)


def test_preflight_warns_on_openai_without_key(monkeypatch, tmp_path):
    monkeypatch.setenv("SENSI_PREFS_PATH", str(tmp_path / "prefs.json"))
    # Force OpenAI model selection
    from utils.preferences import save_preferences
    save_preferences({"predictability": {"llm_model": "gpt-4o-mini"}})
    from gui.preflight import compute_warnings
    warnings = compute_warnings(session_api_key="")
    keys = {w.toggle_key for w in warnings}
    assert "pred_surprisal_llm" in keys or "pred_cloze" in keys


def test_preflight_no_warning_on_openai_with_key(tmp_path, monkeypatch):
    monkeypatch.setenv("SENSI_PREFS_PATH", str(tmp_path / "prefs.json"))
    from utils.preferences import save_preferences
    save_preferences({"predictability": {"llm_model": "gpt-4o-mini"}})
    from gui.preflight import compute_warnings
    warnings = compute_warnings(session_api_key="sk-fake")
    keys = {w.toggle_key for w in warnings}
    assert not any(k.startswith("pred_") for k in keys)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/gui/test_preflight.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `gui/preflight.py`**

```python
"""Continuous pre-flight check engine.

Reads current preferences + session-only state (API key) and returns a
list of inline warnings keyed by the toggle they should appear next to.
"""

from dataclasses import dataclass
from pathlib import Path

from utils.preferences import load_preferences


@dataclass(frozen=True)
class Warning:
    toggle_key: str   # which ToggleRow it attaches to
    severity: str     # "amber" or "red"
    message: str


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_EWFG_PARQUET = _PROJECT_ROOT / "dictionaries" / "edu_zipf.parquet"
_LIWC_DIC_GLOB = "*.dic"


def compute_warnings(session_api_key: str = "") -> list[Warning]:
    prefs = load_preferences()
    f = prefs["features"]
    out: list[Warning] = []

    # EWFG missing
    if not _EWFG_PARQUET.exists():
        for band in ("freq_ewfg_k2", "freq_ewfg_3to5", "freq_ewfg_6to8",
                     "freq_ewfg_9to12", "freq_ewfg_overall"):
            if f.get(band, False):
                out.append(Warning(
                    toggle_key=band, severity="amber",
                    message="EWFG parquet not found at dictionaries/edu_zipf.parquet "
                            "— columns will be NaN. See README for acquisition."))

    # OpenAI without API key
    llm_model = prefs["predictability"].get("llm_model", "")
    if llm_model.startswith("gpt-") and not session_api_key:
        for k in ("pred_surprisal_llm", "pred_cloze", "pred_topmatch"):
            if f.get(k, False):
                out.append(Warning(
                    toggle_key=k, severity="amber",
                    message=f"OpenAI model '{llm_model}' selected but no API key "
                            "pasted — paste your key in Predictability."))

    # LIWC backend selected but no .dic
    if prefs["semantic"]["backend"] == "liwc":
        dict_dir = _PROJECT_ROOT / "dictionaries"
        if not any(dict_dir.glob(_LIWC_DIC_GLOB)):
            if f.get("sem_indicators", False):
                out.append(Warning(
                    toggle_key="sem_indicators", severity="amber",
                    message="LIWC backend selected but no .dic file found in "
                            "dictionaries/. Will fall back to Empath at run time."))

    # MFA detection
    if f.get("acoustic_boundaries", False):
        try:
            import subprocess
            subprocess.run(["mfa", "version"],
                           capture_output=True, timeout=3, check=False)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            out.append(Warning(
                toggle_key="acoustic_boundaries", severity="red",
                message="MFA binary not found on PATH. Activate the sensi conda env "
                        "or disable this feature."))

    return out
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/gui/test_preflight.py -v`
Expected: green.

- [ ] **Step 5: Commit**

```bash
git add gui/preflight.py tests/gui/test_preflight.py
git commit -m "gui: preflight engine computes inline warnings from current settings"
```

---

### Task 17: Wire pre-flight warnings into the Features section

**Files:**
- Modify: `gui/section_features.py`

- [ ] **Step 1: Add a `refresh_warnings()` method**

In `FeaturesSection`, add at the end of `__init__` after all subsections:

```python
self.refresh_warnings()
# Re-run on any setting change anywhere in this section
self.settings_changed.connect(self.refresh_warnings)
```

Add method:

```python
def refresh_warnings(self, session_api_key: str = "") -> None:
    """Recompute warnings and attach them inline to toggles."""
    from gui.preflight import compute_warnings
    warnings = compute_warnings(session_api_key=session_api_key)
    # Clear all previous warnings
    for row in self._toggles.values():
        row.set_warning(None)
    # Attach new ones
    for w in warnings:
        row = self._toggles.get(w.toggle_key)
        if row is not None:
            row.set_warning(w.message)
```

- [ ] **Step 2: Hook the API-key line edit to trigger a refresh**

In the Predictability block where `self.api_key_line` is created, after assignment add:

```python
self.api_key_line.textChanged.connect(
    lambda txt: self.refresh_warnings(session_api_key=txt))
```

- [ ] **Step 3: Smoke test**

Run: `python gui.py`. Navigate to Features. Confirm:
- The EWFG band toggles show amber dots if `dictionaries/edu_zipf.parquet` is missing.
- Selecting `gpt-4o-mini` and leaving the API key blank shows amber dots on the LLM predictability toggles.
- Pasting any text into the API key field removes those dots.

- [ ] **Step 4: Commit**

```bash
git add gui/section_features.py
git commit -m "gui: wire preflight warnings as inline indicators on Features toggles"
```

---

## Phase 6: Run dispatch and sections

### Task 18: Run dispatch — background-thread orchestration

**Files:**
- Create: `gui/run_dispatch.py`
- Test: `tests/gui/test_run_dispatch.py`

- [ ] **Step 1: Read the existing dispatch logic**

Run: `grep -n "run_word_level_batch\|run_doc_level_batch" main.py gui_run.py`
Read these functions to understand their signatures.

- [ ] **Step 2: Write failing tests**

Create `tests/gui/test_run_dispatch.py`:

```python
def test_build_args_word_only_includes_correct_flags(monkeypatch, tmp_path):
    monkeypatch.setenv("SENSI_PREFS_PATH", str(tmp_path / "prefs.json"))
    from gui.run_dispatch import build_args_word_level
    args = build_args_word_level()
    # Should always include transcripts/audio/output folders and analyzer
    assert hasattr(args, "transcripts_folder")
    assert hasattr(args, "audio_folder")
    assert hasattr(args, "output_folder")
    assert hasattr(args, "analyzer")


def test_build_args_word_filters_ewfg_bands(monkeypatch, tmp_path):
    monkeypatch.setenv("SENSI_PREFS_PATH", str(tmp_path / "prefs.json"))
    from utils.preferences import save_preferences
    save_preferences({"features": {
        "freq_ewfg_k2": True, "freq_ewfg_3to5": False,
        "freq_ewfg_6to8": False, "freq_ewfg_9to12": False,
        "freq_ewfg_overall": True,
    }})
    from gui.run_dispatch import build_args_word_level
    args = build_args_word_level()
    assert set(args.edu_bands) == {"k2", "overall"}


def test_dispatcher_runs_both_when_both_outputs_checked(monkeypatch, tmp_path):
    monkeypatch.setenv("SENSI_PREFS_PATH", str(tmp_path / "prefs.json"))
    from utils.preferences import save_preferences
    save_preferences({"output": {"word_level_csv": True, "document_summary": True}})

    called = []
    def fake_word(args, **kwargs): called.append("word")
    def fake_doc(args, **kwargs): called.append("doc")

    monkeypatch.setattr("main.run_word_level_batch", fake_word, raising=False)
    monkeypatch.setattr("main.run_doc_level_batch", fake_doc, raising=False)

    from gui.run_dispatch import run
    run(progress_cb=lambda *a: None, done_cb=lambda *a: None,
        session_api_key="", in_thread=False)
    assert "word" in called and "doc" in called
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/gui/test_run_dispatch.py -v`
Expected: ImportError.

- [ ] **Step 4: Implement `gui/run_dispatch.py`**

```python
"""Background-thread dispatcher to main.run_word_level_batch / run_doc_level_batch.

Replaces the orchestration in the old gui_run.py.
"""

import argparse
import threading
from typing import Callable

from utils.preferences import load_preferences


def _label_to_model_id(label: str) -> str:
    from extractors.llm_cloze import LLM_OPTIONS
    for mid, lab in LLM_OPTIONS:
        if lab == label or mid == label:
            return mid
    return label


def _common_args() -> argparse.Namespace:
    prefs = load_preferences()
    s = prefs["shared"]
    ns = argparse.Namespace()
    ns.transcripts_folder = s.get("last_transcripts_folder", "")
    ns.audio_folder       = s.get("last_audio_folder", "")
    ns.output_folder      = s.get("last_output_folder", "")
    ns.analyzer           = s.get("analyzer_name", "")
    ns.dictionary_backend = prefs["semantic"].get("backend", "auto")
    return ns


def _enabled_ewfg_bands() -> list[str]:
    f = load_preferences()["features"]
    bands = []
    for suffix, key in [("k2", "freq_ewfg_k2"),
                         ("3to5", "freq_ewfg_3to5"),
                         ("6to8", "freq_ewfg_6to8"),
                         ("9to12", "freq_ewfg_9to12"),
                         ("overall", "freq_ewfg_overall")]:
        if f.get(key, False):
            bands.append(suffix)
    return bands


def build_args_word_level() -> argparse.Namespace:
    prefs = load_preferences()
    f = prefs["features"]
    ns = _common_args()
    ns.no_spacy   = not (f["structural"] or f["lexling_tokens"]
                          or f["lexling_pos"] or f["lexling_morph"]
                          or f["lexling_dep"])
    ns.no_zipf    = not f["freq_subtlex"]
    ns.edu_bands  = _enabled_ewfg_bands()
    ns.no_edu     = (not ns.edu_bands)
    ns.no_sem     = not f["sem_indicators"]
    ns.no_gpt2    = not f["pred_surprisal_gpt2"]
    ns.no_llm     = not (f["pred_surprisal_llm"]
                          or f["pred_cloze"] or f["pred_topmatch"])
    ns.no_align   = not f["acoustic_boundaries"]
    ns.llm_model  = prefs["predictability"].get("llm_model", "")
    # Lex-ling column filtering — exposed via the writer
    ns.lexling_cols = [k for k in (
        "lexling_tokens", "lexling_pos", "lexling_morph", "lexling_dep"
    ) if f[k]]
    return ns


def build_args_doc_level() -> argparse.Namespace:
    prefs = load_preferences()
    o = prefs["output"]
    f = prefs["features"]
    ns = _common_args()
    ns.no_doc_counts             = not o["doc_counts"]
    ns.no_doc_aggregates         = not o["doc_aggregates"]
    ns.no_doc_lex_diversity      = not o["doc_lex_diversity"]
    ns.no_doc_readability        = not o["doc_readability"]
    ns.no_doc_category_proportions = not o["doc_category_proportions"]
    ns.no_psych                  = not f["sem_composite_psych"]
    ns.llm_model                 = prefs["predictability"].get("llm_model", "")
    ns.edu_bands                 = ["k2", "3to5", "6to8", "9to12", "overall"]
    return ns


def run(
    progress_cb: Callable[[int, int, str], None],
    done_cb: Callable[[str], None],
    session_api_key: str,
    in_thread: bool = True,
) -> None:
    """Dispatch a run. progress_cb(current, total, filename). done_cb(status)."""
    def _go():
        import os
        if session_api_key:
            os.environ["OPENAI_API_KEY"] = session_api_key
        prefs = load_preferences()
        o = prefs["output"]
        try:
            if o.get("word_level_csv", True):
                from main import run_word_level_batch
                run_word_level_batch(build_args_word_level(),
                                      progress_callback=progress_cb)
            if o.get("document_summary", True):
                from main import run_doc_level_batch
                run_doc_level_batch(build_args_doc_level(),
                                     progress_callback=progress_cb)
            done_cb("done")
        except Exception as exc:
            done_cb(f"error: {exc}")
        finally:
            if session_api_key:
                os.environ.pop("OPENAI_API_KEY", None)

    if in_thread:
        threading.Thread(target=_go, daemon=True).start()
    else:
        _go()
```

> **Note for engineer:** the exact `progress_callback` keyword and the precise `no_*` argument names in `main.run_word_level_batch` / `run_doc_level_batch` may differ from what's sketched above. Before running tests, inspect those function signatures in `main.py` and align the `build_args_*` namespaces accordingly. The shape (folders, flags, edu_bands list, llm_model string) is what the tests check.

- [ ] **Step 5: Run tests**

Run: `pytest tests/gui/test_run_dispatch.py -v`
Expected: green.

- [ ] **Step 6: Commit**

```bash
git add gui/run_dispatch.py tests/gui/test_run_dispatch.py
git commit -m "gui: run_dispatch — background thread to run_*_batch with v3 prefs"
```

---

### Task 19: Cost estimate (tiktoken) + pricing config

**Files:**
- Create: `config/openai_pricing.json`
- Create: `gui/cost_estimate.py`
- Test: `tests/gui/test_cost_estimate.py`

- [ ] **Step 1: Create the pricing config**

Create `config/openai_pricing.json`:

```json
{
  "last_updated": "2026-05-11",
  "rates_per_million_tokens": {
    "gpt-4o-mini":  { "input": 0.15, "output": 0.60 },
    "gpt-4o":       { "input": 2.50, "output": 10.00 }
  },
  "notice": "OpenAI pricing changes. Verify current rates at openai.com/api/pricing"
}
```

- [ ] **Step 2: Write failing tests**

Create `tests/gui/test_cost_estimate.py`:

```python
def test_estimate_local_model_is_zero():
    from gui.cost_estimate import estimate
    e = estimate(model_id="llama-3.2-3b", n_input_tokens=10000, n_output_tokens=2000)
    assert e.usd_total == 0.0
    assert e.is_local


def test_estimate_openai_uses_pricing_file():
    from gui.cost_estimate import estimate
    e = estimate(model_id="gpt-4o-mini",
                  n_input_tokens=1_000_000, n_output_tokens=1_000_000)
    # gpt-4o-mini @ $0.15 input + $0.60 output per 1M
    assert abs(e.usd_total - 0.75) < 1e-6
    assert not e.is_local


def test_count_input_tokens_with_tiktoken(tmp_path):
    from gui.cost_estimate import count_input_tokens
    p = tmp_path / "t.txt"
    p.write_text("Hello world. " * 1000, encoding="utf-8")
    n = count_input_tokens([p], model_id="gpt-4o-mini")
    assert 1000 < n < 5000
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/gui/test_cost_estimate.py -v`
Expected: ImportError.

- [ ] **Step 4: Implement `gui/cost_estimate.py`**

```python
"""OpenAI cost estimation using tiktoken + a versioned rate file."""

import json
from dataclasses import dataclass
from pathlib import Path


_PRICING_PATH = Path(__file__).resolve().parent.parent / "config" / "openai_pricing.json"


@dataclass(frozen=True)
class CostEstimate:
    model_id: str
    n_input_tokens: int
    n_output_tokens: int
    usd_input: float
    usd_output: float
    usd_total: float
    rates_last_updated: str
    is_local: bool


def _load_rates() -> dict:
    if not _PRICING_PATH.exists():
        return {"last_updated": "unknown", "rates_per_million_tokens": {}}
    return json.loads(_PRICING_PATH.read_text(encoding="utf-8"))


def estimate(model_id: str, n_input_tokens: int,
             n_output_tokens: int) -> CostEstimate:
    if not model_id.startswith("gpt-"):
        return CostEstimate(
            model_id=model_id, n_input_tokens=n_input_tokens,
            n_output_tokens=n_output_tokens, usd_input=0.0,
            usd_output=0.0, usd_total=0.0,
            rates_last_updated="n/a", is_local=True,
        )
    rates = _load_rates()
    table = rates.get("rates_per_million_tokens", {}).get(model_id, {})
    in_rate = table.get("input", 0.0)
    out_rate = table.get("output", 0.0)
    usd_in = n_input_tokens / 1_000_000 * in_rate
    usd_out = n_output_tokens / 1_000_000 * out_rate
    return CostEstimate(
        model_id=model_id, n_input_tokens=n_input_tokens,
        n_output_tokens=n_output_tokens, usd_input=usd_in,
        usd_output=usd_out, usd_total=usd_in + usd_out,
        rates_last_updated=rates.get("last_updated", "unknown"),
        is_local=False,
    )


def count_input_tokens(transcript_paths: list[Path], model_id: str) -> int:
    import tiktoken
    try:
        enc = tiktoken.encoding_for_model(model_id)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    n = 0
    for p in transcript_paths:
        try:
            n += len(enc.encode(p.read_text(encoding="utf-8")))
        except (OSError, UnicodeDecodeError):
            continue
    return n
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/gui/test_cost_estimate.py -v`
Expected: green.

- [ ] **Step 6: Commit**

```bash
git add config/openai_pricing.json gui/cost_estimate.py tests/gui/test_cost_estimate.py
git commit -m "cost: tiktoken-based estimate + versioned pricing config"
```

---

### Task 20: Run section UI

**Files:**
- Create: `gui/section_run.py`
- Test: `tests/gui/test_section_run.py`
- Modify: `gui/main_window.py`

- [ ] **Step 1: Write failing tests**

Create `tests/gui/test_section_run.py`:

```python
def test_run_section_has_run_button(qtbot):
    from gui.section_run import RunSection
    w = RunSection(get_session_api_key=lambda: "", status_strip=None)
    qtbot.addWidget(w)
    assert w.run_button is not None


def test_run_button_disabled_when_no_analyzer(qtbot, tmp_path, monkeypatch):
    monkeypatch.setenv("SENSI_PREFS_PATH", str(tmp_path / "prefs.json"))
    from utils.preferences import save_preferences
    save_preferences({"shared": {"analyzer_name": ""}})
    from gui.section_run import RunSection
    w = RunSection(get_session_api_key=lambda: "", status_strip=None)
    qtbot.addWidget(w)
    w.refresh()
    assert not w.run_button.isEnabled()


def test_cost_panel_hidden_when_local_model(qtbot, tmp_path, monkeypatch):
    monkeypatch.setenv("SENSI_PREFS_PATH", str(tmp_path / "prefs.json"))
    from utils.preferences import save_preferences
    save_preferences({"predictability": {"llm_model": "llama-3.2-3b"}})
    from gui.section_run import RunSection
    w = RunSection(get_session_api_key=lambda: "", status_strip=None)
    qtbot.addWidget(w)
    w.refresh()
    assert not w.cost_panel.isVisible()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/gui/test_section_run.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `gui/section_run.py`**

```python
"""Run section: output checkboxes, readiness summary, cost estimate, Run button."""

from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame,
)

from gui.widgets import SectionHeader, ToggleRow
from gui.preflight import compute_warnings
from gui.cost_estimate import estimate, count_input_tokens
from gui.run_dispatch import run as dispatch_run
from utils.preferences import load_preferences, save_preferences


class RunSection(QWidget):
    run_started = Signal()
    run_finished = Signal(str)

    def __init__(self, get_session_api_key, status_strip, parent=None):
        super().__init__(parent)
        self._get_api_key = get_session_api_key
        self._status_strip = status_strip

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 16, 24, 16)
        layout.setSpacing(16)

        # Output
        layout.addWidget(SectionHeader("Output"))
        prefs = load_preferences()
        self.word_csv_toggle = ToggleRow("Word-level CSVs (one per stimulus)")
        self.word_csv_toggle.set_checked(prefs["output"]["word_level_csv"])
        self.word_csv_toggle.toggled.connect(
            lambda v: save_preferences({"output": {"word_level_csv": v}}))
        layout.addWidget(self.word_csv_toggle)
        self.doc_sum_toggle = ToggleRow("Document-level summary (single CSV)")
        self.doc_sum_toggle.set_checked(prefs["output"]["document_summary"])
        self.doc_sum_toggle.toggled.connect(
            lambda v: save_preferences({"output": {"document_summary": v}}))
        layout.addWidget(self.doc_sum_toggle)

        # Readiness
        layout.addWidget(SectionHeader("Readiness"))
        self.readiness_label = QLabel("…")
        layout.addWidget(self.readiness_label)
        self.warnings_label = QLabel("")
        self.warnings_label.setWordWrap(True)
        layout.addWidget(self.warnings_label)

        # Cost panel
        self.cost_panel = QFrame()
        self.cost_panel.setObjectName("Card")
        cp = QVBoxLayout(self.cost_panel)
        cp.addWidget(QLabel("<b>Cost estimate</b>"))
        self.cost_detail = QLabel("…")
        self.cost_detail.setWordWrap(True)
        cp.addWidget(self.cost_detail)
        layout.addWidget(self.cost_panel)

        # Run button
        layout.addStretch(1)
        self.run_button = QPushButton("▶  Run extraction")
        self.run_button.setObjectName("Primary")
        self.run_button.clicked.connect(self._on_run)
        layout.addWidget(self.run_button)

        self.refresh()

    def refresh(self) -> None:
        prefs = load_preferences()
        # Hard errors disable Run
        ok = bool(prefs["shared"].get("analyzer_name", "").strip()) \
             and Path(prefs["shared"].get("last_transcripts_folder", "")).is_dir() \
             and Path(prefs["shared"].get("last_audio_folder", "")).is_dir() \
             and Path(prefs["shared"].get("last_output_folder", "")).is_dir()
        self.run_button.setEnabled(ok)

        warnings = compute_warnings(session_api_key=self._get_api_key())
        if not warnings:
            self.readiness_label.setText("✓  Ready to run.")
            self.warnings_label.setText("")
        else:
            self.readiness_label.setText(f"⚠  {len(warnings)} warnings — review Features.")
            self.warnings_label.setText(
                "\n".join(f"• {w.message}" for w in warnings))

        # Cost panel
        model = prefs["predictability"].get("llm_model", "")
        if model.startswith("gpt-"):
            self.cost_panel.setVisible(True)
            t = Path(prefs["shared"].get("last_transcripts_folder", ""))
            if t.is_dir():
                paths = [p for p in t.iterdir() if p.is_file()]
                n_in = count_input_tokens(paths, model)
                n_out = sum(len(p.read_text(encoding="utf-8", errors="ignore").split())
                             for p in paths if p.is_file())
                est = estimate(model_id=model, n_input_tokens=n_in,
                                n_output_tokens=n_out)
                self.cost_detail.setText(
                    f"Model: {est.model_id}<br>"
                    f"Input tokens: ~ {est.n_input_tokens:,} (tiktoken)<br>"
                    f"Output tokens: ~ {est.n_output_tokens:,} (1/word)<br>"
                    f"Rates (last updated {est.rates_last_updated}): "
                    f"${est.usd_input/max(est.n_input_tokens,1)*1e6:.2f}/1M input, "
                    f"${est.usd_output/max(est.n_output_tokens,1)*1e6:.2f}/1M output<br>"
                    f"<b>Estimated: ≈ ${est.usd_total:.3f}</b><br>"
                    "⚠ OpenAI pricing changes — verify at openai.com/api/pricing"
                )
            else:
                self.cost_detail.setText("Pick a transcripts folder to compute cost.")
        else:
            self.cost_panel.setVisible(False)

    def _on_run(self) -> None:
        self.run_button.setEnabled(False)
        self.run_started.emit()
        if self._status_strip:
            self._status_strip.set_running(0, 1, "starting…")

        def on_progress(i, total, name):
            if self._status_strip:
                self._status_strip.set_running(i, total, name)

        def on_done(status):
            if status == "done":
                if self._status_strip:
                    self._status_strip.set_done(1, 0)
            else:
                if self._status_strip:
                    self._status_strip.set_errored(1)
            self.run_finished.emit(status)
            self.run_button.setEnabled(True)

        dispatch_run(
            progress_cb=on_progress,
            done_cb=on_done,
            session_api_key=self._get_api_key(),
        )
```

- [ ] **Step 4: Wire into `main_window.py`**

Replace the third placeholder with `RunSection(...)`. The Run section needs:
- `get_session_api_key` callback that returns `features_section.session_api_key()`
- `status_strip` reference

In `MainWindow.__init__`, after instantiating `InputsSection` and `FeaturesSection`, pass references:

```python
self.features_section = FeaturesSection()
self.run_section = RunSection(
    get_session_api_key=self.features_section.session_api_key,
    status_strip=self.status_strip,
)
```

And ensure cross-section refresh: `self.features_section.settings_changed.connect(self.run_section.refresh)`.

- [ ] **Step 5: Run tests**

Run: `pytest tests/gui/test_section_run.py -v`
Expected: green.

- [ ] **Step 6: Smoke test**

Run: `python gui.py`. Navigate to Run. Confirm: output checkboxes, readiness summary, cost panel (visible only with OpenAI model selected). Run button disabled when analyzer is empty or any folder is missing.

- [ ] **Step 7: Commit**

```bash
git add gui/section_run.py gui/main_window.py tests/gui/test_section_run.py
git commit -m "gui: Run section with readiness summary and cost panel"
```

---

### Task 21: Logs section

**Files:**
- Create: `gui/section_logs.py`
- Test: `tests/gui/test_section_logs.py`
- Modify: `gui/main_window.py`

- [ ] **Step 1: Write failing tests**

Create `tests/gui/test_section_logs.py`:

```python
def test_logs_section_has_text_viewer(qtbot):
    from gui.section_logs import LogsSection
    w = LogsSection()
    qtbot.addWidget(w)
    assert w.viewer is not None


def test_append_line_shows_in_viewer(qtbot):
    from gui.section_logs import LogsSection
    w = LogsSection()
    qtbot.addWidget(w)
    w.append_line("14:02:13  INFO   Hello")
    assert "Hello" in w.viewer.toPlainText()


def test_past_runs_lists_existing_logs(qtbot, tmp_path):
    (tmp_path / "run_20260510_1644.log").write_text("done", encoding="utf-8")
    (tmp_path / "run_20260510_1115.log").write_text("error", encoding="utf-8")
    from gui.section_logs import LogsSection
    w = LogsSection(output_folder=tmp_path)
    qtbot.addWidget(w)
    items = [w.past_runs_list.item(i).text()
             for i in range(w.past_runs_list.count())]
    assert any("20260510_1644" in s for s in items)
    assert any("20260510_1115" in s for s in items)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/gui/test_section_logs.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `gui/section_logs.py`**

```python
"""Logs section: live current-run viewer + past-runs list."""

from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPlainTextEdit,
    QLineEdit, QListWidget, QListWidgetItem, QPushButton, QLabel,
)
from PySide6.QtGui import QFont

from gui.widgets import SectionHeader
from utils.preferences import load_preferences


class LogsSection(QWidget):
    def __init__(self, output_folder: Path | None = None, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 16, 24, 16)
        layout.setSpacing(12)

        # Header
        header_row = QHBoxLayout()
        header_row.addWidget(SectionHeader("Current run"), 1)
        self.copy_btn = QPushButton("Copy")
        self.copy_btn.clicked.connect(self._copy_log)
        header_row.addWidget(self.copy_btn)
        hr_w = QWidget(); hr_w.setLayout(header_row)
        layout.addWidget(hr_w)

        # Filter / search row
        search_row = QHBoxLayout()
        search_row.addWidget(QLabel("Search:"))
        self.search = QLineEdit()
        self.search.setPlaceholderText("Filter visible lines…")
        self.search.textChanged.connect(self._apply_filter)
        search_row.addWidget(self.search, 1)
        sr_w = QWidget(); sr_w.setLayout(search_row)
        layout.addWidget(sr_w)

        # Viewer
        self.viewer = QPlainTextEdit()
        self.viewer.setReadOnly(True)
        self.viewer.setFont(QFont("JetBrains Mono", 10))
        layout.addWidget(self.viewer, 1)

        # Past runs
        layout.addWidget(SectionHeader("Past runs"))
        self.past_runs_list = QListWidget()
        layout.addWidget(self.past_runs_list, 1)

        self._all_lines: list[str] = []
        self._output_folder = output_folder or self._guess_output_folder()
        self.refresh_past_runs()

    def _guess_output_folder(self) -> Path:
        p = load_preferences()["shared"].get("last_output_folder", "")
        return Path(p) if p else Path.cwd()

    def append_line(self, line: str) -> None:
        self._all_lines.append(line)
        if self._line_matches(line):
            self.viewer.appendPlainText(line)

    def _line_matches(self, line: str) -> bool:
        q = self.search.text().lower().strip()
        return (not q) or (q in line.lower())

    def _apply_filter(self, _q: str) -> None:
        self.viewer.clear()
        for line in self._all_lines:
            if self._line_matches(line):
                self.viewer.appendPlainText(line)

    def _copy_log(self) -> None:
        from PySide6.QtGui import QGuiApplication
        QGuiApplication.clipboard().setText("\n".join(self._all_lines))

    def refresh_past_runs(self) -> None:
        self.past_runs_list.clear()
        if not self._output_folder.is_dir():
            return
        for p in sorted(self._output_folder.glob("run_*.log"), reverse=True):
            self.past_runs_list.addItem(QListWidgetItem(p.name))
```

- [ ] **Step 4: Wire into `main_window.py`**

Replace the fourth placeholder with `LogsSection()`.

- [ ] **Step 5: Run tests**

Run: `pytest tests/gui/test_section_logs.py -v`
Expected: green.

- [ ] **Step 6: Commit**

```bash
git add gui/section_logs.py gui/main_window.py tests/gui/test_section_logs.py
git commit -m "gui: Logs section with live viewer and past-runs list"
```

---

## Phase 7: Theme switching + cleanup

### Task 22: Theme toggle (light / dark / system)

**Files:**
- Modify: `gui/main_window.py` (add top bar with theme buttons)
- Test: `tests/gui/test_main_window.py` (extend)

- [ ] **Step 1: Write failing test**

Append to `tests/gui/test_main_window.py`:

```python
def test_main_window_has_theme_buttons(qtbot):
    from gui.main_window import MainWindow
    w = MainWindow()
    qtbot.addWidget(w)
    assert w.theme_light_btn is not None
    assert w.theme_dark_btn is not None
    assert w.theme_system_btn is not None


def test_clicking_dark_button_persists(qtbot, tmp_path, monkeypatch):
    monkeypatch.setenv("SENSI_PREFS_PATH", str(tmp_path / "prefs.json"))
    from gui.main_window import MainWindow
    from utils.preferences import load_preferences
    w = MainWindow()
    qtbot.addWidget(w)
    w.theme_dark_btn.click()
    assert load_preferences()["shared"]["theme"] == "dark"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/gui/test_main_window.py -v -k "theme"`
Expected: failures.

- [ ] **Step 3: Add top bar with three theme buttons**

In `MainWindow.__init__`, insert a top bar above the body:

```python
top_bar = QFrame()
top_bar.setFixedHeight(36)
tb = QHBoxLayout(top_bar)
tb.setContentsMargins(16, 4, 16, 4)
tb.addWidget(QLabel("SensiExtractor"), 1)
self.theme_light_btn = QPushButton("☀")
self.theme_dark_btn = QPushButton("🌙")
self.theme_system_btn = QPushButton("💻")
for btn, mode in [(self.theme_light_btn, "light"),
                   (self.theme_dark_btn, "dark"),
                   (self.theme_system_btn, "system")]:
    btn.setFixedWidth(40)
    btn.clicked.connect(lambda _, m=mode: self._set_theme(m))
    tb.addWidget(btn)
root.addWidget(top_bar)
root.addWidget(body, 1)
root.addWidget(self.status_strip)
```

Add helper:

```python
def _set_theme(self, mode: str) -> None:
    save_preferences({"shared": {"theme": mode}})
    self._prefs = load_preferences()
    self._apply_theme()
```

And add the imports for `save_preferences`, `QPushButton`, `QFrame` if not already present.

- [ ] **Step 4: Run tests**

Run: `pytest tests/gui/test_main_window.py -v -k "theme"`
Expected: green.

- [ ] **Step 5: Smoke test**

Run: `python gui.py`. Click ☀/🌙/💻 buttons. Confirm the window restyles.

- [ ] **Step 6: Commit**

```bash
git add gui/main_window.py tests/gui/test_main_window.py
git commit -m "gui: top bar with light/dark/system theme buttons"
```

---

### Task 23: Remove deprecated tkinter files

**Files:**
- Delete: `gui_word_tab.py`, `gui_doc_tab.py`, `gui_categories_dialog.py`, `gui_run.py`

- [ ] **Step 1: Confirm no remaining imports**

Run: `grep -rn "from gui_word_tab\|from gui_doc_tab\|from gui_categories_dialog\|from gui_run" .`
Expected: only this file and the now-orphaned files themselves.

- [ ] **Step 2: Delete the four legacy files**

```bash
rm gui_word_tab.py gui_doc_tab.py gui_categories_dialog.py gui_run.py
```

- [ ] **Step 3: Run the full test suite**

Run: `pytest -v`
Expected: green (any regression would be a sign that something still imports the deleted files).

- [ ] **Step 4: Smoke test the app one more time**

Run: `python gui.py`. Click through all four sidebar sections. Confirm everything still works.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "cleanup: remove deprecated tkinter gui_*.py files"
```

---

### Task 24: Update CLAUDE.md and Feature_Extraction_Guide.md

**Files:**
- Modify: `CLAUDE.md`
- Modify: `Feature_Extraction_Guide.md`

- [ ] **Step 1: Update CLAUDE.md**

In `CLAUDE.md`:

- Change the "Project structure" tree: replace the four `gui_*.py` files with the new `gui/` package and list its contents.
- Change the "Running" section so `python gui.py` is described as launching the PySide6 GUI.
- Add a sentence to the "Key design decisions" list:
  - "**PySide6 GUI (since 2026-05-11):** sidebar/content architecture, continuous inline validation, in-session API-key paste, per-band EWFG toggles, inline category picker. Replaces the original tkinter UI. See `docs/superpowers/specs/2026-05-11-gui-redesign-design.md`."

- [ ] **Step 2: Update Feature_Extraction_Guide.md**

Append a new section titled "Batch 2 — PySide6 GUI redesign (added 2026-05-11)" that documents:
- New `gui/` package structure.
- Column rename: `liwc_*` → `psych_*` with the honest-naming reason.
- Per-band EWFG output filtering.
- Preferences schema v2 → v3 migration.
- Where the spec and plan live (paths).

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md Feature_Extraction_Guide.md
git commit -m "docs: update CLAUDE.md and Feature Extraction Guide for batch 2 GUI"
```

---

### Task 25: Final acceptance walkthrough

This is a manual verification task — no new tests or code. Walk through every acceptance criterion in the spec.

**Files:**
- Read: `docs/superpowers/specs/2026-05-11-gui-redesign-design.md` — "Acceptance criteria" section

- [ ] **Step 1: Run the smoke walkthrough**

For each criterion in the spec (numbered 1–10), perform the action and verify the expected behavior. Note any failures.

- [ ] **Step 2: If any criterion fails, file follow-up tasks**

Create new entries in this plan or open new tickets describing what didn't work. Don't claim completion if anything red.

- [ ] **Step 3: Final commit if any small fixes were needed**

```bash
git add -A
git commit -m "qa: acceptance-criterion fixes from final walkthrough"
```

---

## Self-Review

**Spec coverage check (against `docs/superpowers/specs/2026-05-11-gui-redesign-design.md`):**

| Spec section | Task(s) |
|---|---|
| File structure (`gui/` package) | 7, 8, 9, 10, 11, 14, 16, 18, 20, 21 |
| Window shell (min size, status strip, scroll) | 9 |
| Theming (QSS, light/dark/system, single accent) | 7, 22 |
| Inputs section + continuous validation + pairing preview | 10 |
| Features — Output checkboxes | 11 |
| Features — Structural + Lex-ling (4 sub-toggles) | 11 |
| Features — Frequency (SUBTLEX + 5 EWFG bands) | 12 |
| Features — Predictability + in-session API key | 13 |
| Features — Semantic + inline category picker + SOP defaults | 14, 15 |
| Features — Acoustic | 15 |
| Features — Document-level summaries (conditional) | 15 |
| Continuous pre-flight inline warnings | 16, 17 |
| Run section + readiness + cost estimate | 19, 20 |
| Logs section + past runs | 21 |
| Persistent status strip across sections | 9 |
| Preferences schema v3 + v2→v3 migration | 2, 3 |
| EWFG per-band filtering in extractor + main.py | 4 |
| `liwc_*` → `psych_*` column rename + docs | 5, 24 |
| Word/doc pipeline independence | 6 |
| Removed `gui_word_tab.py`, etc. | 23 |
| CLAUDE.md + Feature Extraction Guide updates | 24 |
| Acceptance criteria walk-through | 25 |

**Placeholders found and fixed:** Task 6 step 2 explicitly notes its test sketch must be filled in against the actual function signature once read in step 1 — this is acceptable because the test requirement is fully specified (just the call shape isn't until inspection). Task 18 step 4 has a similar "verify against actual `main.py` signatures" note. Both are flagged inline; the engineer reading them in order will not be surprised.

**Type consistency:** `freq_ewfg_*` band keys, `psych_*` column names, `pred_surprisal_gpt2/llm`, `output_*` keys, and `lexling_*` keys are consistent across Tasks 2, 3, 4, 5, 11, 12, 13, 15, 16, 17, 18.

**Scope:** The plan covers one coherent feature (GUI redesign + supporting backend changes) — appropriate for a single execution cycle.

---

Plan complete and saved to `docs/superpowers/plans/2026-05-11-gui-redesign.md`. Two execution options:

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
