# Predictability Model Unification — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Merge the GPT-2 surprisal and LLM-cloze extractors into one multi-model predictability feature: the user selects models from a roster and metrics from three toggles, and each model emits namespaced `<slug>_<Metric>` columns from one forward pass.

**Architecture:** A data-driven model **roster** is the single source of truth (slug, hf_id, gated, native_ctx). One `extractors/predictability.py` scores each selected model via a shared local HF backend (GPT-2 loads as an ordinary `AutoModelForCausalLM`, so its values are unchanged) and writes only the requested, model-prefixed columns. Prefs move model/metric selection into a `predictability` settings section (v18); the planner gets one dynamic `predictability` stage whose columns are computed from `models × metrics`.

**Tech Stack:** Python, pandas, torch, transformers (`AutoModelForCausalLM`/`AutoTokenizer`), PySide6 (GUI), pytest.

Spec: `docs/superpowers/specs/2026-07-10-predictability-model-unification-design.md`.

## Global Constraints

- **TDD for extractor math** (`extractors/`): failing test first, watch it fail, minimal code. GUI/prefs/planner/docs may be test-after where cheaper, but migration and planner-activation logic get tests-first.
- **Slugs are the only model identifier** in prefs, columns, doc aggregates, and filenames. They must be column- and filename-safe: `gpt2`, `llama32_3b`, `llama3_8b`, `llama3_70b`.
- **Clean break** on column names — no `Surprisal` / `LLM_*` aliases. New: `<slug>_Surprisal`, `<slug>_Cloze`, `<slug>_TopMatch`; doc means `mean_<slug>_<metric>`.
- **Predictability columns are produced opt-in** (only the selected ones are written) — never produced-then-dropped, so no per-column drop map.
- **Preserve the recent correctness semantics**: document-first word → NaN, PUNCT → NaN, each subword charged to exactly one token, `(model, text-digest)` score cache reused across the word and doc passes.
- **`native_ctx` is carried as roster data only** in this project; no context-size control (that is Project B).
- Prefs migration here is **not purely additive** (it removes keys) — precedent: the v14 struct-id rename. Migration runs in the existing pre-current finalize block in `utils/preferences.py`.
- Run pytest with the project env on PATH: `env`, `env\Library\bin`, `env\Scripts` (Windows) so `libsndfile`/MKL resolve. Full suite baseline: **672 passed, 2 skipped**.

---

### Task 1: Model roster

**Files:**
- Create: `extractors/predictability_models.py`
- Test: `tests/extractors/test_predictability_models.py`

**Interfaces:**
- Produces: `ModelSpec(slug, label, hf_id, gated, native_ctx, note)` frozen dataclass; `ALL_SPECS: list[ModelSpec]`; `all_specs() -> list[ModelSpec]`; `spec_for(slug) -> ModelSpec` (raises `KeyError` on unknown); `is_gated(slug) -> bool`; `slug_for_hf_id(hf_id) -> str | None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/extractors/test_predictability_models.py
import re
import pytest
from extractors.predictability_models import (
    all_specs, spec_for, is_gated, slug_for_hf_id, ALL_SPECS,
)

_SAFE = re.compile(r"^[a-z0-9_]+$")

def test_roster_has_gpt2_and_three_llamas():
    slugs = {s.slug for s in all_specs()}
    assert slugs == {"gpt2", "llama32_3b", "llama3_8b", "llama3_70b"}

def test_slugs_are_column_and_filename_safe():
    for s in ALL_SPECS:
        assert _SAFE.match(s.slug), s.slug

def test_gpt2_is_free_and_1024():
    g = spec_for("gpt2")
    assert g.gated is False and g.hf_id == "gpt2" and g.native_ctx == 1024

def test_llamas_are_gated_with_ceilings():
    assert is_gated("llama3_8b") and spec_for("llama3_8b").native_ctx == 8192
    assert spec_for("llama32_3b").native_ctx == 131072
    assert spec_for("llama3_70b").native_ctx == 8192

def test_slug_for_hf_id_maps_stored_model():
    assert slug_for_hf_id("meta-llama/Meta-Llama-3-8B") == "llama3_8b"
    assert slug_for_hf_id("gpt2") == "gpt2"
    assert slug_for_hf_id("something/unknown") is None

def test_spec_for_unknown_raises():
    with pytest.raises(KeyError):
        spec_for("nope")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/extractors/test_predictability_models.py -q`
Expected: FAIL (module missing).

- [ ] **Step 3: Write minimal implementation**

```python
# extractors/predictability_models.py
"""Roster of local language models for the predictability feature.

Single source of truth for the GUI model list, column/aggregate prefixes,
gating, and (for the future context-size project) each model's native
context ceiling. `slug` is the only identifier used downstream; it is
column- and filename-safe.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    slug: str          # column/filename-safe id; the sole downstream identifier
    label: str         # GUI display text
    hf_id: str         # HuggingFace repo id passed to from_pretrained
    gated: bool        # needs a HF token + accepted license
    native_ctx: int    # max_position_embeddings ceiling (data only here)
    note: str          # short GUI hint


ALL_SPECS: list[ModelSpec] = [
    ModelSpec("gpt2", "GPT-2", "gpt2", False, 1024,
              "free, no token, ~500 MB, CPU-capable baseline"),
    ModelSpec("llama32_3b", "Llama-3.2-3B", "meta-llama/Llama-3.2-3B",
              True, 131072, "gated, free; ≥8 GB VRAM or CPU"),
    ModelSpec("llama3_8b", "Llama-3-8B", "meta-llama/Meta-Llama-3-8B",
              True, 8192, "gated, free; ≥12 GB VRAM"),
    ModelSpec("llama3_70b", "Llama-3-70B", "meta-llama/Meta-Llama-3-70B",
              True, 8192, "gated, free; ≥48 GB VRAM"),
]

_BY_SLUG = {s.slug: s for s in ALL_SPECS}
_BY_HF_ID = {s.hf_id: s.slug for s in ALL_SPECS}


def all_specs() -> list[ModelSpec]:
    return list(ALL_SPECS)


def spec_for(slug: str) -> ModelSpec:
    return _BY_SLUG[slug]


def is_gated(slug: str) -> bool:
    return _BY_SLUG[slug].gated


def slug_for_hf_id(hf_id: str) -> str | None:
    return _BY_HF_ID.get(hf_id)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/extractors/test_predictability_models.py -q`
Expected: PASS.

Note (implementer): confirm the `native_ctx` values against each model's `config.max_position_embeddings` if the weights are present; the values above are the published ceilings.

- [ ] **Step 5: Commit**

```bash
git add extractors/predictability_models.py tests/extractors/test_predictability_models.py
git commit -m "feat(predictability): model roster (slug/hf_id/gated/native_ctx)"
```

---

### Task 2a: Unified extractor — column production (pure, fake backend)

**Files:**
- Create: `extractors/predictability.py`
- Test: `tests/extractors/test_predictability.py`

**Interfaces:**
- Consumes: roster helpers from Task 1.
- Produces: `add_predictability(df, text, *, models: list[str], metrics: list[str], backends: dict | None = None) -> pd.DataFrame`. Column names: `f"{slug}_Surprisal"`, `f"{slug}_Cloze"`, `f"{slug}_TopMatch"`. A backend is any object with `.score(text, tokens) -> list[dict]` where each dict has `{p, surprisal, top_match}` (top_match may be None for the context-less first token). Also re-exports `map_tokens_to_subwords` / `_scores_from_subwords` / `_chunk_slices` (moved here in Task 2b) — for 2a, define the metric-writing logic against injected `backends`.

- [ ] **Step 1: Write the failing test**

```python
# tests/extractors/test_predictability.py
import math
import numpy as np
import pandas as pd
import pytest
from extractors.predictability import add_predictability

class _Fake:
    """Backend returning fixed per-token scores; first token is context-less."""
    def score(self, text, tokens):
        out = []
        for i, _ in enumerate(tokens):
            if i == 0:
                out.append({"p": float("nan"), "surprisal": float("nan"),
                            "top_match": None})
            else:
                out.append({"p": 0.5, "surprisal": 1.0, "top_match": 1})
        return out

def _df():
    return pd.DataFrame({"token": ["The", "cat", "."],
                         "upos": ["DET", "NOUN", "PUNCT"]})

def test_namespaced_columns_per_model_and_metric():
    out = add_predictability(_df(), "The cat.", models=["gpt2", "llama3_8b"],
                             metrics=["surprisal", "cloze"],
                             backends={"gpt2": _Fake(), "llama3_8b": _Fake()})
    for slug in ("gpt2", "llama3_8b"):
        assert f"{slug}_Surprisal" in out.columns
        assert f"{slug}_Cloze" in out.columns
        assert f"{slug}_TopMatch" not in out.columns   # topmatch not requested

def test_first_word_and_punct_are_nan():
    out = add_predictability(_df(), "The cat.", models=["gpt2"],
                             metrics=["surprisal", "cloze", "topmatch"],
                             backends={"gpt2": _Fake()})
    assert math.isnan(out["gpt2_Surprisal"].iloc[0])   # context-less first word
    assert math.isnan(out["gpt2_Surprisal"].iloc[2])   # PUNCT
    assert out["gpt2_Surprisal"].iloc[1] == 1.0
    assert math.isnan(out["gpt2_TopMatch"].iloc[0])     # top_match None → NaN
    assert out["gpt2_TopMatch"].iloc[1] == 1

def test_no_models_or_no_metrics_writes_no_columns():
    base = _df()
    a = add_predictability(base, "x", models=[], metrics=["surprisal"],
                           backends={})
    b = add_predictability(base, "x", models=["gpt2"], metrics=[],
                           backends={"gpt2": _Fake()})
    assert list(a.columns) == list(base.columns)
    assert list(b.columns) == list(base.columns)

def test_unknown_slug_rejected():
    with pytest.raises(KeyError):
        add_predictability(_df(), "x", models=["nope"], metrics=["surprisal"],
                           backends={"nope": _Fake()})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/extractors/test_predictability.py -q`
Expected: FAIL (module/function missing).

- [ ] **Step 3: Write minimal implementation**

Create `extractors/predictability.py` with the column-writing core. `spec_for(slug)` validates the slug (raises `KeyError` on unknown). PUNCT masking mirrors the old `add_llm_cloze`; `top_match is None → np.nan`. Metrics map: `surprisal→_Surprisal` (round 4), `cloze→_Cloze` (round 6), `topmatch→_TopMatch` (int or NaN).

```python
"""Unified multi-model predictability extractor (surprisal / cloze / top-match).

Replaces extractors/surprisal.py and the old extractors/llm_cloze.py. Each
selected roster model is scored once; only the requested metrics are written,
prefixed by the model slug. See the design spec §4.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from extractors.predictability_models import spec_for

_METRIC_COLS = {"surprisal": "Surprisal", "cloze": "Cloze", "topmatch": "TopMatch"}

def add_predictability(df, text, *, models, metrics, backends=None):
    metrics = [m for m in metrics if m in _METRIC_COLS]
    if not models or not metrics:
        return df.copy()
    for slug in models:
        spec_for(slug)          # validate; KeyError on unknown
    out = df.copy()
    upos = out["upos"].tolist()
    tokens = [str(t) for t in out["token"].tolist()]
    for slug in models:
        backend = (backends or {}).get(slug) or _get_backend(slug)
        scores = backend.score(text, tokens)
        _write_metric_columns(out, slug, metrics, upos, scores)
    return out

def _write_metric_columns(out, slug, metrics, upos, scores):
    cols = {m: [] for m in metrics}
    for pos_tag, score in zip(upos, scores):
        for m in metrics:
            if pos_tag == "PUNCT":
                cols[m].append(np.nan); continue
            if m == "surprisal":
                cols[m].append(round(score["surprisal"], 4)
                               if score["surprisal"] == score["surprisal"] else np.nan)
            elif m == "cloze":
                cols[m].append(round(score["p"], 6)
                               if score["p"] == score["p"] else np.nan)
            else:  # topmatch
                tm = score["top_match"]
                cols[m].append(np.nan if tm is None else int(tm))
    for m in metrics:
        out[f"{slug}_{_METRIC_COLS[m]}"] = cols[m]

def _get_backend(slug):   # real backend wired in Task 2b
    raise NotImplementedError("real backend added in Task 2b")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/extractors/test_predictability.py -q`
Expected: PASS (all four tests; real-backend path unused via `backends=`).

- [ ] **Step 5: Commit**

```bash
git add extractors/predictability.py tests/extractors/test_predictability.py
git commit -m "feat(predictability): unified extractor column production (fake backend)"
```

---

### Task 2b: Real backend + GPT-2-via-Auto + retire old extractors

**Files:**
- Modify: `extractors/predictability.py` (add `_HuggingFaceBackend`, `_get_backend`, moved helpers)
- Delete: `extractors/surprisal.py`, `extractors/llm_cloze.py`
- Modify: any importer of the deleted modules (grep first)
- Test: add to `tests/extractors/test_predictability.py`; move/keep the still-relevant cases from `tests/extractors/test_llm_cloze.py` and `tests/extractors/test_surprisal.py`

**Interfaces:**
- Consumes: roster (`spec_for`).
- Produces: `_get_backend(slug)` returns a cached `_HuggingFaceBackend(spec_for(slug).hf_id)`; module-level `map_tokens_to_subwords`, `_scores_from_subwords`, `_chunk_slices`, `_check_context_limit`, `select_model`, `has_hf_token`, `is_gated_local_model` re-homed here (moved verbatim from the old files). `_SCORE_CACHE` keyed by `(hf_id, text-digest)`.

- [ ] **Step 1: Write the failing test (GPT-2 equivalence)**

```python
def test_gpt2_via_auto_matches_legacy_surprisal():
    # AutoModelForCausalLM("gpt2") is GPT2LMHeadModel — identical logits.
    # Pin: the first content word's surprisal is finite and stable; the
    # document-first token is NaN. Guards against a silent load-path drift.
    import pandas as pd
    df = pd.DataFrame({"token": ["The", "cat", "sat"],
                       "upos": ["DET", "NOUN", "VERB"]})
    out = add_predictability(df, "The cat sat", models=["gpt2"],
                             metrics=["surprisal"])
    import math
    assert math.isnan(out["gpt2_Surprisal"].iloc[0])
    assert out["gpt2_Surprisal"].iloc[1] > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/extractors/test_predictability.py::test_gpt2_via_auto_matches_legacy_surprisal -q`
Expected: FAIL (`_get_backend` raises NotImplementedError).

- [ ] **Step 3: Implement the real backend**

Move `_HuggingFaceBackend` from the old `llm_cloze.py` into `predictability.py` unchanged (it already loads via `AutoModelForCausalLM`, chunks via `_chunk_slices`, and returns `{p, surprisal, top_match}` with the first-token-None rule via `_scores_from_subwords`). Also move `map_tokens_to_subwords`/`_token_surprisals` (from `surprisal.py`), `_scores_from_subwords`, `_chunk_slices`, `_check_context_limit`, `select_model`, `_available_vram_gb`, `has_hf_token`, `is_gated_local_model`, and the `_SCORE_CACHE`. Implement:

```python
_BACKEND_CACHE = {}
def _get_backend(slug):
    hf_id = spec_for(slug).hf_id
    b = _BACKEND_CACHE.get(hf_id)
    if b is None:
        b = _HuggingFaceBackend(hf_id)
        _BACKEND_CACHE[hf_id] = b
    return b
```

Wire `add_predictability` to memoize `backend.score` per `(hf_id, digest)` via `_SCORE_CACHE` (mirror the old `add_llm_cloze` caching), so the doc pass reuses the word pass.

- [ ] **Step 4: Delete old modules and fix imports**

```bash
git rm extractors/surprisal.py extractors/llm_cloze.py
grep -rn "extractors.surprisal\|extractors.llm_cloze\|from extractors import surprisal\|add_surprisal\|add_llm_cloze\|LLM_OPTIONS" --include=*.py . | grep -v tests/
```
Update every hit to import from `extractors.predictability` / the roster. Port the still-relevant tests from `test_llm_cloze.py` / `test_surprisal.py` (chunk-slices, subword mapping, HF token, device-tier cascade, first-word NaN, cache) into `test_predictability.py`; delete the two old test files.

- [ ] **Step 5: Run the extractor tests**

Run: `python -m pytest tests/extractors/test_predictability.py -q`
Expected: PASS (downloads gpt2 weights on first run; ~cached in `hf_cache/`).

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat(predictability): unified HF backend; GPT-2 via Auto; retire surprisal.py/llm_cloze.py"
```

---

### Task 3: Prefs schema v18 + migration

**Files:**
- Modify: `utils/preferences.py`
- Test: `tests/utils/test_preferences_migration.py`

**Interfaces:**
- Produces: `SCHEMA_VERSION_V18 = 18`; `DEFAULT_PREFS` with `predictability = {"models": [], "metrics": {"surprisal": True, "cloze": False, "topmatch": False}}` and no `pred_surprisal_gpt2/_llm/pred_cloze/pred_topmatch` feature keys, no `predictability.llm_model`. Migration maps old selections forward.

- [ ] **Step 1: Write the failing migration tests**

```python
def test_v17_gpt2_surprisal_toggle_becomes_gpt2_model(tmp_path, monkeypatch):
    import json
    from utils import preferences as P
    p = tmp_path / "prefs.json"
    p.write_text(json.dumps({"schema_version": 17,
        "features": {"pred_surprisal_gpt2": True},
        "predictability": {"llm_model": ""}}), encoding="utf-8")
    monkeypatch.setenv("SENSI_PREFS_PATH", str(p))
    prefs = P.load_preferences()
    assert prefs["schema_version"] == P.SCHEMA_VERSION
    assert "gpt2" in prefs["predictability"]["models"]
    assert prefs["predictability"]["metrics"]["surprisal"] is True
    assert "pred_surprisal_gpt2" not in prefs["features"]

def test_v17_llm_toggles_become_stored_model_and_metrics(tmp_path, monkeypatch):
    import json
    from utils import preferences as P
    p = tmp_path / "prefs.json"
    p.write_text(json.dumps({"schema_version": 17,
        "features": {"pred_surprisal_llm": True, "pred_cloze": True,
                     "pred_topmatch": False},
        "predictability": {"llm_model": "meta-llama/Meta-Llama-3-8B"}}),
        encoding="utf-8")
    monkeypatch.setenv("SENSI_PREFS_PATH", str(p))
    prefs = P.load_preferences()
    assert prefs["predictability"]["models"] == ["llama3_8b"]
    assert prefs["predictability"]["metrics"]["surprisal"] is True
    assert prefs["predictability"]["metrics"]["cloze"] is True
    assert prefs["predictability"]["metrics"]["topmatch"] is False
    assert "llm_model" not in prefs["predictability"]

def test_v17_llm_toggle_unknown_model_defaults_llama32_3b(tmp_path, monkeypatch):
    import json
    from utils import preferences as P
    p = tmp_path / "prefs.json"
    p.write_text(json.dumps({"schema_version": 17,
        "features": {"pred_cloze": True},
        "predictability": {"llm_model": ""}}), encoding="utf-8")
    monkeypatch.setenv("SENSI_PREFS_PATH", str(p))
    prefs = P.load_preferences()
    assert prefs["predictability"]["models"] == ["llama32_3b"]
    assert prefs["predictability"]["metrics"]["cloze"] is True

def test_v18_fresh_default_predictability_off(tmp_path, monkeypatch):
    from utils import preferences as P
    monkeypatch.setenv("SENSI_PREFS_PATH", str(tmp_path / "none.json"))
    prefs = P.load_preferences()
    assert prefs["predictability"]["models"] == []
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/utils/test_preferences_migration.py -q -k v17_gpt2 or v17_llm or v18_fresh`
Expected: FAIL.

- [ ] **Step 3: Implement v18 defaults + migration**

Add v18 constants after the v17 block: build `_FEATURES_DEFAULT_V18` = `_FEATURES_DEFAULT_V17` minus the four `pred_*` keys; `predictability` settings section = `{"models": [], "metrics": {"surprisal": True, "cloze": False, "topmatch": False}}`. Set `SCHEMA_VERSION = SCHEMA_VERSION_V18`, `DEFAULT_PREFS = DEFAULT_PREFS_V18`, and add `SCHEMA_VERSION_V18` to the deep-merge version tuple in `load_preferences`.

In the pre-current finalize block (where v17 already coerces env_comp/llm_model), add — for `version < SCHEMA_VERSION_V18`:

```python
from extractors.predictability_models import slug_for_hf_id
f = merged.get("features", {})
pred = merged.setdefault("predictability", {})
models = list(pred.get("models") or [])
metrics = pred.setdefault("metrics", {"surprisal": False, "cloze": False, "topmatch": False})
if f.pop("pred_surprisal_gpt2", False):
    if "gpt2" not in models: models.append("gpt2")
    metrics["surprisal"] = True
llm_on = f.pop("pred_surprisal_llm", False)
cloze_on = f.pop("pred_cloze", False)
top_on = f.pop("pred_topmatch", False)
if llm_on or cloze_on or top_on:
    slug = slug_for_hf_id(pred.get("llm_model", "")) or "llama32_3b"
    if slug not in models: models.append(slug)
    metrics["surprisal"] = metrics.get("surprisal", False) or llm_on
    metrics["cloze"] = cloze_on
    metrics["topmatch"] = top_on
pred["models"] = models
pred.pop("llm_model", None)
```

(Guard the import so `utils.preferences` doesn't import torch — `predictability_models` is pure, no heavy deps, so a top-level import in the finalize function is fine.)

- [ ] **Step 4: Run the migration tests + the whole migration file**

Run: `python -m pytest tests/utils/test_preferences_migration.py -q`
Expected: PASS. Fix any prior test that asserted the old `pred_*`/`llm_model` keys (update to the v18 shape).

- [ ] **Step 5: Commit**

```bash
git add utils/preferences.py tests/utils/test_preferences_migration.py
git commit -m "feat(prefs): v18 predictability models/metrics section + migration"
```

---

### Task 4: Planner — dynamic predictability stage/cluster + emit

**Files:**
- Modify: `gui/planner/registry.py`, `gui/planner/plan.py`, `gui/planner/emit.py`
- Test: `tests/planner/test_plan.py`, `tests/planner/test_emit.py`, `tests/planner/test_schema_drift.py`

**Interfaces:**
- Consumes: prefs v18 (`predictability.models` / `.metrics`), roster.
- Produces: one Stage `predictability` (run=`add_predictability`), one dynamic FeatureCluster; `plan.stage_configs["predictability"] = {"models": [...], "metrics": [...]}`; `ns.predictability = {"models": [...], "metrics": {...}}`; active columns `<slug>_<Metric>`.

- [ ] **Step 1: Write the failing plan test**

```python
def test_predictability_stage_active_and_columns(monkeypatch):
    from gui.planner import plan_from_prefs
    from utils.preferences import DEFAULT_PREFS
    import copy
    prefs = copy.deepcopy(DEFAULT_PREFS)
    prefs["features"]["master_predictability"] = True
    prefs["output"]["word_level_csv"] = True
    prefs["predictability"] = {"models": ["gpt2", "llama3_8b"],
                               "metrics": {"surprisal": True, "cloze": True,
                                           "topmatch": False}}
    p = plan_from_prefs(prefs)
    assert "predictability" in p.stages_to_run
    cols = set(p.emitted_columns_for("predictability"))  # helper or plan field
    assert cols == {"gpt2_Surprisal", "gpt2_Cloze",
                    "llama3_8b_Surprisal", "llama3_8b_Cloze"}

def test_predictability_inactive_when_no_metric(monkeypatch):
    from gui.planner import plan_from_prefs
    from utils.preferences import DEFAULT_PREFS
    import copy
    prefs = copy.deepcopy(DEFAULT_PREFS)
    prefs["features"]["master_predictability"] = True
    prefs["output"]["word_level_csv"] = True
    prefs["predictability"] = {"models": ["gpt2"],
                               "metrics": {"surprisal": False, "cloze": False,
                                           "topmatch": False}}
    assert "predictability" not in plan_from_prefs(prefs).stages_to_run
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/planner/test_plan.py -q -k predictability`
Expected: FAIL.

- [ ] **Step 3: Implement the dynamic stage/cluster**

In `registry.py`: delete the `gpt2_surprisal` and `llm_predict` Stages and the `GPT-2 surprisal` / `LLM predictability` FeatureClusters. Add one Stage `predictability` (run=`add_predictability`, consumes `word_df`, produces `word_df.predictability`). Add one FeatureCluster `Predictability` with `master_toggle="master_predictability"`, `output_csv="features"`, `requires_stages=("spacy_parse", "predictability")`, and — since columns are dynamic — no static `toggle_keys`/`emits_columns`. Represent activation via a small predicate the planner already supports (mirror `available_when` used for the LIWC gate) or a dedicated field; the plan is active when `master_predictability` and `models` non-empty and ≥1 metric.

In `plan.py`: compute the predictability stage's active columns from `prefs["predictability"]` as `[f"{slug}_{Cap}" for slug in models for metric,Cap in selected]`, expose them (a `plan.predictability_columns` field or `emitted_columns_for`). Set `stage_configs["predictability"] = {"models": models, "metrics": [selected metric names]}`. Include the columns in the session-units source so the units CSV still lists them.

If wiring dynamic columns through the declarative cluster proves invasive, use the spec §12 fallback: keep the cluster static-but-considered (gate only the stage), and compute active columns in `plan.py`/`emit.py` directly from `predictability` — do NOT block on a perfect declarative fit.

In `emit.py`: replace the `ns.no_surprisal` / `ns.no_llm_cloze` / `ns.llm_model` lines with:
```python
_p = prefs.get("predictability", {})
ns.predictability = {
    "models": list(_p.get("models") or []),
    "metrics": dict(_p.get("metrics") or {}),
}
```

- [ ] **Step 4: Update the schema-drift guard**

`tests/planner/test_schema_drift.py` asserts registry toggle keys ⊆ prefs feature keys. The predictability cluster no longer contributes `pred_*` keys — update the guard's expected set (remove those four). Run: `python -m pytest tests/planner -q` → PASS.

- [ ] **Step 5: Run planner tests**

Run: `python -m pytest tests/planner -q`
Expected: PASS (fix any emit test referencing `ns.no_surprisal`/`no_llm_cloze`/`llm_model`).

- [ ] **Step 6: Commit**

```bash
git add gui/planner tests/planner
git commit -m "feat(planner): dynamic predictability stage + ns.predictability"
```

---

### Task 5: main.py wiring + retire CLI flags

**Files:**
- Modify: `main.py`
- Test: `tests/test_main_cli.py`, `tests/test_run_extract.py`

**Interfaces:**
- Consumes: `ns.predictability` (from emit) or prefs fallback; `add_predictability`.
- Produces: `run()` writes `<slug>_<Metric>` columns; no `--no-surprisal`/`--no-llm-cloze`/`--llm-model` argparse flags.

- [ ] **Step 1: Write the failing test**

```python
def test_removed_predictability_flags_are_gone():
    from main import build_parser
    p = build_parser()
    import pytest
    for flag in ("--no-surprisal", "--no-llm-cloze", "--llm-model"):
        with pytest.raises(SystemExit):
            p.parse_args(["--transcript", "t.txt", flag, "x"])

def test_run_reads_predictability_from_namespace(monkeypatch, tmp_path):
    # A namespace carrying predictability config drives add_predictability.
    import main, pandas as pd
    called = {}
    monkeypatch.setattr("extractors.predictability.add_predictability",
        lambda df, text, **k: called.setdefault("k", k) or df.assign(**{}))
    # ... build a minimal args with .predictability set, .no_parse etc.,
    # a tiny transcript; call main.run(args); assert called["k"]["models"].
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_main_cli.py -q -k predictability`
Expected: FAIL (flags still parse).

- [ ] **Step 3: Implement**

In `build_parser`: remove the `--no-surprisal`, `--no-llm-cloze`, `--llm-model` arguments. In `run()`: replace the `add_surprisal` and `add_llm_cloze` blocks with one block:
```python
pconf = getattr(args, "predictability", None)
if pconf is None:
    pconf = load_preferences().get("predictability", {})
models = list(pconf.get("models") or [])
metrics = [m for m, on in (pconf.get("metrics") or {}).items() if on]
if models and metrics and df is not None:
    from extractors.predictability import add_predictability
    print(f"[predict] {len(models)} model(s) × {len(metrics)} metric(s)…")
    df = add_predictability(df, text, models=models, metrics=metrics)
```
Remove the `config_snapshot` keys `no_surprisal`/`no_llm_cloze`/`llm_model` (replace with `predictability`). Remove `select_model` import usage here.

- [ ] **Step 4: Run CLI tests**

Run: `python -m pytest tests/test_main_cli.py tests/test_run_extract.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add main.py tests/test_main_cli.py
git commit -m "feat(cli): run() uses add_predictability; retire predictability flags"
```

---

### Task 6: Doc-level predictability aggregates

**Files:**
- Modify: `utils/doc_summary.py`, `gui/planner/registry.py` (doc-agg cluster), `main.py` (doc_toggles)
- Test: `tests/utils/test_doc_summary.py`

**Interfaces:**
- Consumes: word-level `<slug>_<Metric>` columns; toggle `doc_agg_predictability`.
- Produces: `mean_<slug>_<metric>` columns (mean over non-NaN).

- [ ] **Step 1: Write the failing test**

```python
def test_doc_predictability_means(monkeypatch):
    import pandas as pd, math
    from utils.doc_summary import build_doc_summary
    df = pd.DataFrame({
        "token": ["a", "b"], "upos": ["NOUN", "NOUN"],
        "gpt2_Surprisal": [2.0, 4.0], "gpt2_Cloze": [0.5, 0.1],
    })
    s = build_doc_summary(df, stimulus_id="x", dictionary_used="Empath",
                          text="a b", toggles={"predictability_means": True})
    assert s["mean_gpt2_surprisal"] == 3.0
    assert math.isclose(s["mean_gpt2_cloze"], 0.3)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/utils/test_doc_summary.py -q -k predictability`
Expected: FAIL.

- [ ] **Step 3: Implement**

In `doc_summary.py`, add a `predictability_means` branch that scans columns matching `^(?P<slug>[a-z0-9_]+)_(Surprisal|Cloze|TopMatch)$` and emits `mean_<slug>_<metric-lower>` via the existing `_mean` helper. Add `predictability_means` to `DOC_FEATURE_KEYS`. In `main.py` `run_doc_level_batch`, replace the `mean_surprisal`/`mean_llm_cloze` toggle wiring with `'predictability_means': not args.no_doc_predictability` (add the arg, defaulting on) or drive from the doc toggle dict; ensure `extract_args` forces predictability on when the aggregate is wanted (mirror the existing `extract_args.no_freq = False` pattern — set `extract_args.predictability` to the union needed). In `registry.py`, replace the `doc_agg_surprisal` + `doc_agg_llm` doc clusters with one `doc_agg_predictability`.

- [ ] **Step 4: Run doc-summary + doc-batch tests**

Run: `python -m pytest tests/utils/test_doc_summary.py tests/integration/test_pipeline_batch_doc.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add utils/doc_summary.py gui/planner/registry.py main.py tests/utils/test_doc_summary.py
git commit -m "feat(doc): mean_<slug>_<metric> aggregates via doc_agg_predictability"
```

---

### Task 7: GUI — Predictability card redesign

**Files:**
- Modify: `gui/section_word_features.py`, `gui/preflight.py`
- Test: `tests/gui/test_section_word_features.py`, `tests/gui/test_preflight.py`

**Interfaces:**
- Consumes: roster, prefs v18.
- Produces: model checkboxes writing `predictability.models`; 3 metric checkboxes writing `predictability.metrics`; HF-token box gated on selected-model gating; preflight warnings.

- [ ] **Step 1: Write the failing GUI tests**

```python
def test_model_rows_persist_selection(qtbot, prefs_tmp):
    from gui.section_word_features import WordFeaturesSection
    from utils.preferences import load_preferences
    w = WordFeaturesSection(); qtbot.addWidget(w)
    w.model_checkboxes["gpt2"].setChecked(True)
    assert "gpt2" in load_preferences()["predictability"]["models"]

def test_metric_toggles_persist(qtbot, prefs_tmp):
    from gui.section_word_features import WordFeaturesSection
    from utils.preferences import load_preferences
    w = WordFeaturesSection(); qtbot.addWidget(w)
    w.metric_checkboxes["cloze"].setChecked(True)
    assert load_preferences()["predictability"]["metrics"]["cloze"] is True

def test_hf_box_shown_when_gated_model_selected_without_token(qtbot, prefs_tmp, monkeypatch):
    import extractors.predictability as m
    monkeypatch.setattr(m, "has_hf_token", lambda: False)
    from gui.section_word_features import WordFeaturesSection
    w = WordFeaturesSection(); qtbot.addWidget(w); w.show()
    w.model_checkboxes["llama3_8b"].setChecked(True)
    assert w.hf_token_row.isVisible() is True
    w.model_checkboxes["llama3_8b"].setChecked(False)
    w.model_checkboxes["gpt2"].setChecked(True)   # free model
    assert w.hf_token_row.isVisible() is False
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/gui/test_section_word_features.py -q -k "model_rows or metric_toggles or hf_box"`
Expected: FAIL.

- [ ] **Step 3: Rebuild the Predictability card**

Replace lines ~132–229 of `section_word_features.py` (the current predictability block, from `# --- Predictability ---` through `layout.addWidget(pred_card)`). New structure, following the existing `ClusterCard` + `_make_toggle` patterns:
- Build `self.model_checkboxes: dict[str, QCheckBox]` — one per `all_specs()`, label `f"{spec.label} — {spec.note}"`; on toggle, read all checked slugs and `save_preferences({"predictability": {"models": [...]}})`, then call `self._refresh_llm_credential_visibility()`.
- Build `self.metric_checkboxes: dict[str, QCheckBox]` for `surprisal/cloze/topmatch`; on toggle, `save_preferences({"predictability": {"metrics": {...}}})`.
- Keep the credentials help ⓘ (`llm_credentials`). Remove the OpenAI **api_key_row** entirely (dead post-OpenAI-removal). Keep `hf_token_row` + `hf_token_detected_label`.
- Rewrite `_refresh_llm_credential_visibility()` to show the HF box when `any(is_gated(slug) for slug in checked) and not has_hf_token()`; drop the `is_openai`/combo logic and the `_on_model_change` method (and its callers). Keep `session_hf_token()`.

Update `gui/main_window.py` if it references `session_api_key` from this widget (grep) — replace with a no-op or remove the api-key threading (it is inert post-OpenAI-removal).

- [ ] **Step 4: Preflight warnings**

In `gui/preflight.py`, replace the single-model llm block with: gather `models = prefs["predictability"]["models"]`, `metrics = [m for m,on in prefs["predictability"]["metrics"].items() if on]`. Warn (amber) if: a gated model is selected and no token (`is_gated(slug) and not session_hf_token and not has_hf_token()`); `models and not metrics`; `metrics and not models`. Update `tests/gui/test_preflight.py` accordingly.

- [ ] **Step 5: Run GUI tests**

Run: `python -m pytest tests/gui/test_section_word_features.py tests/gui/test_preflight.py tests/gui/test_main_window.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add gui/section_word_features.py gui/preflight.py gui/main_window.py tests/gui
git commit -m "feat(gui): predictability model multi-select + metric toggles + gating"
```

---

### Task 8: Docs, sherlock config, README, master_plan

**Files:**
- Create: `docs/features/pred_surprisal.md` (merge of the two surprisal docs)
- Delete: `docs/features/pred_surprisal_gpt2.md`, `docs/features/pred_surprisal_llm.md`
- Modify: `docs/features/pred_cloze.md`, `docs/features/pred_topmatch.md`, `docs/features/llm_credentials.md`, `gui/feature_docs.py` (`_SHARED_DOC_MAP`), `README.md`, `docs/master_plan.md`, `sherlock/config.yaml`
- Test: `tests/gui/test_feature_docs.py`

- [ ] **Step 1: Merge the surprisal docs**

Combine `pred_surprisal_gpt2.md` + `pred_surprisal_llm.md` into `pred_surprisal.md`: model-agnostic meaning/EEG/math (the metric is identical across models), a short "Models" note pointing at the roster, the merged chunking/long-text section, and **all** citations from both (dedupe). `git rm` the two originals.

- [ ] **Step 2: Update feature-doc routing**

In `gui/feature_docs.py`, map the new metric/model toggle keys to docs: the surprisal metric → `pred_surprisal`; cloze → `pred_cloze`; topmatch → `pred_topmatch`; model rows / card → `llm_credentials` (or a roster doc). Update `tests/gui/test_feature_docs.py` for the new mapping (drop the old `pred_surprisal_gpt2`/`_llm` keys). De-model-specialize wording in `pred_cloze.md` / `pred_topmatch.md` / `llm_credentials.md` (GPT-2 = no token; columns are `<slug>_<Metric>`).

- [ ] **Step 3: Regenerate sherlock config**

Run the export used to generate it (see `utils/config_yaml.py` / `test_export_config.py` for the entry point), or hand-edit `sherlock/config.yaml`: replace the four `pred_*` feature lines and `settings.predictability.llm_model` with the `predictability: {models, metrics}` settings block; drop `run.openai_api_key_file`. Verify `python -m pytest tests/utils/test_config_yaml.py tests/test_export_config.py -q` passes (update expectations if the exporter enumerates predictability differently).

- [ ] **Step 4: README + master_plan**

README predictability section: model roster + namespaced `<slug>_<Metric>` columns. `master_plan.md`: new §3 completed-batch row noting the **breaking** column rename, prefs v18, and the retired CLI flags; bump the prefs-version references to v18; note Project B (context size) as the next follow-on.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "docs(predictability): merge surprisal docs; roster docs; sherlock/README/master_plan"
```

---

### Task 9: Full-suite integration + verification

**Files:**
- Test: `tests/integration/test_pipeline_batch.py` (extend)

- [ ] **Step 1: Write the integration test**

```python
def test_word_batch_emits_namespaced_predictability_columns(tmp_path):
    # Text-only batch, gpt2 only (free, no token), surprisal+cloze on.
    # Assert the word CSV has gpt2_Surprisal and gpt2_Cloze and NOT
    # Surprisal / LLM_* / gpt2_TopMatch.
    ...
```

- [ ] **Step 2: Run it**

Run: `python -m pytest tests/integration/test_pipeline_batch.py -q -k namespaced`
Expected: PASS.

- [ ] **Step 3: Full suite**

Run: `python -m pytest -q`
Expected: green (≥ 672 baseline minus retired duplicates, plus new tests). Fix any stragglers referencing old columns/keys.

- [ ] **Step 4: Real smoke test**

Run a text-only CLI/GUI run with gpt2 + surprisal on; confirm `gpt2_Surprisal` appears with the document-first word NaN. (Audio-dependent paths still deferred to the user's manual MFA pass.)

- [ ] **Step 5: Commit**

```bash
git add tests/integration/test_pipeline_batch.py
git commit -m "test(predictability): namespaced-column integration + full-suite green"
```

---

## Self-Review

**Spec coverage:** §3 roster→T1; §4 extractor→T2a/T2b; §5 prefs v18→T3; §6 planner/emit→T4, main/CLI→T5, sherlock→T8; §7 GUI→T7; §8 doc aggregates→T6; §9 docs→T8; §10 tests→every task + T9; §11 breaking changes→T8 (master_plan/README) + covered by migration T3; §12 dynamic-planner risk→T4 (with the spec fallback inlined). All covered.

**Placeholder scan:** the only intentionally-sketched code is T5 Step 1's second test body and T9 Step 1 (both marked with `...` where the harness scaffolding is mechanical and follows existing batch-test patterns in the same files); all logic-bearing steps show real code.

**Type consistency:** `add_predictability(df, text, *, models, metrics, backends=None)`, backend `.score(text, tokens) -> [{p,surprisal,top_match}]`, roster `spec_for/is_gated/slug_for_hf_id`, `ns.predictability={"models","metrics"}`, columns `<slug>_<Cap>`, doc means `mean_<slug>_<lower>` — consistent across T1–T9.
