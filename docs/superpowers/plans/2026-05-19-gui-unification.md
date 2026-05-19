# GUI Validation Unification + Cluster UX Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `gui/planner/` the single source of truth for every GUI surface that has an opinion about feature dependencies. Add two declarative fields to `FeatureCluster` (`master_toggle`, `available_when`), promote them through the registry, then collapse the parallel logic in `gui/preflight.py`, `gui/section_run.py`, and the section pages into planner consumers. Migrate prefs v6 → v7 (additive). Add a session-units sidecar.

**Architecture:** Two new fields on `FeatureCluster` (frozen tuple-typed). One new GUI widget (`ClusterCard`). One new utility (`utils/session_units.py`). One prefs migration step. Five existing files repointed at the planner. No edits to `main.py`'s argparse or any extractor.

**Tech Stack:** Python 3.10, PySide6 (existing), pytest. No new external dependencies.

**Spec:** [`docs/superpowers/specs/2026-05-19-gui-unification-design.md`](../specs/2026-05-19-gui-unification-design.md). **Upstream:** [`docs/superpowers/specs/2026-05-14-planner-design.md`](../specs/2026-05-14-planner-design.md), [`docs/feature_dependencies.md`](../../feature_dependencies.md).

---

## File Structure

**New files:**

| File | Responsibility |
|---|---|
| `utils/session_units.py` | `build_session_units(plan, prefs) -> list[dict]` — column→unit table per run. |
| `tests/utils/test_session_units.py` | Unit tests for the table builder. |
| `tests/gui/test_preflight_invariants.py` | New invariant tests for the planner-aware preflight. |
| `tests/gui/test_cluster_card.py` | Widget tests for `ClusterCard`. |

**Modified files:**

| File | Change |
|---|---|
| `gui/planner/stages.py` | +2 fields on `FeatureCluster`. |
| `gui/planner/plan.py` | `_active_clusters` checks both new fields; helper `_get_dotted(prefs, "a.b") -> value`. |
| `gui/planner/registry.py` | Each cluster declares `master_toggle` / `available_when` where appropriate. |
| `gui/planner/emit.py` | `_categories_for_level` reads unified `categories_<backend>` key. |
| `gui/preflight.py` | Rewrite (~200 → ~120 lines). Planner consumer, env/asset-only. |
| `gui/section_run.py` | Cost panel gated on `llm_predict in plan.stages_to_run`. |
| `gui/section_word_features.py` | Cluster cards wrap prosody / EWFG / Composite psych (LIWC gate). |
| `gui/section_doc_features.py` | Cluster cards; doc-categories shows count-only with hint. |
| `gui/section_subword_features.py` | Syllables card splits into Intervals + Labels + Prosody. |
| `gui/widgets.py` | New `ClusterCard` widget. |
| `gui/category_picker.py` | Drops `level` arg; reads `semantic.categories_<backend>`. |
| `utils/preferences.py` | +v7 defaults + `_migrate_to_v7`. |
| `main.py` | Writes `<output>/<run_ts>_session_units.csv` after each batch. |
| `tests/planner/test_stages.py` | Cover both new fields. |
| `tests/planner/test_plan.py` | Activation tests for both gates. |
| `tests/planner/test_schema_drift.py` | New invariant: every `master_*` key in clusters exists in v7 prefs. |
| `tests/utils/test_preferences_migration.py` | v6→v7 round-trip. |
| `tests/gui/test_preflight.py` | Repoint stale assertions; add invariant. |
| `tests/gui/test_category_picker.py` | Unified-key reads. |
| `tests/gui/test_section_*` | Cluster-card master gating tests. |
| `docs/master_plan.md` | New row dated 2026-05-19. |

**Unchanged files (verify):** all of `extractors/*.py`, `main.py`'s argparse + batch-runner core, `gui/run_dispatch.py`, `gui/planner/__init__.py`, `utils/run_logger.py`, `utils/textgrid_parser.py`.

---

## Task 1: Add `master_toggle` + `available_when` to `FeatureCluster`

**Files:** Modify `gui/planner/stages.py`, `gui/planner/plan.py`, `tests/planner/test_stages.py`, `tests/planner/test_plan.py`.

**TDD.** Add two failing tests first; then add the fields and activation logic.

- [ ] **Step 1: Write failing tests** in `tests/planner/test_stages.py`:

```python
def test_feature_cluster_accepts_master_toggle():
    c = FeatureCluster(
        name="word prosody",
        toggle_keys=("feat_prosody_pitch_mean",),
        output_csv="features",
        requires_stages=("word_prosody",),
        emits_columns=("Pitch_Mean",),
        master_toggle="master_word_prosody",
    )
    assert c.master_toggle == "master_word_prosody"
    assert c.available_when is None  # default


def test_feature_cluster_accepts_available_when():
    c = FeatureCluster(
        name="psych composite",
        toggle_keys=("feat_psych_authentic",),
        output_csv="documents_summary",
        requires_stages=("doc_liwc",),
        emits_columns=("psych_authentic",),
        available_when=("semantic.backend", "liwc"),
    )
    assert c.available_when == ("semantic.backend", "liwc")
```

Add activation tests in `tests/planner/test_plan.py`:

```python
def test_master_toggle_gates_cluster():
    """Cluster is inactive when its master_toggle is False even if a child is on."""
    # Use SUBTLEX cluster as baseline — but we need a cluster with a master.
    # Simulate by patching CLUSTERS with a master-bearing one, OR rely on
    # the real registry's master_word_prosody once Task 3 lands. For Task 1,
    # write a direct unit test of _active_clusters with a synthetic cluster.
    ...


def test_available_when_gates_cluster():
    """Cluster is inactive when available_when predicate is False."""
    ...
```

These will use a small fixture that monkeypatches `CLUSTERS` with a hand-built test cluster — keeps the test isolated from registry changes that come in Task 3.

- [ ] **Step 2: Run the tests to confirm they fail**

```bash
./env/python.exe -m pytest tests/planner/test_stages.py tests/planner/test_plan.py -v
```

Expected: the two new shape tests fail with TypeError (unexpected keyword arg).

- [ ] **Step 3: Add the fields** to `gui/planner/stages.py`:

```python
@dataclass(frozen=True)
class FeatureCluster:
    name: str
    toggle_keys: tuple[str, ...]
    output_csv: str
    requires_stages: tuple[str, ...]
    emits_columns: tuple[str, ...]
    master_toggle: str | None = None
    available_when: tuple[str, str] | None = None
```

- [ ] **Step 4: Add activation logic** in `gui/planner/plan.py`:

```python
def _get_dotted(prefs: dict, path: str):
    cur = prefs
    for part in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def _active_clusters(prefs: dict, active_csvs: frozenset[str]) -> list[FeatureCluster]:
    actives = []
    for c in CLUSTERS:
        if c.available_when is not None:
            path, expected = c.available_when
            if _get_dotted(prefs, path) != expected:
                continue
        if c.master_toggle is not None and not _toggle_value(prefs, c.master_toggle):
            continue
        if c.output_csv not in active_csvs:
            continue
        if any(_toggle_value(prefs, k) for k in c.toggle_keys):
            actives.append(c)
    return actives
```

- [ ] **Step 5: Run tests**

```bash
./env/python.exe -m pytest tests/planner/ -v
```

Expected: all planner tests pass, including the new activation tests.

- [ ] **Step 6: Commit**

```
planner: add master_toggle + available_when to FeatureCluster

Two declarative fields enabling the GUI to express cluster-level
master toggles (with grayed-out children) and environment-availability
gates (e.g. psych composite requires LIWC backend). Both default to
None — existing clusters behave identically.

_active_clusters now early-outs on available_when (hard env gate),
then master_toggle (user gate), then the existing output_csv +
any-child checks.

Refs spec §5-6.
```

---

## Task 2: Prefs schema v7 + migration

**Files:** Modify `utils/preferences.py`. Add `tests/utils/test_preferences_migration.py` (or extend existing).

- [ ] **Step 1: Add v7 constants**

```python
SCHEMA_VERSION_V7 = 7

# New master_* feature keys
_MASTERS_V7_FEATURES = {
    "master_word_prosody":         True,
    "master_psych_composite":      True,
    "master_syllable_intervals":   True,
    "master_syllable_labels":      True,
    "master_syllable_prosody":     True,
    "master_phoneme_intervals":    True,
}

# New master_* output keys
_MASTERS_V7_OUTPUT = {
    "master_doc_prosody":          True,
    "master_doc_articulation_rate": True,
}

_FEATURES_DEFAULT_V7 = {**_FEATURES_DEFAULT_V6, **_MASTERS_V7_FEATURES}
_OUTPUT_DEFAULT_V7   = {**_OUTPUT_DEFAULT_V6,   **_MASTERS_V7_OUTPUT}

DEFAULT_PREFS_V7: dict[str, Any] = {
    **DEFAULT_PREFS_V6,
    "schema_version": SCHEMA_VERSION_V7,
    "features": dict(_FEATURES_DEFAULT_V7),
    "output":   dict(_OUTPUT_DEFAULT_V7),
    # Semantic categories: unified per-backend
    "semantic": {
        **DEFAULT_PREFS_V6["semantic"],
        "categories_liwc":   list(_SOP_CATEGORIES_LIWC),
        "categories_empath": list(_SOP_CATEGORIES_EMPATH),
    },
}

SCHEMA_VERSION = SCHEMA_VERSION_V7
DEFAULT_PREFS: dict[str, Any] = DEFAULT_PREFS_V7
```

- [ ] **Step 2: Add `_migrate_to_v7`**

```python
def _migrate_to_v7(prefs: dict) -> dict:
    """v6 → v7: add master_* defaults; union word+doc categories."""
    out = copy.deepcopy(prefs)

    feats = out.setdefault("features", {})
    for k, v in _MASTERS_V7_FEATURES.items():
        feats.setdefault(k, v)

    outs = out.setdefault("output", {})
    for k, v in _MASTERS_V7_OUTPUT.items():
        outs.setdefault(k, v)

    sem = out.setdefault("semantic", {})
    for backend in ("liwc", "empath"):
        word_key = f"word_categories_{backend}"
        doc_key  = f"doc_categories_{backend}"
        unified_key = f"categories_{backend}"
        if unified_key not in sem:
            union = list(dict.fromkeys(
                sem.get(word_key, []) + sem.get(doc_key, [])
            ))
            if not union:
                # Both empty (or absent) — fall back to SOP default for backend
                union = list(_SOP_CATEGORIES_LIWC if backend == "liwc"
                             else _SOP_CATEGORIES_EMPATH)
            sem[unified_key] = union
        sem.pop(word_key, None)
        sem.pop(doc_key, None)

    out["schema_version"] = SCHEMA_VERSION_V7
    return out
```

Wire into `MIGRATIONS`:

```python
MIGRATIONS = {
    1: _migrate_to_v2,
    # ... existing ...
    6: _migrate_to_v7,
}
```

- [ ] **Step 3: Tests**

In `tests/utils/test_preferences_migration.py`:

```python
def test_v6_to_v7_unions_categories_lossless():
    v6 = {
        "schema_version": 6,
        "semantic": {
            "backend": "empath",
            "word_categories_empath": ["positive_emotion", "work"],
            "doc_categories_empath":  ["family", "money"],
            "word_categories_liwc":   [],
            "doc_categories_liwc":    [],
        },
        "features": {}, "output": {},
    }
    v7 = _migrate_to_v7(v6)
    assert v7["schema_version"] == 7
    assert set(v7["semantic"]["categories_empath"]) == {
        "positive_emotion", "work", "family", "money"}
    assert "word_categories_empath" not in v7["semantic"]
    assert "doc_categories_empath"  not in v7["semantic"]


def test_v6_to_v7_sets_master_defaults_true():
    v7 = _migrate_to_v7({"schema_version": 6, "features": {}, "output": {}})
    for k in ("master_word_prosody", "master_psych_composite",
             "master_syllable_intervals", "master_syllable_prosody"):
        assert v7["features"][k] is True
    assert v7["output"]["master_doc_prosody"] is True
```

- [ ] **Step 4: Run tests**

```bash
./env/python.exe -m pytest tests/utils/test_preferences_migration.py -v
./env/python.exe -m pytest -q
```

Both should be green.

- [ ] **Step 5: Commit**

```
prefs: schema v7 — cluster master toggles + unified categories

Adds 8 master_* keys (6 features + 2 output) defaulting to True so
existing user prefs continue to behave identically until a master is
toggled off.

Unifies semantic.{word,doc}_categories_<backend> into a single
semantic.categories_<backend> via lossless union. The word-level page
becomes the canonical editor; doc-level page reads the same key.

Migration is single-step (v6 → v7). Tests cover union, defaults, and
the schema-version bump.

Refs spec §7.
```

---

## Task 3: Wire `master_toggle` + `available_when` onto every cluster in `registry.py`

**Files:** Modify `gui/planner/registry.py`. Update `tests/planner/test_schema_drift.py`.

- [ ] **Step 1: Add master_toggle / available_when to each cluster**

Mapping table (apply in registry):

| Cluster | master_toggle | available_when |
|---|---|---|
| word structural identifiers | — | — |
| lex-ling | — | — |
| SUBTLEX frequency | — | — |
| EWFG grade-band frequency | — | — |
| GPT-2 surprisal | — | — |
| LLM predictability | — | — |
| semantic indicators | — (already gated by `sem_indicators`) | — |
| word acoustic alignment | — | — |
| word derived timing | — | — |
| **word prosody** | `master_word_prosody` | — |
| **phoneme intervals** | `master_phoneme_intervals` | — |
| **syllabification + labels** | (split — see below) | — |
| **syllable prosody** | `master_syllable_prosody` | — |
| doc counts | — (gated by `out_count_*`) | — |
| doc word-feature aggregates | — (gated by `doc_aggregates`) | — |
| doc lex diversity | — (gated by `doc_lex_diversity`) | — |
| doc readability | — (gated by `doc_readability`) | — |
| **doc LIWC composite** | `master_psych_composite` | `("semantic.backend", "liwc")` |
| doc category proportions | — | — |
| **doc prosody** | `master_doc_prosody` | — |
| **doc articulation rate** | `master_doc_articulation_rate` | — |

Additionally, **split the existing "syllabification + labels" cluster** into two:
- `syllabification intervals` — toggle_keys: `feat_syllable_onset, feat_syllable_end, feat_syllable_duration`; emits: `Onset, End, Duration`. master_toggle: `master_syllable_intervals`.
- `syllable labels` — toggle_keys: `feat_syllable_label, feat_syllable_arpabet`; emits: `Label, ARPABET`. master_toggle: `master_syllable_labels`.

Both clusters keep the same `requires_stages=("spacy_parse", "mfa_align", "syllabify_pairs")` and `output_csv="syllables"`. The pre-Task-3 cluster's 22 entries become 23.

- [ ] **Step 2: Update schema-drift invariant**

In `tests/planner/test_schema_drift.py`, add:

```python
def test_every_master_toggle_key_exists_in_prefs_schema():
    """Each cluster's master_toggle (when present) must be in the v7 schema."""
    from utils.preferences import _FEATURES_DEFAULT_V7, _OUTPUT_DEFAULT_V7
    all_pref_keys = set(_FEATURES_DEFAULT_V7) | set(_OUTPUT_DEFAULT_V7)
    masters = {c.master_toggle for c in CLUSTERS if c.master_toggle}
    missing = masters - all_pref_keys
    assert not missing, f"Cluster master_toggle(s) not in v7 schema: {sorted(missing)}"
```

- [ ] **Step 3: Update existing tests** to account for the split syllable cluster (22 → 23 clusters).

- [ ] **Step 4: Run**

```bash
./env/python.exe -m pytest tests/planner/ -v
./env/python.exe -m pytest -q
```

- [ ] **Step 5: Commit**

```
planner: master toggles + LIWC gate + syllable split in registry

Eight clusters gain a master_toggle (word prosody, syllable
prosody/intervals/labels, phoneme intervals, doc prosody/articulation
rate, psych composite). The psych composite also gets
available_when=("semantic.backend", "liwc") so it deactivates entirely
when the user has Empath selected.

The old "syllabification + labels" cluster splits into two clusters:
syllabification intervals (Onset/End/Duration) and syllable labels
(Label/ARPABET). This is what enables the per-sub-cluster master
toggles the user has been asking for.

Schema-drift invariant now covers master_toggle ↔ v7 prefs schema.

Refs spec §7, feature_dependencies.md §2.
```

---

## Task 4: Rewrite `gui/preflight.py` as a planner consumer

**Files:** Modify `gui/preflight.py`, `tests/gui/test_preflight.py`. Add `tests/gui/test_preflight_invariants.py`.

- [ ] **Step 1: Rewrite preflight.compute_warnings**

Replace the file's `compute_warnings` function. New behavior per spec §11:

```python
from gui.planner import plan_from_prefs


def compute_warnings(session_api_key: str = "") -> list[Warning]:
    prefs = load_preferences()
    plan = plan_from_prefs(prefs)
    out: list[Warning] = []

    # Empty-plan info hint
    if not plan.output_csvs:
        out.append(Warning(
            toggle_key=None, severity="info",
            message="No outputs selected — pick at least one CSV in the Outputs section.",
        ))
        return out

    # Asset/env checks, each gated on planner stage list
    if "ewfg_freq" in plan.stages_to_run and not _ewfg_parquet_exists():
        out.append(Warning(...))
    if "llm_predict" in plan.stages_to_run:
        model = prefs.get("predictability", {}).get("llm_model", "")
        if model.startswith("gpt-") and not session_api_key:
            out.append(Warning(...))
    if "sem_domain" in plan.stages_to_run and prefs["semantic"].get("backend") == "liwc" \
            and not _liwc_dic_exists():
        out.append(Warning(...))
    if "mfa_align" in plan.stages_to_run and not _mfa_on_path():
        out.append(Warning(...))
    if any(s in plan.stages_to_run for s in ("word_prosody", "syllable_prosody",
                                              "doc_prosody", "praat_contours")):
        if not _parselmouth_importable():
            out.append(Warning(...))
    if "doc_articulation_rate" in plan.stages_to_run and not _pyphen_importable():
        out.append(Warning(...))
    if "syllabify_pairs" in plan.stages_to_run \
            and prefs["features"].get("feat_syllable_label", False) \
            and not _pyphen_importable():
        out.append(Warning(...))

    return out
```

Delete entirely:
- The prosody-needs-acoustic block (L94-108)
- The sub-word-needs-acoustic block (L180-185)
- The `_any_acoustic`, `_any_subword`, `_any_prosody` helpers and key tuples
- The L144-163 categories-vs-master-toggle block (Task 7 makes the picker show its own grayed-out state)

- [ ] **Step 2: Repoint existing tests** in `tests/gui/test_preflight.py`. For each test asserting the sub-word/prosody/acoustic stale warning is present, flip the assertion to "warning absent". Delete tests that exercise only the deleted blocks.

- [ ] **Step 3: Add invariant test** in `tests/gui/test_preflight_invariants.py`:

```python
def test_no_preflight_warning_says_enable_a_toggle():
    """Preflight should never instruct the user to flip a toggle —
    the planner handles all toggle-dependency resolution."""
    # Iterate a representative matrix of prefs configurations
    for prefs_override in _representative_prefs_matrix():
        with _temp_prefs(prefs_override):
            warnings = compute_warnings(session_api_key="")
            for w in warnings:
                assert "enable" not in w.message.lower(), \
                    f"Preflight told user to enable something: {w.message!r}"


def test_empty_plan_shows_info_hint_only():
    with _temp_prefs({"output": {k: False for k in _OUTPUT_CSV_MAP}}):
        warnings = compute_warnings(session_api_key="")
        assert len(warnings) == 1
        assert warnings[0].severity == "info"
        assert "No outputs selected" in warnings[0].message
```

- [ ] **Step 4: Run tests**

```bash
./env/python.exe -m pytest tests/gui/test_preflight.py tests/gui/test_preflight_invariants.py -v
./env/python.exe -m pytest -q
```

- [ ] **Step 5: Commit**

```
preflight: planner consumer, env/asset checks only

Deletes the two stale toggle-dependency warnings (prosody-needs-
acoustic and sub-word-needs-acoustic) — the planner already pulls
mfa_align/praat_contours from each cluster's requires_stages, so the
warnings were instructing users to fix non-bugs.

Surviving env/asset checks (EWFG parquet, OpenAI key, LIWC .dic, MFA
binary, parselmouth, pyphen) now gate on the planner's stage list,
firing only when the relevant stage is actually scheduled.

New invariant: no preflight warning message contains the substring
'enable'. Catches future regressions where toggle-dependency logic
sneaks back in.

Empty-plan info hint added.

Refs spec §11.
```

---

## Task 5: Cost panel consumes planner state

**Files:** Modify `gui/section_run.py`. Add tests in `tests/gui/test_run_dispatch.py` (or new file).

- [ ] **Step 1: Gate the cost panel** in `gui/section_run.py:refresh`:

```python
# Cost panel — gated on the planner saying llm_predict is in the plan
plan = planner.plan_from_prefs(prefs)
if "llm_predict" in plan.stages_to_run:
    model = prefs["predictability"].get("llm_model", "")
    if model.startswith("gpt-"):
        self.cost_panel.setVisible(True)
        # ... existing token-counting + cost rendering ...
    else:
        self.cost_panel.setVisible(False)
else:
    self.cost_panel.setVisible(False)
```

- [ ] **Step 2: Test** — assert that with `pred_*` all off but `llm_model="gpt-4o-mini"`, the cost panel is hidden:

```python
def test_cost_panel_hidden_when_no_llm_features_active(...):
    save_preferences({
        "predictability": {"llm_model": "gpt-4o-mini"},
        "features": {"pred_surprisal_llm": False, "pred_cloze": False,
                     "pred_topmatch": False, ...},
        "output": {"word_level_csv": True, ...},
    })
    # Construct the RunSection widget headlessly and call refresh()
    # Assert self.cost_panel.isVisible() == False
```

- [ ] **Step 3: Run + Commit**

```
gui: cost panel gated on planner state

The Run-tab cost panel previously rendered whenever the configured
model started with 'gpt-', regardless of whether any LLM-using
feature toggle was actually on. Now it queries the planner: shown
only when 'llm_predict' is in plan.stages_to_run AND the model is
an OpenAI one.

Refs spec §11 (cost panel gating).
```

---

## Task 6: `ClusterCard` widget + Syllables split in section pages

**Files:** Modify `gui/widgets.py`, `gui/section_subword_features.py`, `gui/section_word_features.py`, `gui/section_doc_features.py`. Add `tests/gui/test_cluster_card.py`.

- [ ] **Step 1: Add `ClusterCard`** to `gui/widgets.py`. Subclasses or composes `Card`:

```python
class ClusterCard(Card):
    """Card with a master ToggleRow + grayed-out children when master is off
    or when `available` is False."""

    settings_changed = Signal()

    def __init__(self, title: str, subtitle: str,
                 master_key: str, master_label: str,
                 *, master_section: str = "features",
                 available: bool = True,
                 disabled_subtitle: str | None = None,
                 parent=None):
        super().__init__(title, subtitle, parent)
        self._master_key = master_key
        self._master_section = master_section
        self._available = available
        prefs = load_preferences()
        initial = prefs.get(master_section, {}).get(master_key, True)

        self._master = ToggleRow(label=master_label)
        self._master.set_checked(initial and available)
        self._master.setEnabled(available)
        self._master.toggled.connect(self._on_master_toggled)
        self.add_widget(self._master)
        self._children_widgets: list[QWidget] = []

        if not available and disabled_subtitle:
            self.add_widget(_make_subtitle(disabled_subtitle))

        self._update_children_enabled(initial and available)

    def add_child(self, widget: QWidget) -> None:
        self._children_widgets.append(widget)
        self.add_widget(widget)
        widget.setEnabled(self._master.is_checked() and self._available)

    def _on_master_toggled(self, v: bool) -> None:
        save_preferences({self._master_section: {self._master_key: v}})
        self._update_children_enabled(v and self._available)
        self.settings_changed.emit()

    def _update_children_enabled(self, enabled: bool) -> None:
        for w in self._children_widgets:
            w.setEnabled(enabled)
```

- [ ] **Step 2: Rewrite `gui/section_subword_features.py`** to use three syllable `ClusterCard`s + one phoneme `ClusterCard`. Each gets the appropriate master key from Task 3.

- [ ] **Step 3: Modify `gui/section_word_features.py`** to wrap:
  - Word-prosody card → `ClusterCard(master_key="master_word_prosody")`
  - Composite-psych card → `ClusterCard(master_key="master_psych_composite", available=prefs["semantic"]["backend"]=="liwc", disabled_subtitle="Requires LIWC backend")`

- [ ] **Step 4: Modify `gui/section_doc_features.py`** for doc-prosody and doc-articulation-rate masters.

- [ ] **Step 5: Tests** in `tests/gui/test_cluster_card.py`:

```python
def test_cluster_card_disables_children_when_master_off(qtbot, prefs_tmp):
    save_preferences({"features": {"master_word_prosody": False,
                                    "feat_prosody_pitch_mean": True}})
    card = ClusterCard("Prosody", "Word-level prosody",
                       master_key="master_word_prosody",
                       master_label="Enable word prosody")
    child = ToggleRow(label="Pitch_Mean")
    card.add_child(child)
    assert child.isEnabled() is False


def test_cluster_card_available_false_disables_master(qtbot, prefs_tmp):
    card = ClusterCard("Psych", "LIWC psych composite",
                       master_key="master_psych_composite",
                       master_label="Enable psych composite",
                       available=False,
                       disabled_subtitle="Requires LIWC backend")
    # The master itself is disabled (not interactable)
    assert card._master.isEnabled() is False
```

- [ ] **Step 6: Run + Commit**

```
gui: ClusterCard widget + Syllables card split

New widget wraps a master ToggleRow over a child grid; children grey
out and become non-interactable when the master is off or when the
cluster's available_when predicate is False. Applied across all
toggleable clusters on the three section pages.

Syllables card splits into three independent ClusterCards (Intervals,
Labels, Prosody) so the user can grossly toggle prosody on/off
without flipping the timing/label columns one at a time.

Composite Psych card grays out and shows 'Requires LIWC backend'
subtitle when the user has Empath selected as the semantic backend.

Refs spec §9, §15.
```

---

## Task 7: Unify semantic categories

**Files:** Modify `gui/category_picker.py`, `gui/section_word_features.py`, `gui/section_doc_features.py`, `gui/planner/emit.py`. Update `tests/gui/test_category_picker.py`.

- [ ] **Step 1: Update `CategoryPicker`** to drop the `level` arg and read `semantic.categories_<backend>`:

```python
def __init__(self, backend: str, parent=None):
    super().__init__(parent)
    self._backend = backend
    self._prefs_key = f"categories_{backend}"
    ...
```

- [ ] **Step 2: Update `section_doc_features.py`** — replace the doc-level Configure-categories button with a read-only count label + link text "Edit on Word-level features page". The doc-level page still has the master toggle `doc_category_proportions` (output-keyed).

- [ ] **Step 3: Update `emit.py:_categories_for_level`**:

```python
def _categories_for_level(prefs: dict, level: str) -> str | None:
    backend = prefs.get("semantic", {}).get("backend", "auto")
    if backend == "auto":
        return None
    cats = prefs.get("semantic", {}).get(f"categories_{backend}", [])
    return ",".join(cats) if cats else None
```

Note: `level` is no longer used inside this function; keep it in the signature for caller compatibility.

- [ ] **Step 4: Update tests** — remove the level-based assertions in `test_emit.py:test_sem_categories_routes_by_level` (rewrite to assert both levels return the same unified value).

- [ ] **Step 5: Run + Commit**

```
gui: unify semantic categories — single source per backend

semantic.{word,doc}_categories_<backend> merges into a single
semantic.categories_<backend> (migrated in Task 2). The CategoryPicker
loses its level arg; the word-level page hosts the canonical editor
UI; the doc-level page shows a read-only count label with a hint to
edit on the word-level page.

emit.to_namespace's _categories_for_level reads the unified key.

Refs spec §7.
```

---

## Task 8: `session_units.csv` sidecar + `docs/master_plan.md`

**Files:** Add `utils/session_units.py`, `tests/utils/test_session_units.py`. Modify `main.py`, `docs/master_plan.md`.

- [ ] **Step 1: Implement `utils/session_units.py`**

```python
"""Per-run column → unit table writer.

Reads the Plan's stage_configs (pitch_units, pause_threshold, etc.) +
hard-coded constants (Volume in dB, durations in seconds) to produce a
machine-readable sidecar listing units for every emittable column.
"""
from gui.planner.stages import Plan

# Static column → unit map; Pitch_* fields override via plan.stage_configs.
_STATIC_UNITS = {
    ("features", "Word_Duration"):       "seconds",
    ("features", "Word_Gap"):            "seconds",
    ("features", "Onset"):               "seconds",
    ("features", "End"):                 "seconds",
    ("features", "Volume_Mean"):         "dB",
    ("features", "Volume_Min"):          "dB",
    ("features", "Volume_Max"):          "dB",
    ("features", "Volume_Std"):          "dB",
    ("features", "Volume_Delta"):        "dB",
    ("syllables", "Onset"):              "seconds",
    ("syllables", "End"):                "seconds",
    ("syllables", "Duration"):           "seconds",
    ("syllables", "Volume_Mean"):        "dB",
    ("syllables", "Volume_Delta"):       "dB",
    ("phonemes", "Onset"):               "seconds",
    ("phonemes", "End"):                 "seconds",
    ("phonemes", "Duration"):            "seconds",
    ("documents_summary", "doc_pitch_mean"):         "<pitch_units>",
    # ... etc
}


def build_session_units(plan: Plan, prefs: dict) -> list[dict]:
    pitch_units = plan.stage_configs.get("praat_contours", {}).get("pitch_units") \
                  or prefs.get("prosody", {}).get("pitch_units", "semitones")
    rows = []
    for (csv, col), unit in _STATIC_UNITS.items():
        if csv not in plan.output_csvs:
            continue
        if unit == "<pitch_units>":
            unit = pitch_units
        rows.append({"output_csv": csv, "column": col, "unit": unit})
    # Pitch_* columns (configurable units)
    for csv, col_prefix in [("features", "Pitch_"), ("syllables", "Pitch_")]:
        if csv not in plan.output_csvs:
            continue
        for col in ("Pitch_Mean", "Pitch_Min", "Pitch_Max", "Pitch_Std", "Pitch_Delta"):
            rows.append({"output_csv": csv, "column": col, "unit": pitch_units})
    return rows
```

- [ ] **Step 2: Wire into `main.py`** — at the end of each batch (`run_word_level_batch`, `run_doc_level_batch`), write `<output>/<run_ts>_session_units.csv` from the rows. Use the existing `run_ts` attribute on `args`.

- [ ] **Step 3: Tests** in `tests/utils/test_session_units.py`:

```python
def test_pitch_units_propagate_from_prefs():
    plan = ...  # synthetic plan with praat_contours in stages_to_run
    prefs = {"prosody": {"pitch_units": "Hz"}, ...}
    rows = build_session_units(plan, prefs)
    pitch_rows = [r for r in rows if r["column"].startswith("Pitch_")]
    assert all(r["unit"] == "Hz" for r in pitch_rows)


def test_no_rows_for_inactive_csv():
    plan = ...  # output_csvs={"features"} only
    rows = build_session_units(plan, prefs)
    assert not any(r["output_csv"] == "syllables" for r in rows)
```

- [ ] **Step 4: Update `docs/master_plan.md`**

Append a row dated 2026-05-19:

```markdown
| 2026-05-19 | **GUI validation unification + cluster UX** | The 2026-05-14 planner became the single source of truth for which stages run; this batch makes it the single source for everything that *consumes* that truth: preflight, the Run-tab cost panel, the cluster master toggles, and unit metadata. Two new declarative fields on `FeatureCluster` (`master_toggle`, `available_when`) enable per-cluster master toggles in the GUI (grayed-out children when off) and the LIWC-only psych composite gating. `gui/preflight.py` was rewritten to consume `plan.stages_to_run`; the cost panel hides when no LLM features are active; the Syllables card split into Intervals + Labels + Prosody sub-clusters. Semantic categories unified into `semantic.categories_<backend>` (word-level page is canonical editor; doc-level shows count + hint). Per-run `<output>/<run_ts>_session_units.csv` records column → unit. Prefs v6 → v7 additive migration. Spec/plan: `docs/superpowers/specs/2026-05-19-gui-unification-design.md`, `docs/superpowers/plans/2026-05-19-gui-unification.md`. |
```

- [ ] **Step 5: Run full suite + Commit**

```
gui: session_units sidecar + master_plan batch row

Per-run <output>/<run_ts>_session_units.csv lists every active output
column with its unit (seconds, dB, Hz/semitones based on prefs). Lets
downstream readers know units without parsing the manifest JSON or
trusting filename conventions.

master_plan.md records the 2026-05-19 GUI-unification batch as
complete.

Refs spec §10.
```

---

## Self-review

**Spec coverage:**
- §1-2 (problem, goal) → Tasks 1-8 collectively
- §4-6 (architecture, data model, algorithm) → Tasks 1, 3
- §7 (migration v6→v7) → Task 2
- §8 (adapter change) → Task 7
- §9 (ClusterCard) → Task 6
- §10 (session units) → Task 8
- §11 (preflight rewrite) → Task 4
- §12 (extensibility) → enforced by the architecture itself + schema-drift test in Task 3
- §13 (testing) → tests live with the task that introduces the surface
- §14 (rollout) → tasks ordered so each commit leaves the codebase working

**Placeholder scan:** no TBD/TODO. Each task has concrete code or precise enough pseudo-code that a subagent can finish it.

**Scope:** 8 tasks. No edits to `main.py` argparse, no extractor changes, no `gui/run_dispatch.py` changes. The new field on `FeatureCluster` is two lines; everything downstream rides on those two lines.

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-05-19-gui-unification.md`.** Recommended execution: **subagent-driven-development** (one task at a time, spec-compliance review then code-quality review between tasks, full suite `./env/python.exe -m pytest -q` must stay green between tasks). Same cadence as the planner batch.
