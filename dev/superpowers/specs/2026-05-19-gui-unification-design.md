# GUI Validation Unification + Cluster UX Design

**Date:** 2026-05-19
**Upstream input:** [`docs/superpowers/specs/2026-05-14-planner-design.md`](2026-05-14-planner-design.md) — the planner this design extends.

---

## 1. Problem

The 2026-05-14 planner batch made `gui/planner/` the canonical answer to "what stages run given which prefs toggles". But the migration was bounded to the dispatcher (`gui/run_dispatch.py`). Other GUI surfaces still hold their own copies of the same dependency rules, which causes a recurring class of bugs:

1. **Stale warnings.** `gui/preflight.py:180-185` tells the user *"Sub-word features require MFA alignment — enable Onset/End in the Acoustic card on the Word-level features page"* when they have sub-word sidecars on and word-level off. The planner already pulls `mfa_align` from the sub-word cluster's `requires_stages` — the warning is wrong by design. Same pattern at `preflight.py:94-108` for prosody.

2. **Cost panel renders when no LLM-using feature is on.** `gui/section_run.py:115-116` shows the OpenAI cost estimate whenever the configured model name starts with `gpt-`, even if the user has turned off every predictability toggle.

3. **No cluster-level master toggle.** Toggling word-level prosody on/off requires flipping 10 individual `feat_prosody_*` checkboxes. Same for the 9 syllable keys and the 15 doc-prosody keys. The user has been asking for a single master with grayed-out children.

4. **Syllables card mixes concerns.** `gui/section_subword_features.py:28-46` bundles 9 keys spanning timing, label, and prosody into one grid. The user can't master-toggle prosody independently of intervals.

5. **Units (Hz/semitones, dB, seconds) aren't exported with data.** A downstream CSV reader has no way to know which units each numeric column carries without trawling the prefs JSON in `<output>/logs/`.

6. **Semantic categories are split into `word_categories_<backend>` and `doc_categories_<backend>`** — the picker has to be configured twice for the same conceptual list. Almost no real workflow wants different category lists at the two levels.

7. **Naming inconsistency** — word-level cluster is called "Semantic indicators"; doc-level is called "Category proportions". Conceptually the same feature group with different output shapes.

8. **`feat_psych_*` composite is togglable on Empath** even though it only computes when LIWC is the active backend.

These are not eight independent bugs. They are **eight symptoms of the same architectural gap**: the planner became authoritative for the dispatcher but not for the GUI's validation, layout, or export surfaces.

## 2. Goal

Make the planner the single source of truth for every GUI surface that has an opinion about feature dependencies, and add the small primitives (cluster master toggle, availability gate, unified categories, units sidecar) the GUI needs to express the planner's truth cleanly. Touching a feature toggle should be safe in isolation — the planner already guarantees that for the pipeline, but the GUI doesn't yet reflect it.

## 3. Non-goals (v1)

- **No `main.py` restructuring.** Adapter shape (Plan → Namespace) unchanged.
- **No new extractors.** All math stays where it is.
- **No CLI surface changes.** `main.py` argparse stays.
- **No category-picker visual redesign.** Picker becomes single-source; the search/scroll UI is unchanged.
- **No new feature levels** (sentence-level, etc.). Existing four levels (word, sub-word, sentence-placeholder, doc) only.

## 4. Architecture

Five additions, sized to fit on top of the existing planner:

```
gui/planner/
├── stages.py              # +2 fields on FeatureCluster (master_toggle, available_when)
├── plan.py                # _active_clusters consults both new fields
├── registry.py            # Each cluster declares its master + LIWC gate
└── emit.py                # _categories_for_level reads the unified key

gui/
├── preflight.py           # REWRITE — planner consumer, env/asset checks only
├── section_run.py         # Cost panel gated on llm_predict in plan
├── section_word_features.py / section_doc_features.py / section_subword_features.py
│                           # Each cluster wrapped in ClusterCard
├── widgets.py             # NEW: ClusterCard
└── category_picker.py     # Reads unified semantic.categories_<backend>

utils/
├── preferences.py         # v6 → v7 migration (additive)
└── session_units.py       # NEW: per-run units sidecar

main.py                    # Writes <output>/<run_ts>_session_units.csv per batch
```

**Files NOT touched:** all of `extractors/*.py`, `gui/run_dispatch.py`'s top-level structure, `gui/planner/__init__.py` API surface (still re-exports the same four symbols), `utils/run_logger.py` (manifest JSON unchanged), `utils/textgrid_parser.py`, `utils/transcript_prep.py`.

## 5. Data-model additions

```python
@dataclass(frozen=True)
class FeatureCluster:
    name: str
    toggle_keys: tuple[str, ...]
    output_csv: str
    requires_stages: tuple[str, ...]
    emits_columns: tuple[str, ...]
    # NEW:
    master_toggle: str | None = None
    """Prefs key (under features or output, resolved by OUTPUT_KEYED_TOGGLES).
    When non-None, the cluster is inactive unless this is True. Children
    are grayed out in the GUI when the master is off."""
    available_when: tuple[str, str] | None = None
    """(dotted_pref_path, expected_value). When set, the cluster is inactive
    if prefs at that path != expected_value. Used for the LIWC-only psych
    composite — keeps the predicate declarative (no callables) so the
    dataclass remains hashable + serialisable."""
```

`available_when` uses a tuple of `(pref_path, value)` rather than a callable for two reasons: (a) the dataclass stays hashable/frozen-friendly; (b) the GUI can read the same predicate to derive the grayed-out state without re-implementing the check.

The current 21 clusters get masters where the user has been asking for them; legacy single-key clusters (like SUBTLEX, where the only toggle is already its own master) leave `master_toggle=None`.

## 6. Activation algorithm change

`plan._active_clusters(prefs, active_csvs)` adds two early-out checks before the existing toggle-any check:

```python
# 1. NEW: available_when gate
if c.available_when is not None:
    pref_path, expected = c.available_when
    if _get_dotted(prefs, pref_path) != expected:
        continue

# 2. NEW: master toggle
if c.master_toggle is not None and not _toggle_value(prefs, c.master_toggle):
    continue

# 3. Existing: output_csv must be in active set
if c.output_csv not in active_csvs:
    continue

# 4. Existing: any child toggle on
if any(_toggle_value(prefs, k) for k in c.toggle_keys):
    actives.append(c)
```

The check ordering matters: `available_when` first (hard environment gate), then `master_toggle` (user gate), then output + child checks (existing logic). Tests in `tests/planner/test_plan.py` lock the contract.

## 7. Prefs migration v6 → v7

Three additive changes:

- **New keys.** For each cluster that's getting a master, add `master_<name>: True` to `_FEATURES_DEFAULT_V7` or `_OUTPUT_DEFAULT_V7` (the section is determined by `OUTPUT_KEYED_TOGGLES`). Defaulting to True means existing user prefs continue to behave identically until the user touches the master.
- **Unified categories.** `semantic.word_categories_<backend>` ∪ `semantic.doc_categories_<backend>` → `semantic.categories_<backend>`. Union is lossless: anyone who curated different lists at each level still gets every pick they made.
- **Delete the old category keys** after migration. Single touchpoint: `_migrate_to_v7` in `utils/preferences.py`.

Master keys to add (eight in total):

| Master key | Cluster | Section |
|---|---|---|
| `master_word_prosody` | word prosody | features |
| `master_psych_composite` | doc LIWC composite | features |
| `master_syllable_intervals` | syllabification intervals (new sub-cluster) | features |
| `master_syllable_labels` | syllable labels (new sub-cluster) | features |
| `master_syllable_prosody` | syllable prosody | features |
| `master_phoneme_intervals` | phoneme intervals | features |
| `master_doc_prosody` | doc prosody | output |
| `master_doc_articulation_rate` | doc articulation rate | output |

EWFG already has a partial master via the band-toggle set; SUBTLEX and surprisal are single-toggle clusters that don't need a separate master. Structural + lex-ling are passthrough clusters (no extractor); the master toggle would be cosmetic — leave them out for v1.

## 8. Adapter changes (emit.py)

One change to `_categories_for_level`:

```python
# was: key = f"{'doc' if level == 'doc' else 'word'}_categories_{backend}"
key = f"categories_{backend}"
```

The `level` argument stays in the function signature because callers pass `"word"` / `"doc"` for output-CSV selection elsewhere in `to_namespace`. The semantic-categories key is now level-independent.

## 9. GUI: ClusterCard widget

`gui/widgets.py` gets a new widget:

```python
class ClusterCard(Card):
    """A Card with a master ToggleRow on top + a child grid below.

    When the master is OFF or `available` is False, child rows are
    setEnabled(False) and visually styled as inactive (CSS class on
    self). Toggling the master saves the master key to prefs and
    propagates the visual state.
    """
    def __init__(self, title: str, subtitle: str,
                 master_key: str, master_label: str,
                 master_section: str = "features",
                 available: bool = True,
                 ...): ...
```

Section pages compose `ClusterCard`s and add child `ToggleRow`s to each. The master state and `available` flag are read from prefs each render, so changing semantic backend (toggling LIWC vs Empath on the word page) immediately propagates to the psych cluster's grayed-out state.

The widget does not bypass the registry — the registry is the source of truth; the widget is the visual reflection.

## 10. Session units sidecar

A new file `utils/session_units.py`:

```python
def build_session_units(plan: Plan, prefs: dict) -> list[dict]:
    """Return rows of (output_csv, column, unit) for every unit-bearing
    column in active output CSVs. Reads plan.stage_configs for the
    configured units (pitch_units, pause_threshold).

    The returned list is what main.py's per-batch finaliser writes to
    <output>/<run_ts>_session_units.csv.
    """
```

Each entry is `{"output_csv": "features", "column": "Pitch_Mean", "unit": "semitones"}` (or `"Hz"` based on prefs). Constant columns (e.g. `Volume_Mean` is always dB) are hard-coded in `session_units.py`; configurable ones (`Pitch_*`) read from the plan.

`main.py` calls `build_session_units` at the end of `run_word_level_batch` and `run_doc_level_batch` and writes the CSV next to the manifest.

## 11. Preflight rewrite

Two warnings DELETED:
- L94-108: prosody-needs-acoustic (planner handles via `word_prosody.requires_stages`)
- L180-185: sub-word-needs-acoustic (planner handles via sub-word clusters' `requires_stages`)

Six warnings KEPT but reframed:
- EWFG parquet missing → gate on `"ewfg_freq" in plan.stages_to_run`
- OpenAI API key missing → gate on `"llm_predict" in plan.stages_to_run` (combined with the existing "is the model GPT?" check)
- LIWC `.dic` missing → gate on `sem_domain` in plan AND backend == "liwc"
- MFA binary missing → gate on `"mfa_align" in plan.stages_to_run`
- parselmouth import failed → gate on any prosody stage in plan
- pyphen import failed (artic rate) → gate on `"doc_articulation_rate" in plan.stages_to_run`
- pyphen import failed (syllable label) → gate on `"syllabify_pairs" in plan.stages_to_run` AND `feat_syllable_label`

One warning REWRITTEN:
- L144-163: categories-vs-master-toggle becomes "categories disabled" UX (Task 7) — the picker greys out when the master is off, removing the need for a warning.

One info-level warning ADDED:
- Empty plan ("No outputs selected — pick at least one CSV in the Outputs section").

Invariant test: after the rewrite, no preflight warning message contains the substring "enable" — preflight never instructs the user to flip another toggle. Catches future regressions.

## 12. Extensibility

Adding a new feature post-batch:

1. Write the extractor.
2. Add a `Stage` + a `FeatureCluster` (with optional `master_toggle` + `available_when`) to `gui/planner/registry.py`.
3. Add the toggle keys to `_FEATURES_DEFAULT_V7` / `_OUTPUT_DEFAULT_V7`.
4. Add a `ClusterCard` to the appropriate section page; ClusterCard composes from the same fields you already declared on the cluster.

No edits to `gui/preflight.py`, `gui/section_run.py`, `gui/category_picker.py`, `main.py`, `gui/run_dispatch.py`, or any test outside the new feature's tests. The GUI surfaces all derive from the registry.

## 13. Testing strategy

**13.1 Planner unit tests** — extended (Tasks 1, 3): `master_toggle` and `available_when` activation gates have their own tests; schema-drift gains a new invariant that every `master_*` key in clusters exists in v7 prefs.

**13.2 Migration tests** — new (Task 2): round-trip v6 prefs through `_migrate_to_v7`, assert category union is lossless, assert master defaults are True.

**13.3 Preflight tests** — repointed + new invariant (Task 4): stale assertions flip from "warning present" to "warning absent"; one invariant test asserts no preflight message contains "enable".

**13.4 Section page tests** — new (Task 6): ClusterCard correctly disables children when the master is off; available_when grays out the LIWC cluster on Empath backend.

**13.5 Category-picker tests** — extended (Task 7): word-level pick and doc-level read return the same selection.

**13.6 Session-units tests** — new (Task 8): plan + prefs produce the expected rows; pitch_units pref propagates correctly.

**13.7 Existing integration tests** continue to pass — `run_word_level_batch` / `run_doc_level_batch` see the same Namespace shape; the planner produces equivalent plans for v7 prefs as for v6 prefs after migration.

## 14. Migration plan (rollout)

Eight tasks, ordered so every commit leaves the codebase in a working state:

1. Add `master_toggle` + `available_when` fields to `FeatureCluster`; update `_active_clusters`. Tests.
2. Prefs schema v7 with migration. Tests.
3. Wire `master_toggle` + `available_when` onto every cluster in `registry.py`. Schema-drift invariant.
4. Rewrite `gui/preflight.py` as a planner consumer.
5. Cost panel reads planner state.
6. `ClusterCard` widget + Syllables split + cluster wraps across section pages.
7. Unify semantic categories.
8. `session_units.csv` sidecar + `docs/master_plan.md` row.

The risky steps (2 and 6) come early enough that they can be reverted without losing later work, and late enough that the planner machinery is already extended to consume their output.

## 15. Open design questions resolved

- **Categories migration policy:** union (lossless). Anyone who curated different lists at each level still has every pick they made; they can prune from the unified picker if they want.
- **Available-when value:** a `(pref_path, expected_value)` 2-tuple, not a callable. Keeps `FeatureCluster` frozen + hashable + serialisable.
- **Empty-plan UX:** one info-level preflight warning, not an error.
- **Cluster naming:** both semantic clusters share `"Semantic categories — "` prefix with level-specific suffix. Prefs keys unchanged; only display strings change.

---

## Self-review

**Placeholder scan:** No TBD / TODO / vague requirements.

**Internal consistency:** Sections 5 (data model), 6 (algorithm), 9 (widget), 11 (preflight rewrite) all reference the same two new fields and the same activation order. Verified.

**Scope:** Bounded. Eight tasks. No edits to `main.py`'s argparse or to any extractor. The planner package gains two fields and a one-line condition.

**Risk:** Lowest-risk first (data-model + migration); highest-risk last (Syllables card split, ClusterCard wraps). Schema-drift invariant catches registry-vs-prefs drift.

---

*End of design.*
