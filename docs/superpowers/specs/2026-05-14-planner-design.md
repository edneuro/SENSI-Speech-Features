# Planner Module Design

**Date:** 2026-05-14
**Upstream input:** [`docs/feature_dependencies.md`](../../feature_dependencies.md) — the bottom-up feature/stage inventory this design operationalizes.

---

## 1. Problem

`gui/run_dispatch.py` translates user intent (which output CSVs and columns are wanted) into the `argparse.Namespace` of `no_*` flags that `main.py` consumes. It does this through ~200 lines of hand-rolled cascading rules: "if feature X is on, force stage Y on; if output Z is on, force stage Y on too; unless toggle W is off; …".

Every feature batch shipped to date has introduced at least one bug in this layer:

- Sub-word features didn't force `no_parse=False` → `syllabify_pairs` crashed.
- Syllable-prosody features didn't force `no_prosody=False` → contours weren't computed, columns silent-NaN'd.
- Sidecar outputs were gated on `word_level_csv` → sidecars-only runs did nothing.
- `_word_columns_to_drop` ran on the in-memory df → stripped `Onset/End/doc_id/sentence_id/token_id` before the sidecar block could read them, silent empty output.

The root cause is the same across all of these: **the feature → stage dependency graph is not declared anywhere**. It's encoded implicitly across imperative rules that get patched after each bug surfaces. Every new feature is a chance to forget one.

## 2. Goal

Make the feature → stage dependency graph **explicit data** in one file, and route the GUI through a thin planner that walks the graph. Registering a new feature becomes a single edit (one new entry in the graph), with zero touches to the GUI dispatcher, `main.py`, or any existing extractor.

## 3. Non-goals (v1)

- **No `main.py` restructuring.** The planner emits the same `argparse.Namespace` `main.py:run_word_level_batch / run_doc_level_batch` consume today. Migration is bounded.
- **No GUI redesign.** Toggle widgets stay where they are; only the dispatcher's gating logic gets replaced.
- **No new feature levels** (e.g. sentence-level CSV). The architecture must make that addition seamless when it lands, but the rung itself is not built in v1.
- **No CLI flag changes.** `main.py`'s argparse surface is unchanged.

## 4. Architecture

A new `gui/planner/` package sits between the GUI and `main.py`:

```
gui/planner/
├── stages.py       # Stage, FeatureCluster dataclasses + StageContext (~80 LoC)
├── registry.py     # STAGES + CLUSTERS lists (the dependency graph) (~250 LoC)
├── plan.py         # Plan dataclass + plan_from_prefs() (~150 LoC)
└── emit.py         # Plan → argparse.Namespace adapter (~120 LoC)
```

Files that are **not touched**:

- All of `extractors/*.py` — the actual feature math.
- All of `main.py` — the batch runner consuming `no_*` flags.
- `gui/section_*.py` — toggle widgets.

Files that change:

- `gui/run_dispatch.py` — the 200-line `build_args_word_level / build_args_doc_level` functions get replaced by a ~15-line call into the planner.
- `tests/gui/test_run_dispatch.py` — repointed at `plan_from_prefs` (same prefs in → same `no_*` out, just derived differently).

## 5. Data model

Two dataclasses, both `@dataclass(frozen=True)`:

```python
@dataclass(frozen=True)
class Stage:
    """One unit of pipeline execution."""
    name: str                       # e.g. "spacy_parse", "mfa_align", "word_prosody"
    consumes: tuple[str, ...]       # Artifact names read ("audio", "word_df", "contours", ...)
    produces: tuple[str, ...]       # Artifact names produced
    run: Callable                   # The extractor function reference

@dataclass(frozen=True)
class FeatureCluster:
    """A user-facing grouping of columns sharing a stage and a toggle group."""
    name: str                       # Human-readable; for logs/errors
    toggle_keys: tuple[str, ...]    # Prefs keys; cluster active iff ANY is True
    output_csv: str                 # "features" | "syllables" | "phonemes" | "documents_summary"
    requires_stages: tuple[str, ...]  # Stage names the cluster depends on
    emits_columns: tuple[str, ...]  # Column names this cluster contributes to output_csv
```

**Stage** = "what runs". **FeatureCluster** = "what users toggle". The separation is load-bearing: per-column toggles drive **column inclusion in the CSV**, never **whether a stage runs**. A stage runs iff any toggle in any cluster pointing at it is on AND that cluster's `output_csv` is among the requested outputs.

`stages.py` also defines a `StageContext` — a dict-like object stages read/write via `ctx["word_df"] = ...`. This makes the artifact flow between stages explicit (replacing the current `args._phones_df`, `args._contours` attribute-mutation pattern, though that pattern survives in v1 since `main.py` is unchanged).

## 6. The planning algorithm

`plan.py:plan_from_prefs(prefs: dict) -> Plan`:

```
1. Compute the active output-CSV set from prefs["output"]:
     "features"           if output.word_level_csv
     "syllables"          if output.syllables_csv
     "phonemes"           if output.phonemes_csv
     "documents_summary"  if output.document_summary
2. For each FeatureCluster:
     active iff (cluster.output_csv in active_csvs)
                AND (any toggle_key in prefs["features"] is True)
3. required_stages = union of cluster.requires_stages for active clusters
4. Walk Stage.consumes backward over required_stages:
     for each stage in required_stages,
       for each artifact in stage.consumes,
         add any Stage whose `produces` includes that artifact.
     Repeat until stable. (DAG closure.)
5. Topologically sort required_stages by consumes/produces.
6. Per-CSV column_drops set:
     column_drops[csv] = (all columns from all clusters with that csv)
                       − (columns from ACTIVE clusters with that csv)
7. Return Plan(stages_to_run, output_csvs, column_drops, stage_configs).
```

`stage_configs` is a small dict passed through from prefs — pitch_floor, pitch_ceiling, pitch_units, pause_threshold, llm_model, etc. Already exists in today's `Namespace`; planner just routes it.

The planner is **pure** — deterministic from prefs, no side effects. Fast to unit-test: feed in a prefs dict, assert on the returned `Plan`.

```python
@dataclass(frozen=True)
class Plan:
    stages_to_run: tuple[str, ...]                       # Ordered stage names
    output_csvs: frozenset[str]                          # CSVs to write
    column_drops: dict[str, frozenset[str]]              # CSV name → columns to omit
    stage_configs: dict[str, dict[str, Any]]             # Per-stage parameter dict
```

## 7. Adapter to existing main.py

`emit.py:to_namespace(plan: Plan, prefs: dict, level: str) -> argparse.Namespace` produces exactly the `Namespace` `main.py:run_word_level_batch` (or `run_doc_level_batch`) expects today:

```python
ns.no_parse      = "spacy_parse"     not in plan.stages_to_run
ns.no_align      = "mfa_align"       not in plan.stages_to_run
ns.no_prosody    = "word_prosody"    not in plan.stages_to_run
ns.no_freq       = "word_freq"       not in plan.stages_to_run
ns.no_edu_freq   = "ewfg_freq"       not in plan.stages_to_run
ns.no_surprisal  = "gpt2_surprisal"  not in plan.stages_to_run
ns.no_llm_cloze  = "llm_predict"     not in plan.stages_to_run
ns.no_sem_domain = "sem_domain"      not in plan.stages_to_run
ns.no_word_csv   = "features"        not in plan.output_csvs
ns.no_syllables  = "syllables"       not in plan.output_csvs
ns.no_phonemes   = "phonemes"        not in plan.output_csvs
# Doc-level analogs for the doc namespace
ns.no_doc_counts            = "doc_counts" not in plan.stages_to_run
ns.no_doc_mean_zipf         = "doc_mean_zipf" not in plan.stages_to_run
# ... (one line per existing no_* flag)
ns.feature_toggles = dict(prefs["features"])  # main.py's per-column drop reads this
ns.output_toggles  = dict(prefs["output"])
ns.edu_bands       = _enabled_ewfg_bands(prefs)
ns.sem_categories  = _resolve_categories(prefs, level)
ns.pitch_units     = prefs["prosody"]["pitch_units"]
ns.pitch_floor     = prefs["prosody"]["pitch_floor_hz"]
ns.pitch_ceiling   = prefs["prosody"]["pitch_ceiling_hz"]
ns.pause_threshold = prefs["prosody"]["pause_threshold_s"]
ns.llm_model       = prefs["predictability"]["llm_model"] or None
```

Two notes on correctness:

- **The `level` argument** ("word" or "doc") selects which subset of stage names map to which `no_*` flags. `mfa_align` running for the doc batch sets `no_align=False` on the doc namespace too.
- **`feature_toggles` is passed through verbatim.** `main.py`'s `_word_columns_to_drop` / `_syllable_columns_to_drop` / `_phoneme_columns_to_drop` / `_doc_columns_to_drop` already consume this dict to drive per-column drops. The planner's `column_drops` is informational for tests (it lets the test assert on what *should* be dropped) but doesn't need to be plumbed into `main.py` in v1.

## 8. GUI integration

`gui/run_dispatch.py:_go()` collapses to:

```python
def _go():
    import os
    if session_api_key:
        os.environ["OPENAI_API_KEY"] = session_api_key
    prefs = load_preferences()
    plan  = planner.plan_from_prefs(prefs)
    run_ts = _dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    try:
        word_outputs = {"features", "syllables", "phonemes"} & plan.output_csvs
        if word_outputs:
            ns = planner.emit.to_namespace(plan, prefs, level="word")
            ns.run_ts = run_ts
            ns.skip_no_match = False
            run_word_level_batch(ns, on_pair_done=progress_cb)
        if "documents_summary" in plan.output_csvs:
            ns = planner.emit.to_namespace(plan, prefs, level="doc")
            ns.run_ts = run_ts
            ns.skip_no_match = bool(word_outputs)
            run_doc_level_batch(ns, on_pair_done=progress_cb)
        done_cb("done")
    except Exception as exc:
        done_cb(f"error: {exc}")
    finally:
        if session_api_key:
            os.environ.pop("OPENAI_API_KEY", None)
```

The `build_args_word_level()` and `build_args_doc_level()` functions (lines 158-325 of the current dispatcher) get **deleted** once their behavior is provably equivalent to the planner output. The `_any_structural_on`, `_any_acoustic_on`, `_any_psych_on`, `_any_doc_counts_on`, `_any_word_prosody_on`, `_any_doc_prosody_on`, `_any_subword_on`, `_any_syllable_prosody_on` helpers (and the `_WORD_PROSODY_KEYS`, `_DOC_PROSODY_KEYS`, `_SUBWORD_SYLLABLE_KEYS`, `_SUBWORD_PHONEME_KEYS`, `_SUBWORD_PROSODY_KEYS` tuples) get deleted — the registry subsumes them.

## 9. Extensibility workflow

The user requirement: "adding new features or levels must be seamless and not introduce bugs." This design makes registering a new feature a **two-file edit** (registry + GUI widget). Concrete example — adding "doc-level speech rate" (words per second per sentence):

1. Write the extractor: `extractors/speech_rate.py:add_speech_rate(df, word_alignment) -> df`.
2. Add **one** `Stage` to `registry.py`:
   ```python
   Stage(name="speech_rate",
         consumes=("word_df", "word_alignment"),
         produces=("doc_summary.speech_rate",),
         run=speech_rate.add_speech_rate),
   ```
3. Add **one** `FeatureCluster` to `registry.py`:
   ```python
   FeatureCluster(name="doc speech rate",
                  toggle_keys=("feat_doc_speech_rate",),
                  output_csv="documents_summary",
                  requires_stages=("speech_rate",),
                  emits_columns=("doc_speech_rate",)),
   ```
4. Bump prefs schema (v7) with `"feat_doc_speech_rate": False`.
5. Add the toggle widget in `gui/section_doc_features.py`.

Files touched: `extractors/speech_rate.py` (new), `gui/planner/registry.py` (+12 lines), `utils/preferences.py` (+1 key), `gui/section_doc_features.py` (+1 widget). **No edits to `main.py`, `gui/run_dispatch.py`, or any existing test.** The planner picks the new cluster up automatically.

Adding a new level (e.g. sentence-level CSV) is similar but adds one new value to the `output_csv` literal and one new gate in `_go()` — bounded change, no cascade.

## 10. Testing strategy

Three test layers:

**10.1 Planner unit tests** (`tests/planner/test_plan.py`)

Feed in prefs dicts, assert on the returned `Plan`. Each historical bug becomes a regression case:

- Sub-word features on, all word toggles off, sidecars wanted → `Plan.stages_to_run` includes `spacy_parse`, `mfa_align`; `features` not in `output_csvs`.
- Only syllable-prosody on, no word-level prosody → `Plan.stages_to_run` includes `praat_contours` and `word_prosody` (because syllable prosody needs Contours, which the planner pulls in transitively).
- Doc category proportions wanted, `sem_indicators` off but `doc_categories_*` non-empty → cluster is inactive (no toggle on), but preflight warning fires; that part stays in preflight, not planner.
- All toggles off → empty stage list, empty output_csvs.

Fast, no I/O. Probably 20-30 tests total.

**10.2 Adapter unit tests** (`tests/planner/test_emit.py`)

Feed in a `Plan`, assert on the resulting `Namespace.no_*` flags. Locks the migration contract: changing planner internals can't break `main.py`'s expectations without a test going red.

**10.3 Existing integration + end-to-end tests** keep working unchanged

`tests/integration/test_pipeline_batch.py` still runs `run_word_level_batch` with hand-built `Namespace`. The fact that the production code path now goes planner→emit→Namespace is invisible to them. They serve as the back-stop that the adapter contract is correct.

The current `tests/gui/test_run_dispatch.py` tests (~25 tests targeting `build_args_word_level / build_args_doc_level`) get **repointed** at `plan_from_prefs` — same prefs in, same `no_*` flags out, just derived through the planner. This is a mechanical rewrite, not a redesign.

**10.4 Schema-drift guard**

One static test: every prefs key in `_FEATURES_DEFAULT_V6` (and every future version) is referenced by some `FeatureCluster.toggle_keys` OR is explicitly listed as "intentionally unrouted". This prevents the registry from silently drifting from the prefs schema as features are added.

## 11. Migration plan (rollout)

Single batch of commits on `master` (matches the project's existing cadence):

1. Add `gui/planner/` package skeleton: `Stage`, `FeatureCluster`, `Plan` dataclasses + empty registry + planner + emit stubs. Tests pass (nothing is wired yet).
2. Populate `STAGES` from §0 of `feature_dependencies.md`. Add stage-graph traversal unit tests.
3. Populate `CLUSTERS` from §1, §2, §4 of `feature_dependencies.md`. Add planner unit tests covering each cluster's gate.
4. Wire `emit.to_namespace`. Add adapter unit tests. Regression-test the recently-fixed bug paths.
5. Replace `gui/run_dispatch.py:build_args_word_level / build_args_doc_level` with planner calls. Existing tests continue to pass (since the adapter produces the same Namespace).
6. Delete the obsolete hand-rolled functions + helpers.
7. Add the schema-drift guard test.
8. Update `docs/master_plan.md` to record the batch as complete.

Estimated 5-8 implementation tasks at the cadence used for prosody and phoneme/syllable. The risk to existing behavior is bounded: every commit in 1-4 leaves the codebase in a working state with the old dispatcher still in charge. Only commit 5 flips the switch, and the entire existing test suite gates that flip.

## 12. Open design questions deferred to implementation

- **`Sem_Categories` summary column placement.** Today it's always emitted when sem_indicators runs. The cluster could either include it among `emits_columns` (always on if cluster active) or split it as its own cluster with its own toggle. Recommend: leave it always-on for v1 (preserve current behavior); add a toggle later if requested.
- **Whether `word_prosody`'s 10 columns split into `pitch_basics + delta` sub-stages.** Decision in brainstorm: keep as one stage; per-column toggles drive CSV-column drops only. The micro-cost of computing unused columns when the cluster is active is negligible since Parselmouth (the heavy stage) is already separate.
- **`stage_configs` dict shape.** Initial cut: a flat dict keyed by parameter name (`pitch_floor`, `pitch_units`, …) mirroring the existing `Namespace` fields. Refactor to nested `dict[stage_name, dict[param, value]]` if/when stages start needing isolated config namespaces.

---

## Self-review

**Placeholder scan:** No "TBD" / "TODO" / vague requirements. Every section is concrete.

**Internal consistency:** Sections 6 (planning algorithm), 7 (adapter), 8 (GUI integration) reference the same `Plan` shape and the same Stage names — verified.

**Scope:** Bounded. Doesn't touch `main.py` or extractors. Single batch of 5-8 implementation tasks. Test strategy in §10 keeps risk bounded.

**Ambiguity:** §10.1 says "preflight warnings stay in preflight, not planner" — making explicit that this design doesn't subsume `gui/preflight.py`. The planner consumes prefs to decide *what runs*; preflight consumes prefs to decide *what to warn about*. They're sibling consumers of the same data, not nested.

---

*End of design.*
