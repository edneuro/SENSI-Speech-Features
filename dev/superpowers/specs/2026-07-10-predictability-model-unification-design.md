# Predictability model unification — design

**Date:** 2026-07-10
**Status:** approved-for-planning
**Batch:** Project A of a two-part effort. Project B (per-model configurable
context size) is a separate later cycle that rides on this roster.

---

## 1. Motivation & scope

Today the language-model predictability features are split across two
extractors for historical reasons only:

- `extractors/surprisal.py` — GPT-2, emits the single `Surprisal` column.
- `extractors/llm_cloze.py` — local Llama models, emits `LLM_Surprisal`,
  `LLM_Cloze`, `LLM_TopMatch`.

GPT-2 is itself an LLM; the split is not principled. It also means GPT-2
cannot produce cloze/top-match, the naming is asymmetric (`Surprisal` vs
`LLM_*`), and there is no clean place to attach a per-model context-size
control (Project B).

This project **unifies both into a single multi-model predictability
feature**: the user selects one or more models from a roster and one or more
metrics, and each selected model emits its own namespaced columns from one
forward pass. This preserves and generalizes the existing GPT-2-vs-Llama
comparison capability.

**In scope:** the roster, the unified extractor, prefs v18 + migration, the
planner/GUI/doc/sherlock wiring, the docs, and the tests.

**Out of scope (explicit follow-ons):**
- **Project B** — per-model configurable context size (max context, clamped
  to each model's `native_ctx`). This spec carries `native_ctx` in the roster
  as data so B is a small addition, but adds no context-size control.
- A YAML feature-config + runnable entry point that replaces
  `preferences.json` and promotes the sherlock machinery to a first-class
  global path (retiring some `sherlock/` files). Its own future project;
  Project A stays compatible with it (config is already serialized prefs).

## 2. Locked decisions

1. **Multi-model, namespaced columns.** Columns are the cross product of
   selected models × selected metrics.
2. **Roster** = `gpt2` + the three Llamas; all local Hugging Face causal LMs.
3. **Metric selection is global** — three checkboxes (surprisal, cloze,
   top-match) apply to every selected model.
4. **Clean break** on column names: `Surprisal` / `LLM_Surprisal` /
   `LLM_Cloze` / `LLM_TopMatch` are removed. New names are `<slug>_Surprisal`
   / `<slug>_Cloze` / `<slug>_TopMatch`. Pre-release, so no external consumers.
5. **CLI predictability flags are retired**, not replaced. No `--models` /
   `--metrics`. The real interfaces are the GUI (prefs) and the sherlock YAML.
6. **Single doc-level aggregate toggle** ("predictability means"), emitting a
   `mean_<slug>_<metric>` for each word-level predictability column present.
7. **Docs:** merge the two surprisal docs into one model-agnostic
   `pred_surprisal.md`; keep `pred_cloze.md`, `pred_topmatch.md`, and
   `llm_credentials.md` independent.

## 3. Model roster — `extractors/predictability_models.py`

Single source of truth for the GUI list, column prefixes, gating, and (for B)
the context ceiling. A frozen dataclass and a module-level list:

```
ModelSpec(slug, label, hf_id, gated, native_ctx, note)
```

| slug | hf_id | gated | native_ctx |
|---|---|---|---|
| `gpt2` | `gpt2` | False | 1024 |
| `llama32_3b` | `meta-llama/Llama-3.2-3B` | True | 131072 |
| `llama3_8b` | `meta-llama/Meta-Llama-3-8B` | True | 8192 |
| `llama3_70b` | `meta-llama/Meta-Llama-3-70B` | True | 8192 |

- `slug` is column- and filename-safe and is the sole identifier used in
  prefs, columns, and doc aggregates.
- Helper API: `all_specs()`, `spec_for(slug)`, `is_gated(slug)`,
  `slug_for_hf_id(hf_id)` (used by the v18 migration to map a stored
  `llm_model`). This replaces `LLM_OPTIONS` and `is_gated_local_model`.
- `native_ctx` is carried now (data only); no control reads it yet.

## 4. Extractor — `extractors/predictability.py`

Folds `surprisal.py` + `llm_cloze.py` into one module. `surprisal.py` and the
old `llm_cloze.py` are deleted; their still-needed pure helpers
(`map_tokens_to_subwords`, `_scores_from_subwords`, `_chunk_slices`, the
`_HuggingFaceBackend`, the `(model, text-digest)` score cache) move here
unchanged.

**GPT-2 becomes an ordinary HF causal LM.** It loads via
`AutoModelForCausalLM.from_pretrained("gpt2")` — the same concrete class
(`GPT2LMHeadModel`) the old code used, so logits and therefore surprisal are
identical by construction (guarded by an equivalence test, §10). This removes
GPT-2's bespoke code path entirely.

**Public API:**

```
add_predictability(df, text, *, models: list[str], metrics: list[str],
                   backends: dict | None = None) -> pd.DataFrame
```

- `models` = roster slugs; `metrics` ⊆ `{"surprisal", "cloze", "topmatch"}`.
- For each slug: get/build its backend (cached), score the text once
  (`backend.score(text, tokens)` → per-token `{p, surprisal, top_match}`,
  already implemented), then write **only** the requested metric columns,
  namespaced `<slug>_Surprisal` / `<slug>_Cloze` / `<slug>_TopMatch`.
- Predictability columns are **produced opt-in** (like the embedding `.mat`
  sidecars), never produced-then-dropped — so there is no per-column drop map
  to maintain, unlike the spaCy structural columns.
- Preserved value semantics from the recent correctness sweep: document-first
  word → NaN (no context), PUNCT → NaN, each subword charged to exactly one
  token, cross-pass memoization so the word and doc passes score each
  (model, document) once.
- `backends` injection point kept for tests (fake backends).

Gated-model load failures raise the existing actionable error; a `gpt-*` slug
cannot occur (removed from the roster in the prior OpenAI-removal work).

## 5. Prefs schema — v18

The `predictability` **settings** section becomes:

```json
"predictability": {
  "models": [],
  "metrics": { "surprisal": true, "cloze": false, "topmatch": false }
}
```

Removed keys:
- `features`: `pred_surprisal_gpt2`, `pred_surprisal_llm`, `pred_cloze`,
  `pred_topmatch`.
- `predictability.llm_model`.

`master_predictability` stays (card master toggle). Default per the v15 opt-in
policy: `models` empty ⇒ predictability off on a fresh install.

**Migration v17 → v18** (finalize step, applies to any pre-v18 source):
1. If `pred_surprisal_gpt2` was on → append `gpt2` to `models`, set
   `metrics.surprisal = True`.
2. If any of `pred_surprisal_llm` / `pred_cloze` / `pred_topmatch` was on →
   append `slug_for_hf_id(stored llm_model)` (default `llama32_3b` if unset or
   unknown, e.g. a coerced-empty gpt-* value) to `models`, and set the
   matching metric(s): `surprisal_llm → surprisal`, `cloze → cloze`,
   `topmatch → topmatch`.
3. Deduplicate `models` (preserve order); drop the four old feature keys and
   `llm_model`.

Not purely additive (removes keys), but there is precedent (v14 struct-id
rename); and post-v15 most installs have these off, so migration usually
yields an empty `models` list.

## 6. Planner / emit / main.py / CLI / sherlock

**Planner (`gui/planner/`).** Replace the two stages (`gpt2_surprisal`,
`llm_predict`) and the two clusters (`GPT-2 surprisal`, `LLM predictability`)
with one `predictability` stage + one cluster. The cluster is **dynamic**: its
activation and its emitted columns come from `predictability.models` ×
`predictability.metrics`, not from static boolean feature keys. This is the one
real architectural change to the otherwise-declarative planner:

- Activation: active iff `models` non-empty **and** ≥1 metric on **and**
  `master_predictability` on.
- `emits_columns`: computed as `models × metrics` → `<slug>_<Metric>` (used by
  the session-units CSV and any drift/coverage checks).
- The schema-drift guard test is updated: predictability no longer contributes
  `pred_*` boolean toggle keys (they moved to the settings section).

**emit.py.** Set `ns.predictability = {"models": [...], "metrics": {...}}`.
Remove `ns.no_surprisal`, `ns.no_llm_cloze`, `ns.llm_model`.

**main.py.** Replace the two extractor blocks with one:
```
if <predictability active>:
    df = add_predictability(df, text, models=..., metrics=...)
```
`run()` reads the config from `getattr(args, "predictability", None)`, falling
back to `load_preferences()["predictability"]` when absent — the same
prefs-fallback `run()` already uses for `sem_categories`, so a bare CLI batch
still respects the GUI-configured selection. Remove the retired argparse flags
(`--no-surprisal`, `--no-llm-cloze`, `--llm-model`) and the `_resolve`/
`select_model` CLI plumbing tied to them.

**Doc batch.** `build_doc_summary` gains predictability-mean handling that
loops the present `<slug>_<Metric>` columns and emits `mean_<slug>_<metric>`,
gated by the single `doc_agg_predictability` toggle (replacing
`doc_agg_surprisal` + `doc_agg_llm`).

**sherlock.** `export_config.py` serializes the prefs dicts generically, so the
new `predictability` settings and the removal of the `pred_*` feature keys flow
through automatically; `run_extract.py` builds its namespace via the same emit
path, so it keeps working. Deliverables: regenerate the committed
`sherlock/config.yaml`; drop the now-dead `run.openai_api_key_file` and
`settings.predictability.llm_model` lines.

## 7. GUI — Predictability card

`gui/section_word_features.py` predictability card is redesigned:

- **Models**: a checkable list, one row per roster entry — `label`, `note`,
  `native_ctx`, and a "free, no token" / "gated" badge. Toggling a row writes
  `predictability.models`. (Project B adds a per-row context spinner; the row
  layout reserves space.)
- **Metrics**: three checkboxes (Surprisal, Cloze, Top-match) → 
  `predictability.metrics`.
- **HF token box**: shown when any *selected* model is gated and no token is
  detected (generalizes today's single-model logic; reuses `has_hf_token()` /
  the session-token box).
- Info-icons via the existing feature-docs system, mapped to the new metric
  docs and a card-level roster doc.
- Cost panel: stays hidden (all local/free), per the prior OpenAI-removal work.

**Preflight** (`gui/preflight.py`) warnings:
- a gated model is selected but no HF token is available;
- models selected but no metric on (nothing will be produced);
- a metric on but no model selected (nothing will be produced).

## 8. Doc-level aggregates

One toggle `doc_agg_predictability`. When on, the document summary emits
`mean_<slug>_<metric>` for every word-level `<slug>_<Metric>` column present in
that run (mean over non-NaN values, matching the existing `_mean` semantics).
Replaces `doc_agg_surprisal` and `doc_agg_llm`. Prefs migration maps either old
toggle being on → `doc_agg_predictability` on.

## 9. Docs

- **Merge** `pred_surprisal_gpt2.md` + `pred_surprisal_llm.md` →
  `pred_surprisal.md` (model-agnostic: the metric is the same regardless of
  model; keep every citation from both). Update `_SHARED_DOC_MAP` /
  `feature_docs.py` so the metric toggle and each model row route here.
- **Keep** `pred_cloze.md`, `pred_topmatch.md`, `llm_credentials.md` as
  independent docs; de-model-specialize their wording and update the roster
  (GPT-2 = no token).
- Update README (predictability section: model roster + namespaced columns)
  and add a master_plan §3 completed-batch row noting the **breaking** column
  rename and prefs v18.

## 10. Testing (TDD for the extractor)

- `tests/extractors/test_predictability.py` (new, pure parts): column
  namespacing per slug; metric subsetting (only requested columns appear);
  multi-model loop produces both models' columns; first-word/PUNCT NaN
  preserved; empty models / empty metrics → no columns; unknown slug rejected.
- **GPT-2 equivalence**: `AutoModelForCausalLM("gpt2")` surprisal ≡ the prior
  `GPT2LMHeadModel` path on a fixed short input (guards against silent value
  drift).
- **Migration** v17→v18: gpt2-only, llm-only (each metric), both, and the
  empty/default cases; `doc_agg_*` mapping; drop of old keys.
- **Planner**: drift guard updated; a plan test asserting the dynamic
  `predictability` stage activates on `models`+`metrics` and emits the right
  `<slug>_<Metric>` set; emit test for `ns.predictability`.
- **GUI**: the new model list + metric toggles persist to prefs; HF-token box
  visibility follows selected-model gating; preflight warnings fire.
- **Integration**: a word-level batch with two models × two metrics writes the
  four `<slug>_<Metric>` columns; a doc batch writes the matching
  `mean_<slug>_<metric>` columns.
- Full suite green (currently 672 passed, 2 skipped).

## 11. Breaking changes & migration

- **Output CSV columns renamed** (clean break): `Surprisal` / `LLM_Surprisal`
  / `LLM_Cloze` / `LLM_TopMatch` → `<slug>_Surprisal` / `<slug>_Cloze` /
  `<slug>_TopMatch`. Doc-summary means likewise → `mean_<slug>_<metric>`.
- **Prefs v18** migration converts old selections; no user action needed.
- **CLI flags removed**: `--no-surprisal`, `--no-llm-cloze`, `--llm-model`.
- Documented prominently in the master_plan row and the merged docs.

## 12. Open risks / notes

- **Dynamic planner cluster** is the main deviation from the declarative
  registry. The plan must settle exactly how a cluster activates on
  non-boolean config and reports dynamic `emits_columns` without breaking the
  drift guard or the session-units builder. Fallback if it proves invasive:
  keep the cluster static-but-always-considered and compute active columns in
  `emit.py`/`main.py` from `models`×`metrics` (the planner then only gates the
  stage, not the columns).
- **Runtime cost** scales with the number of selected models (each is a full
  forward pass). This is inherent and surfaced in the GUI (the model list makes
  the count explicit); no cost panel since all are local/free.
- **`native_ctx` values** are the published `max_position_embeddings`; verify
  against each model's config at implementation time (esp. Llama-3.2-3B's
  128K) so Project B clamps correctly.
