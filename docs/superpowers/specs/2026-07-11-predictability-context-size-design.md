# Per-model predictability context size — design

**Date:** 2026-07-11
**Status:** approved-for-planning
**Batch:** Project B of the predictability effort. Rides on Project A's roster
(merged): each `ModelSpec` already carries `native_ctx`.

---

## 1. Motivation & scope

Each predictability model has a native context window (`native_ctx` in the
roster: GPT-2 1024, Llama-3-8B/70B 8192, Llama-3.2-3B 131072). The backend
currently uses the full native window and only chunks (`_chunk_slices`) when a
stimulus exceeds it. This project lets the user **cap the context per model**,
clamped to that model's native ceiling. A smaller cap makes chunking kick in
sooner — each word conditions on less preceding text: faster and lower memory
on long texts, and a way to fix the preceding-context budget for comparability
— at the cost of fidelity.

**In scope:** the per-model cap (prefs v19 + additive migration), threading it
into the backend's effective context limit, the GUI spinner per model row,
validation/clamping, and tests.

**Out of scope:** changing chunking strategy (stride stays `window // 2`); the
acoustic/phoneme models (no LM context); BERT's fixed 510 window.

## 2. Locked decisions

1. **Per-model-id caps.** Stored per roster slug.
2. **Default = native max** (no cap). Nothing changes until the user lowers a
   value; current output is unaffected.
3. **Clamp to `[FLOOR, native_ctx]`** with `FLOOR = 128` tokens. A value above
   the model's native ceiling is clamped down; the GUI won't allow > native.
4. **Unit** = max context tokens (subwords) fed to the model per forward pass.

## 3. Prefs schema — v19 (additive)

Add a `context` map to the `predictability` section:

```json
"predictability": {
  "models": [],
  "metrics": {"surprisal": false, "cloze": false, "topmatch": false},
  "context": {}
}
```

- `context` maps `slug → int` (max context tokens). A slug ABSENT from the map
  (the default) means "use the model's native ceiling". So `{}` = every model
  at native max = today's behavior.
- **Migration v18 → v19** (additive): add `context: {}` if missing. No value
  transforms. `SCHEMA_VERSION = 19`; `DEFAULT_PREFS_V19` adds the empty map;
  v19 joins the deep-merge version tuple.

## 4. Effective-limit resolution (single source of truth)

A pure helper on the roster/extractor computes the effective context for a
slug from the stored cap:

```
effective_context(slug, context_map) -> int
  native = spec_for(slug).native_ctx
  raw = context_map.get(slug)            # None → uncapped
  if raw is None: return native
  return max(FLOOR, min(int(raw), native))   # clamp to [FLOOR, native]
```

`FLOOR = 128`. This is the ONLY place clamping happens; the GUI, the extractor,
and any validation all call it so they never disagree.

## 5. Extractor threading

- `_HuggingFaceBackend.__init__(hf_id, max_context=None)`: after reading
  `native = model.config.max_position_embeddings`, set
  `self._context_limit = native if max_context is None else min(native, max_context)`.
  (The existing GPT-2 fp32/CPU pin and everything else is unchanged.)
- `_get_backend(slug, max_context=None)`: cache key becomes
  `(hf_id, max_context)` so two different caps for the same model don't collide
  in `_BACKEND_CACHE`.
- `add_predictability(df, text, *, models, metrics, context=None, backends=None)`:
  `context` is the slug→cap map (default None → all native). For each slug it
  resolves `effective_context(slug, context or {})` and passes it to
  `_get_backend(slug, effective)`. The score cache key already includes the
  model id; extend it to include the effective context so a re-cap re-scores.
- The chunking path (`score()`) is unchanged — it already chunks whenever
  `len(ids) > self._context_limit`; lowering `_context_limit` just makes that
  fire sooner.

## 6. Planner / emit

`emit.py` already sets `ns.predictability = {"models", "metrics"}`; add
`"context"` from `prefs["predictability"]["context"]`. `main.py`'s `run()`
predictability block passes `context=pconf.get("context")` to
`add_predictability`. No new planner stage/cluster (context doesn't affect
activation or columns — same `<slug>_<Metric>` set regardless of cap).

## 7. GUI

On each model row in the Predictability card (`gui/section_word_features.py`),
add a context spinner:
- Range `[FLOOR, spec.native_ctx]`, default = `spec.native_ctx` (or the stored
  cap if set). Label e.g. "context: [____] tok (max N)".
- On change: write `predictability.context[slug] = value` (or delete the key
  when the value equals `native_ctx`, to keep "default = native" as absence).
- Enabled only when the model's checkbox is checked (greyed otherwise), since
  the cap is meaningless for an unselected model.
- Values are pre-clamped by the spinner's range, so an out-of-range value can't
  be entered; `effective_context` is the backstop for hand-edited prefs.

## 8. Testing (TDD for the extractor math)

- `effective_context`: uncapped (absent slug) → native; capped below native →
  the cap; above native → native; below floor → FLOOR; exactly native → native.
- Backend: `_HuggingFaceBackend(hf_id, max_context=k)` sets `_context_limit =
  min(native, k)`; a stimulus longer than the cap triggers `_chunk_slices`
  (assert via a fake/monkeypatched model or the slice helper). `max_context=
  None` preserves native.
- `_get_backend` cache: different `max_context` for the same hf_id yields
  distinct cached backends.
- `add_predictability`: passes the resolved effective context per model; score
  cache re-scores when the cap changes.
- Migration v18→v19: adds `context: {}`; existing values preserved; idempotent.
- GUI: spinner persists per-slug to prefs; deletes the key at native; greyed
  when unchecked; range capped at native.
- Full suite green.

## 9. Breaking changes

None. Purely additive (v19 adds an empty map; default behavior identical).

## 10. Open risks / notes

- `max_position_embeddings` occasionally differs from the roster's `native_ctx`
  (e.g. a model config reporting a rounded value). `effective_context` clamps
  to the ROSTER `native_ctx` (the user-facing ceiling); the backend additionally
  min's against the actual `model.config` value, so the true model limit is
  never exceeded even if the roster is optimistic. Verify the roster values at
  implementation time.
- Lowering the cap reduces per-word preceding context and therefore CHANGES the
  emitted surprisal/cloze values for stimuli longer than the cap — by design,
  and only when the user opts in. Documented in the feature docs.
