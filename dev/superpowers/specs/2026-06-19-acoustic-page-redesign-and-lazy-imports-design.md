# Acoustic page redesign + lazy extractor imports — design

**Date:** 2026-06-19
**Status:** Approved (brainstorming) — pending implementation plan

## Problem

Two issues, both confirmed:

1. **Confusing acoustic page layout.** `gui/section_acoustic_features.py` puts the
   four feature toggles in one card and *all* params in a separate card below.
   Nothing visually links a param to the feature(s) it drives, and the params are
   not 1:1 with features (some are shared) — so the user can't tell which option
   belongs to which feature.

2. **Slow GUI startup.** Building the GUI pages imports `gui.planner`
   (via `section_run.py`, `section_doc_features.py`, `preflight.py`), which imports
   `gui/planner/registry.py`. `registry.py` eagerly imports all 18 extractors at
   module top level, dragging in torch + transformers + spaCy. Measured cost:
   `import gui.planner` = **51.39 s**. These heavy models are *not* used at
   startup — they're only needed when a stage actually runs.

   Guiding principle (user): **don't load the heavy ML models unless we know we're
   going to use them; load them at the point we know we will.**

## Part A — Acoustic page layout

Scope: `gui/section_acoustic_features.py` only. The Prosody card (same pattern on
the Word Features page) is explicitly out of scope for now.

Chosen layout: **indented list in one card** (no new master toggle — output is
still gated by the Run tab's "Acoustic .mat", unchanged).

```
┌─ Acoustic features ───────────────────────┐
│ Analysis rate (Hz): [128]  ← all series   │
│ ───────────────────────────────────────── │
│ ☑ Amplitude envelope                       │
│      Method:[rms] Comp:[0.3] Window:[1.0]  │
│ ☑ Onset envelope (HWR derivative)          │
│      (derived from envelope — no params)   │
│ ☑ Log-mel spectrogram                      │
│      Mel bands:[16]  Preset:[speech]       │
│ ☐ MFCC                                     │
│      n_mfcc:[13]  (+ mel bands/preset ↑)   │
└────────────────────────────────────────────┘
```

Structure:
1. **Shared analysis grid** at top: `Analysis rate (Hz)` spinbox + caption
   "applies to every series below". Always enabled.
2. Divider, then four feature blocks. Each block = the feature `ToggleRow`
   immediately followed by an indented params row (or a caption for features with
   no own params).

### Contextual greying

A feature's params enable/disable live, following what `extract_acoustic()` in
`extractors/acoustic_ts.py` actually consumes:

| Param widgets | Consumed by | Enabled when |
|---|---|---|
| Analysis rate | all | always |
| Env method / comp / window | envelope + onset-env (`acoustic_ts.py` ~L173) | envelope OR onset-env on |
| Mel bands (`n_mels`) | log-mel + MFCC | log-mel OR MFCC on |
| Mel preset | log-mel + MFCC | log-mel OR MFCC on |
| `n_mfcc` | MFCC only | MFCC on |

Each feature toggle's `toggled` signal calls a `_sync_param_enabled()` method that
applies the table above. Run once at construction to set the initial state.

### No behavior change

Same preference keys (`features.feat_acoustic_ts_*`, `acoustic_ts.*`), same config
dict shape, same widget value ranges/defaults. This is layout + a greying method
only — nothing about what gets computed or saved changes.

## Part B — Lazy extractor imports (`gui/planner/registry.py`)

`plan_from_prefs` (planning) reads only each `Stage`'s `name/consumes/produces`
metadata; the `run` callables fire only at execution time in `run_dispatch`. So the
eager top-level extractor imports are pure startup tax.

Fix: a `_lazy(module_path, attr)` helper returning a small callable that
imports-and-caches the target function on first **call**, then delegates:

```python
def _lazy(module_path, attr):
    cache = {}
    def call(*args, **kwargs):
        fn = cache.get("fn")
        if fn is None:
            from importlib import import_module
            fn = getattr(import_module(module_path), attr)
            cache["fn"] = fn
        return fn(*args, **kwargs)
    return call
```

Remove the 18 top-level `from extractors.* import …` lines; each
`Stage(run=add_surprisal)` becomes `Stage(run=_lazy("extractors.surprisal", "add_surprisal"))`.
`Stage` metadata builds instantly; torch/spaCy/transformers load only when a stage
actually runs (Run button), where a wait is already expected.

**Risk to verify in implementation:** nothing may rely on `Stage.run` identity
(e.g. a test asserting `stage.run is add_surprisal`). Grep before committing; if any
exist, adjust them to call through the lazy wrapper instead.

## Testing

Per repo TDD scope (extractors math = TDD; GUI/refactor = test-after):

- **Part A (GUI):** test-after. One test asserting greying rules: MFCC off →
  `n_mfcc` disabled; envelope + onset-env both off → env params disabled; log-mel
  off but MFCC on → mel params still enabled.
- **Part B (refactor):** a regression test asserting that after `import gui.planner`,
  `torch`/`transformers`/`spacy` are **not** in `sys.modules` — so the slow-startup
  regression can't silently return. Optional timing sanity check.

## Out of scope

- Prosody card restructure (same pattern, deferred).
- Any change to extractor math, preference schema, or output formats.
