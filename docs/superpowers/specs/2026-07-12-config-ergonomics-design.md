# Config ergonomics: headless defaults templates + persistent HF token — design

**Date:** 2026-07-12
**Status:** approved by user (decisions settled in-conversation, same day as Project C shipped)

## 1. Goal

Two ergonomics gaps left after Project C (two-file YAML config):
(a) a GUI-skipping user has no file to copy/edit — a fresh checkout has no
YAMLs and `run.py` silently runs minimal defaults; (b) the HuggingFace token
for gated Llama models is session-only in the GUI and requires hand-authoring
`shared.hf_token_file` for headless runs.

## 2. User-approved decisions (do not re-litigate)

1. **Committed, generated templates**: `config/features_defaults.yaml` +
   `config/run_defaults.yaml`, tracked in git, generated from the code
   defaults (`DEFAULT_PREFS`) — code stays the single source of truth; the
   templates are build artifacts. A **drift-guard test** asserts the
   committed templates stay in sync with `DEFAULT_PREFS` (semantic equality
   via parsed YAML, so a human-facing template banner is allowed). Headless
   flow: `cp config/features_defaults.yaml config/features.yaml`, edit, run.
2. **Reset is unchanged** — it keeps regenerating `features.yaml` from
   `DEFAULT_PREFS` in code (identical content to the template, by the guard).
3. **HF token file**: `config/hf_token.txt` (gitignored) becomes the
   conventional token location. The GUI token box **saves** the pasted token
   there (replacing today's session-only behavior) and the UI copy says so
   explicitly ("saved to config/hf_token.txt" — no more "never saved to
   disk"). Env var / `hf auth login` remain as detection fallbacks.
   Name is `hf_token.txt` (HuggingFace account token, not Llama-specific).
4. Master-toggle removal and the Run-tab redesign (#7/#8 in the user's list)
   are **deferred** until after the manual-validation pass on real stimuli —
   record them in the master-plan roadmap with the design constraints noted
   there, no code now.

## 3. Judgment calls (accepted design, implementer follows these)

- **Relative token paths resolve against the config dir**: the shipped
  default for `shared.hf_token_file` becomes `"hf_token.txt"` (relative), and
  every consumer resolves a relative value against `get_config_dir()` —
  absolute paths and `~`/`$VARS` still work as today. This makes the same
  default correct on Windows dev machines and on Sherlock.
- **No schema bump**: `hf_token_file` shipped days ago with default `""`;
  fresh installs get the new default via `DEFAULT_PREFS`. A stored `""` in an
  existing `run.yaml` is left alone (deep-merge keeps user values) — the GUI
  self-heals it: saving a token via the GUI writes the file AND sets
  `shared.hf_token_file` to point at it.
- **One shared credentials helper**: the token-file→`HF_TOKEN` logic
  currently in `run.py:load_credentials` moves to a small shared util
  (`utils/credentials.py`) used by `run.py` AND the GUI run path
  (`gui/run_dispatch.py`), so GUI batch runs pick up the saved token without
  the paste box. `extractors.predictability.has_hf_token()` (and the
  preflight gated-model warning) additionally treat a readable, non-empty
  configured token file as "token present".
- **Template generation**: `scripts/export_default_config.py` writes both
  templates via the existing `split_prefs` + `write_features_yaml`/
  `write_run_yaml`, then prepends a template banner ("copy to
  features.yaml/run.yaml and edit; regenerate with this script"). The
  drift-guard test compares `read_yaml(template)` to `split_prefs(DEFAULT_PREFS)`
  and its failure message says to rerun the script. Adding a prefs key
  without regenerating templates = red test.
- `run.py`'s missing-features.yaml warning gains "copy
  config/features_defaults.yaml to config/features.yaml to start from the
  documented defaults".
- `.gitignore`: add `config/hf_token.txt`; the templates are tracked (no
  ignore rule matches them).
- Docs touched: CLAUDE.md (config/ line), README (headless quickstart
  mention), sherlock/RUNBOOK.md (copy `run_defaults.yaml` → `run.yaml` on
  Sherlock; token file location), `docs/features/llm_credentials.md`
  (token now persistable via GUI / `config/hf_token.txt`), master_plan
  (§3 row + roadmap entries for the deferred #7/#8 gating redesign and the
  parked utils→scripts tool move).

## 4. Out of scope

- No changes to master toggles, output toggles, Run-tab layout (deferred).
- No move of `utils/download_models.py` etc. to `scripts/` (parked; roadmap
  note only).
- `unrelated/` directory untouched.

## 5. Testing expectations

- TDD for the utils-layer logic (path resolution, credentials helper,
  has_hf_token extension); test-after for GUI/scripts/docs (per
  `feedback-tdd-scope`).
- Drift-guard test for both templates; regression tests for GUI token save
  (writes file + sets pref + label state) and for run_dispatch exporting the
  file token into `HF_TOKEN`.
- Full suite green at the end (baseline: **755 passed, 1 skipped**).
- Test invocation (bash): `PATH="$(pwd)/env/Library/bin:$PATH" ./env/python.exe -m pytest`.
