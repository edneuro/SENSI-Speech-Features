# Design: GUI HuggingFace-token support for the local Llama LLM-cloze backend

**Date:** 2026-07-07
**Status:** approved (design), pending implementation plan

## Problem

Selecting a local Llama model for LLM-cloze in the GUI does not work end-to-end:

1. **(already fixed)** The shipped model IDs were wrong (`meta-llama/Llama-3-8B`
   vs the real `meta-llama/Meta-Llama-3-8B`), so the request 404'd. Corrected in
   `extractors/llm_cloze.py` (`LLM_OPTIONS`, `select_model`) + tests + `validate_cloze.py`.
2. Meta-Llama models are **gated** — they require an accepted license and a
   HuggingFace token. The GUI has no way to supply a token, so `transformers`
   downloads fail unless the user has separately set `HF_TOKEN` / run `hf auth login`.
3. When the model can't load (gated / not-found / offline), `_HuggingFaceBackend`
   swallows the error in a bare `except` and retries without quantization,
   producing a confusing failure instead of an actionable message.

This design covers the **GUI / local-run** path only. The Sherlock path (offline
compute nodes, login-node pre-download) is a **separate issue** and out of scope
here.

## Goal

A user running the pipeline locally through the GUI on a GPU machine can select a
local Llama model, supply a HuggingFace token (or have one auto-detected), and get
LLM-cloze columns — with clear guidance when something is missing.

## Non-goals

- Sherlock/HPC token handling or docs (separate issue).
- Persisting the token to disk (session-only, mirroring the OpenAI key).
- Auto-accepting the Meta license (impossible — it's Meta's gate).
- Changing the model-selection / VRAM auto-pick logic.

## Existing patterns this mirrors

The OpenAI API-key input is the template; the HF token reuses its shape exactly:

- **Input widget:** `gui/section_word_features.py` `api_key_row` — password
  `QLineEdit` + "Show" toggle, placeholder "…never saved to disk", built once and
  shown/hidden by `_refresh_api_key_visibility` (visible iff `model_id` starts with
  `gpt-`).
- **Getter:** `session_api_key()` returns the field text only when its row is visible.
- **Wiring:** `gui/main_window.py` passes `get_session_api_key=self.word_features.session_api_key`
  to `SectionRun`; `gui/section_run.py` calls it and forwards `session_api_key=…`
  into `run_dispatch`.
- **Run injection:** `gui/run_dispatch.py:66,101` sets `os.environ["OPENAI_API_KEY"]`
  before the run and pops it after.
- **Preflight:** `gui/preflight.py` `compute_warnings(session_api_key=…)` surfaces
  an OpenAI-key warning on the relevant toggle.

## Components

### A. Token detection — `extractors/llm_cloze.py`

`has_hf_token() -> bool`: lazily import `huggingface_hub` and return
`bool(huggingface_hub.get_token())`. `get_token()` already resolves `HF_TOKEN`,
`HUGGING_FACE_HUB_TOKEN`, and a token saved by a prior `hf auth login`. Lazy import
keeps GUI startup cost unchanged. Also add a helper `is_gated_local_model(model_id)`
→ `model_id.startswith("meta-llama/")` so the GUI and preflight share one rule.

### B. GUI token box — `gui/section_word_features.py`

Add `hf_token_row` (+ `hf_token_line`, "Show" toggle), a direct clone of
`api_key_row`, placeholder "Paste HuggingFace token — never saved to disk".
Add a sibling `hf_token_detected_label` ("✓ HuggingFace token detected").

Rename `_refresh_api_key_visibility` → `_refresh_llm_credential_visibility`
(keep behavior for the OpenAI row) and extend:

- `model_id` starts with `gpt-` → show `api_key_row`; hide HF widgets.
- `is_gated_local_model(model_id)`:
  - `has_hf_token()` true → show `hf_token_detected_label`; hide the box.
  - else → show `hf_token_row`; hide the label.
- otherwise → hide all three.

`hf_token_line.textChanged` re-runs preflight (mirrors the OpenAI field).

Add `session_hf_token() -> str`: return `hf_token_line.text()` only when
`hf_token_row` is visible, else `""` (mirrors `session_api_key`).

### C. Token → run

- `gui/main_window.py`: pass `get_session_hf_token=self.word_features.session_hf_token`
  into `SectionRun` (alongside the existing api-key getter).
- `gui/section_run.py`: store it; forward `session_hf_token=self._get_hf_token()`
  into the `run_dispatch` call (next to `session_api_key`).
- `gui/run_dispatch.py`: accept `session_hf_token`; if non-empty, set
  `os.environ["HF_TOKEN"] = session_hf_token` before the run and pop it in the
  same `finally`/cleanup path that pops `OPENAI_API_KEY`. Session-only; never persisted.

### D. Preflight warning — `gui/preflight.py`

`compute_warnings` gains awareness of the HF token (accept a `session_hf_token`
param mirroring `session_api_key`). When the selected model
`is_gated_local_model(...)` and neither `has_hf_token()` nor a session token is
present, emit a warning on the LLM-cloze toggle key: "Local Llama needs a
HuggingFace token + accepted license — paste a token above or run `hf auth login`."

### E. Clear backend error — `extractors/llm_cloze.py` `_HuggingFaceBackend.__init__`

Current code wraps the quantized load in `try/except Exception` and, on *any*
error, retries `from_pretrained` without quantization — masking gated/404/offline
failures. Change:

- Attempt the 4-bit quantized load; catch **only** bitsandbytes/quantization-related
  errors (ImportError / the bitsandbytes runtime error) → fall back to fp16.
- Let auth/not-found/offline errors from `from_pretrained` propagate, wrapped in a
  `RuntimeError` with an actionable message: which model id failed, and that the
  fixes are — accept the license at its HF page, provide a token (GUI box, `HF_TOKEN`,
  or `hf auth login`), or (offline) pre-download the weights.

### F. Docs (local only) — `docs/features/pred_cloze.md`

Add a "Setting up the local Llama backend (local runs)" section: accept the license
→ create a Read token → paste it in the GUI box *or* run `hf auth login` → note the
GPU/VRAM requirement and the OpenAI backend as the zero-setup alternative. Add a
one-line cross-link from `pred_surprisal_llm.md`. **No Sherlock content.**

## Data flow

```
model dropdown ─┐
                ├─► _refresh_llm_credential_visibility ─► shows OpenAI box | HF box | "✓ detected" | nothing
has_hf_token() ─┘
HF box text ─► session_hf_token() ─► SectionRun ─► run_dispatch ─► os.environ["HF_TOKEN"] (set, run, pop)
                                    └─► preflight warning if gated & no token
run ─► add_llm_cloze ─► _cached_backend ─► _HuggingFaceBackend(model_id)
                                             ├─ success → cached, reused across docs
                                             └─ gated/404/offline → RuntimeError with guidance (E)
```

## Testing

Per project TDD scope (test-after for GUI/prefs; E is error-handling, not extractor
math):

- `tests/extractors/test_llm_cloze.py`: `has_hf_token()` true/false via monkeypatched
  `get_token`; backend error path raises the actionable message on a simulated
  gated/not-found load (mock `from_pretrained` to raise), and still falls back to
  fp16 on a bitsandbytes-only error.
- `pytest-qt` (word-features page): selecting an OpenAI model shows the API-key box;
  selecting a gated model with no token shows the HF box; with a token detected shows
  the "✓" label; `session_hf_token()` returns text only when the box is visible.
- `tests/gui/test_run_dispatch.py`: `HF_TOKEN` is set from `session_hf_token` before
  the run and popped after (mirror the existing `OPENAI_API_KEY` assertions).

## Risks / notes

- `huggingface_hub.get_token()` API name: verify against the container's
  `huggingface_hub` version at implementation time; fall back to reading `HF_TOKEN`
  env + the token file if the helper name differs.
- Keep the token strictly session-scoped — no `save_preferences` call ever receives it.
