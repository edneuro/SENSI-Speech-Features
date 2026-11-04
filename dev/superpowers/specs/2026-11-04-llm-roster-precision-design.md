# GPT-2 family, model precision, beginning-of-text token — design

Status: draft for owner approval (2026-11-04).

## Why

A collaborator compared her GPT-2 **Large** surprisal with ours (GPT-2
small) and found large gaps exactly where the bigger model is confident
(*when he* **grew** *up*: 0.14 vs 7.11 bits). Researchers need to pick the
model size and to know, precisely, how each model was run. Today:

- only the 124M-parameter GPT-2 is offered;
- the Llama models silently load in **4-bit** when a GPU is present (then
  fp16, then CPU), so their values depend on the machine and on whether
  `bitsandbytes` works;
- every model is scored without its beginning-of-text token, so the first
  word is `NaN` and Llama is run without the token it was trained to see.

## Goals

1. Add **GPT-2 Medium, Large, XL** to the roster, downloaded on first use
   only (as GPT-2 is today; setup downloads nothing for them).
2. **Precision is explicit.** GPT-2 family: fp32, fixed. Llama: a
   per-model choice — **fp16 (default), 8-bit, 4-bit**; fp16 is the
   highest offered. The run says what actually ran.
3. **Beginning-of-text token** for every model; the first word gets a
   value.
4. Docs state each model's precision, size and memory; embeddings docs
   (word *and* sentence) explain how the vectors are used.

## Non-goals

- No fp16 for GPT-2 (measured on this CPU: fp16 82 s per 1024-subword
  window vs fp32 0.55 s; fp32 is the exact reference).
- No bf16 option (measured on GPT-2: 0.22 bits mean shift, up to 1.14).
- No new model families, no OpenAI backend.
- No silent precision fallback: a precision that cannot run is an error
  with a fix, never a quiet downgrade.

## Decisions (owner, 2026-11-04)

| Topic | Decision |
|---|---|
| GPT-2 family precision | **fp32 everywhere.** Uses a GPU if present, still fp32. |
| Llama precision | per-model choice **fp16 (default) / 8-bit / 4-bit** |
| Llama on a CPU-only machine | **runs fp32 and says so** (run log + Run page + console) |
| First word | **beginning-of-text token for every model, in this batch** |
| Downloads | on first use only; nothing new in setup |

## Behaviour

### Roster (`sensi/extractors/predictability_models.py`)

| slug | label | hf_id | params | download (fp32 safetensors) | native_ctx | precisions |
|---|---|---|---|---|---|---|
| `gpt2` | GPT-2 | `gpt2` (unchanged: keeps existing caches) | 124M | 0.55 GB | 1024 | fp32 |
| `gpt2_medium` | GPT-2 Medium | `openai-community/gpt2-medium` | 355M | 1.52 GB | 1024 | fp32 |
| `gpt2_large` | GPT-2 Large | `openai-community/gpt2-large` | 774M | 3.25 GB | 1024 | fp32 |
| `gpt2_xl` | GPT-2 XL | `openai-community/gpt2-xl` | 1.5B | 6.43 GB | 1024 | fp32 |
| `llama32_3b` … `llama3_70b` | unchanged | unchanged | 3.2B / 8B / 70B | ≈ 2 bytes × params (weights stored 16-bit) | unchanged | fp16, 8bit, 4bit |

GPT-2 download sizes are the HuggingFace `model.safetensors` sizes
(checked 2026-11-04). `ModelSpec` gains `params` (count), `download_gb`,
`precisions` (tuple, first = default). Column names follow the slug
(`gpt2_large_Surprisal`, …), so the GUI list, columns, doc aggregates,
context spinner and YAML comments pick the new models up from the roster.

### Precision → how a model loads

| model | chosen | GPU present | CPU only |
|---|---|---|---|
| GPT-2 family | fp32 (fixed) | fp32 on GPU | fp32 on CPU |
| Llama | fp16 | fp16 on GPU | **fp32 on CPU** |
| Llama | 8-bit / 4-bit | bitsandbytes 8/4-bit on GPU | **fp32 on CPU** |

- If 8-/4-bit fails to load (e.g. a broken `bitsandbytes`), the run stops
  for that model with a message naming the failure and suggesting fp16 —
  no fallback to another precision.
- Every model prints one line when it loads, e.g.
  `[predict] llama3_8b: fp16 on GPU` / `[predict] llama32_3b: no GPU —
  running fp32 on CPU (~13 GB RAM)`, and the run manifest records
  `slug, hf_id, precision, device` for each model scored.
- A first-use download prints the model and its size before it starts
  (`[predict] gpt2_large: downloading 3.25 GB on first use…`).
- The old `_load_tiers` cascade and `_FULL_PRECISION_CPU_HF_IDS` pin are
  replaced by this table (update their tests and
  `sherlock/validate_llama_predictability.py` if it relies on them).
- The roster notes stop quoting the old 4-bit VRAM figures; memory comes
  from `params × bytes(precision)` (e.g. Llama-3-70B in fp16 ≈ 140 GB).

### Settings

`predictability.precision`: `{slug: "fp16" | "8bit" | "4bit"}`, same shape
as `predictability.context` — a slug absent from the map uses its default
(fp16 for Llama). Only Llama slugs may appear; config_checks rejects other
values and GPT-2 slugs. Prefs schema **v24** (additive). YAML comment lists
the choices. Existing users' Llama runs change from "4-bit if a GPU" to
fp16 — called out in the master plan and the docs.

### GUI (Predictability card)

Each model row gains a precision dropdown after the context spinner: Llama
rows offer fp16 / 8-bit / 4-bit; GPT-2 rows show a fixed, disabled "fp32".
The row note gives size and memory (e.g. "free, no token, 3.25 GB download
on first use, ~3.3 GB RAM"). The Run page disk/RAM estimate uses the
model's download size and `params × bytes(precision)` (fp32 = 4, fp16 = 2,
8-bit = 1, 4-bit = 0.5), with fp32 on a CPU-only machine.

### Beginning-of-text token

- Every forward-pass window starts with the model's BOS token
  (`tokenizer.bos_token_id`: `<|endoftext|>` for GPT-2,
  `<|begin_of_text|>` for Llama 3). The token is never scored and maps to
  no word.
- Long texts: each chunked window is `BOS + up to (limit − 1)` subwords, so
  the context cap still bounds the forward pass.
- The first word now has Surprisal / Cloze / TopMatch values. Only rows
  with no scorable subword (PUNCT, unmappable) stay `NaN`.
- Values change: measured on LJ002-0321 with GPT-2, the second word moved
  7.58 → 5.33 bits; later words change little. Documented as a change.

## Docs

- `pred_surprisal.md` *Models*: a table per model — parameters, download
  size, precision, where it runs, RAM — and a *Precision* subsection with
  the measured fp16/bf16 shifts on GPT-2, why GPT-2 stays fp32, what 8-bit
  and 4-bit trade, and the CPU fp32 rule. Replace every "first word is
  NaN" statement (surprisal, cloze, topmatch, design notes) with the BOS
  behaviour.
- `llm_credentials.md`: first-use downloads and sizes.
- `master_bert_sentence.md`: a *How these vectors are used* section like
  the word-level docs (sentence-to-sentence similarity, encoding models).
- Rebuild the manual with `--refresh-examples --refresh-screenshots`.

## Tests (what must hold)

- roster: new slugs, hf_ids, precisions; GPT-2 family fixed fp32.
- load-plan resolution: the table above, as a pure function
  `(spec, chosen, cuda) -> (dtype/quant, device)`; 8-/4-bit load failure
  raises, no fallback.
- prefs v24 migration (absent map = defaults) and config_checks (bad value,
  GPT-2 slug in the map).
- BOS: fake tokenizer/model — window starts with BOS, first word scored,
  chunk windows each start with BOS and stay ≤ limit.
- manifest records precision/device.

## Left to the builder

Widget details of the dropdown; exact message wording; where in the
manifest the per-model record lives; how estimates read the new fields.
