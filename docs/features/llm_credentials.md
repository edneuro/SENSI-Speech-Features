---
key: llm_credentials
name: LLM access & credentials
levels: [word]
libraries: [transformers, torch, openai, huggingface_hub]
status: reference
depth: shallow
---

## Getting access & credentials

The LLM predictability columns — [`LLM_Cloze`](pred_cloze.md),
[`LLM_Surprisal`](pred_surprisal_llm.md), and
[`LLM_TopMatch`](pred_topmatch.md) — all come from a single forward pass of
one language model. Whichever backend that model uses needs credentials, and
neither is bundled; set one up before running. What you do here applies to all
three columns at once.

Pick **one** backend:

- **OpenAI** (`gpt-4o`, `gpt-4o-mini`) — zero download, paid API, needs an API key.
- **Local Llama** (`meta-llama/*`) — free, runs on your machine/GPU, but the
  weights are *gated* and need a HuggingFace token plus one-time license
  approval.

### OpenAI backend — API key

1. Create an API key at <https://platform.openai.com/api-keys> (requires an
   account with billing enabled — these calls are paid; see *Cost / runtime*
   in [`LLM_Surprisal`](pred_surprisal_llm.md#cost--runtime-notes-from-llm_options)).
2. Make it available to the run in one of two ways:
   - Paste it into the **API key** box in the GUI's Predictability card
     (session-only, never written to disk), **or**
   - Set the `OPENAI_API_KEY` environment variable before launching, e.g.
     `export OPENAI_API_KEY=sk-…` (bash) or `$env:OPENAI_API_KEY="sk-…"`
     (PowerShell).

When `OPENAI_API_KEY` is set, `gpt-4o-mini` is auto-selected. No download.

### Local Llama backend — gated, free, one-time setup

The Meta-Llama weights are free but **gated**: HuggingFace releases them only
after you accept Meta's license *and* Meta approves your request.

1. **Sign in** at <https://huggingface.co> (create a free account if needed).
2. **Open the model page** for the variant you want, e.g.
   <https://huggingface.co/meta-llama/Meta-Llama-3-8B>, and fill in the access
   form to **accept Meta's license**.
3. **Wait for approval.** Access is granted **manually, not instantly** —
   you'll receive a **confirmation email** and the page will change to show
   "You have been granted access to this model". This typically takes anywhere
   from a few minutes to a day or two. Until then, downloads return HTTP 403.
4. **Create a token.** Settings → Access Tokens → *New token* with the **Read**
   role: <https://huggingface.co/settings/tokens>.
5. **Provide the token** to the run in any one of these ways (all auto-detected):
   - Paste it into the **HF token** box that appears in the GUI's Predictability
     card when a gated Llama model is selected (session-only, never written to
     disk); once detected the box is replaced by "✓ HuggingFace token detected",
     **or**
   - Run `hf auth login` once in the environment (stored under `~/.cache`), **or**
   - Set the `HF_TOKEN` (or `HUGGING_FACE_HUB_TOKEN`) environment variable.

**Device placement is automatic**, in this order: 4-bit on the GPU (needs a
working *bitsandbytes* build — an 8B model fits in ~6 GB VRAM this way) →
fp16 on the GPU if bitsandbytes is unavailable (≈16 GB VRAM for 8B; this is the
common path on Windows, where bitsandbytes is often broken) → CPU if there's no
CUDA GPU. If a GPU *is* present but both GPU loads fail, the run logs a loud
warning and falls back to CPU rather than using the GPU silently — watch the
log for `loaded on CPU` if inference is unexpectedly slow. Running on
Sherlock/HPC is documented separately.
