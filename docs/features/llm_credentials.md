---
key: llm_credentials
name: LLM access & credentials
levels: [word]
libraries: [transformers, torch, huggingface_hub]
status: reference
depth: shallow
---

## Getting access & credentials

The predictability metrics — [`Cloze`](pred_cloze.md),
[`Surprisal`](pred_surprisal.md), and
[`TopMatch`](pred_topmatch.md) — come from a single forward pass of
each **local** model you select in the GUI's Predictability card
(roster: GPT-2 + three Llama-3 variants; see
[Models](pred_surprisal.md#models)). Weights are not bundled; set
access up once per gated model before running. **GPT-2 needs no
credentials at all** — it's free, ungated, and downloads automatically
on first use. Everything below applies only to the three gated
Llama-3 models.

> **Why local-only (no OpenAI).** Scoring an *existing* string for per-token
> log-probabilities needs the legacy OpenAI `/v1/completions` `echo=True,
> logprobs` primitive, which OpenAI has retired and never served for the
> `gpt-4o` family. There is no current OpenAI endpoint that returns the
> probabilities these columns require, so SensiExtractor uses local
> Hugging Face causal LMs only. (Selecting a `gpt-*` model id raises an
> explanatory error rather than failing mid-run.) There is no API-key
> input anywhere in the GUI for this feature — only local models are ever
> offered.

### Local Llama backends — gated, free, one-time setup

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
