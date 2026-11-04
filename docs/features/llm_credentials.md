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
(roster: four GPT-2 sizes + three Llama-3 variants; see
[Models](pred_surprisal.md#models)). Weights are not bundled and **nothing
is downloaded at setup**: each model's weights download into
`data/hf_cache/` the first time a run uses it (GPT-2 0.55 GB, GPT-2 medium
1.52 GB, GPT-2 large 3.25 GB, GPT-2 XL 6.43 GB, Llama-3.2-3B 6.42 GB,
Llama-3-8B 16.0 GB, Llama-3-70B 142 GB), with a console line and a
"First-use download" line on the Run page. Set access up once per gated
model before running. **The GPT-2 models need no credentials at all** —
they are free and ungated. Everything below applies only to the three gated
Llama-3 models.

> **Why local-only (no OpenAI).** Scoring an *existing* string for per-token
> log-probabilities needs the OpenAI `/v1/completions` `echo=True,
> logprobs` primitive, which OpenAI does not offer and never served for the
> `gpt-4o` family. There is no current OpenAI endpoint that returns the
> probabilities these columns require, so SENSI SpeechFeatures uses local
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
   - Paste it into the **HF token** box in the GUI's Predictability card, then
     click **Save** to persist it to `config/hf_token.txt` (gitignored) and
     point `shared.hf_token_file` at it — headless runs (`python run.py`,
     Sherlock) then pick it up automatically without repasting. Leaving it
     unsaved keeps the token session-only, for that GUI run alone. The box is
     shown whenever no token is already detected — you don't need to select a
     model first; once a token is detected (saved to the configured file, or
     via one of the fallbacks below) it is replaced by "✓ HuggingFace token
     detected", **or**
   - Run `hf auth login` once in the environment (stored under `~/.cache`), **or**
   - Set the `HF_TOKEN` (or `HUGGING_FACE_HUB_TOKEN`) environment variable.

**Precision and device placement.** With a GPU, a Llama runs at the
precision chosen on its row of the Predictability card — fp16 by default,
or 8bit / 4bit (these need a working *bitsandbytes*; if the load fails the
run stops and suggests fp16, it never switches precision silently) — and
the GPT-2 models run fp32. With no CUDA GPU every model runs fp32 on the
CPU, and the console prints the RAM that needs (for example about 32 GB for
Llama-3-8B). See [Precision](pred_surprisal.md#precision) for the measurements
behind this and for what the console line and the run manifest record.
Running on Sherlock/HPC is documented separately.
