---
key: pred_surprisal_llm
name: LLM surprisal
levels: [word]
gui_toggle: pred_surprisal_llm
output_columns: [LLM_Surprisal]
libraries: [transformers, torch, openai]
status: implemented
depth: deep
---

## Meaning
Same metric as [`Surprisal`](pred_surprisal_gpt2.md) — negative log₂
probability of each word given preceding context, in **bits** — but
computed with a modern LLM instead of GPT-2:

    LLM_Surprisal(wᵢ) = -log₂ P_{LLM}(wᵢ | w₁ … wᵢ₋₁)

The active LLM is configurable from the GUI's Predictability card
(default `gpt-4o-mini` if `OPENAI_API_KEY` is set, otherwise the
largest Llama-3 that fits the GPU). Better LMs produce surprisal
estimates that predict reading time and N400 amplitude more tightly
([Goodkind 2018], [Wilcox 2020]) — `LLM_Surprisal` is the column to
use for new analyses; `Surprisal` (GPT-2) is kept as a baseline.

## EEG rationale
See [`Surprisal` § EEG rationale](pred_surprisal_gpt2.md#eeg-rationale)
— the surprisal-theory story is identical; what changes is the
estimator quality. Empirically, larger / better LMs improve N400
prediction up to a point and then plateau ([Schrimpf 2021],
[Caucheteux 2022]). Using `LLM_Surprisal` lets you take advantage of
that improvement; using both `Surprisal` and `LLM_Surprisal` lets you
study how estimator quality affects your specific neural fit.

## Implementation in SensiExtractor
Source: `extractors/llm_cloze.py:add_llm_cloze` + the two backends in
the same file.

**What the code does:**

1. Selects a backend based on `model_id` (or via `select_model()` —
   prefer OpenAI if `OPENAI_API_KEY` set, otherwise a Llama-3 sized to
   available VRAM).
2. Runs a single forward pass over the **whole transcript**:
   - Local backend (`_HuggingFaceBackend`): autoregressive LM forward
     pass; 4-bit quantization via `BitsAndBytesConfig` when supported,
     fp16 fallback otherwise.
   - OpenAI backend (`_OpenAIBackend`): calls `/v1/completions` with
     `echo=True, max_tokens=0, logprobs=5` to score the prompt itself.
3. Per spaCy token, finds the contiguous run of LLM subwords whose
   character offsets overlap that token, sums their log-probabilities,
   and converts to bits: `-Σ log_e P / log_e 2`.
4. `upos == "PUNCT"` → `NaN`.

This **also produces** [`LLM_Cloze`](pred_cloze.md) and
[`LLM_TopMatch`](pred_topmatch.md) from the same forward pass — they're
three views of the same per-token prediction. Enabling any of the three
in the GUI triggers the full extractor; cost (and runtime) is the same
whether you enable one or all three.

**Why** *(⚠️ flags my interpolation):*

- *Three columns from one forward pass* — cloze probability,
  surprisal, and top-match are all cheap byproducts of the same scoring
  call. Computing them separately would triple cost and runtime for no
  benefit.
- *Auto-selection prefers OpenAI when API key is set* — ⚠️ assumption
  is that a user who set the key intends to use it (otherwise why
  paste it?). Local Llama is the free / private fallback.
- *4-bit quantization via bitsandbytes on the local backend* —
  Llama-3-8B fits in ~6 GB VRAM at 4-bit (vs ~16 GB at fp16), so most
  researcher-grade GPUs can run it. fp16/CPU is a graceful fallback.

## Cost / runtime notes (from `LLM_OPTIONS`)
| Model | Cost / stim | r vs human cloze | Requirement |
|---|---|---|---|
| gpt-4o-mini | $0.10–0.30 | ~0.83 | `OPENAI_API_KEY` |
| gpt-4o      | $1–3       | ~0.85 | `OPENAI_API_KEY` |
| Llama-3-70B | free       | ~0.84 | ≥ 48 GB VRAM |
| Llama-3-8B  | free       | ~0.79 | ≥ 12 GB VRAM |
| Llama-3.2-3B | free      | ~0.76 | ≥ 8 GB VRAM (or CPU) |

⚠️ The `r` values above are project conventions in the GUI dropdown;
they reference cloze-prediction correlations rather than EEG fit
specifically. Treat them as relative-ordering hints, not absolute
benchmarks.

## Library
See `extractors/llm_cloze.py:_HuggingFaceBackend.score` for the local
path and `_OpenAIBackend.score` for the API path. Both return one
dict per spaCy token with keys `{p, surprisal, top_match}`.

## Papers

**Surprisal ↔ LM quality:**
- <a name="goodkind-2018"></a>**[Goodkind 2018]** Goodkind & Bicknell. *Predictive power of word surprisal for reading times is a linear function of language model quality.* CMCL 2018. [arXiv:1804.10742](https://arxiv.org/abs/1804.10742)
- <a name="wilcox-2020"></a>**[Wilcox 2020]** Wilcox, Gauthier, Hu, Qian, Levy. *On the Predictive Power of Neural Language Models for Human Real-Time Comprehension Behavior.* CogSci 2020. [arXiv:2006.01912](https://arxiv.org/abs/2006.01912)

**Modern LM ↔ neural data:**
- <a name="schrimpf-2021"></a>**[Schrimpf 2021]** Schrimpf et al. *The neural architecture of language: Integrative modeling converges on predictive processing.* PNAS. [doi:10.1073/pnas.2105646118](https://doi.org/10.1073/pnas.2105646118)
- <a name="caucheteux-2022"></a>**[Caucheteux 2022]** Caucheteux & King. *Brains and algorithms partially converge in natural language processing.* Communications Biology. [doi:10.1038/s42003-022-03036-1](https://doi.org/10.1038/s42003-022-03036-1)
- <a name="goldstein-2022"></a>**[Goldstein 2022]** Goldstein et al. *Shared computational principles for language processing in humans and deep language models.* Nature Neuroscience. [doi:10.1038/s41593-022-01026-4](https://doi.org/10.1038/s41593-022-01026-4)

[Goodkind 2018]: #goodkind-2018
[Wilcox 2020]: #wilcox-2020
[Schrimpf 2021]: #schrimpf-2021
[Caucheteux 2022]: #caucheteux-2022
[Goldstein 2022]: #goldstein-2022

## Datasets
<!-- auto-populated from docs/datasets/*.md once the dataset registry exists -->
