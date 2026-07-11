---
key: pred_cloze
name: LLM cloze probability
levels: [word]
gui_toggle: pred_cloze
output_columns: [LLM_Cloze]
libraries: [transformers, torch, openai]
status: implemented
depth: deep
---

## Meaning
The conditional probability that the active LLM assigns to each word
given preceding context — i.e. the raw P, not its negative log:

    LLM_Cloze(wᵢ) = P_{LLM}(wᵢ | w₁ … wᵢ₋₁)    ∈ (0, 1]

Mathematical exponential inverse of [`LLM_Surprisal`](pred_surprisal_llm.md):

    LLM_Cloze = 2^(-LLM_Surprisal)

PUNCT rows are `NaN`, and so is the document's first word (no
preceding context to condition on). Both columns come from the same
forward pass —
enabling either triggers the full LLM extractor.

## EEG rationale
"Cloze probability" is the historical name from offline human-cloze
norming ([Taylor 1953]): show humans the context up to position `i`
and ask them to fill in the missing word; tally the proportion who
produce each candidate. The N400 was originally characterized as
inversely proportional to (human) cloze probability ([Kutas 1984],
[DeLong 2005]).

`LLM_Cloze` is the **modern machine analogue** — an LLM's conditional
probability over the actual continuation. Where the relationship to
the N400 is concerned:

- For *amplitude* prediction, **surprisal** (the log) is theoretically
  preferred ([Smith 2013]) because cognitive cost is linear in
  surprisal, not in probability — and language-model surprisal has
  been shown to predict N400 amplitude better than cloze probability
  itself ([Michaelov 2024]).
- For comparison with classic human-cloze experiments, the raw
  probability is what aligns with the published metric.
- For analyses that need an interpretable [0–1] regressor (logistic
  models, threshold-based contrasts of "predictable" vs "not"), the
  probability form is more natural than bits.

⚠️ When in doubt, use [`LLM_Surprisal`](pred_surprisal_llm.md) for
ERP-amplitude modelling and `LLM_Cloze` only when you specifically
need the cloze metric.

## Implementation in SensiExtractor
Source: `extractors/llm_cloze.py:add_llm_cloze`. Backend:
`_HuggingFaceBackend` (local Hugging Face causal LM). OpenAI is not a
supported backend — see [LLM access & credentials](llm_credentials.md).

See [`LLM_Surprisal` § Implementation](pred_surprisal_llm.md#implementation-in-sensiextractor)
— `LLM_Cloze` is computed in the same scoring call as
`LLM_Surprisal` and [`LLM_TopMatch`](pred_topmatch.md). Per spaCy
token:

```python
p = math.exp(sum(subword_log_probs))      # ← this column
surprisal = -sum(subword_log_probs) / math.log(2)
top_match = (argmax_subword == actual_subword)  # at the first subword
```

PUNCT rows are masked to `NaN`. The probability is stored rounded to
6 decimal places.

## Papers

**Cloze procedure & N400:**
- <a name="taylor-1953"></a>**[Taylor 1953]** Taylor. *"Cloze procedure": A new tool for measuring readability.* Journalism Quarterly. [doi:10.1177/107769905303000401](https://doi.org/10.1177/107769905303000401) — original cloze methodology.
- <a name="kutas-1984"></a>**[Kutas 1984]** Kutas & Hillyard. *Brain potentials during reading reflect word expectancy and semantic association.* Nature. [doi:10.1038/307161a0](https://doi.org/10.1038/307161a0) — N400 ∝ −cloze probability.
- <a name="delong-2005"></a>**[DeLong 2005]** DeLong, Urbach, Kutas. *Probabilistic word pre-activation during language comprehension inferred from electrical brain activity.* Nature Neuroscience. [doi:10.1038/nn1504](https://doi.org/10.1038/nn1504)
- <a name="smith-2013"></a>**[Smith 2013]** Smith & Levy. *The effect of word predictability on reading time is logarithmic.* Cognition. [doi:10.1016/j.cognition.2013.02.013](https://doi.org/10.1016/j.cognition.2013.02.013) — cost is in log P, not P; favours `LLM_Surprisal` over `LLM_Cloze` for amplitude modelling.

**LM-derived cloze:**
- <a name="hofmann-2022"></a>**[Hofmann 2022]** Hofmann, Remus, Biemann, Radach, Kuchinke. *Language models explain word reading times better than empirical predictability.* Frontiers in Artificial Intelligence. [doi:10.3389/frai.2021.730570](https://doi.org/10.3389/frai.2021.730570) — LM probabilities outperform human cloze-completion probabilities at predicting fixation durations and gaze durations in English and German eye-tracking corpora.
- <a name="michaelov-2024"></a>**[Michaelov 2024]** Michaelov, Bardolph, Van Petten, Bergen, Coulson. *Strong Prediction: Language Model Surprisal Explains Multiple N400 Effects.* Neurobiology of Language 5(1), 107–135. [doi:10.1162/nol_a_00105](https://doi.org/10.1162/nol_a_00105) — direct comparison: LM surprisal explains N400 amplitude better than cloze probability or distributional semantic similarity.

[Taylor 1953]: #taylor-1953
[Kutas 1984]: #kutas-1984
[DeLong 2005]: #delong-2005
[Smith 2013]: #smith-2013
[Hofmann 2022]: #hofmann-2022
[Michaelov 2024]: #michaelov-2024

## Getting access & credentials

The local Llama backend needs a HuggingFace token + license approval before
it runs. This is shared with `LLM_Surprisal` and `LLM_TopMatch` — see
**[LLM access & credentials](llm_credentials.md)** for the full step-by-step.

## Datasets
<!-- auto-populated from docs/datasets/*.md once the dataset registry exists -->
