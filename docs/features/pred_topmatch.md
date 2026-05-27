---
key: pred_topmatch
name: LLM top-1 prediction match
levels: [word]
gui_toggle: pred_topmatch
output_columns: [LLM_TopMatch]
libraries: [transformers, torch, openai]
status: implemented
depth: shallow
---

## Meaning
Binary flag (`0` or `1`) indicating whether the LLM's **most likely
next word** at this position matched the actual word in the
transcript:

    LLM_TopMatch(wᵢ) = 1   if  argmaxⱼ P_{LLM}(j | w₁ … wᵢ₋₁) == subword(wᵢ)
                       0   otherwise

Comparison is made at the **first subword** of the spaCy token (BERT
or LLM subwords don't always align 1:1 with spaCy tokens). PUNCT rows
are `NaN`.

## EEG rationale
Whereas [`LLM_Cloze`](pred_cloze.md) and
[`LLM_Surprisal`](pred_surprisal_llm.md) are graded measures of
predictability, `LLM_TopMatch` is a **discrete** prediction-correctness
flag. It's useful when:

- The downstream model wants a binary regressor — e.g. an "expected"
  vs "violation" contrast — without picking a probability threshold.
- You want to count *strict-prediction hits* across a stimulus
  (proportion of positions where the LLM nailed the next word).
- You're comparing against psycholinguistic experiments that operate
  on "predictable" vs "unpredictable" rather than continuous cloze.

⚠️ For ERP-amplitude regressions, the graded
[`LLM_Surprisal`](pred_surprisal_llm.md) carries more information per
row and is the better default — `LLM_TopMatch` is binary by design.

## Implementation in SensiExtractor
Source: `extractors/llm_cloze.py:add_llm_cloze` and the per-backend
`score()` methods.

Computed in the same forward pass as `LLM_Cloze` and
`LLM_Surprisal`: per subword, the backend already computes
`argmax(log_probs)` over the vocabulary. For each spaCy token, the
flag is set by comparing the argmax at the token's **first subword**
position to the actual id at that position:

```python
top_match = 1 if sub_top1[first_sub] == ids[first_sub] else 0
```

The OpenAI backend uses `top_logprobs=5` and compares the stripped
top-1 token string against the stripped actual subword string —
necessary because the OpenAI endpoint returns text rather than ids.

`upos == "PUNCT"` → `NaN`. The output is stored as `int` for `0` / `1`
and remains `NaN` (so a `Nullable Int64` view in pandas) for
punctuation.

## Library
See `extractors/llm_cloze.py:add_llm_cloze`. Both backends emit
`{"p": ..., "surprisal": ..., "top_match": 0|1}` per spaCy token.

## Papers
The exact "top-1 prediction match" framing is more common in NLP
benchmarking than in EEG papers, but the underlying notion — comparing
model argmax to human production / actual continuation — appears in:

- <a name="goldstein-2022"></a>**[Goldstein 2022]** Goldstein et al. *Shared computational principles for language processing in humans and deep language models.* Nature Neuroscience. [doi:10.1038/s41593-022-01026-4](https://doi.org/10.1038/s41593-022-01026-4) — uses next-word prediction agreement between humans and GPT-2 as a covariate against ECoG.
- <a name="schrimpf-2021"></a>**[Schrimpf 2021]** Schrimpf et al. *The neural architecture of language.* PNAS. [doi:10.1073/pnas.2105646118](https://doi.org/10.1073/pnas.2105646118) — top-N agreement as a complementary measure to per-token probability.

[Goldstein 2022]: #goldstein-2022
[Schrimpf 2021]: #schrimpf-2021

## Datasets
<!-- auto-populated from docs/datasets/*.md once the dataset registry exists -->
