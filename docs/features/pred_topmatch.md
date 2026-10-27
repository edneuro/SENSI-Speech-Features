---
key: pred_topmatch
name: Top-1 prediction match
levels: [word]
gui_toggle: pred_topmatch
output_columns: ["<slug>_TopMatch"]
libraries: [transformers, torch]
status: implemented
depth: shallow
---

> GPT-2 needs no credentials. The gated Llama-3 roster models need a
> HuggingFace token before they run — see
> **[LLM access & credentials](llm_credentials.md)**.

## Meaning
Binary flag (`0` or `1`) indicating whether a selected model's **most
likely next word** at this position matched the actual word in the
transcript:

    <slug>_TopMatch(wᵢ) = 1   if  argmaxⱼ P_model(j | w₁ … wᵢ₋₁) == subword(wᵢ)
                          0   otherwise

Comparison is made at the **first subword** of the spaCy token (model
subwords don't always align 1:1 with spaCy tokens). PUNCT rows are
`NaN`, and so is the document's first word — it has no preceding
context, so the model makes no prediction for it. Enabling Top-match
for several roster models writes one column per model, e.g.
`gpt2_TopMatch` and `llama3_8b_TopMatch` side by side.

## EEG rationale
Whereas [`<slug>_Cloze`](pred_cloze.md) and
[`<slug>_Surprisal`](pred_surprisal.md) are graded measures of
predictability, `<slug>_TopMatch` is a **discrete** prediction-correctness
flag. It's useful when:

- The downstream model wants a binary regressor — e.g. an "expected"
  vs "violation" contrast — without picking a probability threshold.
- You want to count *strict-prediction hits* across a stimulus
  (proportion of positions where the model nailed the next word).
- You're comparing against psycholinguistic experiments that operate
  on "predictable" vs "unpredictable" rather than continuous cloze.

For ERP-amplitude regressions, the graded
[`<slug>_Surprisal`](pred_surprisal.md) is the better default.
`<slug>_TopMatch` is binary by design, so it discards everything below the
top guess: every word the model did not rank first scores the same, yet N400
facilitation is graded even among unpredictable words [Szewczyk 2022]. No
paper we know of tests top-match against surprisal directly; this is a
recommendation, not a measured result.

## Implementation in SensiExtractor
Source: `extractors/predictability.py:add_predictability` (the
unified, roster-driven extractor — `extractors/llm_cloze.py` is now a
thin compatibility shim over this module) and
`_HuggingFaceBackend.score`.

Computed in the same forward pass as `<slug>_Cloze` and
`<slug>_Surprisal` for a given model: per subword, the backend already
computes `argmax(log_probs)` over the vocabulary. For each spaCy
token, the flag is set by comparing the argmax at the token's **first
subword** position to the actual id at that position:

```python
top_match = 1 if sub_top1[first_sub] == ids[first_sub] else 0
```

The comparison is done at the token's first subword, in vocabulary-id
space (`argmax` id vs. the actual next-token id) from the local model's
logits.

`upos == "PUNCT"` → `NaN`, and the document's first word is `NaN` too
(no preceding context, so no prediction to match). The output is stored
as `int` for `0` / `1` and remains `NaN` (a `Nullable Int64` view in
pandas) for those rows.

## Library
See `extractors/predictability.py:_HuggingFaceBackend.score`. Every
backend — GPT-2 included, via `AutoModelForCausalLM` — emits
`{"p": ..., "surprisal": ..., "top_match": 0|1}` per spaCy token.

## Papers
The exact "top-1 prediction match" framing is more common in NLP
benchmarking than in EEG papers. The closest neuroscience precedent
is work that links transformer next-word predictions to brain
responses during natural listening:

- <a name="goldstein-2022"></a>**[Goldstein 2022]** Goldstein et al. *Shared computational principles for language processing in humans and deep language models.* Nature Neuroscience 25(3), 369–380. [doi:10.1038/s41593-022-01026-4](https://doi.org/10.1038/s41593-022-01026-4) — ECoG study showing that humans and autoregressive transformer models share next-word prediction as a computational principle during natural language processing.

- <a name="szewczyk-2022-tm"></a>**[Szewczyk 2022]** Szewczyk & Federmeier. *Context-based facilitation of semantic access follows both logarithmic and linear functions of stimulus probability.* Journal of Memory and Language 123, 104311. [doi:10.1016/j.jml.2021.104311](https://doi.org/10.1016/j.jml.2021.104311) — N400 facilitation "combines linear and logarithmic functions"; graded even among unpredictable words.

[Goldstein 2022]: #goldstein-2022
[Szewczyk 2022]: #szewczyk-2022-tm

## Datasets
<!-- auto-populated from docs/datasets/*.md once the dataset registry exists -->
