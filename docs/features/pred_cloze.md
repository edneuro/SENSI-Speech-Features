---
key: pred_cloze
name: Cloze probability (model estimate)
levels: [word]
gui_toggle: pred_cloze
output_columns: ["<slug>_Cloze"]
libraries: [transformers, torch]
status: implemented
depth: deep
---

## Meaning
**Cloze in the literature.** A word's cloze probability is the
proportion of people who, shown a sentence up to a gap, fill the gap with
that word ([Taylor 1953]). If 18 of 30 participants write *toast* after
*She spread butter on her …*, *toast* has a cloze probability of 0.6.

**What this column is.** No participants are involved here:
`<slug>_Cloze` is a **language-model estimate used in place of cloze**,
the probability the model gives to the word that actually comes next. It
is named after cloze because it answers the same question (how expected
is this word, on a 0–1 scale), but it is a different quantity:

- Human cloze counts each person's single best guess; the model spreads
  its probability over its whole vocabulary, so its values for the same
  words tend to be lower. On the target nouns of a large N400 study, the
  mean GPT-2 (XL) probability was 0.38 against a mean human cloze of 0.63
  ([Michaelov 2022]).
- A word nobody writes down has a cloze probability of exactly 0. The
  model never gives exactly 0: *violin* in *She spread butter on her
  violin* gets `gpt2_Cloze` = 0.000008 in the manual's example run.
- The closest model analogue of "most people would guess this word" is
  [`<slug>_TopMatch`](pred_topmatch.md): whether the word was the model's
  top prediction for the next word.
- Which of the two better predicts behaviour or the brain is an empirical
  question. Model probabilities predicted eye-tracking reading times
  better than human cloze ([Hofmann 2022]), and surprisal from large
  models fit N400 amplitude better than human cloze ([Michaelov 2022]).

If your study has human cloze norms for its stimuli, they are a separate
measure; this column does not reproduce them.

The column itself is the model's conditional probability of each word
given the preceding context, i.e. the raw P, not its negative log:

    <slug>_Cloze(wᵢ) = P_model(wᵢ | w₁ … wᵢ₋₁)    ∈ (0, 1]

Mathematical exponential inverse of [`<slug>_Surprisal`](pred_surprisal.md)
(which is defined as `-log₂ P` [Hale 2001]), so

    <slug>_Cloze = 2^(-<slug>_Surprisal)

`<slug>` stands for the model's short name — `gpt2`, `gpt2_medium`,
`gpt2_large`, `gpt2_xl`, `llama32_3b`, `llama3_8b` or `llama3_70b` — and is the prefix of the column name, so
`gpt2_Cloze` is the probability from GPT-2. Each model you tick writes its own
column, and the name always says which model produced it.

PUNCT rows are `NaN`. The document's first word is scored: every forward
pass starts with the model's beginning-of-text token (see
[Surprisal](pred_surprisal.md#what-each-words-surprisal-is-conditioned-on)),
so it has a left context. `<slug>_Cloze`, `<slug>_Surprisal`
and `<slug>_TopMatch` all come from the same forward pass for a given
model — enabling any one of the three metrics for a model triggers
that model's scoring pass. Enabling Cloze for several roster models at
once (see [Models](pred_surprisal.md#models)) writes one column per
model, e.g. `gpt2_Cloze` and `llama3_8b_Cloze` side by side.

## EEG rationale
The N400 was originally characterized as inversely proportional to
(human) cloze probability ([Kutas 1984], [DeLong 2005]).

`<slug>_Cloze` is the **modern machine analogue** — a model's
conditional probability over the actual continuation. Where the
relationship to the N400 is concerned:

- For *amplitude* prediction, **surprisal** (the log) is the usual
  regressor. The log form is established for reading times
  ([Smith 2013]), and GPT-3 surprisal explained single-trial N400
  amplitude better than cloze probability ([Michaelov 2024]). For the
  N400 the shape is not settled, though: [Szewczyk 2022] find the
  facilitation is linear in probability for expected words and
  logarithmic for unexpected ones, with a combination fitting best, so
  surprisal and probability each capture part of it.
- For comparison with classic human-cloze experiments, the raw
  probability is what aligns with the published metric.
- For analyses that need an interpretable [0–1] regressor (logistic
  models, threshold-based contrasts of "predictable" vs "not"), the
  probability form is more natural than bits.

When in doubt, use [`<slug>_Surprisal`](pred_surprisal.md) for
ERP-amplitude modelling (the more common choice) and `<slug>_Cloze` when you
specifically need the cloze metric, or enter both if you want to test the
linear-plus-log shape of [Szewczyk 2022].

## Implementation in SENSI SpeechFeatures
Source: `sensi/extractors/predictability.py:add_predictability` (the
unified, roster-driven extractor). Backend:
`_HuggingFaceBackend`, built on `AutoModelForCausalLM` for every
roster model (GPT-2 included) — local Hugging Face weights only, no
OpenAI backend; see [LLM access & credentials](llm_credentials.md).

See [`Surprisal` § Implementation](pred_surprisal.md#implementation-in-sensi-speechfeatures)
— `<slug>_Cloze` is computed in the same scoring call as
`<slug>_Surprisal` and [`<slug>_TopMatch`](pred_topmatch.md), once per
selected model. Per spaCy token:

```python
p = math.exp(sum(subword_log_probs))      # ← this column
surprisal = -sum(subword_log_probs) / math.log(2)
top_match = (argmax_subword == actual_subword)  # at the first subword
```

PUNCT rows are masked to `NaN`; the first word is scored (beginning-of-text
token). The probability is stored to 6
**significant digits** (e.g. `0.0123457`, or `2.07e-12` for a very
unlikely word), so it is never rounded down to exactly 0 and
`Cloze = 2^(−Surprisal)` holds for every word (rounding to 6 decimal places
would turn any word above about 20 bits of surprisal into `0.0`).

## Papers

**Cloze procedure & N400:**
- <a name="hale-2001"></a>**[Hale 2001]** Hale. *A probabilistic Earley parser as a psycholinguistic model.* NAACL 2001. [ACL Anthology N01-1021](https://aclanthology.org/N01-1021/) — defines surprisal as −log P of a word given its context.
- <a name="taylor-1953"></a>**[Taylor 1953]** Taylor. *"Cloze procedure": A new tool for measuring readability.* Journalism Quarterly. [doi:10.1177/107769905303000401](https://doi.org/10.1177/107769905303000401) — original cloze methodology.
- <a name="kutas-1984"></a>**[Kutas 1984]** Kutas & Hillyard. *Brain potentials during reading reflect word expectancy and semantic association.* Nature. [doi:10.1038/307161a0](https://doi.org/10.1038/307161a0) — N400 ∝ −cloze probability.
- <a name="delong-2005"></a>**[DeLong 2005]** DeLong, Urbach, Kutas. *Probabilistic word pre-activation during language comprehension inferred from electrical brain activity.* Nature Neuroscience. [doi:10.1038/nn1504](https://doi.org/10.1038/nn1504)
- <a name="smith-2013"></a>**[Smith 2013]** Smith & Levy. *The effect of word predictability on reading time is logarithmic.* Cognition. [doi:10.1016/j.cognition.2013.02.013](https://doi.org/10.1016/j.cognition.2013.02.013) — reading-time cost is logarithmic in predictability over six orders of magnitude (reading times, not ERPs).

**LM-derived cloze:**
- <a name="hofmann-2022"></a>**[Hofmann 2022]** Hofmann, Remus, Biemann, Radach, Kuchinke. *Language models explain word reading times better than empirical predictability.* Frontiers in Artificial Intelligence. [doi:10.3389/frai.2021.730570](https://doi.org/10.3389/frai.2021.730570) — LM probabilities outperform human cloze-completion probabilities at predicting fixation durations and gaze durations in English and German eye-tracking corpora.
- <a name="michaelov-2022"></a>**[Michaelov 2022]** Michaelov, Coulson, Bergen. *So Cloze yet so Far: N400 Amplitude is Better Predicted by Distributional Information than Human Predictability Judgements.* IEEE Transactions on Cognitive and Developmental Systems. [doi:10.1109/TCDS.2022.3176783](https://doi.org/10.1109/TCDS.2022.3176783) ([arXiv:2109.01226](https://arxiv.org/abs/2109.01226)) — on the Nieuwland et al. (2018) EEG data, GPT-3, RoBERTa and ALBERT surprisal fit N400 amplitude better than human cloze; reports mean model probabilities and human cloze for the same target nouns.
- <a name="michaelov-2024"></a>**[Michaelov 2024]** Michaelov, Bardolph, Van Petten, Bergen, Coulson. *Strong Prediction: Language Model Surprisal Explains Multiple N400 Effects.* Neurobiology of Language 5(1), 107–135. [doi:10.1162/nol_a_00105](https://doi.org/10.1162/nol_a_00105) — direct comparison: LM surprisal explains N400 amplitude better than cloze probability or distributional semantic similarity.
- <a name="szewczyk-2022"></a>**[Szewczyk 2022]** Szewczyk & Federmeier. *Context-based facilitation of semantic access follows both logarithmic and linear functions of stimulus probability.* Journal of Memory and Language 123, 104311. [doi:10.1016/j.jml.2021.104311](https://doi.org/10.1016/j.jml.2021.104311) — N400 facilitation "combines linear and logarithmic functions"; graded even among unpredictable words.

[Hale 2001]: #hale-2001
[Taylor 1953]: #taylor-1953
[Kutas 1984]: #kutas-1984
[DeLong 2005]: #delong-2005
[Smith 2013]: #smith-2013
[Hofmann 2022]: #hofmann-2022
[Michaelov 2022]: #michaelov-2022
[Michaelov 2024]: #michaelov-2024
[Szewczyk 2022]: #szewczyk-2022

## Getting access & credentials

The GPT-2 models need no credentials — they are free and ungated. The gated Llama-3
models need a HuggingFace token + license approval before they run.
This is shared with `<slug>_Surprisal` and `<slug>_TopMatch` — see
**[LLM access & credentials](llm_credentials.md)** for the full step-by-step.

## Datasets
<!-- auto-populated from docs/datasets/*.md once the dataset registry exists -->
