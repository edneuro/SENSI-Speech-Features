---
key: pred_surprisal
name: Surprisal
levels: [word]
gui_toggle: pred_surprisal
output_columns: ["<slug>_Surprisal"]
libraries: [transformers, torch]
status: implemented
depth: deep
---

## Meaning
Per-word **surprisal** — the negative log probability that a language
model assigns to a word given everything that came before it in the
document, in **bits**:

    <slug>_Surprisal(wᵢ) = -log₂ P_model(wᵢ | w₁ … wᵢ₋₁)        [Hale 2001], [Levy 2008]

`<slug>` stands for the model's short name — `gpt2`, `gpt2_medium`,
`gpt2_large`, `gpt2_xl`, `llama32_3b`, `llama3_8b` or `llama3_70b` — and is the prefix of the column name, so
`gpt2_Surprisal` is surprisal from GPT-2. Each model you tick writes its own
column, and the name always says which model produced it.

Low surprisal means the model expected the word; high surprisal means it
did not (see *Reading the numbers* below for the scale). PUNCT rows are
`NaN`. The document's first word is scored too: every pass starts with the
model's beginning-of-text token, so it has a left context to condition on
(see *What each word's surprisal is conditioned on*).

## Reading the numbers
Surprisal has no upper limit and no fixed "surprising" cut-off, but every
value converts exactly to a probability: `P = 2^(−surprisal)`.

| Surprisal (bits) | Probability the model gave the word | Roughly "1 in …" |
|---|---|---|
| 0 | 1 | certain |
| 1 | 0.5 | 2 |
| 3.3 | 0.1 | 10 |
| 6.6 | 0.01 | 100 |
| 10 | 0.001 | 1,000 |
| 13.3 | 0.0001 | 10,000 |
| 20 | 0.000001 | 1,000,000 |

Each extra bit halves the probability, so 2 bits vs 4 bits is the same
step as 12 vs 14. Real values from the manual's example run (GPT-2):

- *She spread butter on her* **toast** → 5.9 bits; **violin** in the same
  frame → 16.1 bits.
- function words in an expected slot sit low: **on** after *spread butter*
  → 2.6 bits; **hours** after *48* → 0.79 bits.
- a content word after a short, open context sits high: **spread** after
  *She* → 14.3 bits, since almost any verb could follow.
- the document's first word is scored from the beginning-of-text token alone,
  so it is a "how likely is this word to open a text" value: **She** → 9.6
  bits, **Others** → 13.7 bits.
- a word the model did not see coming sits high even when it is ordinary
  English. In *Others who arrived just after the time of* **distribution**
  (LJ002-0321), GPT-2 gave *the* 50 % and **distribution** 13.7 bits;
  after *were often*, **48** gets 18.8 bits (GPT-2's top guess, *greeted*,
  had only 2 %), and **six** after *might also be* gets 12.1 bits. Writing
  the number out does not lower it: *forty-eight* in the same place costs
  22.5 bits, even less expected.

(The 50 %, the 2 % and the *forty-eight* figure are GPT-2 small probabilities
measured for this page with the beginning-of-text token, on the same
sentence; the bit values are from the manual's example run.)

Model surprisal on naturalistic reading corpora spans roughly 0–20 bits
([Wilcox 2020], Fig. 1), with most words well below the top of that
range. What counts as "high" depends on the model, the text and its
genre, so compare words **within** a stimulus (or z-score the column per
stimulus, as N400 studies do with surprisal predictors, [Michaelov 2022])
rather than against a fixed threshold. Values from different models are
on the same bits scale but are not interchangeable: a stronger model is
less surprised by text overall.

The metric itself is **identical across models** — only the estimator
(which language model computes `P`) changes. Enabling multiple models
writes one namespaced column per model (e.g. `gpt2_Surprisal` and
`llama3_8b_Surprisal` side by side), so a single run can directly
compare estimator quality on the same stimulus rather than committing
to one model.

## Models
Surprisal is available for every model in the roster
(`sensi/extractors/predictability_models.py`): the **GPT-2 family**
(`gpt2`, `gpt2_medium`, `gpt2_large`, `gpt2_xl`; free, no token) and three
gated **Llama-3** variants (`llama32_3b`, `llama3_8b`, `llama3_70b`) that
need a HuggingFace token + accepted license — see
**[LLM access & credentials](llm_credentials.md)**. Nothing is downloaded at
setup: a model's weights are fetched the first time a run uses it (the
console and the Run page say how much).

| Slug | Parameters | First-use download | Precision (default; choices) | Runs on | Weights in memory |
|---|---|---|---|---|---|
| `gpt2` | 124 M | 0.55 GB | fp32 only | CPU or GPU | 0.5 GB |
| `gpt2_medium` | 355 M | 1.52 GB | fp32 only | CPU or GPU | 1.4 GB |
| `gpt2_large` | 774 M | 3.25 GB | fp32 only | CPU or GPU | 3.1 GB |
| `gpt2_xl` | 1.5 B | 6.43 GB | fp32 only | CPU or GPU | 6.0 GB |
| `llama32_3b` | 3.21 B | 6.42 GB | fp16 (default), 8bit, 4bit | GPU (CPU: fp32) | 6.4 GB fp16 · 3.2 GB 8bit · 1.6 GB 4bit · 12.8 GB fp32 on CPU |
| `llama3_8b` | 8 B | 16.0 GB | fp16 (default), 8bit, 4bit | GPU (CPU: fp32) | 16 GB fp16 · 8 GB 8bit · 4 GB 4bit · 32 GB fp32 on CPU |
| `llama3_70b` | 71 B | 142 GB | fp16 (default), 8bit, 4bit | GPU (CPU: fp32) | 142 GB fp16 · 71 GB 8bit · 35.5 GB 4bit · 284 GB fp32 on CPU |

"Weights in memory" is parameters × bytes per parameter (fp32 4, fp16 2,
8bit 1, 4bit 0.5; `predictability_models.weights_gb`). It is the weights
alone: activations and the key-value cache come on top, so allow headroom.

Pick one or several from the GUI's Predictability card's model multi-select;
each selected model is scored once per stimulus and contributes its own
`<slug>_Surprisal` column when the Surprisal metric checkbox is on. Whether a
larger model's surprisal fits human data better depends on the measure and on
how large the models are — see *EEG rationale* below; the GPT-2 family is the
free, always-available option, and ticking several sizes writes one column
each so you can compare them on your own data.

### Precision
Each Llama row has a precision dropdown (fp16, 8bit, 4bit); the GPT-2 rows
are fixed at fp32. Stored per model in `prefs["predictability"]["precision"]`
(a slug absent from the map = the model's default).

- **GPT-2 stays fp32.** Measured on GPT-2 small in this project
  (2026-11-04): running it in fp16 moves surprisal by 0.0213 bits on average
  (largest change 0.1187), and on a CPU one 1024-subword window takes 82.31 s
  in fp16 against 0.55 s in fp32; bf16 moves surprisal by 0.2193 bits on
  average (largest 1.1432). So lower precision buys nothing for models this
  small, and fp32 is the value to cite.
- **Llama defaults to fp16**.
  The previous behaviour (silently loading 4-bit where it could) is gone:
  the precision that ran is always the one you chose and is recorded.
- **8-bit and 4-bit** shrink the memory a Llama needs (table above) at the
  cost of values that differ from fp16. The size of that shift was **not
  measured** in this project, so state the precision with any result that
  uses them. If an 8-/4-bit load fails (for example no working
  *bitsandbytes*), the run stops with a message that suggests fp16; it never
  falls back to another precision, because that would change the values
  without telling you.
- **No GPU → fp32 on the CPU for every model**, whatever precision is
  selected, because half precision is very slow on a CPU (the GPT-2 timing
  above). Llama-3-8B in fp32 needs about 32 GB of RAM, which is why the
  console warns first.

What is recorded: the console prints one line per model as it loads, e.g.
`[predict] gpt2: fp32 on CPU`; when your choice was overridden it first adds
`[predict] llama3_8b: no GPU — running fp32 on CPU (~32.0 GB RAM)` (followed
by the usual `[predict] llama3_8b: fp32 on CPU` load line), and
`[predict] <slug>: downloading X GB on first use…` before a first download.
The run manifest (`logs/*_manifest.json`) has a `predictability_models` list,
one `{slug, hf_id, precision, device}` entry per model the run used.

## EEG rationale
Surprisal is a well-replicated word-level predictor of **N400
amplitude** in naturalistic reading and listening ([Frank 2015],
[Michaelov 2024]) and of **reading time** ([Smith 2013],
[Goodkind 2018]). Surprisal Theory ([Hale 2001], [Levy 2008]) proposes
that processing cost is **linear in surprisal** — equivalently,
logarithmic in word probability. Each halving of a word's probability
adds one bit to its surprisal and a fixed increment to processing
cost, regardless of where on the probability scale the halving occurs.
[Michaelov 2024] further shows that language-model surprisal explains
N400 amplitude better than competing predictors such as cloze
probability and distributional semantic similarity.

[Heilbron 2022] used GPT-2-derived surprisal specifically to predict
EEG and source-localized MEG responses during naturalistic audiobook
listening, dissociating syntactic, phonemic and semantic prediction
signatures — the `gpt2_Surprisal` column is the one to use when
reproducing analyses of that kind.

Does a better language model give better surprisal? The literature
disagrees, and the disagreement matters for which model you tick:

- **Better is better (small models, reading time).** Across more than two
  dozen models trained on at most 42M tokens (test perplexity roughly
  30–500), lower perplexity went with surprisal that predicted reading
  times better ([Wilcox 2020]; also [Goodkind 2018]). Models that predict
  the next word better also predict more variance in fMRI / ECoG responses
  to language
  ([Schrimpf 2021], [Goldstein 2022]).
- **Bigger is worse (large models, reading time).** Across GPT-Neo and
  OPT models, surprisal from the **larger** transformers (lower
  perplexity) fit human reading times **worse** ([Oh 2023]): they predict
  named entities and other memorisable sequences better than people do,
  so their surprisal underestimates human difficulty there.
- **Bigger is better (large models, N400).** On EEG from 334 participants
  (94 sentences after filtering), surprisal from GPT-3 (175B parameters),
  RoBERTa and ALBERT fit N400 amplitude better than human cloze
  probability, while GPT-2 XL, Transformer-XL and two RNNs did not; within
  each model type the higher-quality model fit better ([Michaelov 2022]).

⚠️ The first two results cover different ranges of model quality, so they
need not contradict each other: the gain from a better model shows up in
Wilcox's small models and reverses only among Oh's much larger ones. The
third result does contradict the second for its measure — the largest
model was the best fit to the N400, not the worst — so the reversal seen
in reading times cannot be assumed for EEG. The default `gpt2` is the
124M-parameter GPT-2, smaller than the GPT-2 XL that did not beat cloze
in [Michaelov 2022]. Ticking a small and a large model on the same run
writes one column each, so you can test which estimator fits your own
neural data instead of assuming either direction.

## Math / linguistics
A causal language model outputs a probability distribution over its
vocabulary at every position. The probability of an actual
continuation token `wᵢ` given all prior tokens is read off this
distribution; its negative log₂ is the surprisal. When a spaCy token
maps to multiple model subwords (e.g. `"unbelievable"` →
`["un", "bel", "iev", "able"]`), their log-probabilities are **summed**
(equivalent to multiplying probabilities), since they jointly
constitute the word:

    P(w) = P(s₁) · P(s₂ | s₁) · ... · P(sₖ | s₁..sₖ₋₁)

so

    -log₂ P(w) = Σᵢ -log₂ P(sᵢ | s₁..sᵢ₋₁)        (chain rule of probability)

The sum is correct because every model in the roster is autoregressive
and conditions each subword on the prior context (including earlier
subwords of the same spaCy token). This holds identically whether the
backend is GPT-2 or a Llama-3 variant.

## Implementation in SENSI SpeechFeatures
Source: `sensi/extractors/predictability.py:add_predictability` (the unified,
roster-driven extractor); long-text handling is in
`_chunk_slices` / `_HuggingFaceBackend.score`.

**This also produces** [`Cloze`](pred_cloze.md) and
[`TopMatch`](pred_topmatch.md) from the same forward pass per model —
they're three views of the same per-token prediction. Enabling any of
the three metrics for a given model triggers that model's scoring
pass; cost (and runtime) for that model is the same whether you enable
one or all three of its metrics.

### What each word's surprisal is conditioned on
Each word `wᵢ` is scored against the model's distribution over the
next token **conditioned on its own preceding text** `w₁ … wᵢ₋₁` —
exactly the per-word, growing-prefix recipe the metric is defined by.

Because every roster model has **causal (left-only) attention**, one
forward pass over a span of subwords already produces a *separate*
next-token distribution at every position, each conditioned only on
the tokens to its left. So feeding the model `w₁ … w_N` once is
equivalent to running it `N` times with prefixes of growing length —
just faster. The per-position distributions are independent: nothing
is pooled across positions.

For the **very first** word of the document there is no preceding
text, so every forward pass starts with the model's **beginning-of-text
token** (`<|endoftext|>` for GPT-2, `<|begin_of_text|>` for Llama 3). The
token is never scored and belongs to no word; it only gives the first
subword a left context, so the first word gets a Surprisal, Cloze and
TopMatch like any other. ⚠️ That value is "how likely is this word to open
a text", not a continuation of anything, so you may still want to drop the
first word from a regression.

### Short transcripts (fit the model's context window): exact
Every model has a hard architectural context limit
(`model.config.max_position_embeddings`) — GPT-2's is 1024 subwords;
the Llama-3.2-3B/8B/70B variants are 131072/8192/8192 (see
`sensi/extractors/predictability_models.py`'s `native_ctx`). When the
transcript tokenises to at or under that limit for the model in use,
the whole document (plus the one beginning-of-text token) fits in one
forward pass and every word's surprisal is conditioned on its complete
natural preceding text. **No approximation.**

### Long transcripts (exceed the context window): chunked, bounded preceding context
Podcast episodes and audiobook chapters routinely exceed a model's
context window (most visibly GPT-2's 1024-subword limit, or the two
8K-context Llama-3 models) and cannot fit in a single forward pass.
`_HuggingFaceBackend.score` then walks the input via `_chunk_slices`.
Throughout this section, `limit` is the model's context limit (its native
window, or your cap below) and **`window = limit − 1`** is the number of
subwords a forward pass holds after the beginning-of-text token. The stride
(chunk size) is `window // 2`. Each forward pass is the beginning-of-text
token, one chunk and the immediately preceding chunk as left context — at
most `window` subwords plus the token, the most the model can hold. Only the logits for the chunk itself are kept; the
prefix logits are discarded and re-computed (with a longer prefix) on the
previous pass.

For a transcript about twice GPT-2's window (2000 subwords, `limit = 1024`,
so `window = 1023` and a stride of 511):

| Chunk being scored | Forward-pass input (after the BOS token) | Preceding context the chunk's first subword sees | Vs. ideal full-context |
|---|---|---|---|
| subwords 0 – 510     | `input[0:511]`     | none (grows naturally within chunk)     | identical |
| subwords 511 – 1021  | `input[0:1022]`    | 511 subwords (positions 0 – 510)        | identical |
| subwords 1022 – 1532 | `input[511:1533]`  | 511 subwords (positions 511 – 1021)     | positions 0 – 510 lost  |
| subwords 1533 – 1999 | `input[1022:2000]` | 511 subwords (positions 1022 – 1532)    | positions 0 – 1021 lost |

So every position past the first window is scored with **between
`window // 2` and `2 · (window // 2) − 1` preceding subwords** of context
(511 to 1021 for GPT-2),
depending on where it sits inside its chunk. The earliest subwords
drop out of view at every chunk boundary. This is the unavoidable cost
of a fixed context window on long texts — *which* subwords to drop is
the only choice.

Scoring inside each chunk remains correct: causal attention guarantees
each position's conditional distribution depends only on its own
(possibly-truncated) left context.

### Per-model context size (optional cap)
Each model row in the Predictability card has a **context** spinner:
the maximum context (subwords) fed per forward pass, defaulting to the
model's full native window (`native_ctx`). Lowering it makes the
chunking above kick in sooner — each word conditions on less preceding
text, which is **faster and uses less memory on long stimuli**, and
lets you fix a common context length across models for comparability.
The cap counts the beginning-of-text token, so a window holds the token plus
up to `cap − 1` subwords. The cap is clamped to `[128, native_ctx]` (helper
`predictability_models.effective_context`); the effective window is
`min(cap, model's real max)`, so it never exceeds what the model can
hold. Stored per model in `prefs["predictability"]["context"]` (a slug
absent from the map = uncapped = the default).

> ⚠️ Lowering a model's context **changes** the emitted `<slug>_Surprisal`
> / `_Cloze` values for stimuli longer than the cap (each word sees less
> history) — by design, and only when you opt in. Leave it at the
> default to reproduce the full-context values.

### Why a `window // 2` stride
The stride trades preceding-context budget against forward-pass count
(`window = limit − 1`, as defined above):

| Stride | Forward passes for N subwords | Minimum preceding context past the first window |
|---|---|---|
| `window` (no overlap) | ≈ N / window | 0 subwords at every chunk boundary — first position of each chunk has no context, yielding spurious surprisal |
| **`window // 2` (current)** | ≈ N / (window / 2) | `window // 2` subwords |
| 1 (true sliding window) | ≈ N | `window - 1` subwords everywhere |

A 1-subword stride would maximise context but costs roughly `window /
2`× more passes than the current setting. A causal LM's per-token loss
flattens well before its full context window is consumed, so the
marginal accuracy gain from a smaller stride is small. The `window //
2` stride trades up to `window // 2 - 1` subwords of additional
context for a `window // 2`× compute reduction. For GPT-2 this is a
511-subword stride against its 1023-subword window (1024 minus the
beginning-of-text token); the same rule applies to every roster model via its
own `window`.

### Step by step
1. `_get_backend` / `_cached_backend` lazily load an
   `AutoModelForCausalLM` + `AutoTokenizer` for the model's HuggingFace
   repo id (`gpt2`, `openai-community/gpt2-medium` … for the GPT-2
   slugs, `meta-llama/...` for the Llama slugs) at the precision and on
   the device `load_plan` picks (see *Precision*), cached process-wide —
   a load costs minutes for the larger Llamas, so it happens once per run,
   shared across the word-level and document-level passes. A model that is
   not yet in `data/hf_cache/` is downloaded now, with a console line.
2. `_HuggingFaceBackend.score` tokenises the **whole transcript** with
   character offsets, so every subword carries a `(char_start,
   char_end)` span, and runs one forward pass (or the chunked sliding
   window above) to get per-subword log-probabilities and top-1
   predictions. Each pass is fed `[BOS] + subwords`; the logits at the BOS
   position score the first subword.
3. `map_tokens_to_subwords` walks the spaCy tokens in document order
   and assigns each subword to exactly ONE token — the first token
   whose character span it overlaps (a subword straddling a token
   boundary, e.g. `"don"` over spaCy `"do"` + `"n't"`, is never charged
   twice). Per-token log-probabilities are summed and converted to
   bits (`-Σ log_e P / log_e 2`) — the joint log-probability of the
   word; see *Math / linguistics*.
4. `add_predictability` writes the result to `<slug>_Surprisal` (and,
   if requested, `<slug>_Cloze` / `<slug>_TopMatch`) for each selected
   model.
5. Rows with `upos == "PUNCT"` are masked to `NaN`.
6. Scores are memoized per `(hf_id, text/tokens digest)`: when the
   word-level and document-level batches process the same stimulus in
   one run, each model runs its forward pass once and the second pass
   is a cache hit.

**GPT-2 models always run fp32**: their only offered precision is fp32, so
with a GPU they run fp32 on it and without one fp32 on the CPU. The
precision a Llama runs at is the one chosen (fp16 by default) on a GPU, and
fp32 on the CPU otherwise (see *Precision*).

### Why (design notes)
- *The first word is scored, from a beginning-of-text token*: without any
  left context the model cannot assign its first subword a probability (an
  earlier version stored `0` bits, faking a "perfectly predictable" word,
  and then `NaN`). Starting every window with the model's own
  beginning-of-text token is how these models were trained to see the
  start of a text, so the first word gets an honest "probability of opening
  a text with this word". Windows after the first one start with the token
  too, so the context a word sees is the same no matter which chunk it
  falls in. ⚠️ Values for all words differ from runs from
  before 2026-11-04, because every position now has the token in its
  context.
- *PUNCT → `NaN`*: language models do assign probabilities to
  punctuation, but ERP work on predictability is "usually investigated
  on content words only", and in [Frank 2015] the surprisal effect on the
  N400 came mostly from content words; punctuation is not a separate item
  in those analyses. The mask is applied here so
  downstream code doesn't accidentally include punctuation rows;
  remove the mask in `add_predictability` if you specifically want
  punctuation surprisal.
- *One backend implementation for every model*: GPT-2 loads through
  `AutoModelForCausalLM` exactly like the Llama models (it resolves to
  the `GPT2LMHeadModel` class) — there is exactly one scoring path, used
  by every roster model, instead of a GPT-2-specific extractor and a
  separate LLM extractor.

## Library
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, math

model_id = "gpt2"   # or e.g. "meta-llama/Llama-3.2-3B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model     = AutoModelForCausalLM.from_pretrained(model_id)
model.eval()

enc = tokenizer("The cat sat on the mat.", return_tensors="pt")
with torch.no_grad():
    logits = model(enc["input_ids"]).logits[0]
log_probs = torch.log_softmax(logits, dim=-1)
# Surprisal of token i is -log2 P(token_i | tokens_0..i-1)
ids = enc["input_ids"][0].tolist()
for i in range(1, len(ids)):
    lp = log_probs[i - 1, ids[i]].item()
    print(tokenizer.decode([ids[i]]), -lp / math.log(2), "bits")
```

## Papers

**Surprisal theory:**
- <a name="hale-2001"></a>**[Hale 2001]** Hale. *A probabilistic Earley parser as a psycholinguistic model.* NAACL 2001. [PDF](https://aclanthology.org/N01-1021/) — introduces surprisal as a cognitive complexity metric.
- <a name="levy-2008"></a>**[Levy 2008]** Levy. *Expectation-based syntactic comprehension.* Cognition. [doi:10.1016/j.cognition.2007.05.006](https://doi.org/10.1016/j.cognition.2007.05.006) — surprisal theory formalized; linearity prediction.

**Surprisal ↔ N400 / reading time:**
- <a name="smith-2013"></a>**[Smith 2013]** Smith & Levy. *The effect of word predictability on reading time is logarithmic.* Cognition 128(3), 302–319. [doi:10.1016/j.cognition.2013.02.013](https://doi.org/10.1016/j.cognition.2013.02.013) — empirical support for the linear-surprisal prediction in reading time.
- <a name="frank-2015"></a>**[Frank 2015]** Frank, Otten, Galli, Vigliocco. *The ERP response to the amount of information conveyed by words in sentences.* Brain & Language 140, 1–11. [doi:10.1016/j.bandl.2014.10.006](https://doi.org/10.1016/j.bandl.2014.10.006) — N400 amplitude ∝ surprisal in sentence reading.
- <a name="goodkind-2018"></a>**[Goodkind 2018]** Goodkind & Bicknell. *Predictive power of word surprisal for reading times is a linear function of language model quality.* CMCL 2018, 10–18. [ACL Anthology W18-0102](https://aclanthology.org/W18-0102/) — better LMs give better surprisal estimates for reading time.
- <a name="michaelov-2024"></a>**[Michaelov 2024]** Michaelov, Bardolph, Van Petten, Bergen, Coulson. *Strong Prediction: Language Model Surprisal Explains Multiple N400 Effects.* Neurobiology of Language 5(1), 107–135. [doi:10.1162/nol_a_00105](https://doi.org/10.1162/nol_a_00105) — direct comparison: LM surprisal explains N400 amplitude better than cloze probability or distributional semantic similarity.

**Naturalistic EEG/MEG using GPT-2 surprisal:**
- <a name="heilbron-2022"></a>**[Heilbron 2022]** Heilbron, Armeni, Schoffelen, Hagoort, De Lange. *A hierarchy of linguistic predictions during natural language comprehension.* PNAS 119(32), e2201968119. [doi:10.1073/pnas.2201968119](https://doi.org/10.1073/pnas.2201968119) — GPT-2 surprisal regressed against EEG and source-localized MEG during naturalistic audiobook listening; dissociates syntactic, phonemic, and semantic prediction signatures.

**Surprisal ↔ LM quality:**
- <a name="wilcox-2020"></a>**[Wilcox 2020]** Wilcox, Gauthier, Hu, Qian, Levy. *On the Predictive Power of Neural Language Models for Human Real-Time Comprehension Behavior.* CogSci 2020. [arXiv:2006.01912](https://arxiv.org/abs/2006.01912) — across architectures, better next-word prediction correlates with stronger prediction of human reading times and eye movements.

- <a name="michaelov-2022"></a>**[Michaelov 2022]** Michaelov, Coulson, Bergen. *So Cloze yet so Far: N400 Amplitude is Better Predicted by Distributional Information than Human Predictability Judgements.* IEEE Transactions on Cognitive and Developmental Systems. [doi:10.1109/TCDS.2022.3176783](https://doi.org/10.1109/TCDS.2022.3176783) ([arXiv:2109.01226](https://arxiv.org/abs/2109.01226)) — on the Nieuwland et al. (2018) EEG data, GPT-3, RoBERTa and ALBERT surprisal fit N400 amplitude better than human cloze; smaller LMs (incl. GPT-2 XL) did not.

- <a name="oh-2023"></a>**[Oh 2023]** Oh & Schuler. *Why Does Surprisal From Larger Transformer-Based Language Models Provide a Poorer Fit to Human Reading Times?* Transactions of the ACL 11, 336–350. [doi:10.1162/tacl_a_00548](https://doi.org/10.1162/tacl_a_00548) — across five GPT-Neo and eight OPT sizes, lower perplexity goes with a poorer fit to reading times; larger models "memorize" sequences such as named entities.

**Modern LM ↔ neural data:**
- <a name="schrimpf-2021"></a>**[Schrimpf 2021]** Schrimpf et al. *The neural architecture of language: Integrative modeling converges on predictive processing.* PNAS. [doi:10.1073/pnas.2105646118](https://doi.org/10.1073/pnas.2105646118)
- <a name="caucheteux-2022"></a>**[Caucheteux 2022]** Caucheteux & King. *Brains and algorithms partially converge in natural language processing.* Communications Biology. [doi:10.1038/s42003-022-03036-1](https://doi.org/10.1038/s42003-022-03036-1)
- <a name="goldstein-2022"></a>**[Goldstein 2022]** Goldstein et al. *Shared computational principles for language processing in humans and deep language models.* Nature Neuroscience. [doi:10.1038/s41593-022-01026-4](https://doi.org/10.1038/s41593-022-01026-4)

[Hale 2001]: #hale-2001
[Levy 2008]: #levy-2008
[Smith 2013]: #smith-2013
[Frank 2015]: #frank-2015
[Goodkind 2018]: #goodkind-2018
[Michaelov 2024]: #michaelov-2024
[Heilbron 2022]: #heilbron-2022
[Wilcox 2020]: #wilcox-2020
[Michaelov 2022]: #michaelov-2022
[Oh 2023]: #oh-2023
[Schrimpf 2021]: #schrimpf-2021
[Caucheteux 2022]: #caucheteux-2022
[Goldstein 2022]: #goldstein-2022

## Datasets
<!-- auto-populated from docs/datasets/*.md once the dataset registry exists -->
