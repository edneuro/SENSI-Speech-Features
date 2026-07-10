---
key: pred_surprisal_gpt2
name: GPT-2 surprisal
levels: [word]
gui_toggle: pred_surprisal_gpt2
output_columns: [Surprisal]
libraries: [transformers, torch]
status: implemented
depth: deep
---

## Meaning
Per-word **surprisal** — the negative log probability that GPT-2
assigns to a word given everything that came before it in the document,
in **bits**:

    Surprisal(wᵢ) = -log₂ P(wᵢ | w₁ … wᵢ₋₁)

A high-frequency word in a strongly constraining context has low
surprisal (≈ 1–3 bits); a low-frequency or contextually unexpected word
has high surprisal (≈ 8–20 bits). PUNCT rows are `NaN`, and so is the
document's first word (no left context to condition on).

`Surprisal` is the **legacy GPT-2 baseline** column. The
[`LLM_Surprisal`](pred_surprisal_llm.md) column carries the same metric
computed with a modern LLM (GPT-4o-mini / Llama-3 family) and is a
better predictor in absolute terms — but the GPT-2 column is kept for
backward compatibility and cross-study comparability.

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
signatures. `Surprisal` is the column to use when reproducing analyses
of that kind.

## Math / linguistics
GPT-2 outputs a probability distribution over its vocabulary at every
position. The probability of an actual continuation token `wᵢ` given
all prior tokens is read off this distribution; its negative log₂ is
the surprisal. When a SpaCy token maps to multiple GPT-2 subwords (e.g.
`"unbelievable"` → `["un", "bel", "iev", "able"]`), their
log-probabilities are **summed** (equivalent to multiplying
probabilities), since they jointly constitute the word:

    P(w) = P(s₁) · P(s₂ | s₁) · ... · P(sₖ | s₁..sₖ₋₁)

so

    -log₂ P(w) = Σᵢ -log₂ P(sᵢ | s₁..sᵢ₋₁)

The sum is correct because GPT-2 is autoregressive and conditions each
subword on the prior context (including earlier subwords of the same
spaCy token).

## Implementation in SensiExtractor
Source: `extractors/surprisal.py:add_surprisal`; long-text handling
is in `_chunked_logits`.

### What each word's surprisal is conditioned on
Each word `wᵢ` is scored against GPT-2's distribution over the next
token **conditioned on its own preceding text** `w₁ … wᵢ₋₁` — exactly
the per-word, growing-prefix recipe the metric is defined by.

Because GPT-2 has **causal (left-only) attention**, one forward pass
over a span of subwords already produces a *separate* next-token
distribution at every position, each conditioned only on the tokens
to its left. So feeding the model `w₁ … w_N` once is equivalent to
running it `N` times with prefixes of growing length — just faster.
The per-position distributions are independent: nothing is pooled
across positions.

For the **very first** word of the document there is no preceding
text to condition on; the extractor stores `0` as a sentinel (not a
meaningful surprisal — drop or skip that row in regressions).

### Short transcripts (≤ 1024 subwords): exact
GPT-2's hard architectural limit is **1024 subwords per forward pass** —
positional embeddings aren't defined past that. When the transcript
tokenises to ≤ 1024 GPT-2 subwords, the whole document fits in one
pass and every word's surprisal is conditioned on its complete natural
preceding text. **No approximation.**

### Long transcripts (> 1024 subwords): chunked, bounded preceding context
Podcast episodes and audiobook chapters routinely exceed 1024
subwords and cannot fit in a single forward pass. `_chunked_logits`
walks the input in **512-subword steps**. Each forward pass takes one
512-subword chunk plus the immediately preceding 512 subwords as left
context — exactly the 1024 the model can hold. Only the logits for
the chunk itself are kept; the prefix logits are discarded and
re-computed (with a longer prefix) on the previous pass.

For a 2000-subword transcript:

| Chunk being scored | Forward-pass input | Preceding context the chunk's first subword sees | Vs. ideal full-context |
|---|---|---|---|
| subwords 0 – 511     | `input[0:512]`     | none (grows naturally within chunk)     | identical |
| subwords 512 – 1023  | `input[0:1024]`    | 512 subwords (positions 0 – 511)        | identical |
| subwords 1024 – 1535 | `input[512:1536]`  | 512 subwords (positions 512 – 1023)     | positions 0 – 511 lost  |
| subwords 1536 – 1999 | `input[1024:2000]` | 512 subwords (positions 1024 – 1535)    | positions 0 – 1023 lost |

So every position past subword 1024 is scored with **between 512 and
1023 preceding subwords** of context, depending on where it sits
inside its chunk. The earliest subwords drop out of view at every
chunk boundary. This is the unavoidable cost of GPT-2's fixed window
on long texts — *which* subwords to drop is the only choice.

Scoring inside each chunk remains correct: causal attention
guarantees each position's conditional distribution depends only on
its own (possibly-truncated) left context.

### Why a 512-subword stride
The stride trades preceding-context budget against forward-pass
count:

| Stride | Forward passes for N subwords | Minimum preceding context past subword 1024 |
|---|---|---|
| 1024 (no overlap) | ≈ N/1024 | 0 subwords at every chunk boundary — first position of each chunk has no context, yielding spurious surprisal |
| **512 (current)** | ≈ N/512  | 512 subwords |
| 1 (true sliding window) | ≈ N | 1023 subwords everywhere |

A 1-subword stride would maximise context but costs ~512× more passes
than the current setting. GPT-2's per-token loss flattens well before
its full context window is consumed, so the marginal accuracy gain is
small. The 512 stride trades up to 511 subwords of additional context
for a ~512× compute reduction.

### Step by step
1. `_load()` lazily fetches `gpt2` and its tokenizer
   (`GPT2LMHeadModel`, `GPT2TokenizerFast`), cached in module
   globals.
2. `_compute_surprisals` tokenises the **whole transcript** with
   character offsets, so every subword carries a
   `(char_start, char_end)` span.
3. `_subword_log_probs` returns one log P(subwordᵢ | context) per
   subword via a single forward pass (≤ 1024 subwords) or via
   `_chunked_logits` otherwise.
4. `map_tokens_to_subwords` walks the spaCy tokens in document order
   and assigns each GPT-2 subword to exactly ONE token — the first
   token whose character span it overlaps (a subword straddling a
   token boundary, e.g. GPT-2 `"don"` over spaCy `"do"` + `"n't"`, is
   never charged twice). `_token_surprisals` sums each token's subword
   log-probabilities and converts to bits (`-log_e P / log_e 2`) —
   the joint log-probability of the word; see *Math / linguistics*.
5. Rows with `upos == "PUNCT"` are masked to `NaN`.
6. Results are memoized per (text, token sequence): when the
   word-level and document-level batches process the same stimulus in
   one run, GPT-2 runs once and the second pass is a cache hit.

### Why (design notes)
- *`NaN` for the first word*: its first subword has no left context,
  so GPT-2 cannot assign it a probability. Emitting `0` bits there
  (the pre-2026-07 behavior) faked a "perfectly predictable" first
  word; `NaN` follows the pipeline-wide no-value convention. Any word
  whose subwords include position 0 is affected (i.e. exactly the
  document-initial word).
- *PUNCT → `NaN`*: GPT-2 does assign probabilities to punctuation,
  but surprisal regressions in the EEG literature are typically over
  content words. ⚠️ The mask is applied here so downstream code
  doesn't accidentally include punctuation rows; remove the mask in
  `add_surprisal` if you specifically want punctuation surprisal.

## Library
```python
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import torch, math

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
model     = GPT2LMHeadModel.from_pretrained("gpt2")
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

[Hale 2001]: #hale-2001
[Levy 2008]: #levy-2008
[Smith 2013]: #smith-2013
[Frank 2015]: #frank-2015
[Goodkind 2018]: #goodkind-2018
[Michaelov 2024]: #michaelov-2024
[Heilbron 2022]: #heilbron-2022

## Datasets
<!-- auto-populated from docs/datasets/*.md once the dataset registry exists -->
