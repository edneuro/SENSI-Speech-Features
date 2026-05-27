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
has high surprisal (≈ 8–20 bits). PUNCT rows are `NaN`.

`Surprisal` is the **legacy GPT-2 baseline** column. The
[`LLM_Surprisal`](pred_surprisal_llm.md) column carries the same metric
computed with a modern LLM (GPT-4o-mini / Llama-3 family) and is a
better predictor in absolute terms — but the GPT-2 column is kept for
backward compatibility and cross-study comparability.

## EEG rationale
Surprisal is the most-cited word-level predictor of N400 amplitude in
naturalistic reading and listening ([Frank 2015], [Smith 2013],
[Goodkind 2018]). Surprisal Theory ([Hale 2001], [Levy 2008]) proposes
that processing cost is **linear in surprisal**: doubling improbability
doubles the cognitive cost of integrating a word into the unfolding
context. The N400 has been shown to track surprisal more tightly than
raw probability ([Wlotko 2012]).

The Brennan group ([Brennan 2016]) and the Hale group ([Hale 2018])
have used GPT-2-derived surprisal specifically to predict MEG / EEG
during naturalistic story listening, and `Surprisal` is the column to
use when reproducing those analyses.

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
Source: `extractors/surprisal.py:add_surprisal`. Sliding-window for
long texts is in `_chunked_logits`.

**What the code does:**

1. Loads `gpt2` and its tokenizer once per process (`_load`, cached in
   module globals).
2. Tokenizes the **entire transcript** once with character offsets, so
   each GPT-2 subword's `(char_start, char_end)` span is known.
3. **Single forward pass** for transcripts ≤ 1024 GPT-2 tokens.
   Otherwise, **chunked inference** in 512-token steps where each
   chunk uses the prior 512 tokens as context — see "Sliding window"
   below.
4. Reads per-subword log-probabilities from the model's logits
   (`log_softmax`, then index by the actual next-token id).
5. For each spaCy token (in document order), finds the contiguous run
   of GPT-2 subwords whose character spans overlap the spaCy token's
   span, sums their log-probabilities, and converts to bits via
   `-log_e P / log_e 2`.
6. `upos == "PUNCT"` rows are masked to `NaN`.

**Sliding window** (`_chunked_logits`, for texts longer than GPT-2's
1024-token context limit):

- Chunk size: **512 tokens**.
- Each chunk runs with the **preceding 512 tokens as context** (so the
  effective input is up to 1024 tokens, the model's hard limit).
- Logits for the context prefix are discarded; only the 512 logits
  belonging to the current chunk are retained and concatenated to the
  next chunk's output.
- Result: every position past index 511 has at least 512 tokens of
  preceding context — same as for any non-boundary position in a short
  document.

**Why** *(⚠️ flags my interpolation):*

- *512-token stride with 512-token context tail* fully uses GPT-2's
  1024-token window while keeping the sliding step cheap. ⚠️ Other
  reasonable choices (smaller stride / more overlap) would smooth
  boundary effects further but at ~2× compute.
- *PUNCT → NaN*: GPT-2 does have probabilities for punctuation, but the
  extractor masks them so downstream code doesn't include punctuation
  rows in surprisal regressions. ⚠️ This matches a common convention
  in the EEG literature where the regression is over content words; if
  someone wants punctuation surprisal they can re-run the extractor
  without the mask.
- *Position 0 surprisal = 0* (neutral) because there is no prior
  context to condition on — this is a sentinel, not a real value.

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

**Surprisal ↔ EEG / behavior:**
- <a name="smith-2013"></a>**[Smith 2013]** Smith & Levy. *The effect of word predictability on reading time is logarithmic.* Cognition. [doi:10.1016/j.cognition.2013.02.013](https://doi.org/10.1016/j.cognition.2013.02.013) — empirical support for the linear-surprisal prediction.
- <a name="frank-2015"></a>**[Frank 2015]** Frank, Otten, Galli, Vigliocco. *The ERP response to the amount of information conveyed by words in sentences.* Brain & Language. [doi:10.1016/j.bandl.2014.10.006](https://doi.org/10.1016/j.bandl.2014.10.006) — N400 ∝ surprisal.
- <a name="wlotko-2012"></a>**[Wlotko 2012]** Wlotko & Federmeier. *So that's what you meant! Event-related potentials reveal multiple aspects of context use during construction of message-level meaning.* NeuroImage. [doi:10.1016/j.neuroimage.2012.05.054](https://doi.org/10.1016/j.neuroimage.2012.05.054)
- <a name="goodkind-2018"></a>**[Goodkind 2018]** Goodkind & Bicknell. *Predictive power of word surprisal for reading times is a linear function of language model quality.* CMCL 2018. [arXiv:1804.10742](https://arxiv.org/abs/1804.10742) — better LMs give better surprisal estimates for reading time.

**Naturalistic EEG/MEG using GPT-2 surprisal:**
- <a name="brennan-2016"></a>**[Brennan 2016]** Brennan & Hale. *Hierarchical structure guides rapid linguistic predictions during naturalistic listening.* PLOS ONE. [doi:10.1371/journal.pone.0207741](https://doi.org/10.1371/journal.pone.0207741)
- <a name="hale-2018"></a>**[Hale 2018]** Hale, Dyer, Kuncoro, Brennan. *Finding syntax in human encephalography with beam search.* ACL 2018. [arXiv:1806.04127](https://arxiv.org/abs/1806.04127)

[Hale 2001]: #hale-2001
[Levy 2008]: #levy-2008
[Smith 2013]: #smith-2013
[Frank 2015]: #frank-2015
[Wlotko 2012]: #wlotko-2012
[Goodkind 2018]: #goodkind-2018
[Brennan 2016]: #brennan-2016
[Hale 2018]: #hale-2018

## Datasets
<!-- auto-populated from docs/datasets/*.md once the dataset registry exists -->
