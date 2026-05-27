---
key: master_bert_word
name: BERT contextual word embeddings
levels: [word]
gui_toggle: master_bert_word
output_files: ["word/<stim>_bert_word.mat"]
libraries: [transformers, torch]
status: implemented
depth: deep
---

## Meaning
A 768-dimensional vector per word, emitted at every one of BERT's 12
transformer layers. Output shape: `(n_words, 12, 768)`. Each word's vector
encodes its meaning *in context* — the same surface word ("bank") receives
different vectors in "river bank" vs. "savings bank".

## EEG rationale
Contextual embeddings outperform static embeddings (Word2Vec, GloVe) as
predictors of word-level neural responses — both for ECoG/fMRI
([Schrimpf 2021], [Goldstein 2022]) and for MEG/EEG word-by-word reading
([Toneva 2019]). The brain processes meaning relative to the unfolding
sentence, not relative to a corpus-averaged sense, so a vector that
encodes the in-context meaning of "bank" is a better regressor for the
N400 than a single static vector.

The 12-layer stack is shipped intact rather than pre-pooled because
layer choice matters and is task-dependent. BERT's layers encode a
rough linguistic hierarchy — surface form and POS in early layers,
syntactic structure in middle layers, semantic and discourse-level
information in late layers ([Tenney 2019], [Rogers 2020]) — and the
optimal layer for predicting EEG signal is itself an empirical question
([Toneva 2019] sweep across layers and report different best-layer per
ROI). Pre-pooling would discard that signal.

## Math / linguistics
Each layer's vector is the mean of the contextual subword embeddings
that BERT's WordPiece tokenizer produced for that word. For a word `w`
that tokenizes into subwords `s_1, …, s_k` at layer `ℓ`:

    h_ℓ(w) = mean(h_ℓ(s_1), …, h_ℓ(s_k))

The hidden state `h_ℓ(s_i)` is the output of layer `ℓ`'s self-attention +
feed-forward block, which incorporates information from all other tokens
in the same input window via attention.

## Implementation in SensiExtractor
Source: `extractors/bert_embed.py`. Design spec:
`docs/superpowers/specs/2026-05-22-embeddings-design.md`.

**What the code does** (verified against the source):

1. Outer loop is **per sentence** (sentence IDs from spaCy via the
   word-level CSV).
2. For each sentence, BERT input is built as
   `[preceding-sentences-context] + [current-sentence-content-words]`,
   where the context is grown by **prepending complete preceding
   sentences** (oldest dropped first) as long as the total stays within
   the **510-token effective limit** (512 minus `[CLS]` / `[SEP]`).
   Preceding sentences are never split mid-way.
3. **PUNCT tokens are excluded from BERT input** entirely (the spaCy
   `upos == "PUNCT"` mask is applied before tokenization).
4. A single forward pass runs on the resulting window
   (`is_split_into_words=True`, `truncation=True` as a safety net).
5. For each target-sentence word, the WordPiece subwords belonging to it
   are **mean-pooled** at each layer to produce a single `(12, 768)`
   vector. Output rows for PUNCT words stay as zero vectors.
6. **All 12 transformer layers** are emitted —
   `hidden_states[1:13]`, skipping the embedding layer at index 0.
7. The sentence-level sidecar (`<stim>_bert_sentence.mat`) is the
   mean-pool of the non-PUNCT word embeddings within each sentence, at
   each layer.

**Why** *(lifted from the design spec where stated; my interpolation
flagged with ⚠️ — please correct):*

- *Left-context, sentence-complete sliding window* is the spec's chosen
  fix for the 512-token cap (§2 of the spec, verbatim phrase).
- ⚠️ *Right context excluded* — the spec doesn't explicitly justify
  this; my read is that asymmetric (left-only) context fits the causal
  nature of word-by-word EEG responses, but please verify this was the
  actual reasoning.
- ⚠️ *PUNCT excluded from input* — the spec lists it as a fact but
  doesn't justify it. Likely rationale: BERT's WordPiece doesn't
  represent punctuation semantically, and including PUNCT tokens
  perturbs attention over content words.
- *All 12 layers emitted* — implied by the spec listing
  `hidden_states[1:13]`; the choice of layer for downstream EEG
  prediction is left to the analyst.

**Alternatives that were *not* chosen** (for reader context):

| Approach | Why we didn't pick it |
|---|---|
| Hard truncation at 512 tokens | Drops the tail of every long passage. |
| Fixed-stride sliding window (e.g. 256 stride / 512 window) | Heavy overlap, ~2× compute, harder to attribute a vector to a specific word position. |
| Sentence-by-sentence with no prior context | Loses cross-sentence semantic continuity, which is a large part of why contextual embeddings predict EEG/MEG well in the first place ([Toneva 2019]). |
| Pool layers in-extractor (e.g. mean of layers 9–12) | Smaller file but discards per-layer signal the downstream analyst may want. |
| Bidirectional (left + right) context | ⚠️ Would let the embedding "see" future tokens; problematic for causal EEG analyses. |

## Library
```python
from transformers import BertTokenizerFast, BertModel
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
model     = BertModel.from_pretrained("bert-base-uncased", output_hidden_states=True)
```

## Example
```python
import torch

words = ["The", "cat", "sat", "on", "the", "mat"]
enc   = tokenizer(words, is_split_into_words=True, return_tensors="pt")
with torch.no_grad():
    out = model(**enc)

# out.hidden_states[0] is the embedding layer; [1:] are the 12 transformer layers
layers = torch.stack(out.hidden_states[1:])     # (12, 1, seq_len, 768)
layers = layers.squeeze(1).permute(1, 0, 2)     # (seq_len, 12, 768)

# Mean-pool subwords belonging to each word
word_ids = enc.word_ids()
word_vecs = []
for w in range(len(words)):
    positions = [i for i, wid in enumerate(word_ids) if wid == w]
    word_vecs.append(layers[positions].mean(dim=0))
```

## Papers

**Model:**
- <a name="devlin-2019"></a>**[Devlin 2019]** Devlin, Chang, Lee, Toutanova. *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* NAACL 2019. [arXiv:1810.04805](https://arxiv.org/abs/1810.04805)

**BERTology — what the layers encode:**
- <a name="tenney-2019"></a>**[Tenney 2019]** Tenney, Das, Pavlick. *BERT Rediscovers the Classical NLP Pipeline.* ACL 2019. [arXiv:1905.05950](https://arxiv.org/abs/1905.05950)
- <a name="rogers-2020"></a>**[Rogers 2020]** Rogers, Kovaleva, Rumshisky. *A Primer in BERTology: What We Know About How BERT Works.* TACL 2020. [arXiv:2002.12327](https://arxiv.org/abs/2002.12327)

**Contextual embeddings ↔ brain:**
- <a name="toneva-2019"></a>**[Toneva 2019]** Toneva & Wehbe. *Interpreting and improving natural-language processing (in machines) with natural language-processing (in the brain).* NeurIPS 2019. [arXiv:1905.11833](https://arxiv.org/abs/1905.11833) — per-layer fMRI/MEG alignment; demonstrates that the best layer differs by ROI and that contextual depth improves brain prediction.
- <a name="schrimpf-2021"></a>**[Schrimpf 2021]** Schrimpf, Blank, Tuckute, Kauf, Hosseini, Kanwisher, Tenenbaum, Fedorenko. *The neural architecture of language: Integrative modeling converges on predictive processing.* PNAS 2021. [doi:10.1073/pnas.2105646118](https://doi.org/10.1073/pnas.2105646118)
- <a name="goldstein-2022"></a>**[Goldstein 2022]** Goldstein et al. *Shared computational principles for language processing in humans and deep language models.* Nature Neuroscience 2022. [doi:10.1038/s41593-022-01026-4](https://doi.org/10.1038/s41593-022-01026-4)

[Devlin 2019]: #devlin-2019
[Tenney 2019]: #tenney-2019
[Rogers 2020]: #rogers-2020
[Toneva 2019]: #toneva-2019
[Schrimpf 2021]: #schrimpf-2021
[Goldstein 2022]: #goldstein-2022

## Datasets
<!-- auto-populated from docs/datasets/*.md once the dataset registry exists -->
