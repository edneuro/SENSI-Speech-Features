---
key: master_bert_sentence
name: BERT contextual sentence embeddings
levels: [sentence]
gui_toggle: master_bert_sentence
output_files: ["sentence/<stim>_bert_sentence.mat"]
libraries: [transformers, torch]
status: implemented
depth: deep
---

## Meaning
A 768-dimensional vector per sentence at each of BERT's 12 transformer
layers — the sentence-level analogue of the
[word-level BERT embedding](master_bert_word.md). Output shape:
`(n_sentences, 12, 768)`. Each sentence's vector is the **mean of the
contextual embeddings of its non-PUNCT content words** at the same
layer.

## EEG rationale
Sentence-level embeddings are the right granularity for:

- **Sentence-locked ERP analyses** of meaning integration / discourse
  effects, where the regressor is "how similar is this sentence's
  meaning to the preceding one?" ([Toneva 2019], [Caucheteux 2022]).
- **Discourse-coherence analyses** — sentence-to-sentence cosine
  similarity in this space tracks topic continuity / discourse shifts
  more cleanly than word-to-word similarity ([Goldstein 2022]).
- **Cross-sentence prediction** — predicting the next sentence's
  embedding given the current one's is the standard "discourse-level
  surprisal" measure, complementary to word-level
  [`Surprisal`](pred_surprisal.md).

All 12 layers are shipped intact for the same reason as the
[word-level column](master_bert_word.md): optimal-layer choice for
sentence-level EEG / fMRI prediction is task- and ROI-dependent
([Toneva 2019]).

## Implementation in SensiExtractor
Source: `extractors/bert_embed.py:embed_bert_words_and_sentences`

**What the code does** (verified against the source):

The sentence embeddings are produced **in the same forward pass** as
the [word-level embeddings](master_bert_word.md) — see that doc's
implementation section for the full sliding-window / left-context /
PUNCT-handling details.

For each sentence, after the forward pass:

1. The per-word embeddings are computed (mean-pool of WordPiece
   subwords) for every non-PUNCT word in the sentence.
2. The sentence vector at each layer is the **mean** of those
   per-word vectors:

```python
sent_emb[s_pos] = stacked[content_positions].mean(axis=0)
```

where `stacked` is the `(seq_len, n_layers, hidden)` array of BERT
hidden states and `content_positions` are the subword positions
belonging to non-PUNCT words in the current sentence.

3. Sentences whose every content word failed alignment / tokenization
   (e.g. all-PUNCT sentences) produce a zero vector with `is_valid=False`.

**Output `.mat` fields** — flat top-level variables (not a nested struct),
so `embeddings` stays backward-compatible:

| Field | Shape | Meaning |
|---|---|---|
| `embeddings`  | `(n_sentences, 12, 768)` float32 | per-layer sentence vectors |
| `sentence_id` | `(n_sentences,)` int | spaCy sentence id |
| `sentence`    | `{n_sentences}` cellstr | the sentence text |
| `is_valid`    | `(n_sentences,)` logical | False for empty / all-PUNCT sentences |

**Why:**

- *Mean-pool over content words rather than over all subwords* — the
  spec is explicit: aggregation is over non-PUNCT word embeddings.
  ⚠️ This means punctuation is excluded from the sentence embedding
  even though it was already excluded from BERT input; both
  exclusions are consistent.
- *Shipped as `(n_sentences, 12, 768)`* — preserves the per-layer
  structure for downstream layer-sweep analyses, matching the
  word-level shape with `n_sentences` instead of `n_words`.
- ⚠️ *Mean-pool rather than CLS-token / max-pool* — the standard
  choice in EEG / brain-prediction work ([Toneva 2019],
  [Schrimpf 2021]) is mean-pool of content tokens; CLS is fine-tuned
  for classification objectives and doesn't generalize as well to
  unsupervised brain alignment.

## Library
Same setup as the word-level extractor — see
[`master_bert_word`](master_bert_word.md#library).

## Papers
See [`master_bert_word`](master_bert_word.md) for the full reference
list; the sentence-level embedding inherits the same foundational
papers (Devlin 2019, Tenney 2019, Rogers 2020, Toneva 2019,
Schrimpf 2021, Goldstein 2022) plus the sentence-level-specific:

- <a name="caucheteux-2022"></a>**[Caucheteux 2022]** Caucheteux & King. *Brains and algorithms partially converge in natural language processing.* Communications Biology. [doi:10.1038/s42003-022-03036-1](https://doi.org/10.1038/s42003-022-03036-1) — sentence-level LM representations and brain prediction.

[Caucheteux 2022]: #caucheteux-2022

## Datasets
<!-- auto-populated from docs/datasets/*.md once the dataset registry exists -->
