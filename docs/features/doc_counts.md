---
key: doc_counts
name: Document-level count columns
levels: [document]
gui_toggles: [output_out_count_tokens, output_out_count_content_tokens, output_out_count_sentences, output_out_count_paragraphs]
output_columns: [n_tokens, n_content_tokens, n_sentences, n_paragraphs]
libraries: [spaCy]
status: implemented
depth: stub
---

## Meaning
Four basic size descriptors per document, all integers, in the
document-level summary CSV:

| Column | Meaning |
|---|---|
| `n_tokens`         | Total token count, including PUNCT. |
| `n_content_tokens` | Tokens with [`upos`](lexling_pos.md) in `{NOUN, VERB, ADJ, ADV, PROPN}`. |
| `n_sentences`      | Sentence count from spaCy's `Doc.sents`. |
| `n_paragraphs`     | Paragraph count. Currently always `1` — paragraph segmentation is not yet implemented; see [`structural_ids`](structural_ids.md). |

## Usage
- Normalize stimulus-level summaries (`mean_*` aggregates) — most
  document-level metrics are sums divided by one of these.
- Compare stimulus sizes across a corpus: word counts are the basic
  unit of "how much material is in this stimulus."
- Filter / weight downstream analyses by content-token count rather
  than raw token count when controlling for function-word density.

## Implementation in SensiExtractor
Derived in `main.py`'s document-batch finishing step from the
word-level DataFrame: simple counts over the appropriate row filters
(`upos != "PUNCT"` for content tokens, unique `sentence_id` for
sentence count, etc.). PUNCT-aware definitions match the conventions
used by [`Zipf_Freq`](freq_subtlex.md), [`Surprisal`](pred_surprisal_gpt2.md),
and the semantic indicators.
