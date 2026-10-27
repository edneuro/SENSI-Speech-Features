---
key: doc_counts
name: Document-level count columns
levels: [document]
gui_toggles: [output_out_count_tokens, output_out_count_content_tokens, output_out_count_sentences, output_out_count_paragraphs]
output_columns: [n_tokens, n_content_tokens, n_sentences, n_paragraphs, stimulus_id, dictionary_used, transcript_path, audio_path]
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
| `n_paragraphs`     | Paragraph count, from blank-line-separated spans in the raw transcript; a transcript with no blank lines is a single paragraph. See [`structural_ids`](structural_ids.md). |

Every row of the document summary CSV also carries four identifier
columns. They have no toggle and are always written:

| Column | Meaning |
|---|---|
| `stimulus_id`     | The transcript's file stem, original casing (e.g. `1_FB_P` for `1_FB_P.txt`) — the same id every per-level output filename uses, so the document table joins to them directly. |
| `dictionary_used` | Semantic-category backend: `LIWC` if any `Sem_LIWC_*` column is present, otherwise `Empath` (also when no semantic columns were produced). |
| `transcript_path` | Path of the transcript file as the batch found it (keeps the full filename, extension included). |
| `audio_path`      | Path of the paired audio file; empty for text-only items. |

## Usage
- Normalize stimulus-level summaries (`mean_*` aggregates) — most
  document-level metrics are sums divided by one of these.
- Compare stimulus sizes across a corpus: word counts are the basic
  unit of "how much material is in this stimulus."
- Filter / weight downstream analyses by content-token count rather
  than raw token count when controlling for function-word density.

## Implementation in SensiExtractor
Derived in `sensi/utils/doc_summary.py:build_doc_summary` (called from
`sensi/pipeline.py`'s document-batch step) from the word-level DataFrame: simple
counts over the appropriate row filters
(`upos` in `CONTENT_POS` = `{NOUN, VERB, ADJ, ADV, PROPN}` for content
tokens, unique `sentence_id` for
sentence count, etc.). PUNCT-aware definitions match the conventions
used by [`Zipf_Freq`](freq_subtlex.md), [`Surprisal`](pred_surprisal.md),
and the semantic indicators.
