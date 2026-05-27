---
key: doc_aggregates
name: Word-level feature aggregates
levels: [document]
gui_toggles: [output_doc_agg_zipf_freq, output_doc_agg_edu_zipf, output_doc_agg_surprisal, output_doc_agg_llm]
output_columns: [mean_zipf_freq, mean_edu_zipf_K2, mean_edu_zipf_3to5, mean_edu_zipf_6to8, mean_edu_zipf_9to12, mean_edu_zipf_Overall, mean_surprisal, mean_llm_cloze, mean_llm_surprisal, mean_llm_top_match_prop]
libraries: []
status: implemented
depth: stub
---

## Meaning
Document-level **means of word-level lexical features**. Each toggle
controls a family of `mean_*` columns in the document summary CSV:

| Toggle | Columns emitted | Sourced from word-level |
|---|---|---|
| `output_doc_agg_zipf_freq` | `mean_zipf_freq` | [`Zipf_Freq`](freq_subtlex.md) |
| `output_doc_agg_edu_zipf`  | `mean_edu_zipf_{K2,3to5,6to8,9to12,Overall}` | [`Edu_Zipf_*`](edu_zipf.md) |
| `output_doc_agg_surprisal` | `mean_surprisal` | [`Surprisal`](pred_surprisal_gpt2.md) |
| `output_doc_agg_llm`       | `mean_llm_cloze`, `mean_llm_surprisal`, `mean_llm_top_match_prop` | [`LLM_Cloze`](pred_cloze.md), [`LLM_Surprisal`](pred_surprisal_llm.md), [`LLM_TopMatch`](pred_topmatch.md) |

Aggregates are computed **only over rows where the underlying value is
not NaN** — so PUNCT rows are excluded automatically.

> The doc-level page surfaces a warning indicator on a toggle when the
> underlying word-level stage is **not in the current run plan** —
> e.g. enabling `mean_zipf_freq` while SUBTLEX is off at the word
> level will emit NaN. Look for the amber ⚠ next to the toggle.

## Usage
Stimulus-level descriptors for cross-stimulus comparisons or for
matching stimuli on average difficulty. The `mean_llm_top_match_prop`
column is the **proportion** of content words that the LLM nailed
exactly — a single-number "predictability score" for the document.

## Implementation in SensiExtractor
Computed in `main.py`'s document-batch finishing step. Each `mean_*`
column is `np.nanmean(word_df[<source_column>])`. The doc-level
preflight in `gui/section_doc_features.py:_refresh_agg_warnings`
checks the planner's `stages_to_run` and warns when an aggregate is
enabled but its source stage is absent.
