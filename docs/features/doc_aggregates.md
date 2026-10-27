---
key: doc_aggregates
name: Word-level feature aggregates
levels: [document]
gui_toggles: [output_doc_agg_zipf_freq, output_doc_agg_edu_zipf, output_doc_agg_predictability]
output_columns: [mean_zipf_freq, mean_edu_zipf_k2, mean_edu_zipf_3to5, mean_edu_zipf_6to8, mean_edu_zipf_9to12, mean_edu_zipf_overall, "mean_<slug>_<metric>"]
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
| `output_doc_agg_edu_zipf`  | `mean_edu_zipf_{k2,3to5,6to8,9to12,overall}` | [`Edu_Zipf_*`](edu_zipf.md) |
| `output_doc_agg_predictability` | one `mean_<slug>_<metric>` column per `<slug>_<Metric>` word-level column currently selected (e.g. `mean_gpt2_surprisal`, `mean_llama3_8b_cloze`) | [`Surprisal`](pred_surprisal.md), [`Cloze`](pred_cloze.md), [`TopMatch`](pred_topmatch.md) |

`mean_edu_zipf_*` is written only when `data/dictionaries/edu_zipf.parquet`
exists, which it does not by default ([why](edu_zipf.md)).

Unlike the other two rows, `output_doc_agg_predictability` doesn't name
a fixed column set — it scans the word-level DataFrame for every
column matching `<slug>_Surprisal` / `<slug>_Cloze` / `<slug>_TopMatch`
(one per model x metric selected in the Predictability card) and
aggregates each one found. Enabling more models/metrics on the
Word-level features page grows this column family automatically; no
separate doc-level toggle per model.

Aggregates are computed **only over rows where the underlying value is
not NaN** — so PUNCT rows are excluded automatically.

> The doc-level page surfaces a warning indicator on a toggle when the
> underlying word-level stage is **not in the current run plan** —
> e.g. enabling `mean_zipf_freq` while SUBTLEX is off at the word
> level will emit NaN. Look for the amber ⚠ next to the toggle.

## Usage
Stimulus-level descriptors for cross-stimulus comparisons or for
matching stimuli on average difficulty. A `mean_<slug>_topmatch`
column is the **proportion** of content words that model `<slug>`
nailed exactly — a single-number "predictability score" for the
document, per model.

## Implementation in SensiExtractor
Computed in `sensi/utils/doc_summary.py:build_doc_summary`. The
`mean_zipf_freq` / `mean_edu_zipf_*` columns are each
`np.nanmean(word_df[<fixed source column>])`. The predictability
family is dynamic: `_PREDICTABILITY_COL_RE` (`^(?P<slug>[a-z0-9_]+)_
(?P<metric>Surprisal|Cloze|TopMatch)$`) matches every `<slug>_<Metric>`
column actually present in the word-level frame, and each match is
aggregated to `mean_<slug>_<metric-lowercased>`. The doc-level
preflight in `sensi/gui/section_doc_features.py:_refresh_agg_warnings`
checks the planner's `stages_to_run` and warns when an aggregate is
enabled but its source stage (`predictability`) is absent.
