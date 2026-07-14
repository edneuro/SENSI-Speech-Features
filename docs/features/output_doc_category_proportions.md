---
key: output_doc_category_proportions
name: Document-level category proportions
levels: [document]
gui_toggle: output_doc_category_proportions
output_columns: ["count_<cat> and prop_<cat> per active semantic category"]
libraries: [liwc, empath]
status: implemented
depth: deep
---

## Meaning
For each active semantic category (selected on the **Word-level
features → Semantic indicators** card), the document summary CSV gains
two columns:

- `count_<cat>` — number of content tokens that match `<cat>`.
- `prop_<cat>`  — proportion of content tokens that match `<cat>`
  (`count_<cat> / n_content_tokens`).

Example: if "Affect" and "PosEmo" are active LIWC categories, the doc
summary gains `count_LIWC_Affect`, `prop_LIWC_Affect`,
`count_LIWC_PosEmo`, `prop_LIWC_PosEmo`.

Same content-POS filter as the word-level
[`sem_indicators`](sem_indicators.md) — function words and PUNCT are
excluded from both numerator and denominator.

## EEG rationale
Per-category proportions are the **document-level analogue** of the
binary [`Sem_<Dict>_<cat>`](sem_indicators.md) columns. They answer:
"how affect-laden is this stimulus overall?" or "what proportion of
the words in this story are about social interaction?"

These are exactly the right covariates when:
- Matching stimuli for emotional content across an EEG study
  (e.g. matched-pair emotion contrasts).
- Looking at large-scale topic effects across a corpus of stimuli.
- Regressing out gross content differences when isolating
  lower-level acoustic or syntactic effects.

The category set should match what's already configured for the
word-level page — they share a single selection, edited in one place
(Word-level features → Configure word-level categories…).

## Implementation in SensiExtractor
Computed in `main.py`'s document-batch finishing step. Walks the
word-level DataFrame after the
[`semantic_domain`](sem_indicators.md) stage has populated the
per-token `Sem_<Dict>_<cat>` columns, then:

```python
content_mask = ~word_df["Sem_<Dict>_<cat>"].isna()    # drops PUNCT + non-content POS
n_content = content_mask.sum()
for cat in active_categories:
    col = f"Sem_<Dict>_{cat}"
    count = int(word_df[col].sum())                   # 0/1 column, NaN-aware sum
    proportion = count / n_content if n_content else float("nan")
    summary[f"count_{cat}"] = count
    summary[f"prop_{cat}"]  = proportion
```

If [`sem_indicators`](sem_indicators.md) is **not** in the run plan,
no per-category columns exist on the word DataFrame and this aggregate
emits nothing meaningful — the GUI surfaces a warning on the
[`sem_indicators`](sem_indicators.md) word-level card in that case.

## Papers
See [`sem_indicators`](sem_indicators.md) — the per-category rationale
and citations apply directly. The proportion-as-stimulus-descriptor
framing follows [Tausczik 2010](sem_indicators.md#tausczik-2010) and
[Pennebaker 2015](sem_indicators.md#pennebaker-2015).

## Datasets
<!-- auto-populated from docs/datasets/*.md once the dataset registry exists -->
