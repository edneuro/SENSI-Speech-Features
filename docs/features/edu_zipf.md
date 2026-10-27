---
key: edu_zipf
name: Educator's Word Frequency Guide (EWFG) grade-band frequencies
levels: [word]
gui_toggles: [freq_ewfg_k2, freq_ewfg_3to5, freq_ewfg_6to8, freq_ewfg_9to12, freq_ewfg_overall]
output_columns: [Edu_Zipf_K2, Edu_Zipf_3to5, Edu_Zipf_6to8, Edu_Zipf_9to12, Edu_Zipf_Overall]
libraries: [pyarrow]
status: unavailable
depth: deep
---

> ⚠️ **Unavailable in SensiExtractor — no data.** These grade-band
> frequencies are **not produced**. Their source, *The Educator's Word
> Frequency Guide* ([Zeno 1995]), exists **only in print and has never
> been digitized**, so there is no lookup table to query. The toggles are
> **disabled** in the app and the extractor code (kept for if/when the
> norms are obtained) would emit all-`NaN` columns. We may add this later
> if a digital version becomes available.
> Book reference:
> [The Educator's Word Frequency Guide](https://books.google.com/books/about/The_Educator_s_Word_Frequency_Guide.html?id=jsJ1QgAACAAJ).

## Meaning
Per-word Zipf-scale frequencies derived from the *Educator's Word
Frequency Guide* ([Zeno 1995]), a corpus of texts that
American schoolchildren actually encounter, **bucketed by grade band**:

| Column | Grade band |
|---|---|
| `Edu_Zipf_K2`       | Kindergarten through 2nd grade |
| `Edu_Zipf_3to5`     | 3rd–5th grade |
| `Edu_Zipf_6to8`     | 6th–8th grade |
| `Edu_Zipf_9to12`    | 9th–12th grade |
| `Edu_Zipf_Overall`  | Aggregated across all bands |

Each column uses the same [Zipf scale](freq_subtlex.md#math--linguistics)
as `Zipf_Freq`, just computed against a different reference corpus.
Missing-from-lookup rows are `NaN` (PUNCT rows also `NaN`).

## EEG rationale
EWFG is the right reference corpus when the question is **age of
acquisition**, **developmental exposure**, or **literacy-graded
difficulty** — i.e. when SUBTLEX-US's adult-conversational baseline is
the wrong yardstick.

- A word's `Edu_Zipf_K2` measures how often a young child has plausibly
  encountered it in print. Words with high K2 frequency but low
  9to12 frequency (function words, common nouns) behave differently
  from words with low K2 but high 9to12 frequency (acquired-late
  vocabulary). The guide's overall frequency does track age of
  acquisition (AoA): r = −.281 across the 72 target words of
  [Juhasz 2003]. But no paper we know of tests the K2-vs-9to12
  *contrast* as an AoA measure, so treat that use as exploratory; for
  AoA itself, use the rated norms of [Kuperman 2012].
- Age of acquisition independently predicts **eye fixation durations**
  in naturalistic reading ([Juhasz 2003], [Dirix 2017]) and **N400
  amplitude** in single-word ERP studies ([Yum 2019]), with effects
  that persist after controlling for word frequency. The per-band
  columns therefore let you regress out a developmental-exposure
  covariate that pure SUBTLEX cannot capture.
- `Edu_Zipf_Overall` is the closest EWFG analogue to general
  `Zipf_Freq` and is most useful as a sanity-check baseline against
  the SUBTLEX value.

## Implementation in SensiExtractor
Source: `sensi/extractors/educator_freq.py:add_edu_freq`.

**What the code does:**

1. Loads a parquet lookup from
   `data/dictionaries/edu_zipf.parquet` (or a user-supplied path), indexed
   by `word`, with columns
   `edu_zipf_k2, edu_zipf_3to5, edu_zipf_6to8, edu_zipf_9to12, edu_zipf_overall`.
2. Lookups are keyed on `lemma.lower()`; `upos == "PUNCT"` rows are
   sentinel-keyed to `"__missing__"` so they always miss and surface as
   `NaN`.
3. Uses `pandas.DataFrame.reindex` for a vectorized lookup — much
   faster than a per-row apply for long transcripts.
4. **If the parquet lookup file is missing**, the extractor still
   returns the right schema: all five columns are emitted as `NaN` and
   a `logging.warning` is issued. This keeps downstream code stable
   even when EWFG data acquisition is pending.
5. The `bands` argument lets the caller emit only a subset of columns —
   the GUI's per-band toggles flow through to this argument so disabled
   bands are simply not added to the output DataFrame.

**Why:**

- *Lookup keyed on lowercased `lemma`* — matches the
  [SUBTLEX extractor](freq_subtlex.md)'s convention so the two
  frequency columns are comparable lexeme-for-lexeme.
- *Parquet over CSV* — vectorised pandas lookup is fast and the file
  format compresses well for a ~5-column-by-N-row table.
- *Schema preserved when lookup is missing* — the columns are still
  written, filled with `NaN`, so downstream code (planner, GUI preflight
  warnings, doc-level aggregates) can rely on them existing.

## Data acquisition
**No digital source exists.** *The Educator's Word Frequency Guide*
([Zeno 1995]) was published as a print reference and, as far as we have
been able to determine, has never been released in machine-readable
form. SensiExtractor therefore ships **no** `data/dictionaries/edu_zipf.parquet`
with real data, and the extractor's missing-file path (all-`NaN` columns
+ a logged warning) is the permanent state rather than a fallback.

To enable these columns in future, a `data/dictionaries/edu_zipf.parquet`
would need to be built — indexed by `word`, with columns
`edu_zipf_k2, edu_zipf_3to5, edu_zipf_6to8, edu_zipf_9to12,
edu_zipf_overall` — from a digitized copy of the guide (book reference:
<https://books.google.com/books/about/The_Educator_s_Word_Frequency_Guide.html?id=jsJ1QgAACAAJ>).
Until then the toggles remain disabled.

## Library
The lookup is a plain pandas-on-pyarrow read:

```python
import pandas as pd
lookup = pd.read_parquet("data/dictionaries/edu_zipf.parquet").set_index("word")
lookup.loc["house"]
# edu_zipf_k2        4.21
# edu_zipf_3to5      4.93
# edu_zipf_6to8      5.18
# edu_zipf_9to12     5.41
# edu_zipf_overall   5.27
```

## Papers
- <a name="zeno-1995"></a>**[Zeno 1995]** Zeno, Ivens, Millard, Duvvuri. *The Educator's Word Frequency Guide.* Touchstone Applied Science Associates. — the source norms for the per-band frequencies.
- <a name="kuperman-2012"></a>**[Kuperman 2012]** Kuperman, Stadthagen-Gonzalez, Brysbaert. *Age-of-acquisition ratings for 30,000 English words.* Behavior Research Methods 44, 978–990. [doi:10.3758/s13428-012-0210-4](https://doi.org/10.3758/s13428-012-0210-4) — large AoA norms; the construct's empirical basis.
- <a name="juhasz-2003"></a>**[Juhasz 2003]** Juhasz & Rayner. *Investigating the effects of a set of intercorrelated variables on eye fixation durations in reading.* J. Experimental Psychology: Learning, Memory, and Cognition 29(6), 1312–1318. [doi:10.1037/0278-7393.29.6.1312](https://doi.org/10.1037/0278-7393.29.6.1312) — AoA effects on eye fixation durations after controlling for frequency, familiarity, length, and concreteness.
- <a name="dirix-2017"></a>**[Dirix 2017]** Dirix & Duyck. *The first- and second-language age of acquisition effect in first- and second-language book reading.* J. Memory and Language 97, 103–120. [doi:10.1016/j.jml.2017.07.012](https://doi.org/10.1016/j.jml.2017.07.012) — AoA effects on eye movements during naturalistic book reading (GECO corpus).
- <a name="yum-2019"></a>**[Yum 2019]** Yum & Law. *Interactions of age of acquisition and lexical frequency effects with phonological regularity: An ERP study.* Psychophysiology 56(10), e13433. [doi:10.1111/psyp.13433](https://doi.org/10.1111/psyp.13433) — single-trial ERP analysis shows AoA modulates N400 amplitude independently of word frequency.

[Zeno 1995]: #zeno-1995
[Kuperman 2012]: #kuperman-2012
[Juhasz 2003]: #juhasz-2003
[Dirix 2017]: #dirix-2017
[Yum 2019]: #yum-2019

## Datasets
<!-- auto-populated from docs/datasets/*.md once the dataset registry exists -->
