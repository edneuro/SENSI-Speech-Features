---
key: output_doc_readability
name: Readability (Flesch Reading Ease + Flesch–Kincaid Grade Level)
levels: [document]
gui_toggle: output_doc_readability
output_columns: [flesch_reading_ease, flesch_kincaid_grade]
libraries: [textstat]
status: implemented
depth: deep
---

## Meaning
Two classical readability indices computed over the full transcript:

| Column | Formula (English) | Range | Interpretation |
|---|---|---|---|
| `flesch_reading_ease` | `206.835 − 1.015·(words/sentence) − 84.6·(syllables/word)` | typically 0–100 | **Higher = easier**. 60–70 ≈ plain English; below 30 ≈ academic / legal. |
| `flesch_kincaid_grade` | `0.39·(words/sentence) + 11.8·(syllables/word) − 15.59` | typically 0–18 | US grade level required to comprehend the text. 8 ≈ 8th grade. |

The two metrics are derived from the **same two predictors** (sentence
length, syllables per word) but combine them with different
coefficients to put one on a "ease" scale and the other on a "grade
level" scale — see [Kincaid 1975] for the derivation history.

## EEG rationale
Readability indices are passage-level numbers combining sentence
length and syllables-per-word into a single scale. Both ingredients
are established correlates of reading time: longer sentences and
longer / more-polysyllabic words produce longer eye fixations
([Just 1980]). The Flesch and Flesch–Kincaid scores are typically
used as a single-number covariate for matching stimuli on overall
text difficulty (e.g. educational or developmental designs).

The Flesch family is not the most sensitive difficulty measure
available — [`MTLD`](output_doc_lex_diversity.md) and the
[`Edu_Zipf_*`](edu_zipf.md) family carry more information — but it's the
most-cited and most-comparable across studies because everyone has it.

## Implementation in SensiExtractor
Computed in `sensi/pipeline.py`'s document-batch finishing step using the
`textstat` package, which implements both formulas as defined in
[Kincaid 1975] / [Flesch 1948]. Syllable counts use the same
`pyphen` machinery as the [syllable extractor](syllable_timing.md),
so the two are internally consistent.

## Library
```python
import textstat
textstat.flesch_reading_ease(text)        # 0-100, higher = easier
textstat.flesch_kincaid_grade(text)       # US grade level
```

## Papers
- <a name="flesch-1948"></a>**[Flesch 1948]** Flesch. *A new readability yardstick.* J. Applied Psychology. [doi:10.1037/h0057532](https://doi.org/10.1037/h0057532) — original Flesch Reading Ease.
- <a name="kincaid-1975"></a>**[Kincaid 1975]** Kincaid, Fishburne, Rogers, Chissom. *Derivation of new readability formulas (Automated Readability Index, Fog Count, and Flesch Reading Ease formula) for Navy enlisted personnel.* Naval Air Station Memphis. — Flesch–Kincaid Grade Level.
- <a name="just-1980"></a>**[Just 1980]** Just & Carpenter. *A theory of reading: From eye fixations to comprehension.* Psychological Review 87(4), 329–354. [doi:10.1037/0033-295X.87.4.329](https://doi.org/10.1037/0033-295X.87.4.329) — eye-fixation model showing readers pause longer at points of higher processing load (infrequent words, clause boundaries, sentence-end inferences).

[Flesch 1948]: #flesch-1948
[Kincaid 1975]: #kincaid-1975
[Just 1980]: #just-1980

## Datasets
<!-- auto-populated from docs/datasets/*.md once the dataset registry exists -->
