---
key: output_doc_lex_diversity
name: Lexical diversity (TTR + MTLD)
levels: [document]
gui_toggle: output_doc_lex_diversity
output_columns: [TTR, MTLD]
libraries: [lexicalrichness]
status: implemented
depth: deep
---

## Meaning
Two complementary indices of **vocabulary richness** at the document
level:

| Column | Formula | Range | Interpretation |
|---|---|---|---|
| `TTR`  | unique types / total tokens | (0, 1] | Higher = more lexical variety; *strongly* length-dependent. |
| `MTLD` | Measure of Textual Lexical Diversity — length-corrected | ≈ 30–200 typical | Higher = more diversity; designed to be stable across text lengths ([McCarthy 2010]). |

PUNCT rows are excluded; tokens are lowercased before counting types.

## EEG rationale
Lexical diversity is a **stimulus-matching covariate** in cross-corpus
EEG/MEG studies — a low-diversity stimulus (e.g. a children's book
with high repetition) has very different word-level statistics than a
high-diversity stimulus (an academic essay), and any neural contrast
between them needs to control for that. MTLD is the diversity metric
of choice precisely because raw TTR is confounded with text length
([Koizumi 2012], [McCarthy 2010]) — comparing TTRs across stimuli of
different lengths is nearly always misleading.

For ERP studies that pool across many stimuli of varying length, MTLD
is the right per-stimulus covariate to include in a regression.

## Math / linguistics
- **TTR** = `|unique_lemmas| / n_tokens`. As text grows, the
  probability of seeing a new type drops, so TTR monotonically
  decreases — making comparisons across different-length texts
  meaningless without correction.
- **MTLD** walks through the text counting tokens until the running
  TTR drops below a threshold (default 0.72), then resets. The number
  of full "factors" (segments) the text breaks into is the MTLD value
  ([McCarthy 2010]). Higher MTLD = it takes more tokens before TTR
  drops to the threshold = the text uses more types per token.

## Implementation in SensiExtractor
Computed in `main.py`'s document-batch finishing step using the
`lexicalrichness` package, which implements MTLD as defined in
[McCarthy 2010]. Tokens are lowercased and PUNCT is excluded before
the calculation.

## Library
```python
from lexicalrichness import LexicalRichness
lex = LexicalRichness("the cat sat on the mat the cat slept")
print(lex.ttr)    # 0.6
print(lex.mtld()) # ~30 for a short text
```

## Papers
- <a name="mccarthy-2010"></a>**[McCarthy 2010]** McCarthy & Jarvis. *MTLD, vocd-D, and HD-D: A validation study of sophisticated approaches to lexical diversity assessment.* Behavior Research Methods. [doi:10.3758/BRM.42.2.381](https://doi.org/10.3758/BRM.42.2.381) — defines MTLD and shows it's stable across text lengths.
- <a name="koizumi-2012"></a>**[Koizumi 2012]** Koizumi & In'nami. *Effects of text length on lexical diversity measures: Using short texts with less than 200 tokens.* System. [doi:10.1016/j.system.2012.10.013](https://doi.org/10.1016/j.system.2012.10.013) — TTR's length sensitivity in detail.

[McCarthy 2010]: #mccarthy-2010
[Koizumi 2012]: #koizumi-2012

## Datasets
<!-- auto-populated from docs/datasets/*.md once the dataset registry exists -->
