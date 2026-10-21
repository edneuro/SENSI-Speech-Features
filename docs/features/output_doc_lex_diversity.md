---
key: output_doc_lex_diversity
name: Lexical diversity (TTR + MTLD)
levels: [document]
gui_toggle: output_doc_lex_diversity
output_columns: [ttr, mtld]
libraries: [lexicalrichness]
status: implemented
depth: deep
---

## Meaning
Two complementary indices of **vocabulary richness** at the document
level:

| Column | Formula | Range | Interpretation |
|---|---|---|---|
| `ttr`  | unique types / total content-word tokens | (0, 1] | Higher = more lexical variety; *strongly* length-dependent. |
| `mtld` | Measure of Textual Lexical Diversity — length-corrected | ≈ 30–200 typical | Higher = more diversity; designed to be stable across text lengths ([McCarthy 2010]). |

Both are computed over **content words only** (`upos` in `{NOUN, VERB,
ADJ, ADV, PROPN}`), so PUNCT and function words are excluded; tokens are
lowercased before counting types.

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
- **TTR** = `|unique_forms| / n_tokens` over lowercased content-word
  surface forms. As text grows, the probability of seeing a new type
  drops, so TTR monotonically decreases — making comparisons across
  different-length texts meaningless without correction.
- **MTLD** walks through the text counting tokens until the running
  TTR drops to or below a threshold (default 0.72), then counts a
  "factor" and resets. Leftover tokens at the end contribute a
  proportional partial factor, `(1 − TTR_remaining) / (1 − 0.72)`.
  `MTLD = n_tokens / factors`, averaged over a forward and a reverse
  pass ([McCarthy 2010]). Higher MTLD = it takes more tokens before
  TTR drops to the threshold = the text uses more types per token.

## Implementation in SensiExtractor
Computed in `extractors/lexical_diversity.py` (in-house implementation
of the published procedure — equivalent to [McCarthy 2010], not
verified bit-for-bit against the authors' original tool), called from
`utils/doc_summary.py`. Both measures run over **content words only**
(NOUN/VERB/ADJ/ADV/PROPN surface tokens, lowercased); punctuation and
function words are excluded before the calculation.

Edge convention: a text that never completes any factor (running TTR
never reaches 0.72 in either direction — fully unique token
sequences, very short texts) returns **NaN**: MTLD is undefined
there, and fabricating a finite value would make it scale with text
length, exactly the artifact MTLD exists to remove.

## Library
```python
from extractors.lexical_diversity import compute_ttr, compute_mtld
tokens = ["cat", "sat", "mat", "cat", "slept"]   # content words, lowercased
compute_ttr(tokens)    # 0.8
compute_mtld(tokens)   # forward+reverse mean; NaN if no factor completes
```

## Papers
- <a name="mccarthy-2010"></a>**[McCarthy 2010]** McCarthy & Jarvis. *MTLD, vocd-D, and HD-D: A validation study of sophisticated approaches to lexical diversity assessment.* Behavior Research Methods. [doi:10.3758/BRM.42.2.381](https://doi.org/10.3758/BRM.42.2.381) — defines MTLD and shows it's stable across text lengths.
- <a name="koizumi-2012"></a>**[Koizumi 2012]** Koizumi & In'nami. *Effects of text length on lexical diversity measures: Using short texts with less than 200 tokens.* System 40(4), 522–532. [doi:10.1016/j.system.2012.10.012](https://doi.org/10.1016/j.system.2012.10.012) — TTR's length sensitivity in detail.

[McCarthy 2010]: #mccarthy-2010
[Koizumi 2012]: #koizumi-2012

## Datasets
<!-- auto-populated from docs/datasets/*.md once the dataset registry exists -->
