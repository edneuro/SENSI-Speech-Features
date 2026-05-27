---
key: sem_indicators
name: Semantic-domain category indicators
levels: [word]
gui_toggle: sem_indicators
output_columns: ["Sem_<Dict>_<category> (one per active category)", Sem_Categories]
libraries: [liwc, empath]
status: implemented
depth: deep
---

## Meaning
Per-word indicators of membership in **semantic / psychological
categories** from a dictionary backend (LIWC or Empath). For each
active category, the extractor emits one binary column:

- `Sem_LIWC_Affect` = `1` if the word is in LIWC's "Affect" category,
  `0` if it's a content word that doesn't match, `NaN` if it's a
  function word.
- Same shape for any other selected category, with the dictionary
  prefix (`LIWC` or `Empath`).

Plus a roll-up column `Sem_Categories` — a comma-separated list of all
categories matched by the word (or `""` if none, `NaN` if function
word).

**Content-POS filter:** scoring is restricted to `NOUN`, `VERB`, `ADJ`,
`ADV`, `PROPN`. Function words, determiners, punctuation, etc. receive
`NaN` for every category column.

## EEG rationale
Category indicators provide **per-word semantic-class regressors** that
go beyond surface lexical features (frequency, surprisal). They let
you ask "do emotion words elicit a larger N400 than neutral words in
this stimulus?" without hand-tagging the transcript.

Specific uses:

- **Emotional valence / arousal contrasts** — LIWC's
  Positive Emotion / Negative Emotion or Empath's
  affection / sadness categories are common emotion regressors and
  modulate amplitudes of late positive components (LPC) and frontal
  ERPs ([Citron 2014], [Recio 2014]).
- **Concrete vs abstract concepts** — affect concreteness ratings
  predict N400 amplitude ([Barber 2013]); LIWC and Empath both have
  categories that approximate concreteness clusters.
- **Topic / discourse markers** — useful for grouping words into
  thematic streams across long stimuli for time-resolved topic-effect
  analyses.

LIWC categories are **the more established psycholinguistic feature
set** ([Tausczik 2010], [Pennebaker 2015]); Empath is a free
open-source alternative ([Fast 2016]) that auto-generates categories
via word-embedding similarity to seed terms — useful when LIWC is not
available, but its categories are less curated.

## Backends and category selection
Two backends share the same interface:

| Backend | License | Source | Setup |
|---|---|---|---|
| **LIWC** | Commercial; licensed `.dic` file required | LIWC-2022 (or earlier) | Drop a `.dic` file at `dictionaries/liwc-2022.dic` (or any `*.dic`) |
| **Empath** | Free / open-source | `pip install empath` | No external file needed |

**Backend selection logic** (from `detect_backend()`):

- If any `*.dic` exists under `dictionaries/`, use **LIWC**.
- Otherwise, use **Empath**.

Categories are selected per-backend on the **Word-level features →
Semantic indicators** card (via the "Configure word-level categories…"
button) and the same selection is reused at the document level (see
[`doc_category_proportions`](output_doc_category_proportions.md) if/when
that doc exists).

## Implementation in SensiExtractor
Source: `extractors/semantic_domain.py:add_semantic_domain`. Backend
classes: `_LiwcScorer` and `_EmpathScorer` in the same file.

**What the code does:**

1. Reads `backend` (`"liwc"` or `"empath"`) and `categories` (list of
   category names) from the call site (planner → main).
2. Instantiates the matching scorer:
   - `_LiwcScorer` loads the `.dic` via the `liwc` Python package's
     `load_token_parser` (trie-based).
   - `_EmpathScorer` instantiates `Empath()` (loads its embedded JSON
     of categories).
3. Iterates over the word-level DataFrame; for each row:
   - If `upos` is **not** in `{NOUN, VERB, ADJ, ADV, PROPN}` → sentinel
     `None` (will become `NaN` in every category column).
   - Otherwise calls `scorer.categories_for(lemma, categories)` →
     returns a `set[str]` of matched category names.
4. For each requested category, emits a `Sem_<Dict>_<cat>` column
   (1 / 0 / NaN as above).
5. Emits a `Sem_Categories` roll-up column with the sorted
   comma-separated match list.

**Why** *(⚠️ flags my interpolation):*

- *Content-POS filter* — keeping function words out matches the standard
  practice in the LIWC literature, where the dictionary scoring is
  always about content categories ([Pennebaker 2015]). It also keeps the
  column NaN-rate proportional to the proportion of function words in
  the stimulus, which is what most downstream regressions want.
- *Lemma-keyed, lowercased lookup* — same convention as the
  [frequency](freq_subtlex.md) extractors, so the same row in the CSV
  reflects the same lexeme across all lexical-content columns.
- *Auto-pick LIWC when a .dic exists* — ⚠️ the spec doesn't justify
  this explicitly; the assumption is that if a user invested in a LIWC
  license they want it used. Empath stays the no-license fallback.
- *Both binary per-category columns AND the rolled-up `Sem_Categories`
  string* — ⚠️ duplication is intentional. The binary columns are easy
  to regress on; the rolled-up string is a fast eyeball view ("what
  categories did this word match?") and is also what the doc-level
  category-proportions feature aggregates.

## Library
```python
# LIWC
import liwc
parse, categories = liwc.load_token_parser("dictionaries/liwc-2022.dic")
matched = list(parse("happy"))   # → ['Affect', 'PosEmo', 'Tone', ...]

# Empath
from empath import Empath
lex = Empath()
"happy" in lex.cats["joy"]       # → True
```

## Papers

**LIWC:**
- <a name="pennebaker-2015"></a>**[Pennebaker 2015]** Pennebaker, Boyd, Jordan, Blackburn. *The Development and Psychometric Properties of LIWC2015.* UT Austin technical report. [doi:10.15781/T29G6Z](https://doi.org/10.15781/T29G6Z)
- <a name="tausczik-2010"></a>**[Tausczik 2010]** Tausczik & Pennebaker. *The Psychological Meaning of Words: LIWC and Computerized Text Analysis Methods.* J. Language and Social Psychology. [doi:10.1177/0261927X09351676](https://doi.org/10.1177/0261927X09351676)

**Empath:**
- <a name="fast-2016"></a>**[Fast 2016]** Fast, Chen, Bernstein. *Empath: Understanding Topic Signals in Large-Scale Text.* CHI 2016. [arXiv:1602.06979](https://arxiv.org/abs/1602.06979)

**Per-category ERP modulation:**
- <a name="citron-2014"></a>**[Citron 2014]** Citron. *Neural correlates of written emotion word processing: A review of recent electrophysiological and hemodynamic neuroimaging studies.* Brain & Language. [doi:10.1016/j.bandl.2013.10.005](https://doi.org/10.1016/j.bandl.2013.10.005)
- <a name="recio-2014"></a>**[Recio 2014]** Recio, Conrad, Hansen, Jacobs. *On pleasure and thrill: The interplay between arousal and valence during visual word recognition.* Brain & Language. [doi:10.1016/j.bandl.2013.11.001](https://doi.org/10.1016/j.bandl.2013.11.001)
- <a name="barber-2013"></a>**[Barber 2013]** Barber, Otten, Kousta, Vigliocco. *Concreteness in word processing: ERP and behavioral effects in a lexical decision task.* Brain & Language. [doi:10.1016/j.bandl.2013.03.005](https://doi.org/10.1016/j.bandl.2013.03.005)

[Pennebaker 2015]: #pennebaker-2015
[Tausczik 2010]: #tausczik-2010
[Fast 2016]: #fast-2016
[Citron 2014]: #citron-2014
[Recio 2014]: #recio-2014
[Barber 2013]: #barber-2013

## Datasets
<!-- auto-populated from docs/datasets/*.md once the dataset registry exists -->
