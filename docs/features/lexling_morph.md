---
key: lexling_morph
name: Morphological features
levels: [word]
gui_toggle: lexling_morph
output_columns: [feats]
libraries: [spaCy]
status: implemented
depth: shallow
---

## Meaning
Per-token morphological feature bundle in **Universal Dependencies (UD)
FEATS** format ([Nivre 2016]): a pipe-separated list of
`Attribute=Value` pairs describing the token's inflectional properties.
String, not parsed.

Examples:

| Token | `feats` value |
|---|---|
| `cats`    | `Number=Plur` |
| `running` | `Aspect=Prog\|Tense=Pres\|VerbForm=Part` |
| `was`     | `Mood=Ind\|Number=Sing\|Person=3\|Tense=Past\|VerbForm=Fin` |
| `the`     | `Definite=Def\|PronType=Art` |
| `.`       | `NA` (no morphological features for PUNCT) |

The `NA` sentinel is used when spaCy returns an empty morphology bag.

## Usage
- Filter on specific attributes — e.g. `feats.str.contains("Tense=Past")`
  to grab all past-tense verbs for a tense-comparison contrast.
- Build per-attribute one-hot covariates for regression analyses where
  finer-grained morphological controls are needed beyond [POS](lexling_pos.md).

## Implementation in SensiExtractor
Source: `extractors/text_parser.py:parse`. Emitted from spaCy's
`token.morph` via `str(token.morph)`, with the literal string `"NA"` as
fallback when the morph bag is empty. No parsing — the value is a raw
pipe-separated string suitable for grep/regex filtering.

## Library
```python
import spacy
nlp = spacy.load("en_core_web_sm")
for tok in nlp("The cats were running."):
    print(tok.text, str(tok.morph) or "NA")
```

## Papers
- <a name="nivre-2016"></a>**[Nivre 2016]** Nivre et al. *Universal Dependencies v1: A Multilingual Treebank Collection.* LREC 2016. [PDF](https://aclanthology.org/L16-1262/) — the UD scheme, including the FEATS column format used here.

[Nivre 2016]: #nivre-2016

## Datasets
<!-- auto-populated from docs/datasets/*.md once the dataset registry exists -->
