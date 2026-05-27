---
key: lexling_tokens
name: Tokens and lemmas
levels: [word]
gui_toggle: lexling_tokens
output_columns: [token, lemma]
libraries: [spaCy]
status: implemented
depth: shallow
---

## Meaning
The two most basic lexical columns:

| Column | Meaning |
|---|---|
| `token` | Surface form of the word as written in the transcript (`"running"`, `"banks"`, `"."`, `"don't"`). Includes punctuation as separate rows (flagged by `upos == "PUNCT"`). |
| `lemma` | Dictionary-form / base form of the token (`"running" → "run"`, `"banks" → "bank"`, `"are" → "be"`). Useful for collapsing inflectional variants. |

## Usage
- `token` is the input to virtually every downstream extractor
  (frequency lookup, embedding lookup, alignment).
- `lemma` is what you join on if you want to aggregate across
  inflectional forms — e.g. count occurrences of `"run"` across
  `"runs"`, `"running"`, `"ran"`.

## Implementation in SensiExtractor
Source: `extractors/text_parser.py:parse`. Tokens come from spaCy's
default tokenizer (rule-based, splits on whitespace + punctuation
with a few language-specific exceptions for contractions like
`"don't" → ["do", "n't"]`). Lemmas come from spaCy's
`en_core_web_sm` lemmatizer, which combines a lookup table with rule
fallbacks ([Honnibal 2020]).

## Library
```python
import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp("The cats were running.")
for tok in doc:
    print(tok.text, tok.lemma_)
# The The
# cats cat
# were be
# running run
# . .
```

## Papers
- <a name="honnibal-2020"></a>**[Honnibal 2020]** Honnibal, Montani, Van Landeghem, Boyd. *spaCy: Industrial-strength Natural Language Processing in Python.* [doi:10.5281/zenodo.1212303](https://doi.org/10.5281/zenodo.1212303) — the spaCy library and its pipelines.

[Honnibal 2020]: #honnibal-2020

## Datasets
<!-- auto-populated from docs/datasets/*.md once the dataset registry exists -->
