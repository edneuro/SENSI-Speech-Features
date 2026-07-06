---
key: lexling_dep
name: Dependency parse
levels: [word]
gui_toggle: lexling_dep
output_columns: [dep_rel, head_token_id]
libraries: [spaCy]
status: implemented
depth: deep
---

## Meaning
Per-token syntactic-dependency annotation. Together, `dep_rel` and
`head_token_id` encode a labelled directed tree over the sentence in
which every word points to its syntactic governor.

| Column | Meaning |
|---|---|
| `dep_rel` | Dependency relation label from spaCy's tagset (a superset of Universal Dependencies, [De Marneffe 2021]). Common values: `nsubj`, `obj`, `det`, `amod`, `advmod`, `aux`, `prep`, `pobj`, `compound`, `ROOT`. |
| `head_token_id` | The [`token_id`](structural_ids.md) (global, whole-document token index) of this word's syntactic head. Using the global id makes the reference self-contained against `token_id` and unambiguous across sentence boundaries. For the sentence's root token, `head_token_id` equals the word's own `token_id` (it points to itself). |

> The UD-standard `deps` (Enhanced UDs) and `misc` columns are
> intentionally **not** emitted. spaCy's `en_core_web_sm` does not
> produce enhanced UDs, and the only English content for `misc` would
> be `SpaceAfter=No` — too thin to justify a dedicated column. If a
> concrete EEG use case for enhanced UDs surfaces later, a Stanza-based
> extractor can be added under its own toggle.

## EEG rationale
Dependency structure governs which words are **predictively related** at
each point in the sentence — the head of a word constrains its
distributional context, and unexpected head-dependent combinations
generate distinct ERP signatures ([Friederici 2002] on the P600;
[Kuperberg 2007] on syntax-semantics interaction). For naturalistic-EEG
analyses:

- Dependency length (`|token_id - head_token_id|`) is a classic
  syntactic-complexity proxy ([Gibson 2000] — Dependency Locality
  Theory). Both endpoints are global `token_id` values, so the
  difference is a true linear-distance-in-words measure. Empirical
  evaluation against eye-tracking corpora shows
  dependency-length-based predictions account for reading times in
  some contexts (e.g. for nouns), though not uniformly across all
  word classes ([Demberg 2008]).
- `dep_rel == "ROOT"` marks the syntactic centre of each sentence —
  useful for sentence-level anchoring.
- Dependency-conditioned analyses (e.g. all words whose head is a verb)
  let you isolate effects that POS alone cannot.

## Implementation in SensiExtractor
Source: `extractors/text_parser.py:parse`. `dep_rel` is read directly
from spaCy's parser (`token.dep_`). `head_token_id` is computed as
`token.head.i + 1` to convert spaCy's 0-indexed-document position into
the 1-indexed global [`token_id`](structural_ids.md) convention (the
extractor's `token_id` counter increments in document order starting at
1, so a token's `token_id` is exactly `token.i + 1`).

Parser model is spaCy's `en_core_web_sm` default — a transition-based
parser trained on OntoNotes 5.

## Library
```python
import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp("The cats chased mice.")
for tok in doc:
    print(tok.text, tok.dep_, "->", tok.head.text)
# The   det      -> cats
# cats  nsubj    -> chased
# chased ROOT    -> chased
# mice  dobj     -> chased
# .     punct    -> chased
```

## Papers
- <a name="demarneffe-2021"></a>**[De Marneffe 2021]** De Marneffe, Manning, Nivre, Zeman. *Universal Dependencies.* Computational Linguistics. [doi:10.1162/coli_a_00402](https://doi.org/10.1162/coli_a_00402) — the UD scheme used by `dep_rel`.
- <a name="friederici-2002"></a>**[Friederici 2002]** Friederici. *Towards a neural basis of auditory sentence processing.* Trends in Cognitive Sciences. [doi:10.1016/S1364-6613(00)01839-8](https://doi.org/10.1016/S1364-6613(00)01839-8) — neural model of syntactic processing; basis for P600 interpretation.
- <a name="kuperberg-2007"></a>**[Kuperberg 2007]** Kuperberg. *Neural mechanisms of language comprehension: Challenges to syntax.* Brain Research. [doi:10.1016/j.brainres.2006.12.063](https://doi.org/10.1016/j.brainres.2006.12.063) — syntax-semantics interaction in ERPs.
- <a name="gibson-2000"></a>**[Gibson 2000]** Gibson. *The dependency locality theory: A distance-based theory of linguistic complexity.* In *Image, Language, Brain* (MIT Press). — dependency length as a complexity metric.
- <a name="demberg-2008"></a>**[Demberg 2008]** Demberg & Keller. *Data from eye-tracking corpora as evidence for theories of syntactic processing complexity.* Cognition. [doi:10.1016/j.cognition.2008.07.008](https://doi.org/10.1016/j.cognition.2008.07.008) — dependency length and reading-time measures.

[De Marneffe 2021]: #demarneffe-2021
[Friederici 2002]: #friederici-2002
[Kuperberg 2007]: #kuperberg-2007
[Gibson 2000]: #gibson-2000
[Demberg 2008]: #demberg-2008

## Datasets
<!-- auto-populated from docs/datasets/*.md once the dataset registry exists -->
