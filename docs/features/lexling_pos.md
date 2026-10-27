---
key: lexling_pos
name: Part-of-speech tags
levels: [word]
gui_toggle: lexling_pos
output_columns: [upos, xpos]
libraries: [spaCy]
status: implemented
depth: deep
---

## Meaning
Two grammatical category tags per word, from spaCy's POS tagger.

| Column | Tagset | Example values | Use case |
|---|---|---|---|
| `upos` | **Universal POS** (17 tags, [Petrov 2012]) | `NOUN`, `VERB`, `ADJ`, `ADV`, `PRON`, `DET`, `PUNCT`, `ADP`, `CCONJ`, `SCONJ`, `INTJ`, `NUM`, `PART`, `AUX`, `PROPN`, `SYM`, `X` | Coarse-grained, cross-language. Use this for filtering (e.g. content words = `NOUN/VERB/ADJ/ADV/PROPN`) and for cross-corpus comparability. |
| `xpos` | Penn Treebank (~36 tags) | `NN`, `NNS`, `NNP`, `VBD`, `VBN`, `VBZ`, `JJR`, `RBS` | Fine-grained, English-specific. Distinguishes inflectional forms (singular vs plural noun, verb tense, comparative/superlative). |

## EEG rationale
Part of speech is one of the oldest known modulators of word-level ERPs:
function words (closed-class: determiners, prepositions, auxiliaries)
versus content words (open-class: nouns, verbs) elicit reliably
different ERP signatures — including a frontal slow negative shift in
the 350–500 ms epoch specifically for closed-class words
([Brown 1999]). Beyond the open/closed-class distinction, syntactic
violations involving part-of-speech mismatches elicit the P600
([Osterhout 1992]). POS is also the standard covariate to regress out
when isolating semantic effects — without controlling for syntactic
class, an "N400 semantic effect" can be confounded by the
closed/open-class distinction ([Kutas 2011]).

For naturalistic-EEG analyses specifically, POS tags drive standard
content-word masks for downstream extractors:
[semantic indicators](sem_indicators.md), syllable nuclei selection
([P-center](feat_syllable_pcenter.md)), and PUNCT exclusion in
[BERT embedding](master_bert_word.md) input.

## Implementation in SensiExtractor
Source: `sensi/extractors/text_parser.py:parse`. Tags are read directly from
spaCy's tagger output (`token.pos_` for `upos`, `token.tag_` for
`xpos`) — no post-processing. Tagger model is `en_core_web_sm`'s
default, a transition-based tagger trained on OntoNotes 5.

## Library
```python
import spacy
nlp = spacy.load("en_core_web_sm")
for tok in nlp("The cats were running quickly."):
    print(tok.text, tok.pos_, tok.tag_)
# The     DET  DT
# cats    NOUN NNS
# were    AUX  VBD
# running VERB VBG
# quickly ADV  RB
# .       PUNCT .
```

## Papers
- <a name="petrov-2012"></a>**[Petrov 2012]** Petrov, Das, McDonald. *A Universal Part-of-Speech Tagset.* LREC 2012. [arXiv:1104.2086](https://arxiv.org/abs/1104.2086) — the 17-tag UPOS scheme used in `upos`.
- <a name="kutas-2011"></a>**[Kutas 2011]** Kutas & Federmeier. *Thirty years and counting: Finding meaning in the N400 component.* Annual Review of Psychology. [doi:10.1146/annurev.psych.093008.131123](https://doi.org/10.1146/annurev.psych.093008.131123) — POS as standard N400 covariate.
- <a name="brown-1999"></a>**[Brown 1999]** Brown, Hagoort, ter Keurs. *Electrophysiological signatures of visual lexical processing: open- and closed-class words.* J. Cognitive Neuroscience 11(3), 261–281. [doi:10.1162/089892999563382](https://doi.org/10.1162/089892999563382) — distinct ERPs to open- vs. closed-class words.
- <a name="osterhout-1992"></a>**[Osterhout 1992]** Osterhout & Holcomb. *Event-related brain potentials elicited by syntactic anomaly.* J. Memory and Language. [doi:10.1016/0749-596X(92)90039-Z](https://doi.org/10.1016/0749-596X(92)90039-Z) — early syntactic-violation ERP work that hinges on POS structure (P600).

[Petrov 2012]: #petrov-2012
[Kutas 2011]: #kutas-2011
[Brown 1999]: #brown-1999
[Osterhout 1992]: #osterhout-1992

## Datasets
<!-- auto-populated from docs/datasets/*.md once the dataset registry exists -->
