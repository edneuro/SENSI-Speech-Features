---
key: master_word2vec
name: Word2Vec static word embeddings
levels: [word]
gui_toggle: master_word2vec
output_files: ["word/<stim>_word2vec.mat"]
libraries: [gensim]
status: implemented
depth: deep
---

## Meaning
A single 300-dimensional vector per word, looked up from a pretrained
static embedding table (Google News 100B-word corpus). Output shape:
`(n_words, 300)`. The same surface word always gets the same vector
regardless of context — this is the *static* (non-contextual)
counterpart to [BERT](master_bert_word.md).

## EEG rationale
Static distributional vectors capture corpus-level semantic similarity
between words, and similarity in this space is the operational definition
behind many EEG/MEG studies of lexical-semantic processing — the N400,
in particular, scales with semantic relatedness between a target and its
preceding context ([Kutas 2011], [Frank 2015]). Word2Vec is a useful
baseline against contextual embeddings: any predictive advantage that
[BERT](master_bert_word.md) brings over Word2Vec is attributable to
*contextual* information, since both share the same target-word lookup
semantics ([Toneva 2019], [Schrimpf 2021]).

## Math / linguistics
Word2Vec vectors are learned by skip-gram with negative sampling
([Mikolov 2013a]): given a center word, predict its context words,
optimizing a low-dimensional embedding so that words appearing in
similar contexts end up with similar vectors. The Google News 300d
model ships these as a flat lookup table — `word → ℝ³⁰⁰` — with ~3M
vocabulary entries.

The semantic-similarity metric typically used downstream is cosine:

    sim(w₁, w₂) = (v(w₁) · v(w₂)) / (‖v(w₁)‖ · ‖v(w₂)‖)

## Implementation in SensiExtractor
Source: `extractors/word2vec_embed.py`. Design spec:
`docs/superpowers/specs/2026-05-22-embeddings-design.md` §2.

**What the code does** (verified against the source):

1. Loads the pretrained Google News 300d vectors via gensim's
   `KeyedVectors.load_word2vec_format(..., binary=True)`.
2. Default lookup path:
   `hf_cache/word2vec/GoogleNews-vectors-negative300.bin`.
3. If the local file is missing, the model is auto-downloaded via
   `gensim.downloader.load("word2vec-google-news-300")` (~1.7 GB, one
   time, cached by gensim in `~/gensim-data/`).
4. Iterates over the word-level token column and, for each token:
   a. Returns the vector if the exact token is in vocabulary.
   b. Otherwise tries the lowercase form (`token.lower()`).
   c. Otherwise leaves the row as a **zero vector**.
5. Output is `float32`, shape `(n_words, 300)`, row-aligned with the
   word-level CSV.

**Why** *(spec-derived where stated; ⚠️ flags my interpolation):*

- *Google News 300d* is the most widely cited static-embedding baseline
  in the literature, so emitting it gives downstream analysts a feature
  that's directly comparable across studies.
- ⚠️ *Lowercase fallback before zero-fill* — the spec lists this as a
  fact (§2 BERT block doesn't mention it; the §2 Word2Vec block does).
  Likely rationale: Google News vectors are case-sensitive, so
  capitalized sentence-initial words and proper nouns would otherwise
  miss frequent common forms. The fallback recovers them without
  silently lowercasing all rows.
- *OOV → zero vector* matches the spec verbatim. ⚠️ The zero choice
  (rather than a random vector or a learned `<UNK>`) is conservative:
  cosine similarity between a zero vector and anything is zero, so OOV
  rows contribute no spurious similarity signal to downstream EEG
  models. They will, however, lower the norm of any aggregate that
  averages across them — analysts should mask OOV rows before pooling.

**Alternatives that were *not* chosen** (for reader context):

| Approach | Why we didn't pick it |
|---|---|
| GloVe 300d (Common Crawl) | Comparable quality; Google News is more widely cited as the EEG-study baseline. |
| FastText (subword-aware) | Better OOV coverage but the static-embedding role is filled by Google News; FastText's value is mostly for morphologically rich languages. |
| BERT layer-0 (embedding layer) as the "static" embedding | Cheap to compute but it's BERT's WordPiece vocabulary, not whole-word — comparability across studies is worse. |
| Train custom vectors on the stimulus corpus | Tiny corpus → noisy vectors. The whole point of using pretrained vectors is to import the statistics of a huge corpus. |

## Library
```python
import gensim.downloader as gensim_dl
model = gensim_dl.load("word2vec-google-news-300")   # ~1.7 GB on first run
vector = model["bank"]                                # numpy array, shape (300,)
sim    = model.similarity("bank", "river")            # cosine, in [-1, 1]
```

For an offline / pinned setup, drop the `.bin` at
`hf_cache/word2vec/GoogleNews-vectors-negative300.bin` and the
extractor uses it directly without contacting the network.

## Example
```python
from gensim.models import KeyedVectors
import numpy as np

model = KeyedVectors.load_word2vec_format(
    "hf_cache/word2vec/GoogleNews-vectors-negative300.bin", binary=True
)

words = ["The", "cat", "sat"]
mat   = np.zeros((len(words), 300), dtype=np.float32)
for i, tok in enumerate(words):
    if   tok        in model: mat[i] = model[tok]
    elif tok.lower() in model: mat[i] = model[tok.lower()]
    # else: leave zero (OOV)
```

## Papers

**Model:**
- <a name="mikolov-2013a"></a>**[Mikolov 2013a]** Mikolov, Sutskever, Chen, Corrado, Dean. *Distributed Representations of Words and Phrases and their Compositionality.* NeurIPS 2013. [arXiv:1310.4546](https://arxiv.org/abs/1310.4546) — skip-gram with negative sampling; the algorithm behind the Google News 300d vectors.
- <a name="mikolov-2013b"></a>**[Mikolov 2013b]** Mikolov, Chen, Corrado, Dean. *Efficient Estimation of Word Representations in Vector Space.* ICLR Workshop 2013. [arXiv:1301.3781](https://arxiv.org/abs/1301.3781) — original Word2Vec paper.

**Use in language-EEG / brain studies:**
- <a name="kutas-2011"></a>**[Kutas 2011]** Kutas & Federmeier. *Thirty years and counting: Finding meaning in the N400 component of the event-related brain potential (ERP).* Annual Review of Psychology. [doi:10.1146/annurev.psych.093008.131123](https://doi.org/10.1146/annurev.psych.093008.131123) — N400 review; semantic relatedness as the canonical N400 modulator.
- <a name="frank-2015"></a>**[Frank 2015]** Frank, Otten, Galli, Vigliocco. *The ERP response to the amount of information conveyed by words in sentences.* Brain & Language. [doi:10.1016/j.bandl.2014.10.006](https://doi.org/10.1016/j.bandl.2014.10.006) — uses distributional language model predictions to model N400 amplitudes.
- <a name="toneva-2019"></a>**[Toneva 2019]** Toneva & Wehbe. *Interpreting and improving natural-language processing (in machines) with natural language-processing (in the brain).* NeurIPS 2019. [arXiv:1905.11833](https://arxiv.org/abs/1905.11833) — compares contextual vs. static vectors as brain-prediction features.
- <a name="schrimpf-2021"></a>**[Schrimpf 2021]** Schrimpf et al. *The neural architecture of language: Integrative modeling converges on predictive processing.* PNAS. [doi:10.1073/pnas.2105646118](https://doi.org/10.1073/pnas.2105646118) — benchmarks many embedding types against neural data.

[Mikolov 2013a]: #mikolov-2013a
[Mikolov 2013b]: #mikolov-2013b
[Kutas 2011]: #kutas-2011
[Frank 2015]: #frank-2015
[Toneva 2019]: #toneva-2019
[Schrimpf 2021]: #schrimpf-2021

## Datasets
<!-- auto-populated from docs/datasets/*.md once the dataset registry exists -->
