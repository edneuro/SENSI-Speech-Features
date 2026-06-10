---
key: lexical_norms
name: Lexical norms (AoA, concreteness, imageability, familiarity, VAD)
levels: [word, document]
gui_toggles: [feat_norm_aoa_mean, feat_norm_aoa_sd, feat_norm_concreteness, feat_norm_imageability, feat_norm_familiarity, feat_norm_valence, feat_norm_arousal, feat_norm_dominance, doc_agg_lexical_norms]
output_columns: [AoA_Mean, AoA_SD, Concreteness, Imageability, Familiarity, Valence, Arousal, Dominance]
libraries: [pyarrow]
status: implemented
depth: deep
---

> ℹ️ **Setup.** These columns are populated from published norm tables placed
> under `dictionaries/` as parquet files. The quickest way is to run
> `python download_lexical_norms.py` from the project root, which fetches AoA,
> concreteness, and VAD and writes the parquets automatically (MRC is built
> manually — see [`dictionaries/README.md`](../../dictionaries/README.md) for
> all sources, columns, and build steps). If a file is absent, its columns are
> emitted as `NaN` with a warning. The feature is **off by default**; enable
> the *Lexical norms* card once the files are in place.

## Meaning
Per-word psycholinguistic ratings looked up from published norm databases,
keyed by lemma (with a surface-token fallback):

| Column | Norm | What it captures |
|---|---|---|
| `AoA_Mean` | Age of acquisition ([Kuperman 2012]) | mean rated age (years) at which the word is learned |
| `AoA_SD` | " | rater dispersion of the AoA estimate (quality/weight) |
| `Concreteness` | [Brysbaert 2014] | how concrete (perceptible) vs. abstract the referent is (1–5) |
| `Imageability` | MRC ([Coltheart 1981]) | how easily the word evokes a mental image |
| `Familiarity` | MRC | how familiar the word feels to typical readers |
| `Valence` | [Warriner 2013] | pleasantness, 1 (unpleasant) – 9 (pleasant) |
| `Arousal` | " | calm (1) – excited (9) |
| `Dominance` | " | feeling controlled (1) – in control (9) |

Out-of-vocabulary words and punctuation are `NaN` (**never imputed** — a blank
means "no norm available", and downstream code decides how to handle it).

## EEG rationale
These are among the most-used per-word regressors in naturalistic and
single-word language ERP work:

- **AoA** independently predicts **eye-fixation durations** in reading
  ([Juhasz 2003], [Dirix 2017]) and **N400** amplitude ([Yum 2019]), over and
  above word frequency — useful as a developmental-exposure covariate.
- **Concreteness / imageability** modulate the **N400**: concrete/imageable
  words elicit different amplitudes from abstract ones ([Barber 2013]).
- **Valence / arousal** drive emotion effects: arousal appears early in the
  **early posterior negativity** (EPN), with interactive arousal × valence
  effects ([Citron 2012] review; [Recio 2014]).

At the **document level** the same norms become passage covariates (mean
concreteness of a stimulus, mean/range of AoA, proportion of late-acquired
vocabulary) for matching or controlling stimuli.

> ⚠️ **Modeling caution.** Frequency, AoA, word length and concreteness are
> strongly intercorrelated, and **concreteness ↔ imageability correlate ≈0.8**.
> Don't enter collinear norms in one model without checking VIF or
> residualizing/PCA first. The length-like norms are best treated as control
> regressors, not effects of interest.

## Document-level aggregates
Enabling **Aggregates → Lexical-norm aggregates** (`doc_agg_lexical_norms`)
adds, computed over **content words** (`NOUN/VERB/ADJ/ADV/PROPN`):

- `mean_aoa`, `median_aoa`, `max_aoa`, `sd_aoa`
- `prop_late_acquired` — fraction of (AoA-matched) content words with
  `AoA_Mean` above the late-acquired threshold (default **10 yrs**,
  configurable via `lexical_norms.late_acquired_threshold`); an arbitrary
  convention, documented as such.
- `mean_concreteness`, `mean_imageability`, `mean_familiarity`,
  `mean_valence`, `mean_arousal`, `mean_dominance`
- per-norm coverage: `aoa_coverage`, `concreteness_coverage`, … (fraction of
  content tokens that matched the norm — read these to judge how trustworthy
  each mean is).

### Relative AoA (analysis-time, not a stored column)
A useful derived variable for developmental studies is
`relative_aoa = AoA_Mean − participant_age` (years): a word's difficulty
relative to the listener. SensiExtractor never sees participant age, so this
is computed downstream by joining the stimulus `AoA_Mean` with participant
metadata. Suggested uses: an AoA × age interaction term, an above/below-age
word contrast, or a document-level proportion-above-age covariate. Caveat:
AoA is an adult-rated population norm, not a measure of the individual
listener's vocabulary.

## Implementation in SensiExtractor
Source: `extractors/lexical_norms.py:add_lexical_norms`. For each source it
loads a parquet (cached), indexes by `word`, and does a vectorized
`reindex` lookup on `lemma.lower()`, falling back to `token.lower()` where the
lemma misses. PUNCT rows are excluded. **MRC gotcha:** a stored `0` in MRC's
imageability/familiarity fields means "missing" and is converted to `NaN` on
load. Document-level aggregates live in
`utils/doc_summary.py:_lexical_norm_aggregates`.

## Library
```python
from extractors.lexical_norms import add_lexical_norms
df = add_lexical_norms(df, paths={
    'aoa':          'dictionaries/aoa_kuperman.parquet',
    'concreteness': 'dictionaries/concreteness_brysbaert.parquet',
    'mrc':          'dictionaries/mrc.parquet',
    'warriner':     'dictionaries/warriner_vad.parquet',
})
```

## Papers
- <a name="kuperman-2012"></a>**[Kuperman 2012]** Kuperman, Stadthagen-Gonzalez, Brysbaert. *Age-of-acquisition ratings for 30,000 English words.* Behavior Research Methods 44, 978–990. [doi:10.3758/s13428-012-0210-4](https://doi.org/10.3758/s13428-012-0210-4)
- <a name="brysbaert-2014"></a>**[Brysbaert 2014]** Brysbaert, Warriner, Kuperman. *Concreteness ratings for 40 thousand generally known English word lemmas.* Behavior Research Methods 46, 904–911. [doi:10.3758/s13428-013-0403-5](https://doi.org/10.3758/s13428-013-0403-5)
- <a name="coltheart-1981"></a>**[Coltheart 1981]** Coltheart. *The MRC Psycholinguistic Database.* Quarterly J. Experimental Psychology 33A, 497–505. [doi:10.1080/14640748108400805](https://doi.org/10.1080/14640748108400805) — imageability & familiarity.
- <a name="warriner-2013"></a>**[Warriner 2013]** Warriner, Kuperman, Brysbaert. *Norms of valence, arousal, and dominance for 13,915 English lemmas.* Behavior Research Methods 45, 1191–1207. [doi:10.3758/s13428-012-0314-x](https://doi.org/10.3758/s13428-012-0314-x)
- <a name="barber-2013"></a>**[Barber 2013]** Barber, Otten, Kousta, Vigliocco. *Concreteness in word processing: ERP and behavioral effects in a lexical decision task.* Brain & Language 125(1), 47–53. [doi:10.1016/j.bandl.2013.01.005](https://doi.org/10.1016/j.bandl.2013.01.005)
- <a name="juhasz-2003"></a>**[Juhasz 2003]** Juhasz & Rayner. *Investigating the effects of a set of intercorrelated variables on eye fixation durations in reading.* JEP:LMC 29(6), 1312–1318. [doi:10.1037/0278-7393.29.6.1312](https://doi.org/10.1037/0278-7393.29.6.1312)
- <a name="dirix-2017"></a>**[Dirix 2017]** Dirix & Duyck. *The first- and second-language age of acquisition effect in first- and second-language book reading.* JML 97, 103–120. [doi:10.1016/j.jml.2017.07.012](https://doi.org/10.1016/j.jml.2017.07.012)
- <a name="yum-2019"></a>**[Yum 2019]** Yum & Law. *Interactions of age of acquisition and lexical frequency effects… An ERP study.* Psychophysiology 56(10), e13433. [doi:10.1111/psyp.13433](https://doi.org/10.1111/psyp.13433)
- <a name="citron-2012"></a>**[Citron 2012]** Citron. *Neural correlates of written emotion word processing: a review…* Brain & Language 122(3), 211–226. [doi:10.1016/j.bandl.2011.12.007](https://doi.org/10.1016/j.bandl.2011.12.007)
- <a name="recio-2014"></a>**[Recio 2014]** Recio, Conrad, Hansen, Jacobs. *On pleasure and thrill: the interplay between arousal and valence…* Brain & Language 134, 34–43. [doi:10.1016/j.bandl.2014.03.009](https://doi.org/10.1016/j.bandl.2014.03.009)

[Kuperman 2012]: #kuperman-2012
[Brysbaert 2014]: #brysbaert-2014
[Coltheart 1981]: #coltheart-1981
[Warriner 2013]: #warriner-2013
[Barber 2013]: #barber-2013
[Juhasz 2003]: #juhasz-2003
[Dirix 2017]: #dirix-2017
[Yum 2019]: #yum-2019
[Citron 2012]: #citron-2012
[Recio 2014]: #recio-2014

## Datasets
<!-- auto-populated from docs/datasets/*.md once the dataset registry exists -->
