---
key: doc_psych_composite
name: Composite psychological scores (Pennebaker 2014)
levels: [document]
gui_toggles: [feat_psych_authentic, feat_psych_analytic, feat_psych_tone, feat_psych_clout]
output_columns: [psych_authentic, psych_analytic, psych_tone, psych_clout]
libraries: [liwc]
status: implemented
depth: deep
---

## Meaning
Four document-level **composite psychological scores** computed from
LIWC category proportions ([Pennebaker 2014], [Pennebaker 2015]):

| Column | What it captures | Direction |
|---|---|---|
| `psych_authentic` | Honest / personal vs. guarded / impersonal | ↑ = more candid, first-person, present-tense |
| `psych_analytic`  | Analytical / formal vs. narrative / informal | ↑ = more articles & prepositions, fewer pronouns & auxiliaries |
| `psych_tone`      | Emotional valence | ↑ = positive emotion dominates negative emotion |
| `psych_clout`     | Social authority / leadership voice | ↑ = more 2nd/3rd-person, fewer 1st-person singular |

All four are reported on a **0–100 scale** (LIWC's standard convention)
and clamped into that range.

> ⚠️ These are **approximations** of LIWC-22's official summary scores.
> The Pennebaker 2014 formulas implemented here are publicly
> documented, but LIWC-22 applies a proprietary z-score-percentile
> post-processing step that may produce slightly different numbers
> from the same input. For exact LIWC-22 values, run LIWC's desktop
> software. For relative comparisons across stimuli, this
> implementation is generally adequate.

## Backend requirement
**Requires the LIWC backend** — Empath does not have the right
categories. The doc-level page greys out the composite-score card
when the active semantic backend is Empath (controlled by
[`ClusterCard.available`](../../gui/widgets.py)).

## EEG rationale
The composite scores are stimulus-level **discourse-style descriptors**
— useful as covariates when matching stimuli on writing register
(authentic personal narrative vs. formal lecture) or for studying
discourse-level effects:

- Tone correlates with affective ERPs and late positive component
  amplitude across narrative passages ([Citron 2014], [Recio 2014]).
- Analytic vs. narrative writing engages different neural systems for
  comprehension — formal academic text produces longer fixations and
  larger N400/P600 amplitudes than narrative on average
  ([Wallot 2014]).
- Clout and Authentic are less validated in ERP literature but are
  standard discourse-psychology descriptors and can be used as
  exploratory covariates.

## Math / linguistics
From [Pennebaker 2014], implemented in
`extractors/liwc_summary.py:compute_liwc_summary`. All formulas take
the **percentage** of tokens in each LIWC category and combine them
linearly:

    analytic = 30 + 0.8·%article + 0.7·%prep − 0.4·(%pronoun + %ipron
               + %auxverb + %conj + %adverb + %negate)

    clout    = 50 + 0.8·%we + 0.8·%you + 0.5·%shehe + 0.5·%they
               − 0.8·%i − 0.4·%ipron

    authentic = 50 + 0.8·%i + 0.4·%discrep + 0.2·%focuspresent
                − 0.4·%we − 0.2·%focusfuture

    tone     = 50 + %posemo − %negemo

All four are then clamped to `[0, 100]`.

## Implementation in SensiExtractor
Source: `extractors/liwc_summary.py:compute_liwc_summary`. Tokenizes
the full document with a simple `[A-Za-z']+` regex (no spaCy
dependency at this stage), parses each token through the LIWC trie
parser, and tallies per-category counts. Then applies the Pennebaker
formulas above. Empty text → all four are `NaN`.

The four toggles are independent — disabling any one suppresses **only
that column** from the doc summary.

## Library
```python
from extractors.liwc_summary import compute_liwc_summary
scores = compute_liwc_summary(text, "dictionaries/liwc-2022.dic")
# {"authentic": 47.2, "analytic": 78.5, "tone": 62.1, "clout": 35.9}
```

## Papers
- <a name="pennebaker-2014"></a>**[Pennebaker 2014]** Pennebaker, Chung, Frazee, Lavergne, Beaver. *When small words foretell academic success: The case of college admissions essays.* PLOS ONE. [doi:10.1371/journal.pone.0115844](https://doi.org/10.1371/journal.pone.0115844) — the formulas implemented here.
- <a name="pennebaker-2015"></a>**[Pennebaker 2015]** Pennebaker, Boyd, Jordan, Blackburn. *The Development and Psychometric Properties of LIWC2015.* UT Austin technical report. [doi:10.15781/T29G6Z](https://doi.org/10.15781/T29G6Z)
- <a name="citron-2014"></a>**[Citron 2014]** Citron. *Neural correlates of written emotion word processing: A review.* Brain & Language. [doi:10.1016/j.bandl.2013.10.005](https://doi.org/10.1016/j.bandl.2013.10.005)
- <a name="recio-2014"></a>**[Recio 2014]** Recio, Conrad, Hansen, Jacobs. *On pleasure and thrill: The interplay between arousal and valence during visual word recognition.* Brain & Language. [doi:10.1016/j.bandl.2013.11.001](https://doi.org/10.1016/j.bandl.2013.11.001)
- <a name="wallot-2014"></a>**[Wallot 2014]** Wallot, Hollis, van Rooij. *Connected text reading and differences in text reading fluency in adult readers.* PLOS ONE. [doi:10.1371/journal.pone.0136471](https://doi.org/10.1371/journal.pone.0136471) — text type / register affects reading and neural responses.

[Pennebaker 2014]: #pennebaker-2014
[Pennebaker 2015]: #pennebaker-2015
[Citron 2014]: #citron-2014
[Recio 2014]: #recio-2014
[Wallot 2014]: #wallot-2014

## Datasets
<!-- auto-populated from docs/datasets/*.md once the dataset registry exists -->
