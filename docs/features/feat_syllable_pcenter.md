---
key: feat_syllable_pcenter
name: Perceptual centre (P-center)
levels: [word]
gui_toggle: feat_syllable_pcenter
output_columns: [P_Center]
libraries: []
status: implemented
depth: deep
---

## Meaning
**Perceptual onset** of each syllable — the time point listeners
perceive as the syllable's "beat" — rather than the physical acoustic
onset. Scott's ([Scott 1993]) formula, originally from
[Marcus 1981]:

    P_Center = nucleus_onset − k · onset_consonant_duration

Where:
- `nucleus_onset` = onset time of the syllable's vowel (the first
  ARPABET phone whose label ends in a stress digit `0/1/2`).
- `onset_consonant_duration` = total duration of the consonant(s)
  preceding the vowel.
- `k` = an empirical coefficient (default **0.3**, configurable on the
  Sub-word features → P-center card).

A syllable like `"strike"` (`S T R AY1 K`) has a long consonant onset;
its physical `Onset` (start of /s/) is earlier than its `P_Center`
(near the vowel /AY/). A syllable like `"oh"` (`OW1`) has no
preceding consonants, so `P_Center ≈ Onset`.

Stored to 4 decimal places, in seconds from audio start. PUNCT and
unaligned syllables → `NaN`.

## EEG rationale
The **perceptual centre is the right anchor for rhythm-related
analyses** of speech ([Scott 1998], [Villing 2010]). Whereas physical
syllable onsets vary with consonant cluster length (an /st/ onset is
longer than an /n/ onset), perceptual centres are quasi-isochronous
in connected speech — they're what listeners synchronize to when
tapping along.

For EEG specifically:
- **Cortical envelope tracking** anchored on P-centres often produces
  cleaner delta/theta-band responses than tracking on physical onsets
  ([Doelling 2019]).
- **Beat-induced expectations** during music-rhythm-style speech
  analyses use P-centres as the regular pulse ([Lerdahl 1983]).
- In any analysis where the question is "when does the listener
  perceive this syllable to start?", `P_Center` is the answer;
  `Onset` answers "when did the audio waveform start for this
  syllable" — a different question.

## Implementation in SensiExtractor
Source: `extractors/pcenter.py:add_pcenter`. Spec:
`docs/superpowers/specs/2026-05-20-pcenter-design.md`.

**What the code does:**

1. Sorts the phoneme DataFrame by
   `(doc_id, sentence_id, token_id, syllable_id, phoneme_id)`.
2. For each syllable, identifies the first phone whose
   [ARPABET](syllable_labels.md) label ends in `0`, `1`, or `2` (the
   vowel nucleus).
3. Computes `P_Center = vowel.Onset − k · Σ(consonants_before.Duration)`,
   rounded to 4 decimal places.
4. If no vowel nucleus is found (extremely rare; only happens for
   malformed syllables), the syllable's physical `Onset` is used as
   fallback.
5. PUNCT / unaligned syllables → `NaN`.

**Why** *(spec-derived where stated; ⚠️ flags my interpolation):*

- *k = 0.3 default* — the original [Scott 1993] empirically derived
  value for English speech. Configurable for studies that want
  a different weighting.
- ⚠️ *Vowel-nucleus identification by trailing stress digit* — spec
  uses ARPABET convention; all ARPABET vowels carry stress digits and
  no consonants do, so this is a clean discriminator.
- *Fallback to physical Onset* — preserves a usable value for the
  rare malformed-syllable case so the column doesn't sprinkle NaN
  unexpectedly.

## Configuration
The `k` coefficient is configurable on the Sub-word features → P-center
card (`Sub-word features → P-center → "k coefficient (0–1)"` spinbox)
and via the CLI `--pcenter-k` flag. Default `0.3` matches the
[Scott 1993] original. Higher `k` pulls the P-centre earlier (toward
the syllable onset); `k=0` collapses `P_Center` to `nucleus_onset`.

## Library
The implementation is pure pandas / numpy — no third-party library
beyond the ARPABET data already provided by MFA.

## Papers
- <a name="marcus-1981"></a>**[Marcus 1981]** Marcus. *Acoustic determinants of perceptual centre (P-centre) location.* Perception & Psychophysics. [doi:10.3758/BF03204357](https://doi.org/10.3758/BF03204357) — first formalisation of P-centre offset from physical onset.
- <a name="scott-1993"></a>**[Scott 1993]** Scott. *Recognition of an Auditory Pattern Based on Time Marked by a Perceptual Centre.* PhD thesis, University College London. — the linear formula with empirical `k = 0.3` for English.
- <a name="scott-1998"></a>**[Scott 1998]** Scott. *The point of P-centres.* Psychological Research. [doi:10.1007/s004260050050](https://doi.org/10.1007/s004260050050)
- <a name="villing-2010"></a>**[Villing 2010]** Villing, Repp, Ward, Timoney. *Measuring perceptual centers using the phase correction response.* Attention, Perception, & Psychophysics. [doi:10.3758/APP.72.5.1389](https://doi.org/10.3758/APP.72.5.1389) — modern P-centre measurement validation.
- <a name="doelling-2019"></a>**[Doelling 2019]** Doelling, Arnal, Ghitza, Poeppel. *Acoustic landmarks drive delta–theta oscillations to enable speech comprehension by facilitating perceptual parsing.* NeuroImage. [doi:10.1016/j.neuroimage.2014.05.054](https://doi.org/10.1016/j.neuroimage.2014.05.054) — perceptual onsets and cortical tracking.
- <a name="lerdahl-1983"></a>**[Lerdahl 1983]** Lerdahl & Jackendoff. *A Generative Theory of Tonal Music.* MIT Press. — beat / perceptual-pulse framework.

[Marcus 1981]: #marcus-1981
[Scott 1993]: #scott-1993
[Scott 1998]: #scott-1998
[Villing 2010]: #villing-2010
[Doelling 2019]: #doelling-2019
[Lerdahl 1983]: #lerdahl-1983

## Datasets
<!-- auto-populated from docs/datasets/*.md once the dataset registry exists -->
