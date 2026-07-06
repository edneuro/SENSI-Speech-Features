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
analyses** of speech ([Scott 1998], [Villing 2011]). Whereas physical
syllable onsets vary with consonant cluster length (an /st/ onset is
longer than an /n/ onset), perceptual centres are quasi-isochronous
in connected speech — they're what listeners synchronize to when
tapping along.

For EEG specifically:
- **Cortical envelope tracking**: acoustic landmarks like the
  sharpness of temporal envelope fluctuations drive delta/theta
  cortical oscillations and support perceptual parsing of speech
  ([Doelling 2014]). P-centres formalise a specific class of such
  landmarks at the syllable grain.
- In any analysis where the question is "when does the listener
  perceive this syllable to start?", `P_Center` is the answer;
  `Onset` answers "when did the audio waveform start for this
  syllable" — a different question ([Villing 2011]).

## Implementation in SensiExtractor
Source: `extractors/pcenter.py:add_pcenter`

**What the code does:**

1. Sorts the phoneme DataFrame by
   `(doc_id, sentence_id, token_in_sentence_id, syllable_id, phoneme_id)`.
2. For each syllable, identifies the first phone whose
   [ARPABET](syllable_labels.md) label ends in `0`, `1`, or `2` (the
   vowel nucleus).
3. Computes `P_Center = vowel.Onset − k · Σ(consonants_before.Duration)`,
   rounded to 4 decimal places.
4. If no vowel nucleus is found (extremely rare; only happens for
   malformed syllables), the syllable's physical `Onset` is used as
   fallback.
5. PUNCT / unaligned syllables → `NaN`.

**Why:**

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
- <a name="marcus-1981"></a>**[Marcus 1981]** Marcus. *Acoustic determinants of perceptual center (P-center) location.* Perception & Psychophysics 30(3), 247–256. [doi:10.3758/BF03214280](https://doi.org/10.3758/BF03214280) — first formalisation of P-centre offset from physical onset.
- <a name="scott-1993"></a>**[Scott 1993]** Scott. *Recognition of an Auditory Pattern Based on Time Marked by a Perceptual Centre.* PhD thesis, University College London. — the linear formula with empirical `k = 0.3` for English.
- <a name="scott-1998"></a>**[Scott 1998]** Scott. *The point of P-centres.* Psychological Research 61, 4–11. [doi:10.1007/PL00008162](https://doi.org/10.1007/PL00008162)
- <a name="villing-2011"></a>**[Villing 2011]** Villing, Repp, Ward, Timoney. *Measuring perceptual centers using the phase correction response.* Attention, Perception, & Psychophysics 73(5), 1614–1629. [doi:10.3758/s13414-011-0110-1](https://doi.org/10.3758/s13414-011-0110-1) — modern P-centre measurement validation.
- <a name="doelling-2014"></a>**[Doelling 2014]** Doelling, Arnal, Ghitza, Poeppel. *Acoustic landmarks drive delta–theta oscillations to enable speech comprehension by facilitating perceptual parsing.* NeuroImage 85, 761–768. [doi:10.1016/j.neuroimage.2013.06.035](https://doi.org/10.1016/j.neuroimage.2013.06.035) — sharpness of temporal envelope fluctuations cues syllabic rate and drives delta-theta cortical tracking.

[Marcus 1981]: #marcus-1981
[Scott 1993]: #scott-1993
[Scott 1998]: #scott-1998
[Villing 2011]: #villing-2011
[Doelling 2014]: #doelling-2014

## Datasets
<!-- auto-populated from docs/datasets/*.md once the dataset registry exists -->
