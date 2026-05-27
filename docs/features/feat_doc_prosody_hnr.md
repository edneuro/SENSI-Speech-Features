---
key: feat_doc_prosody_hnr
name: Document-level Harmonics-to-Noise Ratio (HNR)
levels: [document]
gui_toggle: feat_doc_prosody_hnr
output_columns: [doc_hnr]
libraries: [parselmouth, praat]
status: implemented
depth: deep
---

## Meaning
Mean **Harmonics-to-Noise Ratio**, in dB, across all voiced frames in
the audio. HNR measures how much of the signal energy is periodic
(harmonics: glottal-pulse-driven) versus aperiodic (noise: breath,
turbulence). Clean studio speech typically sits in **10–25 dB**;
breathy or hoarse voices drop below 10 dB.

A single value per stimulus.

## EEG rationale
HNR is the standard **voice-quality matching covariate** — when
comparing stimuli recorded under different conditions or by different
speakers, gross differences in voice quality (breathy vs. modal vs.
creaky phonation) drive systematically different early auditory
responses ([Banse 1996], [Patel 2018]). Including `doc_hnr` lets you
regress out that confound at the stimulus level.

In clinical / pathological-voice EEG work (e.g. studies of dysphonia,
hearing impairment, ageing voice), HNR is itself the variable of
interest ([Boersma 1993]).

## Math / linguistics
HNR is estimated by Praat as

    HNR(dB) = 10 · log₁₀(periodic_energy / aperiodic_energy)

using the autocorrelation peak relative to the rest of the
autocorrelation function ([Boersma 1993]). Unvoiced frames return
−200 dB (Praat's sentinel for "no harmonic content"); those frames
are excluded from the mean.

## Implementation in SensiExtractor
Computed in `main.py`'s document-batch finishing step from the same
`Contours` object used by the other prosody columns. Per-frame HNR
values come from Praat's `To Harmonicity (cc)` call; the mean
excludes Praat's unvoiced sentinel (−200 dB) — this is the fix from
commit `460801b` referenced in `master_plan.md`.

⚠️ Without the sentinel exclusion, a stimulus with even a few
unvoiced frames would have `doc_hnr` pulled down toward −200 — the
fix ensures the value reflects voiced speech only.

## Library
```python
import parselmouth
snd = parselmouth.Sound("stimulus.wav")
harmonicity = snd.to_harmonicity()
hnr_per_frame = harmonicity.values[0]
hnr_mean = hnr_per_frame[hnr_per_frame > -100].mean()  # mask unvoiced sentinels
```

## Papers
- <a name="boersma-1993"></a>**[Boersma 1993]** Boersma. *Accurate short-term analysis of the fundamental frequency and the harmonics-to-noise ratio of a sampled sound.* IFA Proceedings. [PDF](https://www.fon.hum.uva.nl/paul/papers/Proceedings_1993.pdf) — the HNR algorithm Praat uses.
- <a name="banse-1996"></a>**[Banse 1996]** Banse & Scherer. *Acoustic profiles in vocal emotion expression.* J. Personality and Social Psychology. [doi:10.1037/0022-3514.70.3.614](https://doi.org/10.1037/0022-3514.70.3.614)
- <a name="patel-2018"></a>**[Patel 2018]** Patel, Awan, Barkmeier-Kraemer, et al. *Recommended protocols for instrumental assessment of voice.* American J. Speech-Language Pathology. [doi:10.1044/2018_AJSLP-17-0009](https://doi.org/10.1044/2018_AJSLP-17-0009)

[Boersma 1993]: #boersma-1993
[Banse 1996]: #banse-1996
[Patel 2018]: #patel-2018

## Datasets
<!-- auto-populated from docs/datasets/*.md once the dataset registry exists -->
