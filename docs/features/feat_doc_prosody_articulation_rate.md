---
key: feat_doc_prosody_articulation_rate
name: Articulation rate
levels: [document]
gui_toggle: feat_doc_prosody_articulation_rate
output_columns: [doc_articulation_rate]
libraries: [pyphen]
status: implemented
depth: deep
---

## Meaning
**Syllables per second**, computed over **speaking time only** (audio
duration minus silent pauses). Single value per stimulus.

Typical adult conversational range: **3–6 syllables/sec** ([Tauroza
1990]). Values outside this range usually indicate either a
configuration problem (no aligned words → zero) or an unusual stimulus
(e.g. read poetry at ~2 syl/sec; rapid news broadcast at ~7 syl/sec).

## EEG rationale
Articulation rate is a **stimulus-level speech-rate descriptor** that
is partly under speaker control (slow vs. fast narration) and partly
fixed by the text content (longer words slow it down). For
naturalistic-EEG:

- Stimulus matching: two stimuli that differ in speech rate produce
  systematically different time-locked responses because the same
  fixed-duration post-onset window covers more linguistic content in
  fast speech. Speech-rate matching is essential for any
  cross-stimulus contrast.
- Tracking responses (e.g. cortical envelope tracking): low-frequency
  cortical activity entrains to the syllabic rhythm of speech
  ([Ding 2014] review), and irregular speech rate dissociates this
  entrainment from sound-onset-evoked responses ([Kayser 2015]).

Note this is **articulation rate**, not **speech rate** — the former
excludes pauses, the latter includes them. The two diverge when a
speaker pauses a lot; together with [`doc_pause_*`](doc_prosody_pauses.md),
articulation rate gives a complete picture of timing.

## Math / linguistics
    articulation_rate = total_syllables / (audio_duration − total_pause_duration)

Where:
- `total_syllables` is the sum of syllable counts across all aligned
  content words.
- Pauses are inter-word gaps exceeding the pause-threshold
  (default 0.25 s; configurable via the GUI's pause-threshold
  spinbox).

Syllable counting uses the `pyphen` hyphenation dictionary applied to
each word's surface form — a well-tested approximation but not exact
phonological syllabification. The MFA-derived per-syllable timing
([syllable_timing](syllable_timing.md)) is used when that stage is
also active; otherwise pyphen-based counting is the fallback.

## Implementation in SensiExtractor
Source: `extractors/prosody.py:compute_doc_prosody` — see the
document-batch finishing step in `main.py` which calls it.

**What the code does:**

1. Counts total syllables via `pyphen` over each word's surface form
   (using the same `_count_syllables` helper as the word-level
   prosody extractor).
2. Computes total pause duration as the sum of inter-word gaps
   exceeding the threshold (configurable in the GUI's Prosody card →
   "Pause threshold (sec)").
3. Returns `total_syllables / (audio_duration − total_pause_duration)`.
4. **If `pyphen` is unavailable**, returns `NaN` with a logged
   warning rather than crashing.

⚠️ The pyphen-based syllable count is approximate — a small
systematic error (~5–10 % per language) is expected vs. true
phonological syllabification. For exact counts, use the
[syllable timing extractor](syllable_timing.md) and aggregate from
there.

## Library
```python
import pyphen
dic = pyphen.Pyphen(lang="en_US")
n_syllables_in_unbelievable = dic.inserted("unbelievable").count("-") + 1
# "un-be-liev-a-ble" → 5
```

## Papers
- <a name="tauroza-1990"></a>**[Tauroza 1990]** Tauroza & Allison. *Speech Rates in British English.* Applied Linguistics 11(1), 90–105. [doi:10.1093/applin/11.1.90](https://doi.org/10.1093/applin/11.1.90) — establishes typical conversational ranges.
- <a name="ding-2014"></a>**[Ding 2014]** Ding & Simon. *Cortical entrainment to continuous speech: functional roles and interpretations.* Frontiers in Human Neuroscience 8, 311. [doi:10.3389/fnhum.2014.00311](https://doi.org/10.3389/fnhum.2014.00311) — rate as a modulator of cortical tracking.
- <a name="kayser-2015"></a>**[Kayser 2015]** Kayser, Ince, Gross, Kayser. *Irregular speech rate dissociates auditory cortical entrainment, evoked responses, and frontal alpha.* J. Neuroscience 35(44), 14691–14701. [doi:10.1523/JNEUROSCI.2243-15.2015](https://doi.org/10.1523/JNEUROSCI.2243-15.2015) — speech-rate manipulations and cortical entrainment.

[Tauroza 1990]: #tauroza-1990
[Ding 2014]: #ding-2014
[Kayser 2015]: #kayser-2015

## Datasets
<!-- auto-populated from docs/datasets/*.md once the dataset registry exists -->
