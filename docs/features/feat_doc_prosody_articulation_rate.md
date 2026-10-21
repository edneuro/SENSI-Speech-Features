---
key: feat_doc_prosody_articulation_rate
name: Articulation rate
levels: [document]
gui_toggle: feat_doc_prosody_articulation_rate
output_columns: [doc_articulation_rate]
libraries: [pyphen, montreal-forced-aligner]
status: implemented
depth: deep
---

## Meaning
**Syllables per second of speaking time**: the speech span (first
word's onset to last word's end) minus the silent pauses longer than the
pause threshold. Single value per stimulus.

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
    articulation_rate = total_syllables / (speech_span − total_pause_duration)

Where:
- `speech_span` = last aligned word's `End` − first aligned word's
  `Onset`.
- `total_pause_duration` is the sum of inter-word gaps **longer than**
  the pause threshold (default 0.25 s; the same threshold as
  [`doc_pause_*`](doc_prosody_pauses.md)). Shorter gaps between words
  count as speaking time, which is the standard definition and is what
  makes the threshold matter.
- `total_syllables` counts every aligned word (all parts of speech,
  punctuation excluded), from the pronunciation MFA aligned: one
  syllable per vowel phone. A word MFA could not pronounce (aligned to
  a single `spn` "spoken noise" phone, see [g2p_oov](g2p_oov.md)) is
  counted with the `pyphen` spelling-based estimate instead.

Counting from MFA's phones follows the pronunciation actually used
(e.g. "every" as 2 or 3 syllables, depending on the dictionary entry
MFA chose) rather than the spelling, and counts a contraction that MFA
aligns as one stretch ("wasn't") once.

## Implementation in SensiExtractor
Source: `extractors/prosody.py:compute_doc_prosody` — see the
document-batch finishing step in `main.py` which calls it.

**What the code does:**

1. Takes the MFA-aligned words (rows with `Onset`/`End`), sorted by
   onset, and computes the speech span and the pauses (gaps longer than
   the threshold set in the GUI's Prosody card → "Pause threshold
   (sec)").
2. Counts syllables with `_count_spoken_syllables`: vowel phones in the
   MFA phone tier, plus a `pyphen` count for each `spn` word. When no
   phone tier is available it falls back to `pyphen` over every aligned
   non-punctuation token.
3. Returns `total_syllables / (speech_span − total_pause_duration)`, or
   `NaN` if that speaking time is not positive or `pyphen` is needed but
   unavailable.

The syllable count no longer depends on whether the syllable-timing
outputs are switched on: the phone tier comes from the same alignment
every time.

**History (2026-10-21):** speaking time used to be the sum of each
word's MFA duration. That dropped every short gap between words, so the
pause threshold had no effect, and it counted a contraction sharing one
MFA interval twice. Syllables used to be counted with `pyphen` over the
spelling. On three validation stimuli the new values were 1–10 % higher
(4.35 → 4.40, 3.52 → 3.85, 4.75 → 4.85 syl/s), mostly because MFA's
pronunciations contain more syllables than the spelling-based estimate
(e.g. 39 vs 33 for one stimulus).

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
