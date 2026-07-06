---
key: syllable_timing
name: Syllable timing (Onset / End / Duration)
levels: [word]
gui_toggles: [feat_syllable_onset, feat_syllable_end, feat_syllable_duration]
output_columns: [Onset, End, Duration]
libraries: [montreal-forced-aligner]
status: implemented
depth: deep
---

## Meaning
Per-syllable acoustic timing in the `<stim>_syllables.csv` sidecar:

| Column | Meaning |
|---|---|
| `Onset`    | Time the syllable begins, in seconds from audio start. |
| `End`      | Time the syllable ends, in seconds from audio start. |
| `Duration` | `End − Onset`, in seconds. |

Each row joins back to its word via the foreign-key columns
`(doc_id, sentence_id, token_in_sentence_id, syllable_id)`. Syllables that did not
align (rare — usually an unrecognized phone sequence) are present in
the CSV with `NaN` for all three columns.

## EEG rationale
Syllables are the **natural perceptual unit of rhythm** in speech, and
syllable-locked EEG responses are increasingly used as a finer-grained
alternative to word-locked analyses:

- Syllable onsets drive **delta–theta cortical tracking**
  (1–8 Hz envelope-following responses) — the standard measure of
  speech-rhythm entrainment ([Giraud 2012], [Ding 2014]).
- For analyses of stress or prosody, syllable-level timing is more
  faithful than word-level timing because stress placement falls on
  syllables, not words ([Hayes 2009] — standard phonological
  treatment).
- The combination of `Onset` and [`P_Center`](feat_syllable_pcenter.md)
  gives two complementary syllable anchors — `Onset` for envelope
  alignment, `P_Center` for perceptual-rhythm alignment.

## Implementation in SensiExtractor
Source: `extractors/syllable.py` (downstream of
[`acoustic_boundaries`](acoustic_boundaries.md)). Syllabification is
performed on **MFA's phones tier** via the **Maximum Onset Principle**
(MOP) — phones are grouped into syllables by maximizing the onset
consonant cluster of each syllable subject to language-specific
phonotactic rules ([Hayes 2009]).

Per-syllable timing is then:

    Onset    = onset of the first phone in the syllable
    End      = end of the last phone in the syllable
    Duration = End − Onset

PUNCT and unaligned words contribute no rows. Each word with N
phones generates ≥ 1 syllable rows.

## Library / dependency
- MFA must be installed and the word-level alignment must have run —
  syllable timing is computed from the phones tier of the same
  TextGrid ([`align_with_phones`](acoustic_boundaries.md)).
- `pyphen` is required for the fallback orthographic syllable count
  used when phone-level data is missing — without it the
  [articulation rate](feat_doc_prosody_articulation_rate.md) column
  also degrades to NaN.

## Papers
- <a name="hayes-2009"></a>**[Hayes 2009]** Hayes. *Introductory Phonology.* Wiley-Blackwell. — Maximum Onset Principle reference.
- <a name="giraud-2012"></a>**[Giraud 2012]** Giraud & Poeppel. *Cortical oscillations and speech processing: Emerging computational principles and operations.* Nature Neuroscience. [doi:10.1038/nn.3063](https://doi.org/10.1038/nn.3063) — syllable-rate cortical tracking.
- <a name="ding-2014"></a>**[Ding 2014]** Ding & Simon. *Cortical entrainment to continuous speech: functional roles and interpretations.* Frontiers in Human Neuroscience. [doi:10.3389/fnhum.2014.00311](https://doi.org/10.3389/fnhum.2014.00311)

[Hayes 2009]: #hayes-2009
[Giraud 2012]: #giraud-2012
[Ding 2014]: #ding-2014

## Datasets
<!-- auto-populated from docs/datasets/*.md once the dataset registry exists -->
