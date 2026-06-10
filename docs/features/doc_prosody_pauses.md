---
key: doc_prosody_pauses
name: Document-level pause statistics
levels: [document]
gui_toggles: [feat_doc_prosody_pause_count, feat_doc_prosody_pause_mean_duration, feat_doc_prosody_pause_total_duration, feat_doc_prosody_pause_pct_silence]
output_columns: [doc_pause_count, doc_pause_mean_duration, doc_pause_total_duration, doc_pause_pct_silence]
libraries: []
status: implemented
depth: deep
---

## Meaning
Four document-level summary statistics of **silent pauses** in the
audio. A "pause" is an inter-word silence (`Word_Gap`) longer than a
configurable threshold (**default 0.25 s**; GUI-tunable on the
Document-level features → Prosody card).

| Column | Meaning | Unit |
|---|---|---|
| `doc_pause_count`          | Number of inter-word gaps exceeding the threshold. | integer |
| `doc_pause_mean_duration`  | Mean duration of those pauses. | seconds |
| `doc_pause_total_duration` | Sum of those pauses. | seconds |
| `doc_pause_pct_silence`    | `doc_pause_total_duration / audio_duration` (as percent). | percent (0–100) |

## EEG rationale
Pauses chunk continuous speech into **prosodic phrases** and **clauses**
— each pause boundary is an event-marker in its own right and elicits
the **Closure Positive Shift (CPS)** ([Steinhauer 1999], [Pauker 2011]).
Document-level pause statistics let you:

- **Match stimuli on pausing behaviour** before running a cross-stimulus
  contrast — a stimulus full of long pauses has very different
  envelope structure from one without.
- **Predict listener engagement** — speech with more / longer pauses
  is easier to process at low speech rates, an effect documented in the
  pausing-and-comprehension literature
  ([Goldman-Eisler 1968]); ERP signatures of pause-related boundaries
  appear as the closure-positive shift ([Steinhauer 1999], [Pauker 2011]).
- **Distinguish pause types** (combined with [`articulation_rate`](feat_doc_prosody_articulation_rate.md)): a stimulus
  with low articulation rate AND many pauses is genuinely slow; one
  with high articulation rate AND many pauses is fast-but-bursty.

`doc_pause_pct_silence` is the most-cited single-number summary of
"how much of the audio is silence" — useful for stimulus-level
matching and as a sanity check (if it's 0 %, alignment may have
failed; if > 30 %, you may have unintended long silences at the start
or end of the audio).

## Configuration
The pause threshold is shared with the document-level **Pause
threshold (sec)** spinbox in the Prosody card. Default `0.25 s`
matches the convention in much of the prosody literature
([Goldman-Eisler 1968]), but the right value depends on your
discourse:

- Conversational dialogue: 0.20 – 0.30 s
- Read narrative / audiobook: 0.30 – 0.50 s
- Lecture / formal speech: 0.40 – 0.60 s

The same threshold applies to all four columns — they're not
independently tunable.

## Implementation in SensiExtractor
Source: `extractors/prosody.py` (document-level prosody section). All
four columns derive from the **word-level [`Word_Gap`](acoustic_durations.md)
column** plus the audio duration:

```python
gaps = word_df["Word_Gap"].dropna()
pauses = gaps[gaps > threshold_s]
doc_pause_count          = int(len(pauses))
doc_pause_total_duration = float(pauses.sum())
doc_pause_mean_duration  = float(pauses.mean()) if len(pauses) else float("nan")
doc_pause_pct_silence    = 100.0 * doc_pause_total_duration / audio_duration_s
```

When `Word_Gap` is not present (alignment off), all four columns are
`NaN`. The doc-level page's preflight surfaces a warning in that
case.

## Papers
- <a name="goldman-eisler-1968"></a>**[Goldman-Eisler 1968]** Goldman-Eisler. *Psycholinguistics: Experiments in Spontaneous Speech.* Academic Press. — foundational study of pause distribution in spontaneous conversation, including the canonical hesitation-pause / juncture-pause distinction.
- <a name="steinhauer-1999"></a>**[Steinhauer 1999]** Steinhauer, Alter, Friederici. *Brain potentials indicate immediate use of prosodic cues in natural speech processing.* Nature Neuroscience 2(2), 191–196. [doi:10.1038/5757](https://doi.org/10.1038/5757) — CPS at prosodic boundaries.
- <a name="pauker-2011"></a>**[Pauker 2011]** Pauker, Itzhak, Baum, Steinhauer. *Effects of cooperating and conflicting prosody in spoken English garden path sentences: ERP evidence for the boundary deletion hypothesis.* J. Cognitive Neuroscience 23(10), 2731–2751. [doi:10.1162/jocn.2011.21610](https://doi.org/10.1162/jocn.2011.21610)

[Goldman-Eisler 1968]: #goldman-eisler-1968
[Steinhauer 1999]: #steinhauer-1999
[Pauker 2011]: #pauker-2011

## Datasets
<!-- auto-populated from docs/datasets/*.md once the dataset registry exists -->
