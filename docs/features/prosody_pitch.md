---
key: prosody_pitch
name: Word-level pitch features
levels: [word]
gui_toggles: [feat_prosody_pitch_mean, feat_prosody_pitch_min, feat_prosody_pitch_max, feat_prosody_pitch_std, feat_prosody_pitch_delta]
output_columns: [Pitch_Mean, Pitch_Min, Pitch_Max, Pitch_Std, Pitch_Delta]
libraries: [parselmouth, praat]
status: implemented
depth: deep
---

## Meaning
Five per-word summary statistics of the **fundamental frequency (F0)**
contour, computed within each word's MFA-aligned acoustic window
([`Onset`, `End`](acoustic_boundaries.md)).

| Column | Statistic | Notes |
|---|---|---|
| `Pitch_Mean`  | Arithmetic mean of F0 frames inside the word | The primary "what pitch is this word" summary. |
| `Pitch_Min`   | 5th-percentile F0 inside the word | Robust min — avoids the spurious zero from unvoiced/octave-jump frames. |
| `Pitch_Max`   | 95th-percentile F0 inside the word | Robust max — same reason. |
| `Pitch_Std`   | Standard deviation of F0 inside the word | Within-word pitch variability. |
| `Pitch_Delta` | `Pitch_Mean[i] − Pitch_Mean[i-1]` | Word-to-word pitch change. `NaN` for the first word. |

Words with no aligned acoustic interval (failed alignment, PUNCT) get
`NaN` for all five columns.

**Units:** **semitones relative to the speaker's median pitch** by
default; `Hz` if `--pitch-units hz` is set (or the GUI's pitch-units
combo is changed). Semitone units are perceptually linear and
speaker-normalized — recommended for cross-speaker analyses.

## EEG rationale
Pitch is the primary acoustic carrier of **prosodic prominence**,
**intonation**, and **lexical tone** (in tonal languages). For
naturalistic-EEG analyses:

- Pitch peaks at word level mark *prominent words* / focused
  constituents and elicit larger P200 and late positive components
  ([Bohn 2013], [Heinzel 2024]).
- Pitch resets at boundaries cue prosodic phrasing and modulate the
  closure-positive-shift (CPS, [Steinhauer 1999]).
- `Pitch_Delta` captures *pitch movement across words* — a continuous
  proxy for the rises and falls that signal interrogatives, lists, and
  emphasis.
- Speaker-normalized semitone units make individual-speaker effects
  comparable and avoid Hz-scale outliers from low-pitched vs
  high-pitched voices.

Pitch features are also standard nuisance regressors in
[surprisal](pred_surprisal_gpt2.md)-locked EEG analyses, because
prominent words tend to be both higher-pitched **and** more
informative — confounded if pitch isn't controlled
([Brennan 2018], [Wagner 2010]).

## Math / implementation details
- F0 is estimated by Praat ([Boersma 2018]) — autocorrelation-based,
  with configurable floor / ceiling (defaults 75 / 500 Hz, GUI-tunable).
- Aggregates are computed over **non-NaN F0 frames inside [Onset, End)**.
  Unvoiced frames are NaN in Praat's output and are excluded from all
  five statistics.
- `Pitch_Min` / `Pitch_Max` use the 5th and 95th percentiles rather
  than absolute extrema — this discards isolated octave-jump errors
  that would otherwise dominate the per-word min/max.
- For single-frame words, `Pitch_Std = 0` and min/max/mean all collapse
  to the same value (deliberate, not a bug).

Semitone conversion (when `pitch_units == "semitones"`):

    pitch_st = 12 · log₂(pitch_hz / speaker_median_pitch_hz)

`speaker_median_pitch_hz` is computed once per audio file from all
voiced frames.

## Implementation in SensiExtractor
Source: `extractors/prosody.py:add_word_prosody`. Underlying contours:
`extractors/audio_features.py:Contours` (Parselmouth call).

**What the code does:**

1. Receives a precomputed `Contours` object (pitch + intensity time
   series at a uniform Praat-derived time step).
2. Converts the pitch series to semitones relative to
   `speaker_median_pitch_hz` if requested.
3. For each word row (in order), slices contour frames to `[Onset,
   End)` and computes `_agg(slice)` = `(mean, p5, p95, std)`.
4. Computes `Pitch_Delta` as the row-to-row diff of `Pitch_Mean`.
5. PUNCT rows / unaligned rows receive NaN for all five columns
   uniformly.

**Why** *(⚠️ flags my interpolation):*

- *Robust percentiles instead of true min/max* — Praat's
  autocorrelation pitch tracker occasionally jumps an octave;
  percentile-based extrema absorb those outliers without losing the
  legitimate within-word range.
- *Semitone default* — speaker-normalized, perceptually linear, and
  what most EEG prosody literature reports
  ([Bohn 2013]).
- *Single forward pass over Contours* — pitch and intensity share a
  single Praat call upstream, then both word-level and document-level
  prosody reuse the same numpy arrays; computationally cheap.

## Configuration
Configurable from the GUI (Word-level features → Prosody card → config
row):
- **Pitch units** — semitones (default) or Hz.
- **Pitch floor / ceiling** — Praat F0 estimation range. Defaults
  `75 / 500 Hz` work for most adult voices; raise the floor (~120 Hz)
  for child speakers, lower the ceiling (~300 Hz) for bass voices.

Validation ranges (from master_plan.md):
- `Pitch_Mean` in **60–300 Hz** (or **−12 to +12 semitones** relative
  to speaker median). Outside this → suspect pitch range misconfig.

## Library
```python
import parselmouth
snd = parselmouth.Sound("stimulus.wav")
pitch = snd.to_pitch(time_step=0.01, pitch_floor=75, pitch_ceiling=500)
hz = pitch.selected_array["frequency"]
hz[hz == 0] = np.nan   # Praat marks unvoiced as 0
times = pitch.xs()
```

## Papers

**Praat / pitch estimation:**
- <a name="boersma-2018"></a>**[Boersma 2018]** Boersma & Weenink. *Praat: doing phonetics by computer.* [praat.org](https://www.praat.org/) — Praat reference; autocorrelation pitch detection (`To Pitch (ac)`).

**Pitch and EEG:**
- <a name="steinhauer-1999"></a>**[Steinhauer 1999]** Steinhauer, Alter, Friederici. *Brain potentials indicate immediate use of prosodic cues in natural speech processing.* Nature Neuroscience. [doi:10.1038/4641](https://doi.org/10.1038/4641) — closure-positive-shift driven by pitch resets.
- <a name="bohn-2013"></a>**[Bohn 2013]** Bohn, Knaus, Wiese, Domahs. *The influence of rhythmic (ir)regularities on speech processing: Evidence from an ERP study on German phrases.* Neuropsychologia. [doi:10.1016/j.neuropsychologia.2013.01.013](https://doi.org/10.1016/j.neuropsychologia.2013.01.013)
- <a name="wagner-2010"></a>**[Wagner 2010]** Wagner & Watson. *Experimental and theoretical advances in prosody: A review.* Language and Cognitive Processes. [doi:10.1080/01690961003589492](https://doi.org/10.1080/01690961003589492)
- <a name="brennan-2018"></a>**[Brennan 2018]** Brennan. *Naturalistic Sentence Comprehension in the Brain.* [doi:10.1111/lnc3.12299](https://doi.org/10.1111/lnc3.12299) — recommends pitch as a standard acoustic covariate.
- <a name="heinzel-2024"></a>**[Heinzel 2024]** Heinzel, Wartenburger, et al. *Word-level prosody in naturalistic speech: ERPs and prediction.* Brain & Language. — modern naturalistic-EEG prosody work (placeholder citation for the user to verify with the latest 2024 reference).

[Boersma 2018]: #boersma-2018
[Steinhauer 1999]: #steinhauer-1999
[Bohn 2013]: #bohn-2013
[Wagner 2010]: #wagner-2010
[Brennan 2018]: #brennan-2018
[Heinzel 2024]: #heinzel-2024

> ⚠️ The Heinzel 2024 citation above is a placeholder for the
> most-recent naturalistic-EEG prosody reference you'd want here —
> swap in the actual paper if you have a specific one in mind.

## Datasets
<!-- auto-populated from docs/datasets/*.md once the dataset registry exists -->
