# Acoustic Prosody — Design

**Date:** 2026-05-12
**Status:** Draft for review
**Authors:** Amilcar Malave (PI: Blair Kaneshiro), Claude
**Scope:** SensiExtractor — adds word-level and document-level acoustic prosody features.
**Roadmap item:** Originally listed as "Batch 2 — Acoustic prosody" in `2026-05-10-feature-batch-1-design.md` §11.

---

## 1. Background

SensiExtractor currently emits text-and-timing features per word (spaCy parse, SUBTLEX/EWFG frequency, GPT-2 and LLM surprisal, MFA onset/end). It does not yet compute any prosodic acoustic features. Brain Treebank and related naturalistic-stimulus EEG corpora typically expose per-word pitch (F0) and volume (intensity) summaries, plus their word-to-word deltas. This spec adds those features along with document-level prosody aggregates relevant to EEG analyses of naturalistic speech.

## 2. Goals

1. Add **10 word-level prosody columns** to the per-stimulus word CSV: pitch and volume mean / min / max / std, plus pitch and volume word-to-word deltas.
2. Add **16 document-level prosody columns** to the documents_summary CSV: the same 8 pitch+volume aggregates at the document level, mean absolute deltas, HNR, articulation rate, and 4 pause statistics.
3. Each new column is independently togglable via the GUI (per-column checkbox) and via CLI flags.
4. Configuration (pitch units, pitch search range, pause threshold) is exposed in the GUI and via CLI flags, with defaults persisted to `config/preferences.json`.

Existing column ordering and semantics are unchanged. The prosody additions are purely additive to the schema.

## 3. Non-goals (deferred)

- **Syllable-level prosody** — deferred to the original Roadmap Batch 3 (phoneme/syllable data model), since the syllable-segmentation infrastructure is not yet built.
- **Sentence-level prosody** — deferred to its own future mini-batch. Sentence boundaries from spaCy (`sentence_id`) exist in the word CSV today, but emitting a per-sentence sidecar CSV is a data-model change we should decide separately.
- **Voice quality** — jitter, shimmer, formants. Parselmouth can compute these; the choice is deferred until a downstream EEG analysis specifically calls for them.
- **Within-word slope deltas** — only word-to-word deltas (Brain Treebank style) are computed.
- **Cross-speaker normalization** beyond the per-stimulus semitone reference. SensiExtractor assumes one speaker per stimulus.

## 4. Backend choice

**Praat-Parselmouth** (`praat-parselmouth>=0.4`) is the sole acoustic-analysis library.

Rationale: Parselmouth wraps Praat, which is the industry-standard tool for pitch (autocorrelation-based, the most accurate widely-used algorithm) and is the only common Python interface that exposes HNR directly. Single dependency, cross-platform wheels.

Librosa was considered as an alternative but rejected: librosa's pYIN pitch tracker is less accurate than Praat for natural speech, and librosa does not provide HNR (which we want at the document level).

`pyphen>=0.14` is added as a small secondary dependency for text-based syllable counting (needed only for the articulation-rate column).

## 5. Architecture

Two new modules:

```
extractors/
  audio_features.py    NEW. Wraps the single Parselmouth pass per stimulus.
                       Returns a Contours dataclass:
                         pitch_hz: np.ndarray (1D, NaN at unvoiced frames)
                         intensity_db: np.ndarray (1D)
                         times: np.ndarray (1D, frame timestamps in sec)
                         frame_rate_hz: int (default 100)
                         speaker_median_pitch_hz: float
                         hnr_db: float (whole-stimulus HNR)
  prosody.py           NEW. Pure-pandas aggregators (no Parselmouth):
                         add_word_prosody(df, contours, pitch_units) -> df
                         compute_doc_prosody(df, contours, text,
                                              pause_threshold_s, pitch_units)
                              -> dict[str, float]
```

The split exists so that signal processing happens **once per stimulus** (in `audio_features.py`) and aggregation/summarization is a pure DataFrame operation (in `prosody.py`) that is unit-testable with synthetic contour fixtures — no audio file required for most tests.

### 5.1 `audio_features.extract(audio_path, pitch_floor_hz, pitch_ceiling_hz)`

Single function. Returns a `Contours` dataclass. Internally:

1. Load the audio file via `parselmouth.Sound`.
2. Compute pitch via `Sound.to_pitch_ac(pitch_floor=pitch_floor_hz, pitch_ceiling=pitch_ceiling_hz)`. Frame rate defaults to 100 Hz (Praat default; 10 ms time step). Unvoiced frames return NaN.
3. Compute intensity via `Sound.to_intensity()` (default settings: minimum pitch=100 Hz, time step matches pitch).
4. Resample both contours onto a common time grid (Parselmouth's intensity and pitch use the same time step by default, but defensive resample handles edge cases).
5. Compute whole-stimulus HNR via `Sound.to_harmonicity_cc()`; take the mean over voiced frames.
6. Compute `speaker_median_pitch_hz` = median of voiced pitch frames. Used as the semitone reference if `pitch_units="semitones"`.

If Parselmouth is not importable, `extract` raises `ProsodyUnavailable`; callers handle the exception by emitting NaN for all prosody columns and recording a warning in the run manifest.

### 5.2 `prosody.add_word_prosody(df, contours, pitch_units)`

For each word row with non-NaN `Onset` and `End`:

1. Slice `contours.pitch_hz` and `contours.intensity_db` to the frames within `[Onset, End]`.
2. Convert pitch slice to the target unit:
   - `"hz"`: no-op.
   - `"semitones"`: `12 * log2(hz / contours.speaker_median_pitch_hz)`.
3. Compute the 10 columns. Min/max use `np.nanpercentile(..., [5, 95])` over voiced frames (robust to pitch-tracker glitches at word boundaries). Std uses raw values, no trimming.
4. Words with zero voiced frames in their interval → all `Pitch_*` columns NaN; `Volume_*` columns still computed from intensity frames.
5. Words with NaN `Onset` or `End` (MFA failure) → all 10 prosody columns NaN.

After per-word computation, `Pitch_Delta` and `Volume_Delta` are added by `df[col].diff()` (NaN for the first word of the stimulus, matching `Word_Gap` convention).

### 5.3 `prosody.compute_doc_prosody(df, contours, text, pause_threshold_s, pitch_units)`

Returns a flat `dict[str, float]` with the 16 doc-level columns. Internally:

1. **Pitch/volume aggregates** — computed over the entire `contours.pitch_hz` (voiced frames only) and `contours.intensity_db` arrays, with the same min/max p5/p95 trimming as word level.
2. **Mean absolute deltas** — `df["Pitch_Delta"].abs().mean()`, `df["Volume_Delta"].abs().mean()`. Computed from the per-word delta columns (which `add_word_prosody` has already produced), so requires word-level prosody to have run first within the same stimulus.
3. **HNR** — `contours.hnr_db`.
4. **Articulation rate** — for each non-punctuation word row with non-NaN `Onset`/`End`, count syllables by calling `pyphen.Pyphen(lang='en_US').inserted(token).count('-') + 1` on the **surface form** (not the lemma — "running" has 2 syllables, "run" has 1). Sum across those rows for the numerator. Denominator is the sum of `End - Onset` across the same rows. NaN if `pyphen` is not importable, or if denominator is zero.
5. **Pause stats** — successive non-NaN word rows (after sorting by Onset) with `Onset[i] - End[i-1] > pause_threshold_s` are counted as pauses. Then:
   - `doc_pause_count`: number of such gaps.
   - `doc_pause_mean_duration`: mean of those gaps (NaN if count == 0).
   - `doc_pause_total_duration`: sum of those gaps.
   - `doc_pause_pct_silence`: total pause time divided by total stimulus duration (taken as `End[last] - Onset[first]`).

### 5.4 Integration into the pipeline

**`main.py:run()`** gains one new gated block, inserted after `_merge_alignment(df, align_df)` and before `add_timing_columns(df)`:

```python
if not args.no_prosody and df is not None and 'Onset' in df.columns:
    from extractors.audio_features import extract as extract_contours
    from extractors.prosody import add_word_prosody
    try:
        contours = extract_contours(
            args.audio,
            pitch_floor_hz=args.pitch_floor,
            pitch_ceiling_hz=args.pitch_ceiling,
        )
        df = add_word_prosody(df, contours, pitch_units=args.pitch_units)
        # Stash contours on the args object so doc-level aggregation can reuse
        # them without a second Parselmouth pass.
        args._contours = contours
    except Exception as exc:
        print(f'[prosody] failed: {exc}', file=sys.stderr)
        # Word prosody columns remain absent; downstream _reorder is tolerant.
```

**`main.py:run_doc_level_batch()`** passes `audio_path` (already in `extract_args`) and the stashed `contours` into `build_doc_summary` via a new keyword argument. `build_doc_summary` calls `compute_doc_prosody` when prosody toggles are on; if `contours` is missing (prosody disabled or extraction failed), the prosody dict keys are populated as NaN.

`_COLUMN_ORDER` in `main.py` is extended at the end of the acoustic block:

```python
_COLUMN_ORDER = [
    ...
    'Transcript', 'Onset', 'End',
    'Word_Duration', 'Word_Gap',
    # New prosody columns (Section 6 below):
    'Pitch_Mean', 'Pitch_Min', 'Pitch_Max', 'Pitch_Std',
    'Volume_Mean', 'Volume_Min', 'Volume_Max', 'Volume_Std',
    'Pitch_Delta', 'Volume_Delta',
]
```

## 6. Word-level columns (10)

All columns are float. NaN signals "could not compute" (unvoiced frames for pitch, MFA failure, etc).

| Column | Definition | Unit |
|---|---|---|
| `Pitch_Mean` | Mean of voiced pitch frames in `[Onset, End]` | Hz or semitones (per `pitch_units` pref) |
| `Pitch_Min` | 5th percentile of voiced pitch frames in `[Onset, End]` | same |
| `Pitch_Max` | 95th percentile of voiced pitch frames in `[Onset, End]` | same |
| `Pitch_Std` | Std of voiced pitch frames in `[Onset, End]` | same |
| `Volume_Mean` | Mean intensity over all frames in `[Onset, End]` | dB |
| `Volume_Min` | 5th percentile of intensity frames in `[Onset, End]` | dB |
| `Volume_Max` | 95th percentile of intensity frames in `[Onset, End]` | dB |
| `Volume_Std` | Std of intensity frames in `[Onset, End]` | dB |
| `Pitch_Delta` | `Pitch_Mean[i] - Pitch_Mean[i-1]`; NaN for first word | same as pitch |
| `Volume_Delta` | `Volume_Mean[i] - Volume_Mean[i-1]`; NaN for first word | dB |

**Edge cases:**
- Zero voiced frames in word → all `Pitch_*` NaN; `Volume_*` still valid.
- Single-frame word → min=max=mean for that quantity; std=0.
- NaN `Onset` or `End` (MFA failed) → all 10 columns NaN.
- Punctuation / OOV tokens (no Onset/End) → all 10 columns NaN, consistent with `Word_Duration`/`Word_Gap`.

## 7. Document-level columns (16)

Added by `compute_doc_prosody` into the `documents_summary` DataFrame. All float.

**Prosody summaries (12 columns):**

| Column | Definition | Unit |
|---|---|---|
| `doc_pitch_mean` | Mean of all voiced pitch frames in the stimulus | Hz or st |
| `doc_pitch_min` | 5th percentile | same |
| `doc_pitch_max` | 95th percentile | same |
| `doc_pitch_std` | Std (raw) | same |
| `doc_volume_mean` | Mean intensity over all frames | dB |
| `doc_volume_min` | 5th percentile | dB |
| `doc_volume_max` | 95th percentile | dB |
| `doc_volume_std` | Std (raw) | dB |
| `doc_pitch_delta_mean` | Mean of `\|Pitch_Delta\|` across words (absolute, so rises and falls don't cancel) | same as pitch |
| `doc_volume_delta_mean` | Mean of `\|Volume_Delta\|` across words | dB |
| `doc_hnr` | Whole-stimulus Harmonics-to-Noise Ratio (mean over voiced frames, Praat `to_harmonicity_cc`) | dB |
| `doc_articulation_rate` | Total syllables (via `pyphen` on the surface form of each aligned word) divided by total spoken time (sum of word durations across the same words; pauses excluded). See §5.3 step 4. | syllables/sec |

**Pause statistics (4 columns):**

| Column | Definition | Unit |
|---|---|---|
| `doc_pause_count` | Number of inter-word gaps `> pause_threshold_s` | count |
| `doc_pause_mean_duration` | Mean of those gaps | sec |
| `doc_pause_total_duration` | Sum of those gaps | sec |
| `doc_pause_pct_silence` | Total pause time / total stimulus duration | proportion |

**Edge cases:**
- Stimulus with no voiced frames → all `doc_pitch_*` and `doc_hnr` NaN.
- Stimulus with one word → all pause stats = 0.
- `--no-align` set → all 16 doc prosody columns NaN; warning logged.
- `pyphen` not importable → only `doc_articulation_rate` NaN; other 15 columns unaffected.

## 8. CLI surface

```
# Whole-group toggles
--no-prosody                         Skip all 10 word-level prosody columns
--no-doc-prosody                     Skip all 16 doc-level prosody + pause columns

# Per-column word-level (rarely used from CLI; primarily GUI-driven)
--no-pitch-mean, --no-pitch-min, --no-pitch-max, --no-pitch-std
--no-volume-mean, --no-volume-min, --no-volume-max, --no-volume-std
--no-pitch-delta, --no-volume-delta

# Per-column doc-level (parallels existing --no-doc-* pattern)
--no-doc-pitch-mean, --no-doc-pitch-min, --no-doc-pitch-max, --no-doc-pitch-std
--no-doc-volume-mean, --no-doc-volume-min, --no-doc-volume-max, --no-doc-volume-std
--no-doc-pitch-delta-mean, --no-doc-volume-delta-mean
--no-doc-hnr
--no-doc-articulation-rate
--no-doc-pause-count, --no-doc-pause-mean-duration,
--no-doc-pause-total-duration, --no-doc-pause-pct-silence

# Configuration
--pitch-units {hz, semitones}        Default: semitones
--pause-threshold SECONDS            Default: 0.25
--pitch-floor HZ                     Default: 75
--pitch-ceiling HZ                   Default: 500
```

The volume-units CLI flag is reserved for future use; only dB is supported in this batch.

## 9. GUI surface

### Word-level features page (`gui/section_word_features.py`)

A new **Prosody** card is added directly below the existing **Acoustic** card. The Acoustic card remains as-is (Onset, End, Word_Duration, Word_Gap).

The Prosody card contains:

- A 2×5 grid of per-column toggles, ordered:
  - Row 1 (pitch): Pitch_Mean, Pitch_Min, Pitch_Max, Pitch_Std, Pitch_Delta
  - Row 2 (volume): Volume_Mean, Volume_Min, Volume_Max, Volume_Std, Volume_Delta
- A **Pitch units** combobox: `Hz` / `Semitones (rel speaker median)`.
- Two spinboxes (compact, side-by-side): **Pitch floor (Hz)** (default 75) and **Pitch ceiling (Hz)** (default 500).
- The whole card greys out and shows an inline warning when alignment is disabled (matching existing Acoustic-card behavior, since prosody requires MFA `Onset`/`End`).

### Doc-level features page (`gui/section_doc_features.py`)

A new **Prosody** card is added, parallel to the existing Aggregates/Lex-diversity/Readability cards. Contains:

- 4×4 grid of per-column toggles for the 12 prosody columns + 4 pause columns, grouped:
  - Pitch summaries (4 toggles)
  - Volume summaries (4 toggles)
  - Mean abs deltas (2 toggles)
  - HNR + articulation rate (2 toggles)
  - Pause stats (4 toggles)
- A **Pause threshold (sec)** spinbox (default 0.25).

### Preflight integration (`gui/preflight.py`)

New warning rules:

- Any word-level prosody toggle on AND `feat_acoustic_onset` toggle off → "Prosody features require MFA alignment; enable Onset/End in the Acoustic card."
- Any prosody toggle on AND `praat-parselmouth` not importable → "Install praat-parselmouth to enable prosody features. Run: `pip install praat-parselmouth`."
- `feat_doc_articulation_rate` on AND `pyphen` not importable → "Install pyphen to enable articulation rate."

## 10. Preferences schema bump (v4 → v5)

`utils/preferences.py` gains a v4→v5 migration. The actual v4 layout (verified in code) puts **all** per-column toggles — word- and doc-level alike — into the single `features` dict; doc-level group toggles + `out_count_*` per-column toggles live in `output`. We follow that pattern.

**Additions to `prefs["features"]`** (alongside existing `feat_struct_*`, `feat_acoustic_*`, `feat_psych_*`):

```jsonc
"features": {
  // ... existing v4 keys preserved
  // Word-level prosody — drive Pitch_*, Volume_*, *_Delta in the word CSV
  "feat_prosody_pitch_mean": true,
  "feat_prosody_pitch_min": true,
  "feat_prosody_pitch_max": true,
  "feat_prosody_pitch_std": true,
  "feat_prosody_pitch_delta": true,
  "feat_prosody_volume_mean": true,
  "feat_prosody_volume_min": true,
  "feat_prosody_volume_max": true,
  "feat_prosody_volume_std": true,
  "feat_prosody_volume_delta": true,
  // Doc-level prosody — drive doc_pitch_*, doc_volume_*, doc_hnr,
  // doc_articulation_rate, doc_pause_* in the documents_summary CSV.
  // (The `doc` prefix disambiguates from word-level Pitch_Mean vs
  //  doc-level doc_pitch_mean, which share the underlying concept.)
  "feat_doc_prosody_pitch_mean": true,
  "feat_doc_prosody_pitch_min": true,
  "feat_doc_prosody_pitch_max": true,
  "feat_doc_prosody_pitch_std": true,
  "feat_doc_prosody_volume_mean": true,
  "feat_doc_prosody_volume_min": true,
  "feat_doc_prosody_volume_max": true,
  "feat_doc_prosody_volume_std": true,
  "feat_doc_prosody_pitch_delta_mean": true,
  "feat_doc_prosody_volume_delta_mean": true,
  "feat_doc_prosody_hnr": true,
  "feat_doc_prosody_articulation_rate": true,
  "feat_doc_prosody_pause_count": true,
  "feat_doc_prosody_pause_mean_duration": true,
  "feat_doc_prosody_pause_total_duration": true,
  "feat_doc_prosody_pause_pct_silence": true
}
```

**New top-level config section `prefs["prosody"]`** (not toggles — analysis knobs):

```jsonc
"prosody": {
  "pitch_units": "semitones",       // or "hz"
  "pitch_floor_hz": 75,
  "pitch_ceiling_hz": 500,
  "pause_threshold_s": 0.25
}
```

**Schema version bumped to 5.** A new `_migrate_v4_into_v5(merged, v4)` is added to `utils/preferences.py`, paralleling the existing v3→v4 migration: copy all v4 sections through, then inject defaults for the new `feat_prosody_*`, `feat_doc_prosody_*`, and `prosody` keys. The `load_preferences()` dispatch grows a `version == 4` branch.

**Main.py mapping dictionaries** (used by `run_word_level_batch` and `run_doc_level_batch` to drop unselected columns):

- `_WORD_FEATURE_COLUMN_MAP` (existing) gains 10 entries:
  ```python
  "feat_prosody_pitch_mean":  "Pitch_Mean",
  "feat_prosody_pitch_min":   "Pitch_Min",
  # ... etc. through "feat_prosody_volume_delta": "Volume_Delta"
  ```
- A new module-level dict `_DOC_PROSODY_COLUMN_MAP` is added with the 16 `feat_doc_prosody_*` → `doc_*` mappings. `_doc_columns_to_drop()` is extended to consult both `_DOC_OUTPUT_COLUMN_MAP`, `_DOC_PSYCH_COLUMN_MAP`, and `_DOC_PROSODY_COLUMN_MAP`.

## 11. Error handling

- **Missing `praat-parselmouth`**: at the top of `extractors/audio_features.py`, `import parselmouth` is wrapped in a try/except. If unavailable, `extract` raises `ProsodyUnavailable`. Callers (in `main.py`) catch it, log a warning to the run manifest, and skip prosody for that stimulus (all 10 + 16 columns NaN). The batch continues for other stimuli.
- **Missing `pyphen`**: only affects `doc_articulation_rate`. Caught locally inside `compute_doc_prosody`; sets that single column to NaN.
- **Parselmouth raises mid-stimulus** (corrupted audio, audio shorter than minimum pitch period, etc.): caught by the same exception handler in `main.py`. That stimulus's prosody columns are NaN; error captured in `manifest.json` `warnings[]`. Batch continues.
- **Audio unreadable**: existing MFA path already raises and is recorded in the manifest; prosody never gets a chance to run on that stimulus.

## 12. Testing

### Unit tests

`tests/test_audio_features.py`:
- Integration test on a 3-second toy WAV (a 200 Hz sine tone) — verify pitch contour tracks 200 Hz within 5%.
- Mock-Parselmouth test verifying the wrapper contract (Contours dataclass shape, NaN handling for unvoiced segments).

`tests/test_prosody_word.py` (synthetic Contours; no audio):
- Constructed Contours with known pitch and intensity arrays.
- Aligned DataFrame with 3 words; verify each of the 10 columns produces the expected value.
- Percentile correctness on a known distribution (`np.arange(100)` → p5=4.95, p95=94.05).
- Delta: word 0 is NaN; word 1 = mean[1] - mean[0].
- All-NaN row when Onset/End are NaN.
- All-NaN pitch row, valid volume row when the contour interval has no voiced frames.
- Semitone vs Hz unit toggling: same Contours, different `pitch_units` arg, verify the transformation.

`tests/test_prosody_doc.py`:
- Synthetic Contours + aligned DataFrame → verify all 16 doc columns.
- Pause threshold edge cases (gap exactly at threshold; just below; just above).
- Single-word stimulus → all pause stats 0; articulation rate finite.
- Empty stimulus → all NaN, no crash.
- `pyphen` mocked to return known syllable counts → articulation rate exact.

`tests/test_preflight.py`: extended with the three new prosody-related warnings.

`tests/test_section_word_features.py` and `tests/test_section_doc_features.py`: extended with Prosody card presence, toggle wiring, and prefs persistence.

### Integration test

One end-to-end run on `tests/fixtures/transcripts/test_short.txt` + a tiny synthetic WAV. Assert that all expected columns are present in the word CSV and the doc summary, with finite values where expected and NaN where expected.

### Manual validation before merging

One ISC-Wu Tsai SENSI stimulus end-to-end. Spot-check:
- `Pitch_Mean` values look like Hz (60–300) or semitones (-12 to +12) depending on prefs.
- `Volume_Mean` in 50–75 dB SPL range.
- `doc_hnr` > 0.
- `doc_articulation_rate` in 3–6 syllables/sec range.

## 13. Documentation deliverables

- `Feature_Extraction_Guide.md` — new prosody section with column definitions, unit notes, Parselmouth defaults, citations (Boersma 1993 for HNR; standard phonetics references for articulation rate), and the dependency note about Parselmouth + pyphen.
- `README.md` — one-paragraph "What's new in the prosody batch" note.
- `requirements.txt` — add `praat-parselmouth>=0.4` and `pyphen>=0.14`.

## 14. Decision log

- 2026-05-12: Scope = word + document only. Syllable deferred to Roadmap Batch 3 (phoneme/syllable data model); sentence deferred to its own future mini-batch.
- 2026-05-12: Backend = Praat-Parselmouth only. Rejected librosa (less accurate pitch, no built-in HNR) and a hybrid (negligible gain over Parselmouth-only).
- 2026-05-12: Pitch units = user-configurable (Hz or semitones rel speaker median). Single set of `Pitch_*` columns either way; unit recorded in run manifest.
- 2026-05-12: Volume units = dB SPL via Praat Intensity. CLI/prefs flag reserved for future RMS option.
- 2026-05-12: Delta = word-to-word difference, Brain Treebank style. NaN for first word.
- 2026-05-12: Min/Max for range columns = 5th/95th percentile (robust to pitch-tracker glitches). Std raw.
- 2026-05-12: Speaking rate = articulation rate (syllables/sec excluding pauses), syllables counted via `pyphen` on the transcript.
- 2026-05-12: Pause threshold = 0.25 s default, configurable; pause column set = count, mean duration, total duration, % silence.
- 2026-05-12: Architecture = layered (`audio_features.py` + `prosody.py`). One Parselmouth pass per stimulus, pure-pandas aggregators consume the contours.
- 2026-05-12: Preferences schema bumped v4 → v5 (additive; no destructive migration).
