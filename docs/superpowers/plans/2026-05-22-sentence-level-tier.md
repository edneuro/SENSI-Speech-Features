# Sentence-Level Tier Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Emit `sentence/<stim>_sentences.csv` per stimulus — sentence timing (onset/end from MFA word alignment) and 11 prosody columns (pitch, volume, HNR).

**Architecture:** New `extractors/sentence_prosody.py` aggregates the existing MFA-aligned word DataFrame by sentence. `Contours` gains two optional per-frame HNR arrays (needed for sentence-level HNR). Wired into the word batch in `main.py`. Prefs v8 → v9. New GUI page "Sentence features."

**Tech Stack:** pandas, numpy, Parselmouth (via existing `Contours`), PySide6

---

## File Map

| Action | Path |
|--------|------|
| **Modify** | `extractors/audio_features.py` — add `hnr_frame_db` + `hnr_frame_times` fields to `Contours` |
| **Create** | `extractors/sentence_prosody.py` — sentence-level aggregator |
| **Create** | `tests/extractors/test_sentence_prosody.py` |
| **Modify** | `tests/extractors/test_audio_features.py` — assert `hnr_frame_db` is populated |
| **Modify** | `utils/preferences.py` — v8 → v9 (14 sentence-level feature toggles + `sentences_csv`) |
| **Modify** | `tests/utils/test_preferences_migration.py` — v8→v9 round-trip |
| **Modify** | `gui/planner/registry.py` — new Stage + 2 FeatureClusters |
| **Modify** | `gui/planner/emit.py` — `_WORD_LEVEL_CSVS` + `_OUTPUT_CSV_MAP` + `ns.no_sentences` |
| **Modify** | `tests/planner/test_schema_drift.py` — update to v9 constants |
| **Modify** | `main.py` — `--no-sentences` CLI flag + sentence CSV emission in batch |
| **Create** | `gui/section_sentence_features.py` |
| **Modify** | `gui/main_window.py` — new sidebar item + section |
| **Modify** | `tests/integration/test_output_paths.py` — sentence/ path test |
| **Modify** | `docs/master_plan.md` — update current state |

---

## Task 1: Extend Contours with per-frame HNR

**Files:**
- Modify: `extractors/audio_features.py`
- Modify: `tests/extractors/test_audio_features.py`

- [ ] **Step 1: Write a failing test asserting hnr_frame_db is populated**

Add to the bottom of `tests/extractors/test_audio_features.py`:

```python
def test_extract_populates_hnr_frame_arrays(tmp_path):
    wav = tmp_path / "sine.wav"
    _write_sine_wav(wav, freq_hz=200.0)
    contours = extract(str(wav), pitch_floor_hz=75, pitch_ceiling_hz=500)
    assert contours.hnr_frame_db is not None
    assert contours.hnr_frame_times is not None
    assert contours.hnr_frame_db.ndim == 1
    assert len(contours.hnr_frame_db) == len(contours.hnr_frame_times)
    # Voiced frames should have finite HNR values; some NaN for unvoiced is fine
    finite_mask = np.isfinite(contours.hnr_frame_db)
    assert finite_mask.any(), "Sine wave should have some voiced (finite HNR) frames"
```

- [ ] **Step 2: Run test to confirm it fails**

```
cd /mnt/c/Users/amilcar/Desktop/code/Antigravity/SensiExtractor
conda run -p env python -m pytest tests/extractors/test_audio_features.py::test_extract_populates_hnr_frame_arrays -v
```
Expected: FAIL — `AttributeError: 'Contours' object has no attribute 'hnr_frame_db'`

- [ ] **Step 3: Add hnr_frame_db + hnr_frame_times to Contours**

In `extractors/audio_features.py`, replace the `Contours` dataclass:

```python
@dataclass(frozen=True)
class Contours:
    pitch_hz: np.ndarray
    intensity_db: np.ndarray
    times: np.ndarray
    frame_rate_hz: int
    speaker_median_pitch_hz: float
    hnr_db: float
    hnr_frame_db: np.ndarray | None = None    # per-frame HNR, NaN=unvoiced (harmonicity grid)
    hnr_frame_times: np.ndarray | None = None  # timestamps matching hnr_frame_db
```

- [ ] **Step 4: Populate hnr_frame_db in extract()**

In `extractors/audio_features.py`, inside the `try:` block that computes `hnr_mean`, replace the entire try/except block:

```python
    hnr_mean = float('nan')
    hnr_frame_db_arr: np.ndarray | None = None
    hnr_frame_times_arr: np.ndarray | None = None
    try:
        harmonicity = snd.to_harmonicity_cc(
            time_step=time_step_s,
            minimum_pitch=pitch_floor_hz,
        )
        hnr_vals_raw = np.asarray(harmonicity.values, dtype=np.float64).ravel()
        hnr_times_raw = np.asarray(harmonicity.xs(), dtype=np.float64).ravel()
        # Replace unvoiced sentinel (−200) with NaN
        hnr_vals_clean = np.where(hnr_vals_raw <= -200.0, np.nan, hnr_vals_raw)
        # Whole-stimulus mean over voiced frames (unchanged behaviour)
        voiced_hnr = hnr_vals_clean[np.isfinite(hnr_vals_clean)]
        hnr_mean = float(np.mean(voiced_hnr)) if voiced_hnr.size else float('nan')
        # Per-frame arrays (used by sentence_prosody for windowed HNR)
        hnr_frame_db_arr = hnr_vals_clean
        hnr_frame_times_arr = hnr_times_raw
    except Exception:
        pass  # hnr_mean stays nan; frame arrays stay None
```

And update the `return Contours(...)` call to pass the new fields:

```python
    return Contours(
        pitch_hz=pitch_values,
        intensity_db=intensity_db,
        times=pitch_times,
        frame_rate_hz=int(round(1.0 / time_step_s)),
        speaker_median_pitch_hz=speaker_median,
        hnr_db=hnr_mean,
        hnr_frame_db=hnr_frame_db_arr,
        hnr_frame_times=hnr_frame_times_arr,
    )
```

- [ ] **Step 5: Run the new test + full extractor test suite to confirm pass**

```
conda run -p env python -m pytest tests/extractors/test_audio_features.py -v
```
Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add extractors/audio_features.py tests/extractors/test_audio_features.py
git commit -m "feat(audio): add hnr_frame_db + hnr_frame_times to Contours for sentence-level HNR"
```

---

## Task 2: Sentence prosody extractor

**Files:**
- Create: `extractors/sentence_prosody.py`
- Create: `tests/extractors/test_sentence_prosody.py`

- [ ] **Step 1: Write failing tests**

Create `tests/extractors/test_sentence_prosody.py`:

```python
"""Tests for extractors.sentence_prosody.add_sentence_prosody.

All tests use synthetic Contours — no audio file needed.
"""
import numpy as np
import pandas as pd
import pytest

from extractors.audio_features import Contours


def make_contours(pitch_hz, intensity_db, frame_rate=100, hnr=15.0,
                   hnr_frame_db=None, hnr_frame_times=None):
    """Build a Contours instance. times derived from frame rate."""
    pitch_hz = np.asarray(pitch_hz, dtype=float)
    intensity_db = np.asarray(intensity_db, dtype=float)
    times = np.arange(len(pitch_hz)) / frame_rate
    voiced = pitch_hz[~np.isnan(pitch_hz)]
    median = float(np.median(voiced)) if voiced.size else float('nan')
    if hnr_frame_db is None:
        hnr_frame_db = np.full(len(times), hnr)
        hnr_frame_times = times.copy()
    return Contours(
        pitch_hz=pitch_hz, intensity_db=intensity_db, times=times,
        frame_rate_hz=frame_rate, speaker_median_pitch_hz=median, hnr_db=hnr,
        hnr_frame_db=hnr_frame_db, hnr_frame_times=hnr_frame_times,
    )


def make_word_df(rows):
    """rows = list of (doc_id, sentence_id, sentence_text, token, onset, end).
    upos defaults to NOUN."""
    return pd.DataFrame([
        {
            "doc_id": doc_id,
            "sentence_id": sent_id,
            "sentence": sent_text,
            "token": tok,
            "Onset": on,
            "End": en,
            "upos": "NOUN",
        }
        for doc_id, sent_id, sent_text, tok, on, en in rows
    ])


def test_basic_sentence_timing():
    """Sentence onset = min word Onset, end = max word End."""
    from extractors.sentence_prosody import add_sentence_prosody

    df = make_word_df([
        ("doc1", 0, "Hello world", "Hello", 0.1, 0.5),
        ("doc1", 0, "Hello world", "world", 0.6, 1.0),
        ("doc1", 1, "Goodbye",     "Goodbye", 1.2, 1.8),
    ])
    contours = make_contours(np.full(200, 200.0), np.full(200, 60.0))
    result = add_sentence_prosody(df, contours)

    assert len(result) == 2
    assert result.iloc[0]["Onset"] == pytest.approx(0.1)
    assert result.iloc[0]["End"] == pytest.approx(1.0)
    assert result.iloc[1]["Onset"] == pytest.approx(1.2)
    assert result.iloc[1]["End"] == pytest.approx(1.8)


def test_sentence_text_is_preserved():
    from extractors.sentence_prosody import add_sentence_prosody

    df = make_word_df([
        ("doc1", 0, "The cat sat.", "The", 0.0, 0.3),
        ("doc1", 0, "The cat sat.", "cat", 0.4, 0.7),
    ])
    contours = make_contours(np.full(100, 200.0), np.full(100, 60.0))
    result = add_sentence_prosody(df, contours)

    assert result.iloc[0]["sentence"] == "The cat sat."


def test_pitch_mean_aggregates_over_sentence_window():
    """Uniform 200 Hz pitch across 0–2 s → Pitch_Mean = 200 Hz."""
    from extractors.sentence_prosody import add_sentence_prosody

    contours = make_contours(np.full(200, 200.0), np.full(200, 60.0))
    df = make_word_df([
        ("doc1", 0, "Hello world", "Hello", 0.0, 1.0),
        ("doc1", 0, "Hello world", "world", 1.0, 2.0),
    ])
    result = add_sentence_prosody(df, contours, pitch_units="hz")
    assert result.iloc[0]["Pitch_Mean"] == pytest.approx(200.0)


def test_pitch_delta_nan_for_first_sentence():
    from extractors.sentence_prosody import add_sentence_prosody

    contours = make_contours(
        np.concatenate([np.full(100, 200.0), np.full(100, 250.0)]),
        np.full(200, 60.0),
    )
    df = make_word_df([
        ("doc1", 0, "sent A", "A", 0.0, 1.0),
        ("doc1", 1, "sent B", "B", 1.0, 2.0),
    ])
    result = add_sentence_prosody(df, contours, pitch_units="hz")
    assert pd.isna(result.iloc[0]["Pitch_Delta"])
    assert result.iloc[1]["Pitch_Delta"] == pytest.approx(50.0)


def test_nan_onset_yields_nan_prosody():
    """Words with NaN alignment → sentence onset/end NaN → NaN prosody."""
    from extractors.sentence_prosody import add_sentence_prosody

    contours = make_contours(np.full(100, 200.0), np.full(100, 60.0))
    df = make_word_df([
        ("doc1", 0, "unaligned", "word", float("nan"), float("nan")),
    ])
    result = add_sentence_prosody(df, contours)
    assert pd.isna(result.iloc[0]["Onset"])
    assert pd.isna(result.iloc[0]["Pitch_Mean"])
    assert pd.isna(result.iloc[0]["HNR"])


def test_none_contours_yields_nan_prosody():
    """If contours is None (no audio), timing comes from word_df but prosody is NaN."""
    from extractors.sentence_prosody import add_sentence_prosody

    df = make_word_df([
        ("doc1", 0, "hello", "hello", 0.0, 1.0),
    ])
    result = add_sentence_prosody(df, contours=None)
    assert result.iloc[0]["Onset"] == pytest.approx(0.0)
    assert result.iloc[0]["End"] == pytest.approx(1.0)
    assert pd.isna(result.iloc[0]["Pitch_Mean"])


def test_hnr_aggregated_per_sentence_window():
    """HNR frames 0–1 s = 12 dB, frames 1–2 s = 20 dB. Two sentences should differ."""
    from extractors.sentence_prosody import add_sentence_prosody

    times = np.arange(200) / 100.0       # 0.00, 0.01, …, 1.99
    hnr_vals = np.where(times < 1.0, 12.0, 20.0)
    contours = make_contours(
        np.full(200, 200.0), np.full(200, 60.0),
        hnr_frame_db=hnr_vals, hnr_frame_times=times,
    )
    df = make_word_df([
        ("doc1", 0, "first",  "first",  0.0, 1.0),
        ("doc1", 1, "second", "second", 1.0, 2.0),
    ])
    result = add_sentence_prosody(df, contours, pitch_units="hz")
    assert result.iloc[0]["HNR"] == pytest.approx(12.0, abs=0.5)
    assert result.iloc[1]["HNR"] == pytest.approx(20.0, abs=0.5)


def test_output_columns_present():
    from extractors.sentence_prosody import add_sentence_prosody

    contours = make_contours(np.full(100, 200.0), np.full(100, 60.0))
    df = make_word_df([("doc1", 0, "hi", "hi", 0.0, 1.0)])
    result = add_sentence_prosody(df, contours)

    expected_cols = {
        "doc_id", "sentence_id", "sentence", "Onset", "End",
        "Pitch_Mean", "Pitch_Min", "Pitch_Max", "Pitch_Std", "Pitch_Delta",
        "Volume_Mean", "Volume_Min", "Volume_Max", "Volume_Std", "Volume_Delta",
        "HNR",
    }
    assert expected_cols.issubset(set(result.columns))
```

- [ ] **Step 2: Run tests to confirm they fail**

```
conda run -p env python -m pytest tests/extractors/test_sentence_prosody.py -v
```
Expected: all FAIL — `ModuleNotFoundError: No module named 'extractors.sentence_prosody'`

- [ ] **Step 3: Implement extractors/sentence_prosody.py**

Create `extractors/sentence_prosody.py`:

```python
"""Sentence-level timing and prosody aggregator.

Given an MFA-aligned word DataFrame and Praat Contours, produces a
sentence-level DataFrame with onset/end timing (min/max over words)
and 11 prosody columns: Pitch_{Mean,Min,Max,Std,Delta},
Volume_{Mean,Min,Max,Std,Delta}, HNR.

The 11 columns mirror doc-level prosody but without pause statistics.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from extractors.audio_features import Contours
from extractors.prosody import _agg, _slice, _semitones_from_hz

_COLUMN_ORDER = [
    "doc_id", "sentence_id", "sentence", "Onset", "End",
    "Pitch_Mean", "Pitch_Min", "Pitch_Max", "Pitch_Std", "Pitch_Delta",
    "Volume_Mean", "Volume_Min", "Volume_Max", "Volume_Std", "Volume_Delta",
    "HNR",
]


def add_sentence_prosody(
    word_df: pd.DataFrame,
    contours: Contours | None = None,
    *,
    pitch_units: str = "semitones",
) -> pd.DataFrame:
    """Return a sentence-level DataFrame with timing + 11 prosody columns.

    Requires word_df to have: doc_id, sentence_id, sentence, Onset, End.
    If contours is None or onset/end are NaN, prosody columns are NaN.
    """
    required = {"doc_id", "sentence_id", "sentence", "Onset", "End"}
    missing = required - set(word_df.columns)
    if missing:
        raise ValueError(f"word_df missing columns: {sorted(missing)}")

    # Pre-compute pitch array (apply unit conversion once)
    pitch_arr: np.ndarray | None = None
    if contours is not None:
        pitch_arr = contours.pitch_hz.copy()
        if pitch_units == "semitones":
            pitch_arr = _semitones_from_hz(pitch_arr, contours.speaker_median_pitch_hz)

    # Coerce Onset/End to float, handling pandas.NA safely
    onsets = pd.to_numeric(word_df["Onset"], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    ends   = pd.to_numeric(word_df["End"],   errors="coerce").to_numpy(dtype=float, na_value=np.nan)

    tmp = word_df.copy()
    tmp["_onset_f"] = onsets
    tmp["_end_f"]   = ends

    rows: list[dict] = []
    for (doc_id, sent_id), grp in tmp.groupby(["doc_id", "sentence_id"], sort=False):
        sent_text = grp["sentence"].iloc[0]
        onset = float(grp["_onset_f"].min())
        end   = float(grp["_end_f"].max())

        has_timing = np.isfinite(onset) and np.isfinite(end)
        row: dict = {
            "doc_id":      doc_id,
            "sentence_id": sent_id,
            "sentence":    sent_text,
            "Onset": onset if has_timing else np.nan,
            "End":   end   if has_timing else np.nan,
        }

        if contours is not None and pitch_arr is not None and has_timing:
            p_mean, p_min, p_max, p_std = _agg(
                _slice(contours.times, pitch_arr, onset, end))
            v_mean, v_min, v_max, v_std = _agg(
                _slice(contours.times, contours.intensity_db, onset, end))

            if (contours.hnr_frame_db is not None
                    and contours.hnr_frame_times is not None):
                hnr_frames = _slice(
                    contours.hnr_frame_times, contours.hnr_frame_db, onset, end)
                finite = hnr_frames[np.isfinite(hnr_frames)]
                hnr = float(np.mean(finite)) if finite.size else np.nan
            else:
                hnr = np.nan
        else:
            p_mean = p_min = p_max = p_std = np.nan
            v_mean = v_min = v_max = v_std = np.nan
            hnr = np.nan

        row.update({
            "Pitch_Mean": p_mean, "Pitch_Min": p_min,
            "Pitch_Max":  p_max,  "Pitch_Std": p_std,
            "Volume_Mean": v_mean, "Volume_Min": v_min,
            "Volume_Max":  v_max,  "Volume_Std": v_std,
            "HNR": hnr,
        })
        rows.append(row)

    df = pd.DataFrame(rows)

    # Sentence-to-sentence deltas (NaN for first sentence)
    if len(df) > 1:
        df["Pitch_Delta"]  = df["Pitch_Mean"].diff()
        df["Volume_Delta"] = df["Volume_Mean"].diff()
    elif len(df) == 1:
        df["Pitch_Delta"]  = np.nan
        df["Volume_Delta"] = np.nan
    else:
        df["Pitch_Delta"]  = pd.Series(dtype=float)
        df["Volume_Delta"] = pd.Series(dtype=float)

    present = [c for c in _COLUMN_ORDER if c in df.columns]
    return df[present]
```

- [ ] **Step 4: Run tests to confirm they pass**

```
conda run -p env python -m pytest tests/extractors/test_sentence_prosody.py -v
```
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add extractors/sentence_prosody.py tests/extractors/test_sentence_prosody.py
git commit -m "feat(sentence): add sentence_prosody extractor with 11 prosody columns"
```

---

## Task 3: Prefs v8 → v9

**Files:**
- Modify: `utils/preferences.py`
- Modify: `tests/utils/test_preferences_migration.py`

- [ ] **Step 1: Check what migration tests currently look like**

```
conda run -p env python -m pytest tests/utils/test_preferences_migration.py -v
```
Note the passing test names; a v8→v9 migration test will follow the same pattern.

- [ ] **Step 2: Write a failing migration test**

Add to `tests/utils/test_preferences_migration.py`:

```python
def test_v8_migrates_to_v9_with_sentence_defaults():
    """v8 prefs gain all sentence-level keys with defaults True."""
    from utils.preferences import (
        DEFAULT_PREFS_V8, SCHEMA_VERSION_V9,
        _FEATURES_DEFAULT_V9, _OUTPUT_DEFAULT_V9,
    )
    import copy, json, os
    from pathlib import Path
    from utils.preferences import load_preferences

    # Write a v8 prefs file to a temp location
    tmp_path = Path(os.environ.get("TMPDIR", "/tmp")) / "prefs_v8_test.json"
    tmp_path.write_text(json.dumps({**DEFAULT_PREFS_V8, "schema_version": 8}))
    orig_env = os.environ.get("SENSI_PREFS_PATH")
    os.environ["SENSI_PREFS_PATH"] = str(tmp_path)
    try:
        prefs = load_preferences()
    finally:
        if orig_env is None:
            del os.environ["SENSI_PREFS_PATH"]
        else:
            os.environ["SENSI_PREFS_PATH"] = orig_env
        tmp_path.unlink(missing_ok=True)

    assert prefs["schema_version"] == SCHEMA_VERSION_V9
    f = prefs["features"]
    assert f["feat_sentence_onset"] is True
    assert f["feat_sentence_pitch_mean"] is True
    assert f["master_sentence_prosody"] is True
    assert prefs["output"]["sentences_csv"] is True
```

- [ ] **Step 3: Run test to confirm it fails**

```
conda run -p env python -m pytest tests/utils/test_preferences_migration.py::test_v8_migrates_to_v9_with_sentence_defaults -v
```
Expected: FAIL — `ImportError: cannot import name 'SCHEMA_VERSION_V9'`

- [ ] **Step 4: Add v9 constants and migration to utils/preferences.py**

In `utils/preferences.py`, after the v8 block (after line `DEFAULT_PREFS_V8: dict...`), add:

```python
# ---------------------------------------------------------------------------
# v9 constants (active schema)
#
# Sentence-level tier (2026-05-22) changes vs v8:
#   - 14 new feature toggles: feat_sentence_* (11 prosody + onset + end + text)
#     plus 2 master toggles: master_sentence_timing, master_sentence_prosody
#   - 1 new output toggle: sentences_csv
# ---------------------------------------------------------------------------

SCHEMA_VERSION_V9 = 9

_FEATURES_DEFAULT_V9 = {
    **_FEATURES_DEFAULT_V8,
    # Sentence-level timing
    "feat_sentence_text":           True,
    "feat_sentence_onset":          True,
    "feat_sentence_end":            True,
    # Sentence-level prosody (11 columns)
    "feat_sentence_pitch_mean":     True,
    "feat_sentence_pitch_min":      True,
    "feat_sentence_pitch_max":      True,
    "feat_sentence_pitch_std":      True,
    "feat_sentence_pitch_delta":    True,
    "feat_sentence_volume_mean":    True,
    "feat_sentence_volume_min":     True,
    "feat_sentence_volume_max":     True,
    "feat_sentence_volume_std":     True,
    "feat_sentence_volume_delta":   True,
    "feat_sentence_hnr":            True,
    # Cluster master toggles
    "master_sentence_timing":       True,
    "master_sentence_prosody":      True,
}

_OUTPUT_DEFAULT_V9 = {
    **_OUTPUT_DEFAULT_V8,
    "sentences_csv": True,
}

DEFAULT_PREFS_V9: dict[str, Any] = {
    **DEFAULT_PREFS_V8,
    "schema_version": SCHEMA_VERSION_V9,
    "features": dict(_FEATURES_DEFAULT_V9),
    "output":   dict(_OUTPUT_DEFAULT_V9),
}
```

Update the `ACTIVE SCHEMA ALIASES` block:
```python
SCHEMA_VERSION = SCHEMA_VERSION_V9  # active version
DEFAULT_PREFS: dict[str, Any] = DEFAULT_PREFS_V9  # active defaults
```

Add the migration function (before `load_preferences`):
```python
def _migrate_v8_into_v9(merged: dict, v8: dict) -> None:
    """v8 → v9: additive only.

    v9 defaults (already in merged) supply all sentence-level keys.
    Any forward-compat value the user set in the v8 file is preserved.
    """
    _deep_merge_current(merged, v8)
    merged["schema_version"] = SCHEMA_VERSION_V9
```

Update `load_preferences` to add a new `elif version == SCHEMA_VERSION_V8:` arm and update all existing arms to chain through v9:

```python
    if version == SCHEMA_VERSION_V9:
        _deep_merge_current(merged, user)
    elif version == SCHEMA_VERSION_V8:
        _migrate_v8_into_v9(merged, user)
        _write(path, merged)
    elif version == SCHEMA_VERSION_V7:
        v8_intermediate = deepcopy(DEFAULT_PREFS_V8)
        _migrate_v7_into_v8(v8_intermediate, user)
        _migrate_v8_into_v9(merged, v8_intermediate)
        _write(path, merged)
    elif version == SCHEMA_VERSION_V6:
        v7_intermediate = deepcopy(DEFAULT_PREFS_V7)
        _migrate_v6_into_v7(v7_intermediate, user)
        v8_intermediate = deepcopy(DEFAULT_PREFS_V8)
        _migrate_v7_into_v8(v8_intermediate, v7_intermediate)
        _migrate_v8_into_v9(merged, v8_intermediate)
        _write(path, merged)
    elif version == SCHEMA_VERSION_V5:
        v6_intermediate = deepcopy(DEFAULT_PREFS_V6)
        _migrate_v5_into_v6(v6_intermediate, user)
        v7_intermediate = deepcopy(DEFAULT_PREFS_V7)
        _migrate_v6_into_v7(v7_intermediate, v6_intermediate)
        v8_intermediate = deepcopy(DEFAULT_PREFS_V8)
        _migrate_v7_into_v8(v8_intermediate, v7_intermediate)
        _migrate_v8_into_v9(merged, v8_intermediate)
        _write(path, merged)
    elif version == SCHEMA_VERSION_V4:
        v5_intermediate = deepcopy(DEFAULT_PREFS_V5)
        _migrate_v4_into_v5(v5_intermediate, user)
        v6_intermediate = deepcopy(DEFAULT_PREFS_V6)
        _migrate_v5_into_v6(v6_intermediate, v5_intermediate)
        v7_intermediate = deepcopy(DEFAULT_PREFS_V7)
        _migrate_v6_into_v7(v7_intermediate, v6_intermediate)
        v8_intermediate = deepcopy(DEFAULT_PREFS_V8)
        _migrate_v7_into_v8(v8_intermediate, v7_intermediate)
        _migrate_v8_into_v9(merged, v8_intermediate)
        _write(path, merged)
    elif version == 3:
        v4_intermediate = deepcopy(DEFAULT_PREFS_V4)
        _migrate_v3_into_v4(v4_intermediate, user)
        v5_intermediate = deepcopy(DEFAULT_PREFS_V5)
        _migrate_v4_into_v5(v5_intermediate, v4_intermediate)
        v6_intermediate = deepcopy(DEFAULT_PREFS_V6)
        _migrate_v5_into_v6(v6_intermediate, v5_intermediate)
        v7_intermediate = deepcopy(DEFAULT_PREFS_V7)
        _migrate_v6_into_v7(v7_intermediate, v6_intermediate)
        v8_intermediate = deepcopy(DEFAULT_PREFS_V8)
        _migrate_v7_into_v8(v8_intermediate, v7_intermediate)
        _migrate_v8_into_v9(merged, v8_intermediate)
        _write(path, merged)
    elif version == 2:
        v3_intermediate = deepcopy(DEFAULT_PREFS_V3)
        _migrate_v2_into_v3(v3_intermediate, user)
        v4_intermediate = deepcopy(DEFAULT_PREFS_V4)
        _migrate_v3_into_v4(v4_intermediate, v3_intermediate)
        v5_intermediate = deepcopy(DEFAULT_PREFS_V5)
        _migrate_v4_into_v5(v5_intermediate, v4_intermediate)
        v6_intermediate = deepcopy(DEFAULT_PREFS_V6)
        _migrate_v5_into_v6(v6_intermediate, v5_intermediate)
        v7_intermediate = deepcopy(DEFAULT_PREFS_V7)
        _migrate_v6_into_v7(v7_intermediate, v6_intermediate)
        v8_intermediate = deepcopy(DEFAULT_PREFS_V8)
        _migrate_v7_into_v8(v8_intermediate, v7_intermediate)
        _migrate_v8_into_v9(merged, v8_intermediate)
        _write(path, merged)
    else:
        # v1 flat → v2 → v3 → v4 → v5 → v6 → v7 → v8 → v9
        v2_intermediate = deepcopy(_DEFAULT_PREFS_V2)
        _migrate_v1_into(v2_intermediate, user)
        v3_intermediate = deepcopy(DEFAULT_PREFS_V3)
        _migrate_v2_into_v3(v3_intermediate, v2_intermediate)
        v4_intermediate = deepcopy(DEFAULT_PREFS_V4)
        _migrate_v3_into_v4(v4_intermediate, v3_intermediate)
        v5_intermediate = deepcopy(DEFAULT_PREFS_V5)
        _migrate_v4_into_v5(v5_intermediate, v4_intermediate)
        v6_intermediate = deepcopy(DEFAULT_PREFS_V6)
        _migrate_v5_into_v6(v6_intermediate, v5_intermediate)
        v7_intermediate = deepcopy(DEFAULT_PREFS_V7)
        _migrate_v6_into_v7(v7_intermediate, v6_intermediate)
        v8_intermediate = deepcopy(DEFAULT_PREFS_V8)
        _migrate_v7_into_v8(v8_intermediate, v7_intermediate)
        _migrate_v8_into_v9(merged, v8_intermediate)
        _write(path, merged)
```

Also update the docstring at the top of `preferences.py` to reflect v9:
- Schema version: v9
- New section entry: sentence-level toggles
- Migration chain updated

- [ ] **Step 5: Run migration test to confirm it passes**

```
conda run -p env python -m pytest tests/utils/test_preferences_migration.py -v
```
Expected: all pass including new v8→v9 test

- [ ] **Step 6: Commit**

```bash
git add utils/preferences.py tests/utils/test_preferences_migration.py
git commit -m "feat(prefs): v8 → v9 additive — sentence-level feature toggles + sentences_csv output key"
```

---

## Task 4: Planner registry + emit updates

**Files:**
- Modify: `gui/planner/registry.py`
- Modify: `gui/planner/emit.py`
- Modify: `tests/planner/test_schema_drift.py`

- [ ] **Step 1: Update schema drift test to reference v9 constants**

In `tests/planner/test_schema_drift.py`, change the imports and constants:

```python
from gui.planner.registry import CLUSTERS
from utils.preferences import _FEATURES_DEFAULT_V9, _OUTPUT_DEFAULT_V9


_INTENTIONALLY_UNROUTED_FEATURES: set[str] = {
    # ... existing set unchanged ...
    "master_sentence_timing",    # routes via master_toggle, not toggle_keys
    "master_sentence_prosody",   # routes via master_toggle, not toggle_keys
}

_OUTPUT_KEYS_VIA_CSV_MAP: set[str] = {
    "word_level_csv", "syllables_csv", "phonemes_csv", "document_summary",
    "sentences_csv",    # routed via _OUTPUT_CSV_MAP
    "master_doc_prosody",
    "master_doc_articulation_rate",
    "master_doc_counts",
}
```

And update the two test functions to use `_FEATURES_DEFAULT_V9` / `_OUTPUT_DEFAULT_V9`:
```python
def test_every_feature_pref_key_is_in_some_cluster():
    routed = _all_cluster_toggle_keys()
    feature_keys = set(_FEATURES_DEFAULT_V9.keys())   # was V8
    unrouted = feature_keys - routed - _INTENTIONALLY_UNROUTED_FEATURES
    assert not unrouted, ...

def test_every_output_pref_key_is_either_master_csv_or_in_some_cluster():
    routed = _all_cluster_toggle_keys()
    output_keys = set(_OUTPUT_DEFAULT_V9.keys())       # was V8
    unrouted = output_keys - routed - _OUTPUT_KEYS_VIA_CSV_MAP
    assert not unrouted, ...

def test_no_cluster_references_nonexistent_pref_key():
    all_pref_keys = set(_FEATURES_DEFAULT_V9.keys()) | set(_OUTPUT_DEFAULT_V9.keys())  # was V8
    ...

def test_every_master_toggle_key_exists_in_v8_prefs_schema():
    # rename to _v9_ for clarity
    all_pref_keys = set(_FEATURES_DEFAULT_V9) | set(_OUTPUT_DEFAULT_V9)  # was V8
    ...
```

- [ ] **Step 2: Run schema drift test to confirm it now fails**

```
conda run -p env python -m pytest tests/planner/test_schema_drift.py -v
```
Expected: `test_every_feature_pref_key_is_in_some_cluster` FAILS — the 14 new v9 feature keys are unrouted.

- [ ] **Step 3: Add sentence_prosody Stage to registry.py**

In `gui/planner/registry.py`, add the import at the top:
```python
from extractors.sentence_prosody import add_sentence_prosody as _add_sentence_prosody
```

Add the new stage to `STAGES` (after the `pcenter` stage, before `# --- §4 Document level ---`):

```python
    # --- §3 Sentence level ---
    Stage(name="sentence_prosody",
          consumes=("word_df", "contours"),
          produces=("sentence_df",),
          run=_add_sentence_prosody),
```

Add two new clusters to `CLUSTERS` (after the p-center cluster, before `# --- Document level ---`):

```python
    # --- Sentence level (§3) ---

    # §3.1 Sentence timing — onset, end, sentence text
    FeatureCluster(
        name="sentence timing",
        toggle_keys=("feat_sentence_text", "feat_sentence_onset", "feat_sentence_end"),
        output_csv="sentences",
        requires_stages=("spacy_parse", "mfa_align"),
        emits_columns=("sentence", "Onset", "End"),
        master_toggle="master_sentence_timing",
    ),

    # §3.2 Sentence prosody — 11 columns
    FeatureCluster(
        name="sentence prosody",
        toggle_keys=(
            "feat_sentence_pitch_mean", "feat_sentence_pitch_min",
            "feat_sentence_pitch_max",  "feat_sentence_pitch_std",
            "feat_sentence_pitch_delta",
            "feat_sentence_volume_mean", "feat_sentence_volume_min",
            "feat_sentence_volume_max",  "feat_sentence_volume_std",
            "feat_sentence_volume_delta",
            "feat_sentence_hnr",
        ),
        output_csv="sentences",
        requires_stages=("spacy_parse", "mfa_align", "praat_contours", "sentence_prosody"),
        emits_columns=(
            "Pitch_Mean", "Pitch_Min", "Pitch_Max", "Pitch_Std", "Pitch_Delta",
            "Volume_Mean", "Volume_Min", "Volume_Max", "Volume_Std", "Volume_Delta",
            "HNR",
        ),
        master_toggle="master_sentence_prosody",
    ),
```

- [ ] **Step 4: Update emit.py**

In `gui/planner/emit.py`:

1. Add `"sentences"` to `_WORD_LEVEL_CSVS`:
   ```python
   _WORD_LEVEL_CSVS = frozenset({"features", "syllables", "phonemes", "sentences"})
   ```

2. Add to `_OUTPUT_CSV_MAP` in `plan.py` (not emit.py — it lives in `gui/planner/plan.py`):
   In `gui/planner/plan.py`, update `_OUTPUT_CSV_MAP`:
   ```python
   _OUTPUT_CSV_MAP = {
       "word_level_csv":   "features",
       "syllables_csv":    "syllables",
       "phonemes_csv":     "phonemes",
       "sentences_csv":    "sentences",   # new
       "document_summary": "documents_summary",
   }
   ```

3. In `emit.py`, add `ns.no_sentences` in `to_namespace()`, alongside `ns.no_syllables` and `ns.no_phonemes`:
   ```python
   ns.no_sentences = "sentences" not in plan.output_csvs
   ```

4. Add `sentence_prosody` config in `plan.py:_stage_configs()`:
   ```python
   if "sentence_prosody" in in_plan:
       configs["sentence_prosody"] = {"pitch_units": pitch_units}
   ```

- [ ] **Step 5: Run schema drift tests**

```
conda run -p env python -m pytest tests/planner/test_schema_drift.py -v
```
Expected: all pass

- [ ] **Step 6: Run full planner tests**

```
conda run -p env python -m pytest tests/planner/ -v
```
Expected: all pass (the planner tests are mostly plan correctness / stage-ordering tests)

- [ ] **Step 7: Commit**

```bash
git add gui/planner/registry.py gui/planner/emit.py gui/planner/plan.py tests/planner/test_schema_drift.py
git commit -m "feat(planner): register sentence_prosody stage and clusters for v9"
```

---

## Task 5: main.py — CLI flags + sentence CSV emission

**Files:**
- Modify: `main.py`

- [ ] **Step 1: Add --no-sentences CLI flag and column map**

In `main.py`, in `build_parser()`, after the `--no-phonemes` flag:

```python
    # Sentence-level sidecar master toggle
    p.add_argument('--no-sentences', action='store_true',
                   help='Skip <stim>_sentences.csv emission.')

    # Sentence per-column toggles (default = on)
    p.add_argument('--no-sent-text',          action='store_true')
    p.add_argument('--no-sent-onset',         action='store_true')
    p.add_argument('--no-sent-end',           action='store_true')
    p.add_argument('--no-sent-pitch-mean',    action='store_true')
    p.add_argument('--no-sent-pitch-min',     action='store_true')
    p.add_argument('--no-sent-pitch-max',     action='store_true')
    p.add_argument('--no-sent-pitch-std',     action='store_true')
    p.add_argument('--no-sent-pitch-delta',   action='store_true')
    p.add_argument('--no-sent-volume-mean',   action='store_true')
    p.add_argument('--no-sent-volume-min',    action='store_true')
    p.add_argument('--no-sent-volume-max',    action='store_true')
    p.add_argument('--no-sent-volume-std',    action='store_true')
    p.add_argument('--no-sent-volume-delta',  action='store_true')
    p.add_argument('--no-sent-hnr',           action='store_true')
```

In the column maps section, add:

```python
_SENTENCE_COLUMN_MAP = {
    "feat_sentence_text":          "sentence",
    "feat_sentence_onset":         "Onset",
    "feat_sentence_end":           "End",
    "feat_sentence_pitch_mean":    "Pitch_Mean",
    "feat_sentence_pitch_min":     "Pitch_Min",
    "feat_sentence_pitch_max":     "Pitch_Max",
    "feat_sentence_pitch_std":     "Pitch_Std",
    "feat_sentence_pitch_delta":   "Pitch_Delta",
    "feat_sentence_volume_mean":   "Volume_Mean",
    "feat_sentence_volume_min":    "Volume_Min",
    "feat_sentence_volume_max":    "Volume_Max",
    "feat_sentence_volume_std":    "Volume_Std",
    "feat_sentence_volume_delta":  "Volume_Delta",
    "feat_sentence_hnr":           "HNR",
}


def _sentence_columns_to_drop(features: dict) -> set:
    """Return sentence-CSV columns to drop based on v9 feature toggles."""
    return {col for key, col in _SENTENCE_COLUMN_MAP.items()
            if not features.get(key, True)}
```

- [ ] **Step 2: Add sentence CSV emission to run_word_level_batch**

In `run_word_level_batch`, after the syllable/phoneme sidecar block (the `if phones_df is not None` block) and before the per-column drop step, add:

```python
            # Sentence-level sidecar — emit after word prosody runs
            if not getattr(args, 'no_sentences', False):
                try:
                    from extractors.sentence_prosody import add_sentence_prosody
                    sent_df = add_sentence_prosody(
                        df,
                        getattr(args, '_contours', None),
                        pitch_units=getattr(args, 'pitch_units', 'semitones'),
                    )
                    if len(sent_df) > 0:
                        drop = _sentence_columns_to_drop(
                            getattr(args, 'feature_toggles', {}) or {})
                        if drop:
                            sent_df = sent_df.drop(
                                columns=[c for c in drop if c in sent_df.columns])
                        sent_path = (_level_dir(output_dir, 'sentence')
                                     / f'{pair.stimulus_id}_sentences.csv')
                        sent_df.to_csv(sent_path, index=False)
                except Exception as exc:
                    print(f'[sentence] failed for {pair.stimulus_id}: {exc}',
                          file=sys.stderr)
```

- [ ] **Step 3: Run existing tests to confirm no regressions**

```
conda run -p env python -m pytest tests/test_main_cli.py tests/integration/ -v
```
Expected: all pass (sentence emission is behind `--no-sentences=False` by default, but the integration test fixtures use `--no-align` which means no Onset/End → sentence CSV writes 0 rows → no file written → the path test won't trip)

- [ ] **Step 4: Commit**

```bash
git add main.py
git commit -m "feat(main): add sentence CSV emission to word batch (--no-sentences + per-column flags)"
```

---

## Task 6: GUI page

**Files:**
- Create: `gui/section_sentence_features.py`
- Modify: `gui/main_window.py`

- [ ] **Step 1: Create gui/section_sentence_features.py**

```python
"""Sentence-level features page — timing and prosody toggles."""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QWidget, QVBoxLayout, QGridLayout

from gui.widgets import ClusterCard, ToggleRow
from utils.preferences import load_preferences, save_preferences


class SentenceFeaturesSection(QWidget):
    settings_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._toggles: dict[str, ToggleRow] = {}
        prefs = load_preferences()
        f = prefs["features"]

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 16, 24, 16)
        layout.setSpacing(12)

        # --- Sentence timing ---
        timing_card = ClusterCard(
            "Sentence timing",
            "Onset/End/text from MFA word alignment (min onset, max end per sentence)",
            master_key="master_sentence_timing",
            master_label="Emit sentence timing columns",
        )
        timing_grid = QGridLayout()
        timing_keys = [
            ("feat_sentence_onset", "Onset"),
            ("feat_sentence_end",   "End"),
            ("feat_sentence_text",  "Sentence text"),
        ]
        for i, (key, label) in enumerate(timing_keys):
            timing_grid.addWidget(
                self._make_toggle(key, label, f[key]),
                i // 2, i % 2,
            )
        timing_card.add_child_layout(timing_grid)
        timing_card.settings_changed.connect(self.settings_changed)
        layout.addWidget(timing_card)

        # --- Sentence prosody ---
        prosody_card = ClusterCard(
            "Sentence prosody",
            "Pitch, volume, and HNR aggregated over each sentence's [Onset, End] window",
            master_key="master_sentence_prosody",
            master_label="Emit sentence prosody columns",
        )
        prosody_grid = QGridLayout()
        prosody_keys = [
            ("feat_sentence_pitch_mean",   "Pitch Mean"),
            ("feat_sentence_pitch_min",    "Pitch Min (5th pct)"),
            ("feat_sentence_pitch_max",    "Pitch Max (95th pct)"),
            ("feat_sentence_pitch_std",    "Pitch Std"),
            ("feat_sentence_pitch_delta",  "Pitch Delta"),
            ("feat_sentence_volume_mean",  "Volume Mean"),
            ("feat_sentence_volume_min",   "Volume Min"),
            ("feat_sentence_volume_max",   "Volume Max"),
            ("feat_sentence_volume_std",   "Volume Std"),
            ("feat_sentence_volume_delta", "Volume Delta"),
            ("feat_sentence_hnr",          "HNR"),
        ]
        for i, (key, label) in enumerate(prosody_keys):
            prosody_grid.addWidget(
                self._make_toggle(key, label, f[key]),
                i // 2, i % 2,
            )
        prosody_card.add_child_layout(prosody_grid)
        prosody_card.settings_changed.connect(self.settings_changed)
        layout.addWidget(prosody_card)

        layout.addStretch()

    def _make_toggle(self, key: str, label: str, initial: bool) -> ToggleRow:
        t = ToggleRow(label, initial)
        self._toggles[key] = t

        def _on_change(checked, _key=key):
            save_preferences({"features": {_key: checked}})
            self.settings_changed.emit()

        t.toggled.connect(_on_change)
        return t

    def refresh_from_prefs(self):
        prefs = load_preferences()
        f = prefs["features"]
        for key, toggle in self._toggles.items():
            toggle.blockSignals(True)
            toggle.setChecked(f.get(key, True))
            toggle.blockSignals(False)
```

- [ ] **Step 2: Add Sentence features to main_window.py**

In `gui/main_window.py`:

1. Add the import alongside other section imports:
   ```python
   from gui.section_sentence_features import SentenceFeaturesSection
   ```

2. Update `SECTIONS` list to add "Sentence features":
   ```python
   SECTIONS = ["Inputs", "Word-level features", "Document-level features",
               "Sub-word features", "Sentence features", "Run", "Logs"]
   ```

3. In `__init__`, add after `self.subword_section = SubwordFeaturesSection()`:
   ```python
   self.sentence_section = SentenceFeaturesSection()
   ```

4. Connect settings_changed signal (after the `subword_section.settings_changed.connect(...)` line):
   ```python
   self.sentence_section.settings_changed.connect(self.run_section.refresh)
   ```

5. Add to the `sections` list (between `self.subword_section` and `self.run_section`):
   ```python
   sections = [
       self.inputs_section,
       self.word_features,
       self.doc_features,
       self.subword_section,
       self.sentence_section,   # new
       self.run_section,
       self.logs_section,
   ]
   ```

- [ ] **Step 3: Run GUI smoke test**

```
conda run -p env python -m pytest tests/gui/test_main_window.py -v
```
Expected: pass

- [ ] **Step 4: Commit**

```bash
git add gui/section_sentence_features.py gui/main_window.py
git commit -m "feat(gui): add Sentence features page with timing and prosody toggles"
```

---

## Task 7: Integration test for sentence/ path

**Files:**
- Modify: `tests/integration/test_output_paths.py`

- [ ] **Step 1: Add sentence/ path test**

In `tests/integration/test_output_paths.py`, add a helper and test:

```python
def _word_args_with_align(tmp_path):
    """Args that exercise the sentence sidecar path — needs alignment context.
    Uses --no-align so no MFA is invoked, but sentence CSV should still be
    written (with NaN timings but valid sentence text rows).
    """
    return build_parser().parse_args([
        '--transcripts-folder', str(FIXTURES / 'transcripts'),
        '--audio-folder', str(FIXTURES / 'audio'),
        '--output-folder', str(tmp_path / 'out'),
        '--analyzer', 'TestRunner',
        '--no-align',
        '--no-llm-cloze',
        '--no-edu-freq',
    ])


def test_sentence_csv_lands_in_sentence_subfolder(tmp_path):
    run_word_level_batch(_word_args_with_align(tmp_path))
    out = tmp_path / 'out'
    # Sentence CSV emits even without alignment (NaN Onset/End but rows are present)
    assert (out / 'sentence' / 'story1_sentences.csv').exists()
    assert (out / 'sentence' / 'story2_sentences.csv').exists()


def test_sentence_csv_suppressed_by_no_sentences_flag(tmp_path):
    args = _word_args_with_align(tmp_path)
    args.no_sentences = True
    run_word_level_batch(args)
    out = tmp_path / 'out'
    assert not (out / 'sentence').exists()
```

- [ ] **Step 2: Run integration tests**

```
conda run -p env python -m pytest tests/integration/test_output_paths.py -v
```
Expected: all pass including new sentence tests

- [ ] **Step 3: Run full test suite**

```
conda run -p env python -m pytest -x -q
```
Expected: all pass (or the same 1 intentional skip that was already present)

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_output_paths.py
git commit -m "test(integration): verify sentence/ subfolder creation and --no-sentences suppression"
```

---

## Task 8: Update master_plan.md

- [ ] **Step 1: Update docs/master_plan.md**

Update §2 (Current state) to reflect this batch is active/complete, and §3 (Completed batches) to add a row for the sentence-level tier batch. Update the prefs version comment to v9.

- [ ] **Step 2: Commit**

```bash
git add docs/master_plan.md
git commit -m "docs: record sentence-level tier batch in master_plan"
```
