# Acoustic Level (audio-only TRF features) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a new audio-only `acoustic` level that emits broadband amplitude envelope, onset envelope, log-mel spectrogram, and MFCC as slow time series at a configurable analysis rate, packaged one combined `.mat` per stimulus on a canonical shared grid.

**Architecture:** A pure-function extractor (`extractors/acoustic_ts.py`, librosa + scipy) computes each feature on a shared grid `N = round(duration × rate)`. A standalone batch runner (`run_acoustic_level_batch` in `main.py`) globs the audio folder independently of transcripts and writes `acoustic/<stim>_acoustic.mat` via `scipy.io.savemat`. The level is registered in the planner registry (so the schema-drift guard and master-toggle machinery work) but executes through its own runner, bypassing the word/doc DataFrame pipeline. Prefs bump v12 → v13 (additive). New GUI page + CLI `--level acoustic`.

**Tech Stack:** Python, librosa, scipy.signal / scipy.io, numpy, PySide6 (GUI), pytest / pytest-qt.

**Spec:** `docs/superpowers/specs/2026-06-13-acoustic-level-trf-features-design.md`

**Conventions to honor (from memory):**
- TDD for `extractors/` math (Tasks 1–6); test-after for prefs/planner/GUI/CLI/docs.
- Per-feature docs: every GUI toggle key resolves to a `docs/features/*.md`; the ⓘ icon appears automatically when the file exists.
- Commit per task, directly to `main`.
- Every empirical doc claim gets an inline `[Author Year]` citation, fact-checked; never fabricate references; flag interpolated *why* with ⚠️.

**Naming (locked to avoid collision with the word-level `feat_acoustic_*`/`master_acoustic` Onset/End keys):**
- prefs feature keys: `feat_acoustic_ts_envelope`, `feat_acoustic_ts_onset_env`, `feat_acoustic_ts_logmel`, `feat_acoustic_ts_mfcc`, master `master_acoustic_ts`.
- prefs output key: `acoustic_ts_mat`. Nested config sub-dict: `acoustic_ts`.
- planner stage: `acoustic_ts`; planner output_csv id: `acoustic`; output folder: `acoustic/`.

---

## Task 1: Envelope extractor — `rms` method (mTRFenvelope replica)

**Files:**
- Create: `extractors/acoustic_ts.py`
- Test: `tests/extractors/test_acoustic_ts.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/extractors/test_acoustic_ts.py
import numpy as np
import pytest

from extractors import acoustic_ts


def _tone(sr=16000, dur=1.0, f=200.0, amp=0.5):
    t = np.arange(int(sr * dur)) / sr
    return (amp * np.sin(2 * np.pi * f * t)).astype(np.float64), sr


def test_rms_envelope_length_matches_grid():
    y, sr = _tone(dur=1.0)
    rate = 128
    env = acoustic_ts.amplitude_envelope(y, sr, rate, method="rms", comp=0.3)
    assert env.shape == (round(1.0 * rate),)


def test_rms_envelope_constant_tone_is_flat():
    # A steady tone -> envelope roughly constant across the interior frames.
    y, sr = _tone(dur=1.0, amp=0.5)
    env = acoustic_ts.amplitude_envelope(y, sr, 128, method="rms", comp=1.0)
    interior = env[5:-5]
    assert np.std(interior) < 0.05 * np.mean(interior)


def test_rms_envelope_compression_monotonic():
    # Lower exponent compresses the dynamic range: a quiet vs loud ratio
    # moves toward 1 as comp -> 0.
    y_loud, sr = _tone(amp=0.8)
    y_quiet, _ = _tone(amp=0.1)
    def ratio(comp):
        a = acoustic_ts.amplitude_envelope(y_loud, sr, 128, method="rms", comp=comp).mean()
        b = acoustic_ts.amplitude_envelope(y_quiet, sr, 128, method="rms", comp=comp).mean()
        return a / b
    assert ratio(0.3) < ratio(1.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/extractors/test_acoustic_ts.py -q`
Expected: FAIL — `AttributeError: module 'extractors.acoustic_ts' has no attribute 'amplitude_envelope'`.

- [ ] **Step 3: Write minimal implementation**

```python
# extractors/acoustic_ts.py
"""Audio-only continuous acoustic features for TRF / cortical-tracking analyses.

Pure functions of the audio signal (no transcript, no EEG). Every feature is
returned as a slow time series at a caller-chosen analysis rate on a shared
canonical grid of N = round(duration * rate) samples, so multiple features
stack directly as TRF predictors.

The `rms` amplitude-envelope method faithfully replicates mTRFenvelope.m from
the mTRF-Toolbox (square -> moving-average downsample -> sqrt -> power-law
compress); `hilbert` is an alternative analytic-signal envelope.
"""
from __future__ import annotations

import numpy as np


def _grid_len(n_samples: int, sr: float, rate: float) -> int:
    """Canonical number of output frames for a signal of n_samples at sr."""
    return int(round((n_samples / sr) * rate))


def _fit_to_grid(arr: np.ndarray, n: int) -> np.ndarray:
    """Trim or edge-pad `arr` along axis 0 to exactly `n` rows."""
    if arr.shape[0] == n:
        return arr
    if arr.shape[0] > n:
        return arr[:n]
    pad_width = [(0, n - arr.shape[0])] + [(0, 0)] * (arr.ndim - 1)
    return np.pad(arr, pad_width, mode="edge")


def amplitude_envelope(y: np.ndarray, sr: float, rate: float, *,
                       method: str = "rms", comp: float = 0.3) -> np.ndarray:
    """Broadband amplitude envelope, power-law compressed, at `rate` Hz.

    method="rms" replicates mTRFenvelope.m: square -> frame-average every
    round(sr/rate) samples -> sqrt -> raise to `comp`.
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    n = _grid_len(y.size, sr, rate)
    if method == "rms":
        L = max(1, int(round(sr / rate)))
        n_frames = y.size // L
        framed = y[:n_frames * L].reshape(n_frames, L)
        power = (framed ** 2).mean(axis=1)        # boxcar avg = anti-alias
        env = np.sqrt(power) ** comp
    elif method == "hilbert":
        raise NotImplementedError  # Task 2
    else:
        raise ValueError(f"unknown envelope method: {method!r}")
    return _fit_to_grid(env, n)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/extractors/test_acoustic_ts.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add extractors/acoustic_ts.py tests/extractors/test_acoustic_ts.py
git commit -m "feat(acoustic): rms amplitude envelope (mTRFenvelope replica)"
```

---

## Task 2: Envelope extractor — `hilbert` method

**Files:**
- Modify: `extractors/acoustic_ts.py` (`amplitude_envelope`)
- Test: `tests/extractors/test_acoustic_ts.py`

- [ ] **Step 1: Write the failing test**

```python
def test_hilbert_envelope_length_and_shape_matches_rms():
    y, sr = _tone(dur=1.0)
    rate = 128
    env_h = acoustic_ts.amplitude_envelope(y, sr, rate, method="hilbert", comp=0.3)
    env_r = acoustic_ts.amplitude_envelope(y, sr, rate, method="rms", comp=0.3)
    assert env_h.shape == env_r.shape == (round(1.0 * rate),)


def test_hilbert_and_rms_agree_on_steady_tone():
    # Both should report a near-constant envelope of similar magnitude on a
    # steady tone (correlation of interior frames is high).
    y, sr = _tone(dur=1.0, amp=0.5)
    h = acoustic_ts.amplitude_envelope(y, sr, 128, method="hilbert", comp=1.0)[5:-5]
    r = acoustic_ts.amplitude_envelope(y, sr, 128, method="rms", comp=1.0)[5:-5]
    # Means within 25% of each other.
    assert abs(h.mean() - r.mean()) < 0.25 * r.mean()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/extractors/test_acoustic_ts.py -k hilbert -q`
Expected: FAIL — `NotImplementedError`.

- [ ] **Step 3: Write minimal implementation**

Replace the `elif method == "hilbert": raise NotImplementedError` branch with:

```python
    elif method == "hilbert":
        from scipy.signal import hilbert, resample_poly
        from math import gcd
        analytic = np.abs(hilbert(y))
        # Resample the magnitude envelope from sr to rate with the built-in
        # FIR anti-alias filter. Use integer up/down derived from the ratio.
        target = int(round(rate))
        source = int(round(sr))
        g = gcd(target, source)
        env = resample_poly(analytic, up=target // g, down=source // g) ** comp
```

(Leave the `_fit_to_grid(env, n)` return at the end of the function — it handles the off-by-a-frame length from `resample_poly`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/extractors/test_acoustic_ts.py -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add extractors/acoustic_ts.py tests/extractors/test_acoustic_ts.py
git commit -m "feat(acoustic): hilbert envelope method"
```

---

## Task 3: Onset envelope

**Files:**
- Modify: `extractors/acoustic_ts.py`
- Test: `tests/extractors/test_acoustic_ts.py`

- [ ] **Step 1: Write the failing test**

```python
def test_onset_envelope_nonnegative_and_length_preserved():
    env = np.array([0.0, 1.0, 1.0, 0.5, 2.0, 2.0], dtype=np.float64)
    onset = acoustic_ts.onset_envelope(env)
    assert onset.shape == env.shape
    assert np.all(onset >= 0)


def test_onset_envelope_keeps_rises_drops_falls():
    env = np.array([0.0, 1.0, 1.0, 0.5, 2.0], dtype=np.float64)
    onset = acoustic_ts.onset_envelope(env)
    # diff = [_, +1, 0, -0.5, +1.5]; HWR keeps positives, first elem 0.
    np.testing.assert_allclose(onset, [0.0, 1.0, 0.0, 0.0, 1.5])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/extractors/test_acoustic_ts.py -k onset -q`
Expected: FAIL — `AttributeError: ... 'onset_envelope'`.

- [ ] **Step 3: Write minimal implementation**

```python
def onset_envelope(env: np.ndarray) -> np.ndarray:
    """Half-wave-rectified first difference of the amplitude envelope.

    Keeps only positive-going changes (acoustic onsets); length is preserved
    by prepending 0 so it stays on the same grid as `env`.
    """
    env = np.asarray(env, dtype=np.float64).ravel()
    diff = np.diff(env, prepend=env[:1])
    return np.clip(diff, 0.0, None)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/extractors/test_acoustic_ts.py -q`
Expected: PASS (7 passed).

- [ ] **Step 5: Commit**

```bash
git add extractors/acoustic_ts.py tests/extractors/test_acoustic_ts.py
git commit -m "feat(acoustic): onset envelope (HWR derivative)"
```

---

## Task 4: Log-mel spectrogram + mel presets

**Files:**
- Modify: `extractors/acoustic_ts.py`
- Test: `tests/extractors/test_acoustic_ts.py`

- [ ] **Step 1: Write the failing test**

```python
def test_mel_range_for_preset_returns_speech_and_general():
    assert acoustic_ts.mel_range("speech", sr=16000) == (50.0, 8000.0)
    fmin, fmax = acoustic_ts.mel_range("music", sr=16000)
    assert fmin == 0.0 and fmax == 8000.0  # Nyquist for sr=16000


def test_log_mel_shape_and_center_freqs():
    y, sr = _tone(dur=1.0)
    rate, n_mels = 128, 16
    logmel, freqs = acoustic_ts.log_mel_spectrogram(
        y, sr, rate, n_mels=n_mels, fmin=50.0, fmax=8000.0)
    assert logmel.shape == (round(1.0 * rate), n_mels)
    assert freqs.shape == (n_mels,)


def test_log_mel_tone_energy_in_expected_band():
    # A 200 Hz tone should put its peak mel energy in a low mel band.
    y, sr = _tone(dur=1.0, f=200.0)
    logmel, freqs = acoustic_ts.log_mel_spectrogram(
        y, sr, 128, n_mels=16, fmin=50.0, fmax=8000.0)
    peak_band = int(np.argmax(logmel.mean(axis=0)))
    assert peak_band < 8  # lower half of the 16 bands
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/extractors/test_acoustic_ts.py -k "mel" -q`
Expected: FAIL — missing `mel_range` / `log_mel_spectrogram`.

- [ ] **Step 3: Write minimal implementation**

```python
_MEL_PRESETS = {
    "speech": (50.0, None),   # fmax filled with min(8000, Nyquist) below
    "music":  (0.0, None),    # full band to Nyquist
    "general": (0.0, None),
}


def mel_range(preset: str, sr: float) -> tuple[float, float]:
    """Resolve a preset name to (fmin, fmax) in Hz for a given sample rate."""
    nyq = sr / 2.0
    if preset == "speech":
        return (50.0, min(8000.0, nyq))
    if preset in ("music", "general"):
        return (0.0, nyq)
    raise ValueError(f"unknown mel preset: {preset!r}")


def log_mel_spectrogram(y: np.ndarray, sr: float, rate: float, *,
                        n_mels: int = 16, fmin: float = 50.0,
                        fmax: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Log-mel (dB) spectrogram on the canonical grid, plus mel center freqs.

    hop_length is set so the frame rate equals `rate`; a ~25 ms analysis
    window (next power of two >= 0.025*sr) is used.
    """
    import librosa
    y = np.asarray(y, dtype=np.float64).ravel()
    if fmax is None:
        fmax = sr / 2.0
    hop = max(1, int(round(sr / rate)))
    win = max(hop, int(round(0.025 * sr)))
    n_fft = 1 << (win - 1).bit_length()   # next power of two >= win
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop, win_length=win,
        n_mels=n_mels, fmin=fmin, fmax=fmax, power=2.0)
    logmel = librosa.power_to_db(mel, ref=np.max).T   # (frames, n_mels)
    freqs = librosa.mel_frequencies(n_mels=n_mels, fmin=fmin, fmax=fmax)
    n = _grid_len(y.size, sr, rate)
    return _fit_to_grid(logmel, n), np.asarray(freqs, dtype=np.float64)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/extractors/test_acoustic_ts.py -q`
Expected: PASS (10 passed).

- [ ] **Step 5: Commit**

```bash
git add extractors/acoustic_ts.py tests/extractors/test_acoustic_ts.py
git commit -m "feat(acoustic): log-mel spectrogram + speech/music presets"
```

---

## Task 5: MFCC

**Files:**
- Modify: `extractors/acoustic_ts.py`
- Test: `tests/extractors/test_acoustic_ts.py`

- [ ] **Step 1: Write the failing test**

```python
def test_mfcc_shape_on_grid():
    y, sr = _tone(dur=1.0)
    rate, n_mfcc = 128, 13
    m = acoustic_ts.mfcc(y, sr, rate, n_mfcc=n_mfcc, n_mels=16,
                         fmin=50.0, fmax=8000.0)
    assert m.shape == (round(1.0 * rate), n_mfcc)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/extractors/test_acoustic_ts.py -k mfcc -q`
Expected: FAIL — missing `mfcc`.

- [ ] **Step 3: Write minimal implementation**

```python
def mfcc(y: np.ndarray, sr: float, rate: float, *, n_mfcc: int = 13,
         n_mels: int = 16, fmin: float = 50.0,
         fmax: float | None = None) -> np.ndarray:
    """MFCCs on the canonical grid.

    NOTE: for TRF / cortical-tracking work the log-mel spectrogram is usually
    preferred over MFCCs — the DCT that produces MFCCs removes the tonotopic
    (frequency-band) interpretability that makes mel features map onto
    auditory-cortex organization. MFCCs are provided for completeness, not as
    the primary feature.
    """
    import librosa
    y = np.asarray(y, dtype=np.float64).ravel()
    if fmax is None:
        fmax = sr / 2.0
    hop = max(1, int(round(sr / rate)))
    win = max(hop, int(round(0.025 * sr)))
    n_fft = 1 << (win - 1).bit_length()
    m = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop,
        win_length=win, n_mels=n_mels, fmin=fmin, fmax=fmax)
    n = _grid_len(y.size, sr, rate)
    return _fit_to_grid(m.T, n)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/extractors/test_acoustic_ts.py -q`
Expected: PASS (11 passed).

- [ ] **Step 5: Commit**

```bash
git add extractors/acoustic_ts.py tests/extractors/test_acoustic_ts.py
git commit -m "feat(acoustic): MFCC feature"
```

---

## Task 6: `extract_acoustic` orchestrator (load audio → dict on shared grid)

**Files:**
- Modify: `extractors/acoustic_ts.py`
- Test: `tests/extractors/test_acoustic_ts.py`

- [ ] **Step 1: Write the failing test**

```python
def test_extract_acoustic_all_features_share_grid(tmp_path):
    import soundfile as sf
    y, sr = _tone(dur=1.0)
    wav = tmp_path / "stim.wav"
    sf.write(str(wav), y, sr)
    rate = 128
    out = acoustic_ts.extract_acoustic(str(wav), rate, {
        "envelope": True, "onset_env": True, "logmel": True, "mfcc": True,
        "env_method": "rms", "env_comp": 0.3, "n_mels": 16, "n_mfcc": 13,
        "mel_preset": "speech",
    })
    n = out["t"].shape[0]
    assert out["fs"] == rate
    assert out["envelope"].shape == (n,)
    assert out["onset_envelope"].shape == (n,)
    assert out["logmel"].shape[0] == n and out["logmel"].shape[1] == 16
    assert out["mfcc"].shape[0] == n and out["mfcc"].shape[1] == 13
    assert out["mel_center_freqs"].shape == (16,)
    assert "config" in out


def test_extract_acoustic_respects_disabled_features(tmp_path):
    import soundfile as sf
    y, sr = _tone(dur=0.5)
    wav = tmp_path / "stim.wav"
    sf.write(str(wav), y, sr)
    out = acoustic_ts.extract_acoustic(str(wav), 100, {
        "envelope": True, "onset_env": False, "logmel": False, "mfcc": False,
        "env_method": "rms", "env_comp": 0.3, "n_mels": 16, "n_mfcc": 13,
        "mel_preset": "speech",
    })
    assert "envelope" in out
    assert "onset_envelope" not in out
    assert "logmel" not in out
    assert "mfcc" not in out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/extractors/test_acoustic_ts.py -k extract_acoustic -q`
Expected: FAIL — missing `extract_acoustic`. (If `soundfile` import fails, install it — it ships with librosa; otherwise write the wav via `scipy.io.wavfile.write` with an int16 cast.)

- [ ] **Step 3: Write minimal implementation**

```python
def extract_acoustic(audio_path: str, rate: float, cfg: dict) -> dict:
    """Compute all enabled acoustic time-series features for one audio file.

    Returns a dict of named arrays (only the enabled ones) plus `fs`, `t`,
    `mel_center_freqs` (when logmel/mfcc enabled), and a `config` provenance
    dict. All feature arrays share exactly N = round(duration*rate) rows.
    """
    import librosa
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    y = np.asarray(y, dtype=np.float64)
    n = _grid_len(y.size, sr, rate)
    fmin, fmax = mel_range(cfg.get("mel_preset", "speech"), sr)

    out: dict = {"fs": float(rate), "t": (np.arange(n) / rate).astype(np.float64)}

    env = None
    if cfg.get("envelope", True) or cfg.get("onset_env", True):
        env = amplitude_envelope(y, sr, rate, method=cfg.get("env_method", "rms"),
                                 comp=float(cfg.get("env_comp", 0.3)))
    if cfg.get("envelope", True):
        out["envelope"] = env
    if cfg.get("onset_env", True):
        out["onset_envelope"] = onset_envelope(env)
    if cfg.get("logmel", True):
        logmel, freqs = log_mel_spectrogram(
            y, sr, rate, n_mels=int(cfg.get("n_mels", 16)), fmin=fmin, fmax=fmax)
        out["logmel"] = logmel
        out["mel_center_freqs"] = freqs
    if cfg.get("mfcc", False):
        out["mfcc"] = mfcc(y, sr, rate, n_mfcc=int(cfg.get("n_mfcc", 13)),
                           n_mels=int(cfg.get("n_mels", 16)), fmin=fmin, fmax=fmax)
        if "mel_center_freqs" not in out:
            out["mel_center_freqs"] = librosa_mel_freqs(int(cfg.get("n_mels", 16)), fmin, fmax)

    out["config"] = {
        "analysis_rate": float(rate), "env_method": cfg.get("env_method", "rms"),
        "env_comp": float(cfg.get("env_comp", 0.3)),
        "n_mels": int(cfg.get("n_mels", 16)), "n_mfcc": int(cfg.get("n_mfcc", 13)),
        "fmin": float(fmin), "fmax": float(fmax),
        "mel_preset": cfg.get("mel_preset", "speech"),
        "source_sr": float(sr), "duration_s": float(y.size / sr),
    }
    return out


def librosa_mel_freqs(n_mels: int, fmin: float, fmax: float) -> np.ndarray:
    import librosa
    return np.asarray(librosa.mel_frequencies(n_mels=n_mels, fmin=fmin, fmax=fmax),
                      dtype=np.float64)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/extractors/test_acoustic_ts.py -q`
Expected: PASS (all extractor tests green).

- [ ] **Step 5: Commit**

```bash
git add extractors/acoustic_ts.py tests/extractors/test_acoustic_ts.py
git commit -m "feat(acoustic): extract_acoustic orchestrator on shared grid"
```

---

## Task 7: Prefs v12 → v13 (additive)

**Files:**
- Modify: `utils/preferences.py`
- Test: `tests/utils/test_preferences_migration.py`

**Pattern to follow:** the v12 block (`utils/preferences.py:564-603`) and `_migrate_v9_into_v10` (`:1122-1128`). v10/v11/v12 are handled by the additive branch at `:656` (`if version in (V10, V11, V12)`). Add V13 to that branch and to the finalize stamp.

- [ ] **Step 1: Write the failing test** (append to `tests/utils/test_preferences_migration.py`)

```python
def test_v12_prefs_migrate_to_v13_additively(tmp_path, monkeypatch):
    import json
    from utils import preferences
    p = tmp_path / "preferences.json"
    # minimal v12 file
    v12 = {"schema_version": 12, "features": {}, "output": {}}
    p.write_text(json.dumps(v12), encoding="utf-8")
    monkeypatch.setenv("SENSI_PREFS_PATH", str(p))
    prefs = preferences.load_preferences()
    assert prefs["schema_version"] == 13
    f = prefs["features"]
    assert f["master_acoustic_ts"] is False           # opt-in
    assert f["feat_acoustic_ts_envelope"] is True
    assert f["feat_acoustic_ts_mfcc"] is False
    assert prefs["output"]["acoustic_ts_mat"] is True
    a = prefs["acoustic_ts"]
    assert a["analysis_rate"] == 128 and a["env_method"] == "rms"
    assert a["mel_preset"] == "speech"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/utils/test_preferences_migration.py -k v13 -q`
Expected: FAIL — `schema_version` is 12 / missing keys.

- [ ] **Step 3: Write minimal implementation**

After the v12 block (`:603`) add:

```python
# ---------------------------------------------------------------------------
# v13 constants
#
# Acoustic level (2026-06-13): audio-only continuous TRF features emitted as a
# combined .mat per stimulus. Opt-in: master defaults OFF. Additive only.
# ---------------------------------------------------------------------------

SCHEMA_VERSION_V13 = 13

_ACOUSTIC_TS_FEATURE_KEYS = {
    "master_acoustic_ts":        False,   # opt-in
    "feat_acoustic_ts_envelope": True,
    "feat_acoustic_ts_onset_env": True,
    "feat_acoustic_ts_logmel":   True,
    "feat_acoustic_ts_mfcc":     False,   # off by default
}

_FEATURES_DEFAULT_V13 = {**_FEATURES_DEFAULT_V12, **_ACOUSTIC_TS_FEATURE_KEYS}
_OUTPUT_DEFAULT_V13 = {**_OUTPUT_DEFAULT_V12, "acoustic_ts_mat": True}

DEFAULT_PREFS_V13: dict[str, Any] = {
    **DEFAULT_PREFS_V12,
    "schema_version": SCHEMA_VERSION_V13,
    "features": dict(_FEATURES_DEFAULT_V13),
    "output":   dict(_OUTPUT_DEFAULT_V13),
    "acoustic_ts": {
        "analysis_rate": 128,
        "env_method": "rms",
        "env_comp": 0.3,
        "n_mels": 16,
        "n_mfcc": 13,
        "mel_preset": "speech",
    },
}
```

Then update the active aliases (`:609-610`):

```python
SCHEMA_VERSION = SCHEMA_VERSION_V13  # active version
DEFAULT_PREFS: dict[str, Any] = DEFAULT_PREFS_V13  # active defaults
```

Add `"acoustic_ts"` to `_CURRENT_SECTIONS` (`:817-818`) so its sub-dict merges by inner key:

```python
_CURRENT_SECTIONS = ('shared', 'features', 'predictability',
                     'semantic', 'output', 'prosody', 'lexical_norms',
                     'acoustic_ts')
```

Add V13 to the additive load branch (`:656`):

```python
    if version in (SCHEMA_VERSION_V10, SCHEMA_VERSION_V11,
                   SCHEMA_VERSION_V12, SCHEMA_VERSION_V13):
```

Also update the module docstring schema note (`:3` and `:25-29`) to mention v13 and the `acoustic_ts` section. (Docstring only — no behavior.)

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/utils/test_preferences_migration.py -q`
Expected: PASS (existing migration tests + new v13 test).

- [ ] **Step 5: Commit**

```bash
git add utils/preferences.py tests/utils/test_preferences_migration.py
git commit -m "feat(prefs): v12->v13 additive — acoustic level keys"
```

---

## Task 8: Planner registry + plan output map + schema-drift guard

**Files:**
- Modify: `gui/planner/registry.py`, `gui/planner/plan.py`, `tests/planner/test_schema_drift.py`
- Test: `tests/planner/test_schema_drift.py` (and existing planner tests)

- [ ] **Step 1: Write the failing test** (append to `tests/planner/test_schema_drift.py`)

```python
def test_acoustic_level_routed_when_output_on():
    from gui.planner.plan import plan_from_prefs
    from utils.preferences import DEFAULT_PREFS
    import copy
    prefs = copy.deepcopy(DEFAULT_PREFS)
    prefs["features"]["master_acoustic_ts"] = True
    prefs["output"]["acoustic_ts_mat"] = True
    plan = plan_from_prefs(prefs)
    assert "acoustic" in plan.output_csvs
    assert "acoustic_ts" in plan.stages_to_run


def test_acoustic_level_off_by_default():
    from gui.planner.plan import plan_from_prefs
    from utils.preferences import DEFAULT_PREFS
    plan = plan_from_prefs(DEFAULT_PREFS)
    assert "acoustic" not in plan.output_csvs   # master defaults OFF
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/planner/test_schema_drift.py -q`
Expected: FAIL — `test_every_feature_pref_key_is_in_some_cluster` flags the new `feat_acoustic_ts_*` keys as unrouted, and the new routing tests fail.

- [ ] **Step 3: Write minimal implementation**

In `gui/planner/registry.py`, add the import near the other extractor imports (`:39-40`):

```python
from extractors.acoustic_ts import extract_acoustic as _extract_acoustic
```

Add the stage to `STAGES` (after the embeddings `bert_embed` stage, `:131`):

```python
    # --- §6 Acoustic level (audio-only continuous TRF features) ---
    Stage(name="acoustic_ts",
          consumes=("audio",),
          produces=("acoustic_mat",),
          run=_extract_acoustic),
```

Add the cluster to `CLUSTERS` (after the bert sentence cluster, `:430`):

```python
    # --- Acoustic level (§6) ---
    FeatureCluster(
        name="acoustic time-series",
        toggle_keys=("feat_acoustic_ts_envelope", "feat_acoustic_ts_onset_env",
                     "feat_acoustic_ts_logmel", "feat_acoustic_ts_mfcc"),
        output_csv="acoustic",
        requires_stages=("acoustic_ts",),
        emits_columns=(),   # .mat variables, not CSV columns
        master_toggle="master_acoustic_ts",
    ),
```

In `gui/planner/plan.py`, add to `_OUTPUT_CSV_MAP` (`:43-52`):

```python
    "acoustic_ts_mat":   "acoustic",
```

In `tests/planner/test_schema_drift.py`, add `master_acoustic_ts` to `_INTENTIONALLY_UNROUTED_FEATURES` (`:15-41`) and `acoustic_ts_mat` to `_OUTPUT_KEYS_VIA_CSV_MAP` (`:46-57`):

```python
    "master_acoustic_ts",   # routes via master_toggle field
```
```python
    "acoustic_ts_mat",      # routed via _OUTPUT_CSV_MAP
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/planner/ -q`
Expected: PASS (all planner tests, including the two new routing tests).

- [ ] **Step 5: Commit**

```bash
git add gui/planner/registry.py gui/planner/plan.py tests/planner/test_schema_drift.py
git commit -m "feat(planner): register acoustic level stage + cluster"
```

---

## Task 9: Batch runner + CLI `--level acoustic`

**Files:**
- Modify: `main.py`
- Test: `tests/integration/test_acoustic_level.py` (create)

**Pattern to follow:** `run_doc_level_batch` (`main.py:816+`) for the RunLogger / loop scaffolding, and the embeddings `.mat` write (`main.py:733-734`) for `scipy.io.savemat`. Key difference: glob the audio folder directly — do NOT call `find_pairs`.

- [ ] **Step 1: Write the failing test**

```python
# tests/integration/test_acoustic_level.py
import argparse
from pathlib import Path

import numpy as np
import scipy.io
import soundfile as sf

import main


def _make_audio(folder: Path, name: str, sr=16000, dur=0.5):
    folder.mkdir(parents=True, exist_ok=True)
    t = np.arange(int(sr * dur)) / sr
    sf.write(str(folder / name), 0.4 * np.sin(2 * np.pi * 200 * t), sr)


def _args(audio_dir, out_dir):
    return argparse.Namespace(
        audio_folder=str(audio_dir), output_folder=str(out_dir),
        analyzer="tester", audio_extensions="wav,flac,mp3",
        acoustic_ts={"envelope": True, "onset_env": True, "logmel": True,
                     "mfcc": False, "env_method": "rms", "env_comp": 0.3,
                     "n_mels": 16, "n_mfcc": 13, "mel_preset": "speech",
                     "analysis_rate": 128},
        run_ts="2026-06-13_00-00-00",
    )


def test_acoustic_batch_writes_mat_without_transcript(tmp_path):
    audio = tmp_path / "audio"
    out = tmp_path / "out"
    _make_audio(audio, "stimA.wav")
    main.run_acoustic_level_batch(_args(audio, out))
    mat_path = out / "acoustic" / "stimA_acoustic.mat"
    assert mat_path.is_file()
    data = scipy.io.loadmat(str(mat_path))
    assert "envelope" in data and "onset_envelope" in data and "logmel" in data
    assert "mfcc" not in data           # disabled
    assert float(data["fs"].squeeze()) == 128
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/integration/test_acoustic_level.py -q`
Expected: FAIL — `AttributeError: module 'main' has no attribute 'run_acoustic_level_batch'`.

- [ ] **Step 3: Write minimal implementation**

Add to `main.py` (near `run_doc_level_batch`). The stimulus id is the audio filename stem.

```python
def run_acoustic_level_batch(args, *, on_pair_done=None, should_cancel=None) -> None:
    """Process every audio file in args.audio_folder independently of any
    transcript, writing acoustic/<stim>_acoustic.mat per file."""
    import scipy.io
    from extractors.acoustic_ts import extract_acoustic

    audio_dir = Path(args.audio_folder)
    output_dir = Path(args.output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_exts = [e.strip().lower() for e in args.audio_extensions.split(',') if e.strip()]
    audio_files = sorted(
        p for p in audio_dir.rglob('*')
        if p.is_file() and p.suffix.lower().lstrip('.') in audio_exts
    )

    logger = RunLogger(
        output_folder=output_dir,
        analyzer=getattr(args, 'analyzer', '') or 'unknown',
        config={'level': 'acoustic', 'acoustic_ts': args.acoustic_ts},
        input_transcripts_folder=audio_dir,   # no transcripts; record audio dir
        input_audio_folder=audio_dir,
        text_extensions=[],
        audio_extensions=audio_exts,
        filename_prefix='acoustic_run',
        run_ts=getattr(args, 'run_ts', None),
        skip_no_match=True,
    )

    rate = float(args.acoustic_ts.get('analysis_rate', 128))
    for i, audio_path in enumerate(audio_files):
        if should_cancel and should_cancel():
            logger.add_warning('Run cancelled by user before processing complete')
            break
        start = time.time()
        try:
            out = extract_acoustic(str(audio_path), rate, args.acoustic_ts)
            mat_dir = _level_dir(output_dir, 'acoustic')
            mat_path = mat_dir / f'{audio_path.stem}_acoustic.mat'
            scipy.io.savemat(str(mat_path), out)   # savemat renders dict 'config' as a struct
            # Audio-only: no transcript. Pass the audio path for both the
            # transcript and audio slots; the .mat is the output artifact.
            logger.record_pair_success(
                stimulus_id=audio_path.stem,
                transcript_path=audio_path,
                audio_path=audio_path,
                runtime_seconds=time.time() - start,
                output_csv_path=mat_path,
                output_summary_path=None,
            )
        except Exception as exc:   # noqa: BLE001 — record and continue batch
            logger.add_warning(f'{audio_path.name}: {exc}')
        if on_pair_done:
            on_pair_done(i + 1, len(audio_files), audio_path.stem)

    logger.finalize()
```

`RunLogger`'s verified public API (`utils/run_logger.py`): `__init__(output_folder, analyzer, config, input_transcripts_folder, input_audio_folder, text_extensions, audio_extensions, filename_prefix=, subdir='logs', run_ts=, skip_no_match=)`, `add_warning(msg)`, `record_pair_success(stimulus_id, transcript_path, audio_path, runtime_seconds, output_csv_path, output_summary_path)`, `record_pair_error(...)`, `finalize()`. The runner above uses only methods that exist.

Wire the CLI. Add flags in `build_parser()` near the other level flags (`main.py:124-137`):

```python
    p.add_argument('--analysis-rate', type=float, default=128.0,
                   help='Acoustic level: output sample rate in Hz (match your EEG).')
    p.add_argument('--env-method', choices=['rms', 'hilbert'], default='rms')
    p.add_argument('--env-comp', type=float, default=0.3,
                   help='Envelope power-law compression exponent.')
    p.add_argument('--n-mels', type=int, default=16)
    p.add_argument('--n-mfcc', type=int, default=13)
    p.add_argument('--mel-preset', choices=['speech', 'music', 'general'], default='speech')
    p.add_argument('--no-envelope',  action='store_true')
    p.add_argument('--no-onset-env', action='store_true')
    p.add_argument('--no-logmel',    action='store_true')
    p.add_argument('--mfcc', action='store_true', help='Acoustic level: also emit MFCCs (off by default).')
```

Extend `--level` choices (`main.py:137`): `choices=['word', 'document', 'acoustic']`.

In `main()` (`:1018-1029`), the acoustic level only needs `--audio-folder` + `--output-folder` (no transcripts). Add a branch:

```python
    if args.level == 'acoustic':
        if not (args.audio_folder and args.output_folder):
            raise SystemExit('Acoustic level requires --audio-folder and --output-folder')
        args.acoustic_ts = {
            'envelope': not args.no_envelope, 'onset_env': not args.no_onset_env,
            'logmel': not args.no_logmel, 'mfcc': args.mfcc,
            'env_method': args.env_method, 'env_comp': args.env_comp,
            'n_mels': args.n_mels, 'n_mfcc': args.n_mfcc,
            'mel_preset': args.mel_preset, 'analysis_rate': args.analysis_rate,
        }
        run_acoustic_level_batch(args)
        return
```

Place this branch BEFORE the `is_batch` block so it doesn't trip the transcript-required guard (it has its own folder check).

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/integration/test_acoustic_level.py -q`
Expected: PASS. Then a CLI smoke check:
Run: `python main.py --level acoustic --audio-folder <some_wav_dir> --output-folder /tmp/acout`
Expected: `acoustic/<stim>_acoustic.mat` files appear.

- [ ] **Step 5: Commit**

```bash
git add main.py tests/integration/test_acoustic_level.py
git commit -m "feat(acoustic): audio-only batch runner + CLI --level acoustic"
```

---

## Task 10: GUI dispatch wiring (`emit.to_acoustic_namespace` + run_dispatch branch)

**Files:**
- Modify: `gui/planner/emit.py`, `gui/run_dispatch.py`
- Test: `tests/gui/test_run_dispatch.py`

- [ ] **Step 1: Write the failing test** (append to `tests/gui/test_run_dispatch.py`)

```python
def test_acoustic_namespace_carries_config():
    from gui.planner.emit import to_acoustic_namespace
    prefs = {
        "shared": {"last_audio_folder": "/aud", "last_output_folder": "/out",
                   "analyzer_name": "me"},
        "features": {"feat_acoustic_ts_envelope": True, "feat_acoustic_ts_onset_env": False,
                     "feat_acoustic_ts_logmel": True, "feat_acoustic_ts_mfcc": False},
        "acoustic_ts": {"analysis_rate": 64, "env_method": "hilbert", "env_comp": 0.5,
                        "n_mels": 20, "n_mfcc": 13, "mel_preset": "music"},
    }
    ns = to_acoustic_namespace(prefs)
    assert ns.audio_folder == "/aud" and ns.output_folder == "/out"
    assert ns.acoustic_ts["envelope"] is True and ns.acoustic_ts["onset_env"] is False
    assert ns.acoustic_ts["analysis_rate"] == 64
    assert ns.acoustic_ts["env_method"] == "hilbert"
    assert ns.acoustic_ts["mel_preset"] == "music"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/gui/test_run_dispatch.py -k acoustic -q`
Expected: FAIL — missing `to_acoustic_namespace`.

- [ ] **Step 3: Write minimal implementation**

In `gui/planner/emit.py`, add:

```python
def to_acoustic_namespace(prefs: dict) -> argparse.Namespace:
    """Namespace for main.run_acoustic_level_batch. Audio-only; no transcript
    or word/doc stage flags."""
    shared = prefs.get("shared", {})
    feats = prefs.get("features", {})
    acfg = prefs.get("acoustic_ts", {})
    ns = argparse.Namespace()
    ns.audio_folder     = shared.get("last_audio_folder", "")
    ns.output_folder    = shared.get("last_output_folder", "")
    ns.analyzer         = shared.get("analyzer_name", "")
    ns.audio_extensions = "wav,flac,mp3"
    ns.acoustic_ts = {
        "envelope":  bool(feats.get("feat_acoustic_ts_envelope", True)),
        "onset_env": bool(feats.get("feat_acoustic_ts_onset_env", True)),
        "logmel":    bool(feats.get("feat_acoustic_ts_logmel", True)),
        "mfcc":      bool(feats.get("feat_acoustic_ts_mfcc", False)),
        "env_method":   acfg.get("env_method", "rms"),
        "env_comp":     float(acfg.get("env_comp", 0.3)),
        "n_mels":       int(acfg.get("n_mels", 16)),
        "n_mfcc":       int(acfg.get("n_mfcc", 13)),
        "mel_preset":   acfg.get("mel_preset", "speech"),
        "analysis_rate": float(acfg.get("analysis_rate", 128)),
    }
    return ns
```

In `gui/run_dispatch.py` `_go()` (after the doc-batch branch, `:84`):

```python
            if "acoustic" in plan.output_csvs:
                from main import run_acoustic_level_batch
                ns = planner.to_acoustic_namespace(prefs)
                ns.run_ts = run_ts
                run_acoustic_level_batch(ns, on_pair_done=progress_cb)
```

If `planner.to_acoustic_namespace` isn't exported, add it to `gui/planner/__init__.py` alongside `to_namespace` / `plan_from_prefs`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/gui/test_run_dispatch.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add gui/planner/emit.py gui/planner/__init__.py gui/run_dispatch.py tests/gui/test_run_dispatch.py
git commit -m "feat(acoustic): GUI dispatch for the acoustic level"
```

---

## Task 11: GUI page + sidebar

**Files:**
- Create: `gui/section_acoustic_features.py`
- Modify: `gui/main_window.py`
- Test: `tests/gui/test_section_acoustic_features.py` (create)

**Pattern to follow:** `gui/section_sentence_features.py` (ClusterCard + ToggleRow + per-toggle save) for the toggle block; `gui/section_subword_features.py:106-113` (`QDoubleSpinBox` + `save_preferences({"<section>": {...}})`) for the numeric params; `gui/section_word_features.py:356-359` (`QComboBox`) for the method/preset combos. Params save into the `acoustic_ts` prefs section.

- [ ] **Step 1: Write the failing test**

```python
# tests/gui/test_section_acoustic_features.py
import pytest
pytest.importorskip("PySide6")

from gui.section_acoustic_features import AcousticFeaturesSection


def test_section_constructs_with_four_toggles(qtbot, tmp_path, monkeypatch):
    monkeypatch.setenv("SENSI_PREFS_PATH", str(tmp_path / "preferences.json"))
    sec = AcousticFeaturesSection()
    qtbot.addWidget(sec)
    for key in ("feat_acoustic_ts_envelope", "feat_acoustic_ts_onset_env",
                "feat_acoustic_ts_logmel", "feat_acoustic_ts_mfcc"):
        assert key in sec._toggles


def test_rate_change_persists(qtbot, tmp_path, monkeypatch):
    monkeypatch.setenv("SENSI_PREFS_PATH", str(tmp_path / "preferences.json"))
    from utils.preferences import load_preferences
    sec = AcousticFeaturesSection()
    qtbot.addWidget(sec)
    sec._rate_spin.setValue(64)
    assert load_preferences()["acoustic_ts"]["analysis_rate"] == 64
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/gui/test_section_acoustic_features.py -q`
Expected: FAIL — module does not exist.

- [ ] **Step 3: Write minimal implementation**

Create `gui/section_acoustic_features.py`. Mirror `SentenceFeaturesSection`: a `ClusterCard("Acoustic (audio-only)", …, master_key="master_acoustic_ts", master_label="Emit acoustic .mat sidecar")` holding a grid of four `_make_toggle` ToggleRows (keys + labels below), plus a params `Card` with: `QDoubleSpinBox` `self._rate_spin` (range 1–2000, default from `acoustic_ts.analysis_rate`, `valueChanged → save_preferences({"acoustic_ts": {"analysis_rate": v}})`), a `QComboBox` for `env_method` (`["rms", "hilbert"]`), a `QDoubleSpinBox` for `env_comp` (0.0–2.0, step 0.05), `QSpinBox`/`QDoubleSpinBox` for `n_mels` (1–128) and `n_mfcc` (1–60), and a `QComboBox` for `mel_preset` (`["speech", "music", "general"]`). Each combo/spin persists into `prefs["acoustic_ts"]`. Provide `refresh_from_prefs` like the sibling. Toggle keys + labels:

```python
acoustic_keys = [
    ("feat_acoustic_ts_envelope",  "Amplitude envelope (broadband)"),
    ("feat_acoustic_ts_onset_env", "Onset envelope (HWR derivative)"),
    ("feat_acoustic_ts_logmel",    "Log-mel spectrogram"),
    ("feat_acoustic_ts_mfcc",      "MFCC (off by default)"),
]
```

Wire the page into `gui/main_window.py`:
- Add to `SECTIONS` (`:25-26`), inserting `"Acoustic"` before `"Run"`.
- `from gui.section_acoustic_features import AcousticFeaturesSection` (`:16`).
- Instantiate `self.acoustic_section = AcousticFeaturesSection()` (after `:85`).
- `self.acoustic_section.settings_changed.connect(self.run_section.refresh)`.
- Insert `self.acoustic_section` into the `sections` list (`:111-119`) at the same position as in `SECTIONS` (before `self.run_section`).

> Sidebar order, SECTIONS list, and the `sections` list must all place "Acoustic" at the same index, or pages and labels desync.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/gui/test_section_acoustic_features.py -q`
Expected: PASS. Then launch `python gui.py` and confirm the "Acoustic" page renders with the master toggle, four feature toggles, and the param controls.

- [ ] **Step 5: Commit**

```bash
git add gui/section_acoustic_features.py gui/main_window.py tests/gui/test_section_acoustic_features.py
git commit -m "feat(acoustic): GUI page + sidebar entry"
```

---

## Task 12: Feature-doc routing (`_SHARED_DOC_MAP`)

**Files:**
- Modify: `gui/feature_docs.py`
- Test: `tests/gui/test_feature_docs.py`

- [ ] **Step 1: Write the failing test** (append)

```python
def test_acoustic_ts_toggles_route_to_docs():
    from gui.feature_docs import _doc_path
    assert _doc_path("feat_acoustic_ts_envelope").name == "acoustic_envelope.md"
    assert _doc_path("feat_acoustic_ts_onset_env").name == "acoustic_onset_envelope.md"
    assert _doc_path("feat_acoustic_ts_logmel").name == "acoustic_logmel.md"
    assert _doc_path("feat_acoustic_ts_mfcc").name == "acoustic_mfcc.md"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/gui/test_feature_docs.py -k acoustic_ts -q`
Expected: FAIL — keys fall through to `feat_acoustic_ts_envelope.md` (wrong name).

- [ ] **Step 3: Write minimal implementation**

Add to `_SHARED_DOC_MAP` in `gui/feature_docs.py`:

```python
    # Acoustic level (audio-only continuous TRF features)
    "feat_acoustic_ts_envelope":  "acoustic_envelope.md",
    "feat_acoustic_ts_onset_env": "acoustic_onset_envelope.md",
    "feat_acoustic_ts_logmel":    "acoustic_logmel.md",
    "feat_acoustic_ts_mfcc":      "acoustic_mfcc.md",
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/gui/test_feature_docs.py -q`
Expected: PASS. (The ⓘ icon stays hidden until Task 13 creates the files — that is expected and harmless.)

- [ ] **Step 5: Commit**

```bash
git add gui/feature_docs.py tests/gui/test_feature_docs.py
git commit -m "feat(acoustic): route acoustic toggles to feature docs"
```

---

## Task 13: Feature docs (fact-checked)

**Files:**
- Create: `docs/features/acoustic_envelope.md`, `docs/features/acoustic_onset_envelope.md`, `docs/features/acoustic_logmel.md`, `docs/features/acoustic_mfcc.md`

No test (docs). This task is authored by the human-in-the-loop per the feature-doc authorship preference; the implementer's job is to scaffold the four files with correct frontmatter and verified citations.

- [ ] **Step 1: Fetch and verify sources before writing**

WebFetch / WebSearch and confirm DOIs + claims for: mTRF-Toolbox [Crosse 2016] (Frontiers in Human Neuroscience), speech-envelope cortical tracking [Lalor 2009] and/or [Ding & Simon 2012], spectrotemporal/mel-band TRF [Di Liberto 2015] (Current Biology), loudness power-law / cube-root [Stevens 1957], and the MFCC-vs-mel "DCT removes tonotopic interpretability" point. Do not write a citation you could not verify; if paywalled, ask the user for the abstract (per `feedback-paywalled-papers`).

- [ ] **Step 2: Write the four files**

Each file: YAML frontmatter (`key`, `name`, `levels: [acoustic]`, `libraries: [librosa, scipy]`, `status: implemented`, `depth: deep`) + body sections: **Meaning**, **What it is / how it's computed** (the math; envelope file documents the mTRFenvelope lineage and the rms-vs-hilbert choice + power-law compression), **EEG rationale** (TRF / cortical tracking; envelope is the primary predictor), **How to set the analysis rate to match your EEG** (the configurable-Hz workflow, canonical-grid guarantee, raw-not-normalized note), **Implementation in SensiExtractor** (point at `extractors/acoustic_ts.py` functions; ⚠️-flag any interpolated *why*), **Library** snippet, **Papers** (verified citations), **Datasets** placeholder comment. The MFCC file includes the "log-mel ≻ MFCC for TRF" caution. Speech-vs-music preset trade-off goes in the log-mel file.

- [ ] **Step 3: Verify the icons render**

Run `python gui.py`, open the Acoustic page, click each ⓘ icon, confirm the modal renders the body.

- [ ] **Step 4: Commit**

```bash
git add docs/features/acoustic_envelope.md docs/features/acoustic_onset_envelope.md docs/features/acoustic_logmel.md docs/features/acoustic_mfcc.md
git commit -m "docs(acoustic): per-feature docs for the acoustic level"
```

---

## Task 14: Full-suite verification + master_plan update

**Files:**
- Modify: `docs/master_plan.md`

- [ ] **Step 1: Run the full test suite**

Run: `python -m pytest -q`
Expected: all green (prior baseline was 441 passed, 2 skipped + the new acoustic tests). Investigate any failure before proceeding — do not claim success without the passing output.

- [ ] **Step 2: Update `docs/master_plan.md`**

- Bump §6 "Current version" note to **v13**.
- Add a §3 completed-batches row: `2026-06-13 | Acoustic level (audio-only TRF features) | …` summarizing the new `acoustic/<stim>_acoustic.mat`, the four features, the mTRFenvelope-faithful default, configurable analysis rate, prefs v13, new GUI page + `--level acoustic`. Spec link.
- Add an "awaiting manual validation" line: run `python gui.py` on a real stimulus, confirm the `.mat` loads in MATLAB with aligned feature lengths.
- Note: this batch's docs were written outside the docs-system author-review backlog.

- [ ] **Step 3: Commit**

```bash
git add docs/master_plan.md
git commit -m "docs: record acoustic level batch in master_plan (v13)"
```

---

## Dependency / environment note

`librosa` and `scipy` are required by this batch. Confirm `librosa` is in the env (`python -c "import librosa"`); if missing, `pip install librosa soundfile` inside the activated `./env` and add `librosa` to `requirements.txt` in Task 1's commit. `soundfile` is used by tests to write fixture WAVs.
