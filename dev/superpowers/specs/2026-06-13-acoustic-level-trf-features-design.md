# Acoustic level — audio-only continuous TRF features

**Date:** 2026-06-13
**Status:** spec finalized (no code yet) — ready to implement on go-ahead

## Why
The pipeline has word / sub-word / sentence / document levels, all derived
from a transcript+audio **pair**. Missing is a level that describes the
**audio signal itself** as continuous time series — the predictors TRF /
cortical-tracking analyses regress against EEG (broadband envelope, onset
envelope, log-mel spectrogram). These are a pure function of the audio file;
no transcript is needed and no EEG is ever read here.

Two non-negotiables drive the design:
1. Every feature is output as a slow time series at a **configurable analysis
   sample rate** the user passes in to match their EEG.
2. Features are a pure function of the audio — the extractor never touches EEG.

## Decisions (from brainstorm)
- **Input discovery:** standalone path that globs the **audio folder
  independently** — runs on every audio file regardless of whether a
  transcript exists. (Honors "no transcript needed".)
- **Output packaging:** **one combined `.mat` per stimulus** with named
  variables + `fs` + time vector `t` (+ `mel_center_freqs`, provenance
  `config` struct).
- **Time alignment:** **one canonical shared grid** — all enabled features
  resampled/trimmed to exactly `N = round(duration × rate)` rows on a shared
  `t`, so they stack directly as TRF predictors with no off-by-one frame
  mismatch (and align with resampled EEG).
- **Default analysis rate:** **128 Hz** (common mTRF rate; user overrides).
- **Envelope default = faithful `mTRFenvelope` replica** (`rms`); `hilbert`
  is the optional alternative.
- **Mel range:** two presets — `speech` (fmin 50, fmax 8000) and
  `music`/`general` (fmin 0, fmax Nyquist).
- **Master defaults OFF** (opt-in, like the embedding `.mat` features) so
  existing runs are unaffected. Within the level: envelope / onset / logmel
  pre-enabled, mfcc off.

### mTRFenvelope verification (source-checked, not from memory)
`mTRFenvelope.m` (mTRF-Toolbox, master branch) signature
`mTRFenvelope(x,fsin,fsout,window,comp,buff,...)`, defaults `window=1`,
`comp=log10(2)≈0.301`. Body:
```matlab
x = x.^2;                                     % square (power)
[y,...] = mTRFresample(x,fsin,fsout,window);  % avg neighbours every fsin/fsout, downsample
y = sqrt(y).^comp;                            % sqrt (→RMS) then power-law compress
```
So the `rms` recipe **square → moving-average → sqrt → ^comp** is the faithful
replica, and the proposed default `comp=0.3` ≈ mTRF's `log10(2)`. Confirmed:
the brainstormed steps were correct; nothing contradicts.

## Naming
The level is **`acoustic`** (new output folder `acoustic/`, no collision with
existing level dirs). Prefs keys use an **`acoustic_ts`** prefix to avoid
collision with the existing *word-level* acoustic-alignment keys
(`feat_acoustic_onset`/`feat_acoustic_end`, `master_acoustic` = Onset/End).

## Extractor — `extractors/acoustic_ts.py` (new, pure functions)
Audio loaded once: `librosa.load(path, sr=None, mono=True)` (native rate,
downmixed). Canonical grid `N = round(duration × rate)`, `t = arange(N)/rate`.
A shared `_fit_to_grid(arr, N)` edge-pads/trims every feature to exactly `N`
rows.

- `amplitude_envelope(y, sr, rate, *, method="rms", comp=0.3)` → `(N,)`
  - `rms` (default, mTRFenvelope replica): `x²` → non-overlapping frames of
    `L=round(sr/rate)` → mean per frame (boxcar = anti-alias) → `sqrt` →
    `**comp`. `window=1` fixed (not exposed).
  - `hilbert` (optional): `|scipy.signal.hilbert(y)|` →
    `scipy.signal.resample_poly` to `rate` (built-in FIR anti-alias) →
    `**comp`.
- `onset_envelope(env)` → `(N,)`: half-wave-rectified first difference,
  `clip(diff(env), 0, None)`, length kept at `N` (prepend 0).
- `log_mel_spectrogram(y, sr, rate, *, n_mels=16, fmin, fmax)` → `(N, n_mels)`
  + center freqs: librosa melspectrogram, `hop_length=round(sr/rate)`,
  `win_length≈25 ms`, `n_fft=next_pow2(win_length)`, then `power_to_db`.
  `n_mels` default 16 (feeds a TRF, not an ML model).
- `mfcc(y, sr, rate, *, n_mfcc=13, n_mels, fmin, fmax)` → `(N, n_mfcc)`: from
  the same mel params; `n_mfcc` default 13. Off by default. Docstring notes
  log-mel is usually preferred for cortical-tracking TRF because the DCT
  removes tonotopic interpretability (⚠️ verify/cite at doc time) — provided
  for completeness, not as the primary feature.
- `extract_acoustic(path, rate, cfg)` → dict of enabled arrays + `fs`, `t`,
  `mel_center_freqs`, `config` provenance struct.

Features are emitted **raw** (not z-scored); TRF users standardize predictors
at fit time. Documented.

`win_length`/`n_fft` are internal defaults (~25 ms / next pow2), not exposed,
to avoid knob-creep.

## Output — one combined `.mat` per stimulus
`acoustic/<stim>_acoustic.mat` via `scipy.io.savemat`, written through a lazy
`_level_dir(output_dir, 'acoustic')` (same helper the embeddings batch uses).
Variables (only enabled): `envelope` `(N,)`, `onset_envelope` `(N,)`,
`logmel` `(N×n_mels)`, `mfcc` `(N×n_mfcc)`, plus `fs` (the analysis rate),
`t` `(N,)`, `mel_center_freqs` `(n_mels,)`, and a `config` struct (method,
comp, n_mels, n_mfcc, fmin, fmax, source_sr, duration).

## Orchestration — `run_acoustic_level_batch` in `main.py` (new, parallel path)
Globs the audio folder independently for `audio_exts` (ignores transcripts),
runs `extract_acoustic` per file, writes the `.mat`. Own `RunLogger`
(`filename_prefix='acoustic_run'`). Does **not** go through the word/doc
DataFrame pipeline — the planner DAG buys nothing here since the only
dependency (envelope→onset) is internal to the extractor.

## GUI / planner / prefs wiring
- **Prefs v12 → v13** (additive; migration chain extended — note the active
  schema is already v12 from the lexical-norms batch):
  - `features`: `feat_acoustic_ts_envelope`=True, `feat_acoustic_ts_onset_env`=True,
    `feat_acoustic_ts_logmel`=True, `feat_acoustic_ts_mfcc`=False,
    `master_acoustic_ts`=False.
  - `output`: `acoustic_ts_mat`=True.
  - nested `acoustic_ts` sub-dict (config knobs, **not** drift-guarded — the
    schema-drift test only scans flat `features`/`output`): `analysis_rate`=128,
    `env_method`='rms', `env_comp`=0.3, `n_mels`=16, `n_mfcc`=13,
    `mel_preset`='speech'. Presets: `speech`→(50, 8000), `music`/`general`→(0, Nyquist).
- **Registry** (`gui/planner/registry.py`): one
  `Stage(name="acoustic_ts", consumes=("audio",), produces=("acoustic_mat",))`
  + FeatureCluster(s) with `output_csv="acoustic"` and
  `master_toggle="master_acoustic_ts"`. `acoustic_ts_mat` added to
  `_OUTPUT_CSV_MAP` in `plan.py`; `master_acoustic_ts` added to the
  intentional-unrouted master set in `tests/planner/test_schema_drift.py`.
- **Dispatch** (`gui/run_dispatch.py`): add
  `if "acoustic" in plan.output_csvs: run_acoustic_level_batch(...)`, sharing
  the dispatch `run_ts`.
- **New page** `gui/section_acoustic_features.py` + 8th sidebar item
  "Acoustic": master toggle, 4 feature ToggleRows (each gets a
  `docs/features/*.md` → auto ⓘ icon), analysis-rate numeric field, method
  combo (rms/hilbert), compression-exponent field, n_mels / n_mfcc fields,
  mel-preset combo. Reuses the existing numeric/combo widget pattern from the
  prosody params.
- **CLI** (`main.py` argparse): `--level acoustic`; group-level flags
  `--analysis-rate`, `--env-method`, `--env-comp`, `--n-mels`, `--n-mfcc`,
  `--mel-preset`, `--no-envelope`, `--no-onset-env`, `--no-logmel`, and
  `--mfcc` (enable; off by default — the one non-`--no-*` flag, justified by
  the off-by-default state).

## Docs — `docs/features/` (fact-checked)
Four files routed via `gui/feature_docs.py::_SHARED_DOC_MAP`:
`acoustic_envelope.md`, `acoustic_onset_envelope.md`, `acoustic_logmel.md`,
`acoustic_mfcc.md`. Written for a new researcher: what each feature is, the
math, **how to set the analysis rate to match their EEG**, the mTRF lineage,
rms-vs-hilbert and speech-vs-music trade-offs, why log-mel ≻ MFCC for TRF.
Every DOI and empirical claim fetched/verified (mTRF-Toolbox [Crosse 2016],
envelope tracking [Lalor 2009 / Ding & Simon], spectrotemporal/mel TRF
[Di Liberto 2015], loudness power-law [Stevens], the MFCC tonotopic point);
interpolated *why* flagged with ⚠️, no fabricated refs. (Authoring rules:
`feedback-feature-docs-authorship`, `feedback-feature-docs-citations`.)

## Testing
TDD for the extractor math (`tests/extractors/test_acoustic_ts.py`), synthetic
signals only:
- constant-amplitude tone → envelope ≈ constant (post-compression);
- **all enabled features have exactly `N = round(dur×rate)` rows** (shared-grid
  guarantee) — the key correctness invariant;
- onset env ≥ 0 and ≈ 0 on flat sections, positive at amplitude increases;
- logmel shape `(N, n_mels)`, mfcc shape `(N, n_mfcc)`;
- `rms` ≈ `hilbert` gross envelope shape on a tone;
- compression exponent monotonic effect;
- `mel_preset` changes fmin/fmax (and `mel_center_freqs`).

Integration test: audio-only run (no transcript present) writes
`acoustic/<stim>_acoustic.mat` with the expected variables. Prefs v13
migration test. GUI/CLI test-after (per TDD-scope preference).

## Out of scope (YAGNI)
- Per-feature normalization toggle (raw output; users standardize downstream).
- Exposing `win_length`/`n_fft`/`window` as knobs.
- Routing the acoustic level through the planner DAG (no cross-feature deps).
- Any EEG I/O, alignment-to-EEG, or stereo-channel handling beyond mono downmix.
