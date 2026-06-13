---
key: acoustic_envelope
name: Amplitude envelope (broadband)
levels: [acoustic]
gui_toggles: [feat_acoustic_ts_envelope]
mat_variables: [envelope]
libraries: [librosa, scipy]
status: implemented
depth: deep
---

## Meaning
The **broadband amplitude envelope** is the slow-varying "loudness contour"
of the audio — how the overall energy of the signal rises and falls over
time, with the fast carrier (the individual pressure oscillations) stripped
away. For speech it tracks the syllabic rhythm: each syllable is a bump in
the envelope. It is the single most widely used acoustic predictor in
cortical-tracking / temporal-response-function (TRF) studies.

It is emitted as one column, `envelope`, in the per-stimulus
`acoustic/<stim>_acoustic.mat` sidecar, sampled on the shared analysis grid
(`t`, `fs`) so it lines up sample-for-sample with the other acoustic time
series.

## What it is / how it's computed
SensiExtractor offers two envelope methods (toggle: `env_method`). Both
return a value at every point of the canonical grid `N = round(duration ×
rate)` and then apply a power-law compression `comp` (default `0.3`).

**`rms` (default) — faithful replica of `mTRFenvelope.m`.** This mirrors the
envelope routine of the mTRF-Toolbox [Crosse 2016], the de-facto reference
implementation for speech-TRF work:

1. **Square** the waveform — `y²` is instantaneous signal power.
2. **Boxcar-average / decimate** onto the grid: the signal is split into `N`
   contiguous bins that together span the whole file, and the mean of `y²`
   is taken within each bin. Averaging the ≈`sr/rate` samples per output
   sample is simultaneously the anti-alias filter and the downsampling step.
3. **Square root** — `√(mean power)` is the RMS amplitude of each bin.
4. **Power-law compress** — raise the RMS envelope to the exponent `comp`.

   The mTRF default, which SensiExtractor matches, is `comp = log₁₀(2) ≈
   0.301` (we expose it as `0.3`). This is not arbitrary: Stevens' loudness
   law found that the perceived loudness of a tone grows as a power function
   of intensity with exponent `log₁₀(2)` [Stevens 1955], so the compression
   pulls the raw acoustic envelope toward a *perceptual* loudness contour.
   ⚠️ *Note the base of the exponent: mTRFenvelope (and SensiExtractor) raise
   the RMS amplitude to `log₁₀(2)`, whereas Stevens' exponent is stated on
   intensity; the toolbox adopts this value as a practical compressive
   default rather than a strict re-derivation of the intensity law.*

**`hilbert` — analytic-signal envelope.** Takes the magnitude of the
analytic signal `|hilbert(y)|` (the instantaneous amplitude), then
band-limits and resamples it from `sr` to `rate` with `scipy.signal.
resample_poly` (a polyphase FIR anti-alias filter), and applies the same
`comp` compression. The Hilbert envelope is smoother and is the more common
choice in some auditory-neuroscience pipelines; the `rms` method is closer
to the mTRF lineage. The two are highly correlated for speech.

Lowering `comp` toward 0 flattens the dynamic range (loud and quiet moments
move closer together); `comp = 1.0` disables compression and returns the
linear RMS / Hilbert envelope.

## EEG rationale
Auditory cortex **phase-locks to the speech envelope**: the slow (≈1–8 Hz)
amplitude modulations of speech are reconstructable from, and predictive of,
the ongoing EEG/MEG [Lalor 2010] [Ding & Simon 2012]. In a forward TRF you
regress the EEG onto the envelope to estimate the brain's impulse response
to acoustic energy; in a backward (decoding) model you reconstruct the
envelope from the EEG. Envelope tracking is sensitive to attention,
intelligibility, and hearing status, which is why it is the workhorse
predictor for naturalistic-listening studies.

The envelope is also the **base signal for the [onset
envelope](acoustic_onset_envelope.md)** — its half-wave-rectified
derivative — and a useful covariate to regress out when isolating
higher-level (lexical, semantic) responses.

## How to set the analysis rate to match your EEG
A TRF predictor must be sampled at **the same rate as the EEG you will
regress it against.** Set `analysis_rate` (Hz) to your post-downsampling EEG
rate — commonly **64 or 128 Hz** for envelope-TRF work. Every acoustic
feature in the `.mat` is computed on the single grid `N = round(duration ×
rate)`, so:

- `envelope`, `onset_envelope`, `logmel`, `mfcc` all have exactly `N` rows
  and share the `t` time vector and scalar `fs` — no per-feature resampling
  or alignment is needed on your side.
- To align with the EEG, resample your EEG to `analysis_rate` (or pick
  `analysis_rate` to equal it). Length will be `round(duration × rate)`;
  trim the EEG and predictor to the common length.

The envelope is returned **raw (not z-scored / not normalized)** so you keep
full control: most TRF pipelines z-score each predictor and each EEG channel
before fitting. The `config` struct in the `.mat` records `analysis_rate`,
`env_method`, `env_comp`, and `source_sr` for provenance.

## Implementation in SensiExtractor
Source: `extractors/acoustic_ts.py` → `amplitude_envelope(y, sr, rate, *,
method, comp)`. The `rms` branch places the boxcar average directly on the
canonical grid via `np.linspace(0, y.size, n+1)` bin edges, which avoids the
cumulative drift you would get from a fixed `round(sr/rate)` decimation when
`sr/rate` is non-integer. The `hilbert` branch uses `scipy.signal.hilbert`
+ `resample_poly` with `up/down` reduced by their GCD. Both return
`_fit_to_grid(env, n)` so the output is exactly `N` samples. The audio is
loaded mono at its native sample rate (`librosa.load(..., sr=None)`); the
envelope's frequency content above `rate/2` is removed by the averaging /
FIR step before decimation.

## Library
```python
import librosa
from extractors.acoustic_ts import amplitude_envelope

y, sr = librosa.load("stim.wav", sr=None, mono=True)
env = amplitude_envelope(y, sr, rate=128, method="rms", comp=0.3)
# env.shape == (round(len(y)/sr * 128),)
```

## Papers
- <a name="crosse-2016"></a>**[Crosse 2016]** Crosse, Di Liberto, Bednar, Lalor. *The Multivariate Temporal Response Function (mTRF) Toolbox: A MATLAB Toolbox for Relating Neural Signals to Continuous Stimuli.* Frontiers in Human Neuroscience 10, 604. [doi:10.3389/fnhum.2016.00604](https://doi.org/10.3389/fnhum.2016.00604) — reference TRF toolbox; `mTRFenvelope.m` is the algorithm replicated by the `rms` method.
- <a name="stevens-1955"></a>**[Stevens 1955]** Stevens. *The Measurement of Loudness.* Journal of the Acoustical Society of America 27(5), 815–829. [doi:10.1121/1.1908048](https://doi.org/10.1121/1.1908048) — loudness grows as a power function of intensity with exponent `log₁₀(2)`; the source of the default compression exponent.
- <a name="lalor-2010"></a>**[Lalor 2010]** Lalor & Foxe. *Neural responses to uninterrupted natural speech can be extracted with precise temporal resolution.* European Journal of Neuroscience 31(1), 189–193. [doi:10.1111/j.1460-9568.2009.07055.x](https://doi.org/10.1111/j.1460-9568.2009.07055.x) — EEG TRF to the speech envelope.
- <a name="ding-2012"></a>**[Ding 2012]** Ding & Simon. *Neural coding of continuous speech in auditory cortex during monaural and dichotic listening.* Journal of Neurophysiology 107(1), 78–89. [doi:10.1152/jn.00297.2011](https://doi.org/10.1152/jn.00297.2011) — phase-locked cortical representation of the slow speech envelope.

[Crosse 2016]: #crosse-2016
[Stevens 1955]: #stevens-1955
[Lalor 2010]: #lalor-2010
[Ding & Simon 2012]: #ding-2012

## Datasets
<!-- auto-populated from docs/datasets/*.md once the dataset registry exists -->
