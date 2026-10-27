# Run-tab estimates (disk, peak memory, time) — design

Date: 2026-10-26 · Owner: Amilcar Malave · Status: draft for owner approval

## Intent

Before clicking Run, the owner's manager wants to know what a run will cost:
how much **disk space** the outputs will take, whether the computer has enough
**memory (RAM)** for what's ticked, and roughly **how long** it will take. Today the
Run tab says what files a run writes but nothing about their size, memory or
time. Researchers on laptops can start a run that fills the disk, runs out of
memory halfway (large language models, Word2Vec), or takes far longer than
expected.

## Goals

1. Each **level row** on the Run tab shows an estimated disk size next to its count
   tag (e.g. "Word level · 3 files per stimulus · ≈ 410 MB"), and each listed file
   shows its own estimated size.
2. Under **Readiness**, a summary box with:
   - **Disk space for outputs** (total across levels);
   - **Peak memory (RAM)**: the estimated peak of the most memory-hungry ticked
     item, next to this computer's installed RAM;
   - **Expected time**: a range for the whole run.
3. An orange **warning** only when the peak memory probably won't fit this computer.
4. The time estimate **learns from this computer's past runs**: a wide first
   estimate, labelled as such, that tightens once the machine has measured speeds.

## Non-goals

- Exact numbers. These are estimates with honest ranges, not promises.
- Per-level time (MFA and models are shared across levels; time is a total).
- Blocking Run because of an estimate. Estimates are informational; only the
  existing readiness rules block Run.
- GPU-memory planning beyond what's needed for the warning (see Constraints).

## Anti-goals

- An estimate that is confidently wrong by an order of magnitude with no hint that
  it is rough (e.g. a fixed "2 min" on a slow laptop).
- Noticeably slowing the Run tab (it refreshes often): estimating must not read
  whole audio files or re-parse transcripts on every refresh.
- Writing estimate data into the user's shareable `features.yaml`.

## Behaviour (decided with the owner)

- **Disk** per file and per level, computed from the actual inputs: transcript word
  counts and audio durations, times the known size of each output (e.g. BERT word
  embeddings store 12 layers × 768 float32 values per word ≈ 36.9 KB/word; CSVs by
  columns × rows). Shown as "≈ <size>" with sensible units (KB/MB/GB).
- **Peak RAM**: from a small table of known peaks for the memory-heavy items
  (Word2Vec, BERT, each predictability model in the roster), the largest ticked
  one plus a baseline for the always-on pipeline; compared with
  `utils/system_info.gather_system_info()["system_ram_gb"]`. Warning when the
  estimate exceeds the machine's RAM (GPU memory can substitute for a model's
  weights when a CUDA GPU with enough VRAM is present — the warning must not fire
  falsely in that case).
- **Time**: per-step rates (e.g. seconds per minute of audio for MFA, seconds per
  1000 words for a model). Start from built-in rough rates with a wide range and the
  note "First estimate. It gets more precise as the app learns from runs on this
  computer."; after runs on this machine, use its measured rates and a narrower
  range.
- Before the input folders are chosen, sizes and time show "—".

## Look (owner-approved mockup)

`docs/superpowers/specs/assets/2026-10-26-run-estimates-mockup.html` (example numbers): size in bold after each level's tag; per-file sizes in the accent
colour after each file path; a bordered summary box under the "Readiness" heading
with three label/value rows and the learning note; the orange warning below it only
when needed; the existing ready/not-ready line stays.

## Constraints

- **Golden rule (CLAUDE.md):** the Run tab displays; which artifacts and stages a run
  includes comes from the planner (`plan.output_csvs`, `plan.stages_to_run`,
  `plan.predictability_*`). The estimator consumes the plan; it decides nothing
  about execution.
- **Machine-local learning data** lives outside `features.yaml` (which is shareable):
  in the config directory next to `run.yaml`, git-ignored (e.g. `config/timing_history.json`),
  respecting `SENSI_CONFIG_DIR` like the rest of `utils/preferences.py`. Losing or
  deleting it just resets to first estimates.
- **Recording timings** needs per-step elapsed time and input size per stimulus
  (today the manifest records only a per-stimulus total,
  `utils/run_logger.py:record_pair_success`). Instrument the batch runners/`run()` in
  `main.py` minimally (wall time around each stage that matters for time: MFA batch,
  spaCy, predictability per model, Word2Vec, BERT, prosody, syllabification,
  acoustic). This must not change any output.
- **RAM table values** must come from a checkable source (model card or measured on
  this repo's validation run), noted next to each value in code.
- **Speed:** scanning inputs for word counts/durations is cached by (path, size,
  mtime); audio duration from file headers (`soundfile.info`), not decoding.
- Applies to GUI only; `run.py` may print the same summary before starting (nice to
  have, builder's call).
- Verify in the real GUI on the validation corpus, and compare each disk estimate
  against the files a real run actually writes (should be within ~25 %).


## Calls made during the builder check (not owner decisions; owner to confirm while reading)

1. **Per-file sizes are totals across all stimuli** for that file pattern (the mockup's
   "word/<stim>_bert_word.mat ≈ 400 MB" is the sum over the 10 stimuli); the level size is
   the sum of its files. The ~25 % accuracy check is on level totals.
2. **Counting inputs cheaply:** word count = whitespace tokens of the prepared transcript
   (cached by path/size/mtime); rows for other files derived from it with fixed ratios
   (syllables/word, phonemes/word, sentences from sentence-final punctuation) calibrated once
   on the validation corpus and stored as constants with a comment saying where they came from.
3. **GPU:** reuse `utils/system_info._detect_gpu()` once per GUI session (cached; it imports
   torch lazily). A model counts against VRAM instead of RAM exactly when the extractor would
   put it on the GPU (`extractors/predictability.py` loads on CUDA when available via
   `device_map='auto'`; mirror that rule rather than duplicating a different one).
4. **RAM model:** CORRECTED 2026-10-26 (task review): Word2Vec, BERT and every predictability
   model stay loaded for the rest of the process (module-level caches, never evicted), so
   peak = baseline (pipeline without heavy models, measured on the validation run) + the SUM
   of the heavy items that will run (MFA runs earlier as a subprocess and is not added); no input-length term unless the validation run shows
   one matters (e.g. acoustic time series), in which case add it for that item only.
5. **Timing history:** one rate per (step, model slug if any, device cpu/cuda); recorded by the
   batch runners in `main.py`, so GUI, `run.py` and CLI batch runs all contribute; only
   successful stimuli are recorded; one-time model load/download time is recorded separately
   from per-stimulus compute so a first run doesn't inflate rates.
6. **Two passes:** the time estimate follows the passes the plan will run — the word pass and,
   when Document level is on, the doc pass — with MFA counted once (cached) and predictability
   scores counted once (reused across passes by the score cache); Word2Vec/BERT only in the
   word pass. Verify these reuse claims against the code before relying on them.
7. **Mockup** copied into the repo at the path above; the prose stays authoritative where they differ.
8. **MFA time** (CORRECTED 2026-10-26, final review): MFA's wall time is mostly fixed start-up, so it
   is modelled as one overhead per aligning run + seconds per audio second, not a flat per-second
   rate. It is learned only from ≥ 3 batches of clearly different sizes whose fit needs no clamping,
   and stays at the wide first-estimate spread when the corpus is more than 2× the largest batch seen.

## Left to the builder

- Module layout (suggested: `utils/run_estimates.py` with pure functions taking the
  plan + input stats, plus a small history store), the exact rate model and range
  width, units/rounding, and caching.
- Exact wording of rows beyond the mockup's.
- How many past runs are needed before switching from built-in to measured rates.
- Tests: pure-function tests for disk/RAM/time maths; GUI test-after.
