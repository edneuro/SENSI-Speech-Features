# Run-tab Estimates (disk, peak memory, time) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show estimated disk size per level and per file, peak memory against this
computer's RAM (with a warning when it won't fit), and an expected run time that learns from
this computer's past runs, on the Run tab.

**Architecture:** Pure estimator functions in `utils/run_estimates.py` take the planner's
`Plan`, the prefs, cheap input statistics and a machine-local timing history, and return
numbers; the Run tab only displays them. A small `utils/timing_history.py` records per-step
speeds from real batch runs (instrumented in `main.py`), stored in the config dir. Constants
(bytes per CSV cell, rows per word, RAM peaks, first-estimate speeds) are measured once on the
validation corpus and kept in `utils/estimate_constants.py` with their provenance.

**Tech Stack:** Python 3.10, pandas, numpy, soundfile, psutil, PySide6, pytest/pytest-qt.

**Spec:** `docs/superpowers/specs/2026-10-26-run-estimates-design.md` (owner-approved
2026-10-26, including its "Calls made during the builder check" section). Mockup:
`docs/superpowers/specs/assets/2026-10-26-run-estimates-mockup.html`.

**Branch:** `feat/run-estimates` from `main` (main == origin/main at `b2e5c99`).

## Global Constraints

- Golden rule: the Run tab displays; the estimator consumes `plan.output_csvs`,
  `plan.stages_to_run`, `plan.predictability_word/_doc` and never decides what runs.
- Estimates never block Run; only existing readiness rules do.
- Copy: per-level "≈ <size>" after the count tag; per-file "≈ <size>" after each path;
  summary rows "Disk space for outputs", "Peak memory (RAM)" (value + "this computer: <N> GB"),
  "Expected time"; note while unlearned: "First estimate. It gets more precise as the app
  learns from runs on this computer."; "—" for every value until the input folders are set.
- Warning copy: "⚠ <item> needs ≈ <X> GB of memory; this computer has <Y> GB. The run will
  likely fail or be extremely slow." — only when the estimate exceeds what's available
  (RAM, or VRAM for an item that would run on the GPU).
- Sizes: totals across all stimuli; units B/KB/MB/GB (1024-based), 2 significant digits.
- Timing history file: `<config dir>/timing_history.json` (`utils.preferences.get_config_dir()`;
  honours `SENSI_CONFIG_DIR`), git-ignored; never written into `features.yaml`/`run.yaml`;
  a missing/corrupt file means "no history" (first estimates), never an error.
- Instrumentation must not change any output file (verified by a before/after real run).
- Run-tab refresh must stay fast: input stats cached by (path, size, mtime); audio duration
  from headers (`soundfile.info`), never decoding; GPU detection once per session.
- No `Co-Authored-By`/Claude attribution in commit messages (owner rule).
- Python: `./env/python.exe` with `env`, `env/Scripts`, `env/Library/bin` on `PATH`;
  suite `pytest -q -p no:cacheprovider`; TDD for the estimator maths, test-after for GUI.

## Review Focus

1. **Transcripts with no matching audio** (text-only items): disk estimate must skip
   audio-derived files for them and the time model must not charge MFA/prosody (Task 3/5 tests).
2. **Huge inputs** (a 60-minute stimulus, 10k+ words): BERT word `.mat` dominates and must
   scale linearly — the estimate should show GB, not overflow formatting (Task 3 test).
3. **History from another device**: a rate learned with CUDA must not be used for a CPU-only
   machine/run, and vice versa (Task 5 test).
4. **Corrupt or hand-edited `timing_history.json`**: falls back to first estimates silently
   (Task 1 test).
5. **Machine with a big GPU**: Llama-3-8B on a 24 GB GPU + 16 GB RAM must not warn; the same
   model on CPU-only 16 GB must warn (Task 4 test).

---

### Task 1: Timing history store + batch-runner instrumentation

**Files:**
- Create: `utils/timing_history.py`
- Modify: `main.py` (`run()` stage blocks; `_prealign_batch`; `run_word_level_batch`,
  `run_doc_level_batch`, `run_acoustic_level_batch` per-pair success paths)
- Modify: `.gitignore` (add `config/timing_history.json`)
- Test: `tests/utils/test_timing_history.py`, `tests/test_stage_timing.py`

**Interfaces:**
- Produces:
  - `STEP_NAMES: tuple[str, ...]` = `("mfa_align", "spacy_parse", "predictability",
    "word2vec", "bert", "prosody_contours", "syllabify", "phoneme_surprisal",
    "sentence_prosody", "acoustic_ts", "doc_summary")`.
  - `@dataclass Sample(step: str, model: str | None, device: str, units: float, seconds: float)`
    — `units` = audio seconds for audio steps (mfa_align, prosody_contours, acoustic_ts,
    phoneme_surprisal), words for text steps; `device` in `{"cpu", "cuda"}`.
  - `class StageTimer`: context manager `timer.stage(step, units, model=None, device="cpu")`
    appends a `Sample`; `.samples: list[Sample]`; `.load_seconds: dict[str, float]` for one-time
    model loads (keyed `"<step>:<model>"`), recorded separately from per-stimulus compute.
  - `load_history(path: Path | None = None) -> dict` and
    `append_samples(samples: list[Sample], path: Path | None = None) -> None` (default path
    `get_config_dir() / "timing_history.json"`; keeps the most recent 200 samples per
    (step, model, device); atomic write via `utils.config_yaml._write_text`).
  - `rate(history: dict, step: str, model: str | None, device: str) -> float | None` —
    median seconds per unit over that key's samples, `None` with fewer than 3 samples.
- Main.py: `args._stage_timer: StageTimer` created per batch; each instrumented block in
  `run()` wraps its work in `args._stage_timer.stage(...)` when present (no-op otherwise);
  after each **successful** pair the runner appends that pair's samples; failed pairs append
  nothing.

- [ ] **Step 1: Write failing tests** (`tests/utils/test_timing_history.py`):
  `test_rate_needs_three_samples`, `test_rate_is_median_per_key` (cpu and cuda keys kept apart),
  `test_corrupt_history_file_reads_as_empty` (garbage bytes → `{}` and `rate(...) is None`),
  `test_history_caps_200_per_key`, `test_history_respects_config_dir` (writes under
  `SENSI_CONFIG_DIR`).
- [ ] **Step 2:** Run them — FAIL (module missing).
- [ ] **Step 3: Implement `utils/timing_history.py`.**
- [ ] **Step 4: Instrument `main.py`.** Device for predictability = `"cuda"` when
  `torch.cuda.is_available()` and the extractor's own rule puts the model on GPU (reuse the
  rule in `extractors/predictability.py`, don't restate it); BERT likewise; all others `"cpu"`.
  Model-load time: measure the first call per model in the process and put it in
  `load_seconds`, subtracting it from that stimulus's sample.
- [ ] **Step 5: Test** (`tests/test_stage_timing.py`):
  `test_successful_pair_records_samples` — run `run_word_level_batch` on a tiny text-only
  fixture (one transcript, `no_align=True`) with tmp `SENSI_CONFIG_DIR`; history file gains a
  `spacy_parse` sample with `units == <word count>`; `test_failed_pair_records_nothing`
  (monkeypatch `main.run` to raise).
- [ ] **Step 6:** Full suite passes.
- [ ] **Step 7: Commit** — `feat: record per-step timings from batch runs (machine-local history)`

### Task 2: Calibration run → `utils/estimate_constants.py`

**Files:**
- Create: `scripts/calibrate_estimates.py`, `utils/estimate_constants.py`
- Test: `tests/utils/test_estimate_constants.py`

**Interfaces:**
- Consumes: Task 1 instrumentation.
- Produces (`utils/estimate_constants.py`, each value with a comment naming how it was
  measured — "validation corpus, <date>, <machine>" or a model-card URL):
  - `ROWS_PER_WORD: dict[str, float]` for artifacts `"syllables"`, `"phonemes"`, and
    `SENTENCES_PER_WORD: float`;
  - `CSV_BYTES_PER_CELL: dict[str, float]` per CSV artifact (`"features"`, `"syllables"`,
    `"phonemes"`, `"sentences"`, `"documents_summary"`);
  - `RAM_GB: dict[str, float]` — `"baseline"` (pipeline without heavy models), `"word2vec"`,
    `"bert"`, and one entry per predictability slug in `predictability_models.all_specs()`
    for CPU (fp32/fp16 as the extractor loads it) and `"<slug>@cuda"` for the GPU load tier;
  - `FIRST_ESTIMATE_RATES: dict[str, float]` seconds per unit per step (cpu), plus
    `FIRST_ESTIMATE_SPREAD = (0.33, 3.0)` (low/high multipliers before history exists) and
    `LEARNED_SPREAD = (0.8, 1.25)`.
- `scripts/calibrate_estimates.py`: runs `run.py` on `validation/data` with a scratch config
  that turns on every CPU-feasible feature (copy of `validation/config` if present, otherwise
  build one), then prints measured ratios, bytes per cell (file size / (rows × columns)), stage
  rates (from the timing history it produced) and peak memory per heavy item (measure each
  heavy item in a fresh subprocess: `psutil.Process().memory_info()` — `peak_wset` on Windows,
  `ru_maxrss` on POSIX — around loading + one stimulus). Never touches the live `config/`.

- [ ] **Step 1:** Write and run the calibration script; paste its output into the task report.
- [ ] **Step 2:** Write `utils/estimate_constants.py` from the measured values. Llama RAM/VRAM
  values the machine can't measure: take parameter count × bytes per parameter for the load
  tier the extractor uses (fp16 = 2 B, 4-bit ≈ 0.5 B, fp32 = 4 B) plus 20 % overhead, with the
  model-card URL for the parameter count.
- [ ] **Step 3: Tests** — `test_every_roster_model_has_ram_entries` (cpu and @cuda for every
  slug in `all_specs()`), `test_every_step_has_a_first_rate` (every `STEP_NAMES` entry),
  `test_constants_are_positive`.
- [ ] **Step 4: Commit** — `feat: measured constants for run estimates (validation corpus)`

### Task 3: Input statistics + disk estimate

**Files:**
- Create: `utils/run_estimates.py`
- Test: `tests/utils/test_run_estimates_disk.py`

**Interfaces:**
- Consumes: `LEVEL_ARTIFACTS` (`gui/planner/plan.py`), `Plan`, Task 2 constants,
  `utils.file_pairing.find_pairs`.
- Produces:
  - `@dataclass InputStats(n_stimuli: int, n_with_audio: int, words: int,
    words_with_audio: int, audio_seconds: float)`.
  - `scan_inputs(transcripts_folder: str, audio_folder: str,
    text_exts=("txt","tsv","md"), audio_exts=("wav","flac","mp3")) -> InputStats | None` —
    `None` when the transcripts folder is unset/missing (and for an acoustic-only plan, use
    the audio folder alone); words = whitespace tokens of `prepare_for_aligner(text)`;
    cached by (path, size, mtime) in a module-level dict.
  - `estimate_disk(plan: Plan, prefs: dict, stats: InputStats) -> dict[str, int]` — bytes
    per live artifact in `plan.output_csvs` (absent artifacts not in the dict).
  - `level_bytes(disk: dict[str, int], level_key: str) -> int` (sum over
    `LEVEL_ARTIFACTS[level_key]`).
  - `format_bytes(n: int) -> str` — `"≈ 410 MB"` style, 2 significant digits, 1024-based.
- Formulas (decided here):
  - CSV: rows × active columns × `CSV_BYTES_PER_CELL[artifact]`; rows = words for
    `features` (all stimuli), `words_with_audio × ROWS_PER_WORD[...]` for syllables/phonemes,
    `words × SENTENCES_PER_WORD` for sentences, `n_stimuli` for `documents_summary`; active
    columns = the `emits_columns` of the active clusters for that artifact
    (`gui.planner.plan._active_clusters(prefs, _active_output_csvs(prefs))`) minus
    `plan.column_drops[artifact]`, plus `plan.predictability_columns` for `features`.
  - `.mat`: `word2vec` = words × 300 × 4; `bert_word` = words × 12 × 768 × 4;
    `bert_sentence` = sentences × 12 × 768 × 4; `acoustic` = audio_seconds ×
    `acoustic_ts.analysis_rate` × (1 per enabled envelope/onset-env + `n_mels` if log-mel +
    `n_mfcc` if MFCC) × 8; `phoneme_surprisal_mat` = audio_seconds ×
    `phoneme_surprisal.analysis_sr` × channels × 8 (channels = the number of impulse trains
    the extractor writes — read it from `extractors/phoneme_surprisal.py`). Add a fixed
    `.mat` header allowance of 1 KB per file.

- [ ] **Step 1: Failing tests:** `test_bert_word_bytes_scale_with_words` (10 000 words →
  `bert_word == 10_000*12*768*4 + 1024*n_stimuli`), `test_huge_input_formats_in_gb`
  (`format_bytes(3_700_000_000) == "≈ 3.4 GB"`), `test_text_only_items_add_no_audio_files`
  (stats with `n_with_audio=0` → no `acoustic`, `phoneme_surprisal_mat`, syllables or phonemes
  bytes even if those artifacts are live), `test_level_bytes_sums_its_artifacts`,
  `test_dropped_columns_shrink_features_csv`, `test_scan_inputs_none_without_folder`,
  `test_scan_inputs_cached` (second call doesn't reopen files: monkeypatch `open`).
- [ ] **Step 2:** Run — FAIL. **Step 3:** Implement. **Step 4:** Run — PASS.
- [ ] **Step 5: Commit** — `feat: disk-size estimates per output artifact`

### Task 4: Peak-memory estimate

**Files:**
- Modify: `utils/run_estimates.py`
- Test: `tests/utils/test_run_estimates_ram.py`

**Interfaces:**
- Produces:
  - `@dataclass RamEstimate(peak_gb: float, driver: str, machine_gb: float, gpu_vram_gb: float,
    fits: bool, warning: str | None)` — `driver` = display name of the heaviest item
    ("Llama-3-70B", "Word2Vec", "BERT", or "the pipeline").
  - `estimate_ram(plan: Plan, prefs: dict, system: dict) -> RamEstimate` — `system` is
    `utils.system_info.gather_system_info()` output (callers cache it per session).
  - Rule (corrected: heavy models stay cached for the whole process, so they add up):
    peak = `RAM_GB["baseline"]` + the SUM of the heavy items that will run (Word2Vec if
    `"word2vec"` live; BERT if `"bert_word"` or `"bert_sentence"` live; each model in
    `prefs["predictability"]["models"]` if `plan.predictability_word or plan.predictability_doc`).
    A model that would load on the GPU (same rule as Task 1) counts its `@cuda` value against
    `gpu_vram_gb` instead of RAM; `fits` is False when RAM or the relevant VRAM is exceeded;
    `warning` uses the Global Constraints copy.

- [ ] **Step 1: Failing tests:** `test_baseline_only_without_heavy_items`,
  `test_heavy_items_add_up` (Word2Vec + BERT + gpt2 → baseline + all three), `test_llama8b_on_big_gpu_does_not_warn`
  (system ram 16, vram 24, models ["llama3_8b"] → fits, warning None),
  `test_llama8b_cpu_only_16gb_warns`, `test_predictability_off_ignores_selected_models`
  (master off and no doc means → models don't count).
- [ ] **Step 2–4:** FAIL → implement → PASS.
- [ ] **Step 5: Commit** — `feat: peak-memory estimate with fit warning`

### Task 5: Time estimate

**Files:**
- Modify: `utils/run_estimates.py`
- Test: `tests/utils/test_run_estimates_time.py`

**Interfaces:**
- Consumes: Task 1 `rate()`/`load_history()`, Task 2 rates/spreads, Task 3 `InputStats`.
- Produces:
  - `@dataclass TimeEstimate(low_s: float, high_s: float, learned: bool)`.
  - `estimate_time(plan: Plan, prefs: dict, stats: InputStats, history: dict,
    cuda_available: bool) -> TimeEstimate` — per step, device = "cuda" exactly where Task 1
    records "cuda" (predictability/BERT on a CUDA machine), else "cpu".
  - `STAGE_TO_STEP: dict[str, str]` = {"mfa_align": "mfa_align", "spacy_parse": "spacy_parse",
    "predictability": "predictability", "word2vec_embed": "word2vec", "bert_embed": "bert",
    "praat_contours": "prosody_contours", "syllabify_pairs": "syllabify",
    "phoneme_surprisal": "phoneme_surprisal", "sentence_prosody": "sentence_prosody",
    "acoustic_ts": "acoustic_ts"} plus every `doc_*` stage → "doc_summary"; stages not listed
    (cheap per-row steps) cost nothing.
  - `format_duration(low_s: float, high_s: float) -> str` — e.g. `"≈ 10–40 min"`,
    `"≈ 18 min"` when low and high round to the same value, seconds under 1 min,
    hours over 90 min.
- Rules: for each step implied by `plan.stages_to_run` (map stage → step), seconds = units ×
  rate, where rate = `rate(history, step, model, device)` or the first-estimate rate. Passes:
  the word pass charges every step; when `"documents_summary"` is live the doc pass charges
  again every step except `mfa_align` (cached), `predictability` (score cache), `word2vec` and
  `bert` (word pass only) — confirm each exception against the code (`main._prealign_batch`,
  `extractors/predictability._SCORE_CACHE`) before relying on it. Audio steps use
  `stats.audio_seconds`; text steps `stats.words` (audio-dependent text steps such as
  syllabify use `words_with_audio`). Add one-time model load seconds (history `load_seconds`
  median, else first-estimate). `learned` = every charged step had a history rate; spread =
  `LEARNED_SPREAD` if learned else `FIRST_ESTIMATE_SPREAD`.

- [ ] **Step 1: Failing tests:** `test_first_estimate_uses_wide_spread`,
  `test_learned_rates_narrow_the_range`, `test_cuda_history_not_used_for_cpu`
  (history only has cuda samples → cpu estimate is not learned),
  `test_doc_pass_does_not_recharge_mfa_or_predictability`,
  `test_text_only_items_not_charged_mfa` (stats with `audio_seconds == 0`),
  `test_format_duration_examples`.
- [ ] **Step 2–4:** FAIL → implement → PASS.
- [ ] **Step 5: Commit** — `feat: run-time estimate that learns from this computer`

### Task 6: Run tab shows the estimates

**Files:**
- Modify: `gui/section_run.py` (`_LevelRow` size label + per-file sizes; summary box under the
  Readiness heading; warning label), `gui/theme.py` (summary box / warning QSS)
- Test: `tests/gui/test_section_run_estimates.py`

**Interfaces:**
- Consumes: Tasks 3–5 functions; `gather_system_info()` once per `RunSection` lifetime
  (lazy, cached attribute); `timing_history.load_history()` on each refresh (cheap file read).
- Behaviour: on `refresh()`, compute stats → disk/ram/time; per-level size after the count tag
  (hidden for levels left out); per-file size after each non-struck path; summary rows per
  Global Constraints; the learning note only when `not time.learned`; "—" values when stats is
  `None`; warning label visible only when `ram.warning`. Estimates never change
  `run_button.isEnabled()`.

- [ ] **Step 1: Implement** per the mockup.
- [ ] **Step 2: Tests:** `test_dash_until_folders_set`, `test_level_and_file_sizes_shown`
  (tmp folders with two small transcripts + wav files generated with soundfile; Word level on →
  word row shows "≈"), `test_ram_warning_only_when_it_does_not_fit` (monkeypatch
  `gather_system_info` to 4 GB RAM / no GPU and select llama3_8b with Word predictability on →
  warning visible; 512 GB → hidden), `test_estimates_never_disable_run`,
  `test_first_estimate_note_hidden_when_learned` (seed a history file).
- [ ] **Step 3:** grab() the Run tab to the scratchpad and compare with the mockup.
- [ ] **Step 4:** Full suite passes. **Step 5: Commit** — `feat(gui): Run tab shows disk, memory and time estimates`

### Task 7: Real-run verification + docs

**Files:**
- Modify: `README.md` (Run-tab section: what the estimates mean, that time learns per
  computer, where the history file lives and that deleting it resets it), `docs/master_plan.md`
  (§3 row; §2a note)

- [ ] **Step 1: Accuracy check:** on a scratch config (never the live one) run the validation
  corpus with Word + Sub-word + Document + Acoustic levels and Word2Vec/BERT on; compare
  `estimate_disk` per level with the real folder sizes — each level within 25 %; report the
  table. If a level misses, fix the constant/formula (Task 2/3 files) and re-check.
- [ ] **Step 2: Output unchanged:** run the same config on `main` (pre-branch) and on the
  branch; every CSV/.mat/TextGrid identical (NaN-aware for .mat).
- [ ] **Step 3: Learning check:** after the runs above, the Run tab shows a narrower,
  learned time range without the first-estimate note; compare it with the actual duration of
  a third run (report both).
- [ ] **Step 4:** Docs updates; full suite. **Step 5: Commit** — `docs: run estimates`
