# GPT-2 family, model precision, beginning-of-text token — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Offer GPT-2 Medium/Large/XL, make every model's precision explicit (GPT-2 fp32; Llama fp16/8-bit/4-bit, fp32 on CPU), and score with each model's beginning-of-text token.

**Architecture:** The roster (`predictability_models.py`, no torch) gains per-model size/precision data and one pure function that turns *(model, chosen precision, GPU present)* into *(precision that runs, device)*. The HF backend loads exactly that, prepends BOS to every forward-pass window, and records what it loaded. Prefs v24 adds `predictability.precision`; GUI, estimates, docs and manual read the roster.

**Tech Stack:** Python 3.10, transformers 5.13 (`dtype=` kwarg, not `torch_dtype`), torch 2.7, bitsandbytes (8-/4-bit, GPU only), PySide6.

**Spec:** `dev/superpowers/specs/2026-11-04-llm-roster-precision-design.md`

## Global Constraints

- Precisions are the strings `"fp32"`, `"fp16"`, `"8bit"`, `"4bit"`. GPT-2 family: `("fp32",)` only. Llama: `("fp16", "8bit", "4bit")`, first = default.
- GPU present → chosen precision on GPU. CPU only → **fp32 on CPU** for every model. 8-/4-bit load failure → error naming the failure and suggesting fp16; **never** a fallback to another precision.
- New slugs / hf_ids exactly: `gpt2_medium` → `openai-community/gpt2-medium`, `gpt2_large` → `openai-community/gpt2-large`, `gpt2_xl` → `openai-community/gpt2-xl`. `gpt2` keeps hf_id `gpt2`.
- Download sizes (GB): gpt2 0.55, gpt2_medium 1.52, gpt2_large 3.25, gpt2_xl 6.43; Llama = 2 bytes × params / 1e9. Params: 124e6, 355e6, 774e6, 1.5e9; Llama 3.21e9, 8e9, 71e9 (as in `estimate_constants._LLAMA`). All GPT-2 native_ctx 1024.
- Bytes per parameter: fp32 4, fp16 2, 8bit 1, 4bit 0.5.
- Downloads happen on first use only; nothing new in setup scripts.
- Every forward-pass window starts with `tokenizer.bos_token_id`, never scored, maps to no word; a window is `BOS + ≤ (limit − 1)` subwords.
- Commits: no Claude / co-author attribution. TDD for `sensi/extractors/`; test-after for GUI, prefs, docs.
- Entry points keep `MKL_THREADING_LAYER`; run tests with `./env/python.exe -m pytest -q -p no:cacheprovider` and PATH prefixed by `env`, `env/Scripts`, `env/Library/bin`.

## Review Focus

1. Llama ticked on a CPU-only machine with 4-bit chosen → runs **fp32 on CPU**, prints that, manifest says `fp32`/`cpu` (Task 1 table test + Task 2 notice test).
2. Hand-edited YAML puts a GPT-2 slug or an unknown value in `predictability.precision` → `value_errors` reports it; the extractor still resolves to the model's default (Task 1 + Task 4).
3. A text longer than the context window → every chunk window starts with BOS and is ≤ limit; the union still covers every subword once (Task 2).
4. Changing a model's precision between two runs in one GUI session → re-scores instead of returning cached values from the other precision (Task 2 cache-key test).
5. A tokenizer with `bos_token_id is None` → no crash; first word falls back to `NaN` (Task 2).

---

### Task 1: Roster data and the load-plan rule

**Files:**
- Modify: `sensi/extractors/predictability_models.py`
- Test: `tests/extractors/test_predictability_models.py`

**Interfaces:**
- Produces:
  - `ModelSpec` gains `params: float`, `download_gb: float`, `precisions: tuple[str, ...]` (keep `slug, label, hf_id, gated, native_ctx, note`).
  - `BYTES_PER_PARAM: dict[str, float]`
  - `default_precision(slug: str) -> str` — `precisions[0]`
  - `effective_precision(slug: str, precision_map: dict | None) -> str` — the map's value if it is in the model's `precisions`, else the default.
  - `load_plan(slug: str, precision_map: dict | None, cuda: bool) -> tuple[str, str]` — `(precision_run, device)`; device `"cuda"`/`"cpu"`.
  - `weights_gb(slug: str, precision_run: str) -> float` — `params × BYTES_PER_PARAM[precision_run] / 1e9`.

- [ ] **Step 1: Write the failing tests**

```python
def test_gpt2_family_in_roster():
    for slug, hf, gb in [("gpt2", "gpt2", 0.55), ("gpt2_medium", "openai-community/gpt2-medium", 1.52),
                         ("gpt2_large", "openai-community/gpt2-large", 3.25),
                         ("gpt2_xl", "openai-community/gpt2-xl", 6.43)]:
        s = spec_for(slug)
        assert (s.hf_id, s.download_gb, s.native_ctx, s.precisions, s.gated) == (hf, gb, 1024, ("fp32",), False)

def test_llama_precisions_default_fp16():
    for slug in ("llama32_3b", "llama3_8b", "llama3_70b"):
        assert spec_for(slug).precisions == ("fp16", "8bit", "4bit")
        assert default_precision(slug) == "fp16"

@pytest.mark.parametrize("slug,chosen,cuda,expected", [
    ("gpt2_large", None, True, ("fp32", "cuda")),
    ("gpt2_large", None, False, ("fp32", "cpu")),
    ("llama3_8b", None, True, ("fp16", "cuda")),
    ("llama3_8b", "4bit", True, ("4bit", "cuda")),
    ("llama3_8b", "8bit", True, ("8bit", "cuda")),
    ("llama3_8b", "4bit", False, ("fp32", "cpu")),
    ("llama3_8b", "fp16", False, ("fp32", "cpu")),
])
def test_load_plan_table(slug, chosen, cuda, expected):
    pm = {slug: chosen} if chosen else {}
    assert load_plan(slug, pm, cuda) == expected

def test_effective_precision_ignores_values_the_model_does_not_offer():
    assert effective_precision("gpt2", {"gpt2": "fp16"}) == "fp32"
    assert effective_precision("llama3_8b", {"llama3_8b": "bf16"}) == "fp16"

def test_weights_gb():
    assert weights_gb("llama3_70b", "fp16") == pytest.approx(142.0)
    assert weights_gb("gpt2_xl", "fp32") == pytest.approx(6.0)
```

- [ ] **Step 2: Run** `pytest tests/extractors/test_predictability_models.py -q` — expect failures (unknown slugs/names).
- [ ] **Step 3: Implement** the fields, the three new roster entries, Llama `params`/`download_gb`/`precisions`, and the functions above. Rewrite every `note` to size + memory + access, e.g. `"free, no token, 3.25 GB download on first use"`; drop the old "≥48 GB VRAM" 4-bit figures.
- [ ] **Step 4: Run** the test file — PASS; also run `tests/extractors tests/gui -q` to catch roster-length assumptions (fix tests that hard-code four models).
- [ ] **Step 5: Commit** `feat(predictability): GPT-2 family in the roster; per-model precision rule`

### Task 2: Backend — exact precision, BOS windows, load records

**Files:**
- Modify: `sensi/extractors/predictability.py` (`_HuggingFaceBackend`, `score`, `_get_backend`, `_cached_backend`, `_make_default_backend`, `_scored_by_slug`, `add_predictability`; delete `_load_tiers`, `_FULL_PRECISION_CPU_HF_IDS`, `_is_full_precision_cpu`; update `LLM_OPTIONS` labels to drop VRAM claims)
- Test: `tests/extractors/test_predictability.py` (replace the `_load_tiers` / `_load_tier` tests at ~lines 125–135 and 428–496)

**Interfaces:**
- Consumes: Task 1 `load_plan`, `effective_context`, `spec_for`, `weights_gb`.
- Produces:
  - `add_predictability(df, text, *, models, metrics, context=None, precision=None, backends=None)`
  - `_get_backend(slug: str, max_context: int | None, precision_map: dict | None) -> _Backend`
  - `_cached_backend(model_id: str, max_context: int | None, precision_run: str, device: str) -> _Backend`; `_BACKEND_CACHE` key `(model_id, max_context, precision_run, device)`
  - `_SCORE_CACHE` key `(hf_id, max_context, precision_run, digest)`
  - `_HuggingFaceBackend(model_id, max_context, precision_run, device)` with attributes `precision_run`, `device`
  - `loaded_models() -> list[dict]` — one `{"slug", "hf_id", "precision", "device"}` per backend built in this process, in load order.

Loading per `(precision_run, device)`: `fp32` → `dtype=torch.float32`, then `.to("cuda")` when device is cuda; `fp16` → `dtype=torch.float16, device_map="auto"`; `8bit` / `4bit` → `BitsAndBytesConfig(load_in_8bit=True)` / `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)`, `device_map="auto"`. A failure raises `RuntimeError` whose text names the precision, the original exception, and "choose fp16 for this model on the Predictability card". Print once per load: `[predict] {slug}: {precision_run} on {GPU|CPU}`; when the chosen precision was not fp32 but the run is CPU: `[predict] {slug}: no GPU — running fp32 on CPU (~{weights_gb:.1f} GB RAM)`. Before loading, if `huggingface_hub.try_to_load_from_cache(hf_id, "config.json")` is not a path, print `[predict] {slug}: downloading {download_gb:.2f} GB on first use…`.

BOS in `score`: tokenize as today (`add_special_tokens=False`); keep `ids`/`offsets` for the text only. Every forward pass feeds `[bos] + ids[a:b]`; the logits at window position `j` predict `ids[a + j]`, so every subword (including `ids[0]`) gets a log-prob. Chunking uses `_chunk_slices(len(ids), limit - 1)`. With `bos_token_id is None`, keep today's behaviour (no BOS, `ids[0]` unscored).

- [ ] **Step 1: Write failing tests** (reuse `_patch_transformers`; a fake tokenizer with `bos_token_id = 99` and a fake model that records each input row):
  - `test_first_word_scored_with_bos` — `add_predictability` on `["The","cat","."]` gives a finite surprisal for `"The"`; the fake model's first input row starts with `99`.
  - `test_every_chunk_window_starts_with_bos_and_fits_limit` — `max_context=8`, 30 subwords: each recorded row `[0] == 99` and `len(row) <= 8`; every subword scored exactly once.
  - `test_no_bos_token_keeps_first_word_nan` — `bos_token_id=None`: first token NaN, no crash.
  - `test_cpu_only_llama_loads_fp32_and_says_so(capsys)` — `torch.cuda.is_available → False`, `precision={"llama32_3b": "4bit"}`: `from_pretrained` called with `dtype=torch.float32`, no quantization config; stdout contains `"no GPU — running fp32 on CPU"`; `loaded_models()[-1] == {"slug": "llama32_3b", ..., "precision": "fp32", "device": "cpu"}`.
  - `test_4bit_failure_raises_without_fallback` — CUDA on, 4-bit `from_pretrained` raises: `RuntimeError` mentioning `fp16`; `from_pretrained` called once.
  - `test_gpt2_large_fp32_on_gpu_when_present` — CUDA on: `dtype=torch.float32`, moved to cuda.
  - `test_precision_change_rescores` — score `llama32_3b` with fp16 then 4bit on one text: two backend builds, two `score` calls.
  - `test_first_use_download_notice(capsys)` — `try_to_load_from_cache` patched to return `None`: stdout has `"downloading 3.25 GB on first use"` for `gpt2_large`.
- [ ] **Step 2: Run** them — FAIL.
- [ ] **Step 3: Implement** per the Interfaces and paragraphs above. Remove the tier cascade and its warning.
- [ ] **Step 4: Run** `tests/extractors/test_predictability.py -q` — PASS. Update any test that asserted `NaN` for the first word to the new behaviour.
- [ ] **Step 5: Commit** `feat(predictability): explicit precision per model, BOS-led windows, load records`

### Task 3: Pipeline, timing device, run manifest

**Files:**
- Modify: `sensi/pipeline.py` (word pass call ~l.307 and `_timed_predictability` ~l.154), `sensi/utils/timing_history.py:model_device`, `sensi/utils/run_logger.py:_finalize_impl`
- Test: `tests/test_stage_timing.py`, `tests/utils/test_run_logger.py`

**Interfaces:**
- Consumes: Task 2 `add_predictability(..., precision=)`, `_get_backend(slug, eff, precision_map)`, cache keys, `loaded_models()`; Task 1 `load_plan`.
- Produces: manifest key `"predictability_models"` = `loaded_models()` (list, possibly empty).

- [ ] **Step 1:** Pass `precision=pconf.get('precision')` everywhere `context=` is passed; `_timed_predictability` gains a `precision` parameter and builds its cache-hit key and `_BACKEND_CACHE` key with `load_plan(...)`'s `precision_run`/`device`.
- [ ] **Step 2:** `model_device(step, model, cuda_available)` returns `load_plan(model, prefs_precision, cuda)[1]`; add a keyword `precision_map: dict | None = None` and pass it from callers that have prefs (`run_estimates`, `_timed_predictability`).
- [ ] **Step 3:** `_finalize_impl` writes `manifest["predictability_models"]` from `sensi.extractors.predictability.loaded_models()` (import inside the function; `[]` if the module was never imported — check `sys.modules`, don't import torch for a run without predictability).
- [ ] **Step 4 (tests, after):** `test_manifest_lists_loaded_models` (monkeypatch `loaded_models`), and fix `test_stage_timing.py` fakes for the new `_make_default_backend` signature. Run `tests -q -k "timing or run_logger or pipeline"` — PASS.
- [ ] **Step 5: Commit** `feat(pipeline): pass model precision; manifest records how each model ran`

### Task 4: Prefs v24, value checks, YAML comment

**Files:**
- Modify: `sensi/utils/preferences.py`, `sensi/utils/config_checks.py`, `sensi/utils/config_yaml.py:_key_comments`, `config/features_defaults.yaml` (regenerated)
- Create: `tests/utils/test_preferences_v24.py`
- Modify: `tests/utils/test_preferences_v23.py` (`test_active_version_is_23` → `>= 23` style, as was done for v22)

**Interfaces:**
- Produces: `DEFAULT_PREFS["predictability"]["precision"] == {}`; `SCHEMA_VERSION == 24`.

- [ ] **Step 1:** v24 block after v23 (additive; `_PREDICTABILITY_DEFAULT_V24 = {**v23, "precision": {}}`), add `SCHEMA_VERSION_V24` to the additive version tuple, update the module docstring schema line.
- [ ] **Step 2:** `value_errors`: for `predictability.precision`, each key must be a roster slug whose `precisions` has more than one entry, each value one of that model's `precisions`; messages name the slug and the allowed values.
- [ ] **Step 3:** `_key_comments`: `("predictability", "precision")` → `"optional per-model weight precision for llama* models: fp16 (default) | 8bit | 4bit, e.g. {llama3_8b: 8bit}; GPT-2 models always run fp32; without a GPU every model runs fp32"`.
- [ ] **Step 4 (tests, after):** v24 tests (v23 file gets `precision == {}`; a v24 file keeps `{"llama3_8b": "4bit"}`); config_checks tests: `{"gpt2": "fp16"}` and `{"llama3_8b": "bf16"}` each produce one error; `{"llama3_8b": "8bit"}` none. Run `python scripts/export_default_config.py`, then `tests/utils -q` — PASS.
- [ ] **Step 5: Commit** `feat(prefs): v24 predictability.precision`

### Task 5: Run-page estimates

**Files:**
- Modify: `sensi/utils/estimate_constants.py` (RAM_GB predictability entries, `FIRST_ESTIMATE_RATES`, `FIRST_LOAD_SECONDS`), `sensi/utils/run_estimates.py:estimate_ram` and the disk estimate for model downloads
- Test: `tests/utils/test_run_estimates_ram.py`, `tests/utils/test_run_estimates_disk.py`, `tests/utils/test_estimate_constants.py`

**Interfaces:**
- Consumes: Task 1 `load_plan`, `weights_gb`, `spec.params`, `spec.download_gb`.
- Produces: `estimate_constants.predictability_ram_gib(slug: str, precision_run: str) -> float`.

Rules: `gpt2` fp32 keeps the measured 0.70 GiB; every other model/precision = `weights_gb(slug, precision_run) × _OVERHEAD` converted to GiB (for 4-bit keep the existing `_cuda_4bit_gib` embedding rule for Llama). Rates and first-load seconds for `gpt2_medium/large/xl` scale from the measured GPT-2 figures by `params` (load: by fp32 bytes); Llama keep their scaling. `estimate_ram` uses `load_plan(slug, prefs precision map, cuda)` to choose the figure and RAM vs VRAM. The disk estimate adds `download_gb` for each selected model whose weights are not cached (`try_to_load_from_cache`), labelled "first-use download".

- [ ] **Step 1 (impl)** as above. **Step 2 (tests, after):** `gpt2_xl` selected on a CPU-only system → RAM item ≈ `6.0 × 1.2 / 1.0737` GiB; `llama3_8b` fp16 with a GPU → counted as VRAM ≈ `16 × 1.2 / 1.0737`; CPU-only `llama3_8b` 4-bit → RAM ≈ `32 × 1.2 / 1.0737`. Run `tests/utils -q -k estimate` — PASS.
- [ ] **Step 3: Commit** `feat(estimates): memory and download by model precision`

### Task 6: Predictability card — precision dropdown and notes

**Files:**
- Modify: `sensi/gui/section_word_features.py` (model rows ~l.165–207; new `_on_precision_changed`)
- Test: `tests/gui/test_section_word_features.py`

- [ ] **Step 1:** After the context spinner add a `QComboBox` per model listing `spec.precisions` (labels `fp32`, `fp16`, `8-bit`, `4-bit`), set from `effective_precision(slug, prefs precision map)`; disabled when `len(precisions) == 1`. Tooltip: "Weight precision. fp16 is the most precise choice for Llama; 8-bit and 4-bit use less GPU memory and shift the values. Without a GPU the model runs fp32 on the CPU." Label text stays `f"{spec.label} — {spec.note}"`.
- [ ] **Step 2:** `_on_precision_changed(slug, value)` stores the value in `predictability.precision`, popping the slug when it equals `default_precision(slug)` (same pattern as `_on_context_changed`), then emits `settings_changed`.
- [ ] **Step 3 (tests, after):** seven model rows; GPT-2 combos disabled showing `fp32`; choosing `4-bit` for `llama3_8b` saves `{"llama3_8b": "4bit"}`, choosing `fp16` removes it. Run `tests/gui -q` — PASS.
- [ ] **Step 4: Commit** `feat(gui): per-model precision on the Predictability card`

### Task 7: Docs, manual, real run

**Files:**
- Modify: `docs/features/pred_surprisal.md`, `pred_cloze.md`, `pred_topmatch.md`, `llm_credentials.md`, `master_bert_sentence.md`; `dev/master_plan.md`; `README.md` (the Predictability bullet lists the models)
- Rebuild: `docs/user-manual/` via `python scripts/build_manual.py --refresh-examples --refresh-screenshots`

- [ ] **Step 1:** `pred_surprisal.md` *Models*: one table — slug, parameters, download, precision (default and choices), where it runs, weights memory (from Task 1 `weights_gb`) — and a *Precision* subsection: GPT-2 stays fp32 (measured on GPT-2 small, this project, 2026-11-04: fp16 moves surprisal 0.0213 bits on average, max 0.1187, and on a CPU takes 82.31 s vs 0.55 s per 1024-subword window; bf16 moves it 0.2193 on average, max 1.1432); Llama fp16 default; 8-/4-bit trade memory for value shifts (not measured here — say so); CPU-only machines run fp32; what the console line and manifest record. Replace every "first word is NaN / position 0 unscoreable" statement in the three predictability docs with the BOS behaviour; the PUNCT mask stays. Update the *Step by step*, chunking table (windows now `BOS + limit − 1`) and the "GPT-2 is pinned full-precision on CPU" paragraph.
- [ ] **Step 2:** `llm_credentials.md`: weights download on first use into `data/hf_cache/` with the sizes; nothing at setup. `master_bert_sentence.md`: *How these vectors are used* (sentence-to-sentence similarity by cosine between consecutive sentences; encoding models), same style as `master_word2vec.md`, MATLAB example over `embeddings(:, L, :)` of consecutive sentences, ⚠️ on interpretation.
- [ ] **Step 3:** `master_plan.md` completed-batches row (note: Llama values change — fp16 replaces silent 4-bit; all models' first word now scored).
- [ ] **Step 4: Real run.** `python run.py --config-dir <scratch copy of dev/validation/config>` with `gpt2` only (already cached): first word has a value, manifest lists `{"slug": "gpt2", "precision": "fp32", "device": "cpu"}`, console shows the load line. Do **not** download other models without the owner's OK. Then rebuild the manual with refreshed examples/screenshots; build must print `[manual] OK`.
- [ ] **Step 5:** Full suite — all pass. **Commit** `docs: model precision, GPT-2 family, beginning-of-text token; manual refreshed`
