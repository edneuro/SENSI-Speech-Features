"""Fixed constants behind the Run-tab estimates (disk, peak memory, time).

Measured values come from `python scripts/calibrate_estimates.py --scratch
<dir outside the repo>`, run 2026-10-26 (the "all" memory variant re-run the
same day with `--skip-full --memory-variants all`). "Validation corpus" below
means: dev/validation/data, 10 paired stimuli, 301 whitespace words,
113.8 s audio, every CPU-feasible feature on (predictability: GPT-2 only),
run 2026-10-26 on "jose" (AMD Ryzen AI 9 HX 370, 24 threads, 31.1 GiB RAM,
Windows 11, CPU-only torch 2.7.1, transformers 5.13.0). Values that the
machine could not measure (gated Llama models, any GPU) are computed from the
model card's parameter count, with the URL next to the value.

"Word" everywhere = a whitespace-separated token of the transcript, the unit
sensi/utils/timing_history.py uses for text steps. Memory is in GiB (2**30 bytes),
the unit of sensi.utils.system_info's system_ram_gb / vram_gb.
"""
from __future__ import annotations

from sensi.extractors.predictability_models import all_specs, spec_for, weights_gb

# --- Rows per transcript word (validation corpus, 2026-10-26, jose) ---------
ROWS_PER_WORD: dict[str, float] = {
    "features": 1.1296,    # 340 word-CSV rows / 301 words (spaCy splits punctuation/clitics)
    "syllables": 1.4020,   # 422 rows / 301 words
    "phonemes": 3.6113,    # 1087 rows / 301 words
}
SENTENCES_PER_WORD: float = 0.0897   # 27 sentence rows / 301 words

# --- CSV bytes per cell = file size / (data rows x columns), totals over the
# corpus (validation corpus, 2026-10-26, jose; every column toggle on) --------
CSV_BYTES_PER_CELL: dict[str, float] = {
    "features": 7.482,             # 155177 B / 20740 cells (340 rows x 61 cols)
    "syllables": 8.253,            # 55725 B / 6752 cells
    "phonemes": 4.329,             # 70589 B / 16305 cells
    "sentences": 20.296,           # 8768 B / 432 cells (long sentence-text column)
    "documents_summary": 13.548,   # 11516 B / 850 cells (10 rows x 85 cols)
}


# --- Peak RAM (GiB) ----------------------------------------------------------
# Peak = RAM_GB["baseline"] + the SUM of the heavy items that will run. Word2Vec
# (_MODEL_CACHE), BERT (_BERT_CACHE) and the predictability backends
# (_BACKEND_CACHE, several models at once) stay cached for the whole process
# and sensi/pipeline.py preloads them, so they add up. (MFA is a separate subprocess.)
# Measured items: peak working set (psutil peak_wset) of a fresh run.py
# subprocess on one stimulus with only that heavy item on, minus the baseline
# subprocess (same config, no heavy models). Validation corpus, 2026-10-26,
# jose; largest of the two measured stimuli (1_FB_P, 32 s audio; LJ002-0321,
# 10 s audio).
# Check of the sum rule ("all" variant: Word2Vec + BERT + GPT-2 together in one
# subprocess): measured 5.73 GiB on 1_FB_P (5.69 on LJ002-0321) vs baseline +
# sum of the per-item entries = 0.87 + 3.85 + 0.48 + 0.70 = 5.90 GiB, i.e. the
# sum overestimates by ~3 % (some shared torch/runtime memory is counted twice).

def _gib(n_bytes: float) -> float:
    return round(n_bytes / 2 ** 30, 2)


_OVERHEAD = 1.2   # +20 % over the raw weights for activations, buffers, CUDA context

# Parameter counts and download sizes come from the roster
# (sensi.extractors.predictability_models, no torch import). Llama counts from
# the model cards:
#   llama32_3b  3.21B ("Params: 3B (3.21B)"), https://huggingface.co/meta-llama/Llama-3.2-3B
#   llama3_8b   8B (card and safetensors widget both say "8B"),
#               https://huggingface.co/meta-llama/Meta-Llama-3-8B
#   llama3_70b  71B (safetensors "Model size: 71B params"; the card table says
#               "70B" - the larger figure is used),
#               https://huggingface.co/meta-llama/Meta-Llama-3-70B
# Only the 4-bit rule needs more than the parameter count: (vocab_size,
# hidden_size, tie_word_embeddings). The official meta-llama config.json files
# are gated (HTTP 401 without a token), so they are read from public ungated
# mirrors of the same checkpoints:
#   https://huggingface.co/unsloth/Llama-3.2-3B/raw/main/config.json
#       (vocab 128256, hidden 3072, tie_word_embeddings true, bfloat16)
#   https://huggingface.co/unsloth/llama-3-8b/raw/main/config.json
#       (vocab 128256, hidden 4096, tie_word_embeddings false, bfloat16)
#   https://huggingface.co/NousResearch/Meta-Llama-3-70B/raw/main/config.json
#       (vocab 128256, hidden 8192, tie_word_embeddings false, bfloat16)
_LLAMA_EMBED: dict[str, tuple[int, int, bool]] = {
    "llama32_3b": (128256, 3072, True),
    "llama3_8b": (128256, 4096, False),
    "llama3_70b": (128256, 8192, False),
}

_GPT2_MEASURED_GIB = 0.70   # gpt2 (124M) fp32 on CPU: 1.57 - 0.87 GiB baseline


def _4bit_gib(slug: str) -> float:
    """Llama on a GPU at 4-bit (bitsandbytes). Only nn.Linear layers become
    4-bit (~0.5 B/param); "all other modules ... are converted to
    torch.float16", and transformers leaves lm_head unquantized, so
    embed_tokens + lm_head stay fp16 (2 B/param), counted once when
    tie_word_embeddings is true:
    https://huggingface.co/docs/transformers/v5.13.0/en/quantization/bitsandbytes
    """
    params = spec_for(slug).params
    vocab, hidden, tied = _LLAMA_EMBED[slug]
    embed = vocab * hidden * (1 if tied else 2)
    return _gib(((params - embed) * 0.5 + embed * 2) * _OVERHEAD)


def predictability_ram_gib(slug: str, precision_run: str) -> float:
    """Peak memory (GiB) of one predictability model loaded at `precision_run`
    (the precision it actually runs at, from load_plan). RAM when it runs on
    the CPU, VRAM when on a GPU - the caller decides which pool.

    gpt2 fp32 is the measured figure; every other model/precision is its weight
    bytes x _OVERHEAD (not measured: no GPU / gated / larger models on the
    calibration machine).
    """
    if slug == "gpt2" and precision_run == "fp32":
        return _GPT2_MEASURED_GIB
    if precision_run == "4bit" and slug in _LLAMA_EMBED:
        return _4bit_gib(slug)
    return _gib(weights_gb(slug, precision_run) * 1e9 * _OVERHEAD)


RAM_GB: dict[str, float] = {
    # 0.87 GiB on 1_FB_P vs 0.84 GiB on LJ002-0321: no input-length term at
    # these lengths (acoustic time series of a 32 s stimulus is small).
    "baseline": 0.87,
    "word2vec": 3.85,   # GoogleNews 300-d KeyedVectors (4.72 - 0.87 GiB)
    "bert": 0.48,       # bert-base-uncased, fp32 (1.35 - 0.87 GiB)
    # Predictability models: see predictability_ram_gib (depends on precision).
}


# --- First-estimate rates (seconds per unit, cpu) ---------------------------
# Median seconds/unit over the validation run's samples (the same statistic as
# sensi.utils.timing_history.rate), validation corpus, 2026-10-26, jose. Units:
# audio seconds for mfa_align / prosody_contours / acoustic_ts /
# phoneme_surprisal, transcript words for every other step.
FIRST_ESTIMATE_RATES: dict[str, float] = {
    # MFA: the per-audio-second SLOPE only; the fixed start-up per aligning
    # run is FIRST_MFA_OVERHEAD_SECONDS below (fit and sources there).
    "mfa_align": 0.244,
    "spacy_parse": 0.000714,
    "predictability": 0.00412,      # GPT-2 (cpu)
    "word2vec": 0.0000095,
    "bert": 0.00394,
    "prosody_contours": 0.0106,
    "syllabify": 0.000865,
    "phoneme_surprisal": 0.000353,
    "sentence_prosody": 0.000172,
    "acoustic_ts": 0.00421,
    "doc_summary": 0.000836,
}

# Predictability rates of every other roster model: NOT measured (gated or
# larger). Scaled from the measured GPT-2 rate by parameter count (forward-pass
# cost ~ params): GPT-2 124M params ("124M parameters",
# https://huggingface.co/openai-community/gpt2); counts from the roster.
_GPT2_PARAMS = spec_for("gpt2").params
for _spec in all_specs():
    FIRST_ESTIMATE_RATES[f"predictability:{_spec.slug}"] = (
        FIRST_ESTIMATE_RATES["predictability"] if _spec.slug == "gpt2" else
        round(FIRST_ESTIMATE_RATES["predictability"] * _spec.params / _GPT2_PARAMS, 4))

# MFA wall time = fixed start-up per MFA invocation + slope x audio seconds
# aligned (sensi.utils.timing_history.fit_overhead_slope learns both per machine).
# First estimate: least-squares fit of the three mfa_align samples the
# calibration run recorded (2026-10-26, jose; each one MFA call):
#   10-stimulus validation batch  113.8 s audio -> 114.4 s
#   1_FB_P warm-up child           16.0 s audio ->  90.6 s
#   LJ002-0321 warm-up child       10.1 s audio ->  88.9 s
# -> 86.6 s + 0.244 s per audio second. Three points over a narrow size range
# (10-114 s audio), so the slope is the weak part: a later same-machine run of
# the 113.8 s batch took 79.7 s, ~30 % under the fit. The first-estimate
# spread (below) covers that.
FIRST_MFA_OVERHEAD_SECONDS: float = 86.6

# Fixed, untimed start-up per run (interpreter + torch/spaCy imports, file
# writes): a real headless run of the validation corpus took 50.4 s wall vs
# 40.7 s of timed steps + recorded model loads (and 134 s vs ~124 s for the
# aligning run before it), 2026-10-26, jose. Charged once per estimate.
STARTUP_SECONDS: float = 10.0

# One-time model loads (seconds), measured on the same run (warm disk cache),
# keyed by timing_history.load_key. Not per-unit; counted once per run.
FIRST_LOAD_SECONDS: dict[str, float] = {
    "predictability:gpt2": 7.8,
    "word2vec": 36.5,
    "bert": 0.7,
    # Not a model load: the first phoneme_surprisal stimulus took 5.88 s vs
    # 0.002-0.004 s for each later one (the cohort trie is built once per
    # process, sensi/extractors/phoneme_surprisal._TRIE_CACHE). Read from the
    # timing_history.json samples of the same run.
    "phoneme_surprisal": 5.9,
}

# Other roster models' loads: NOT measured. Scaled from the measured GPT-2
# load by weight bytes read: seconds per byte = 7.8 s / (124M params x 4 B, fp32),
# times the model's on-disk weight size = params x 4 B for the GPT-2 family
# (fp32 checkpoints) or params x 2 B for Llama (bfloat16, the config.json dtype
# cited above). Every device tier reads the same weights from disk; the figure
# excludes 4-bit quantization time. Like the GPT-2 figure this is a
# warm-disk-cache load; a cold first download/read is slower.
_GPT2_WEIGHT_BYTES = _GPT2_PARAMS * 4
for _spec in all_specs():
    if _spec.slug != "gpt2":
        _bytes = _spec.params * (4 if _spec.slug.startswith("gpt2") else 2)
        FIRST_LOAD_SECONDS[f"predictability:{_spec.slug}"] = round(
            FIRST_LOAD_SECONDS["predictability:gpt2"] * _bytes
            / _GPT2_WEIGHT_BYTES, 1)

# Range multipliers around the point estimate: wide before this machine has
# history (built-in rates come from one other machine), narrow after.
FIRST_ESTIMATE_SPREAD: tuple[float, float] = (0.33, 3.0)
LEARNED_SPREAD: tuple[float, float] = (0.8, 1.25)
