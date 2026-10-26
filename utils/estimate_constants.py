"""Fixed constants behind the Run-tab estimates (disk, peak memory, time).

Measured values come from `python scripts/calibrate_estimates.py` (raw output
in .superpowers/sdd/2026-10-26-run-estimates/task-2-report.md). "Validation
corpus" below means: validation/data, 10 paired stimuli, 301 whitespace words,
113.8 s audio, every CPU-feasible feature on (predictability: GPT-2 only),
run 2026-10-26 on "jose" (AMD Ryzen AI 9 HX 370, 24 threads, 31.1 GiB RAM,
Windows 11, CPU-only torch 2.7.1, transformers 5.13.0). Values that the
machine could not measure (gated Llama models, any GPU) are computed from the
model card's parameter count, with the URL next to the value.

"Word" everywhere = a whitespace-separated token of the transcript, the unit
utils/timing_history.py uses for text steps. Memory is in GiB (2**30 bytes),
the unit of utils.system_info's system_ram_gb / vram_gb.
"""
from __future__ import annotations

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
# Peak = RAM_GB["baseline"] + the single largest ticked heavy item (stages run
# one after another in one process; MFA is a separate subprocess). Measured
# items: peak working set (psutil peak_wset) of a fresh run.py subprocess on one
# stimulus with only that heavy item on, minus the baseline subprocess (same
# config, no heavy models). Validation corpus, 2026-10-26, jose; largest of the
# two measured stimuli (1_FB_P, 32 s audio; LJ002-0321, 10 s audio).

def _gib(n_bytes: float) -> float:
    return round(n_bytes / 2 ** 30, 2)


_OVERHEAD = 1.2   # +20 % over the raw weights for activations, buffers, CUDA context

RAM_GB: dict[str, float] = {
    # 0.87 GiB on 1_FB_P vs 0.84 GiB on LJ002-0321: no input-length term at
    # these lengths (acoustic time series of a 32 s stimulus is small).
    "baseline": 0.87,
    "word2vec": 3.85,   # GoogleNews 300-d KeyedVectors (4.72 - 0.87 GiB)
    "bert": 0.48,       # bert-base-uncased, fp32 (1.35 - 0.87 GiB)
    # GPT-2 always loads full-precision on the CPU
    # (extractors/predictability._is_full_precision_cpu): fp32, 1.57 - 0.87 GiB.
    "gpt2": 0.70,
    # ...so its "@cuda" tier is the same CPU load, never VRAM.
    "gpt2@cuda": 0.70,
    # Llama, CPU tier: from_pretrained with no dtype; transformers 5.x defaults
    # to dtype="auto", i.e. the checkpoint's BF16 = 2 B/param.
    # @cuda: the extractor's first GPU tier is 4-bit (bitsandbytes) ~ 0.5 B/param;
    # if 4-bit fails it falls back to fp16 (2 B/param, the CPU figure).
    # 3.21B params ("Params: 3B (3.21B)"), https://huggingface.co/meta-llama/Llama-3.2-3B
    "llama32_3b": _gib(3.21e9 * 2 * _OVERHEAD),
    "llama32_3b@cuda": _gib(3.21e9 * 0.5 * _OVERHEAD),
    # 8B params (card and safetensors widget both say "8B"),
    # https://huggingface.co/meta-llama/Meta-Llama-3-8B
    "llama3_8b": _gib(8e9 * 2 * _OVERHEAD),
    "llama3_8b@cuda": _gib(8e9 * 0.5 * _OVERHEAD),
    # 71B params (safetensors "Model size: 71B params"; card table says "70B" -
    # the larger figure is used), https://huggingface.co/meta-llama/Meta-Llama-3-70B
    "llama3_70b": _gib(71e9 * 2 * _OVERHEAD),
    "llama3_70b@cuda": _gib(71e9 * 0.5 * _OVERHEAD),
}


# --- First-estimate rates (seconds per unit, cpu) ---------------------------
# Median seconds/unit over the validation run's samples (the same statistic as
# utils.timing_history.rate), validation corpus, 2026-10-26, jose. Units:
# audio seconds for mfa_align / prosody_contours / acoustic_ts /
# phoneme_surprisal, transcript words for every other step.
FIRST_ESTIMATE_RATES: dict[str, float] = {
    # One MFA call aligning the whole 10-stimulus batch (113.8 s audio in
    # 114.4 s). MFA has a large fixed startup, so per audio second this is
    # batch-size dependent: high for small batches, an overestimate for long
    # stimuli. Learned history replaces it.
    "mfa_align": 1.005,
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

# Llama predictability on CPU, keyed like timing_history.load_key: NOT measured
# (gated models). Scaled from the measured GPT-2 rate by parameter count
# (forward-pass cost ~ params): GPT-2 124M params ("124M parameters",
# https://huggingface.co/openai-community/gpt2); Llama counts as in RAM_GB.
_GPT2_PARAMS = 124e6
for _slug, _params in (("llama32_3b", 3.21e9), ("llama3_8b", 8e9),
                       ("llama3_70b", 71e9)):
    FIRST_ESTIMATE_RATES[f"predictability:{_slug}"] = round(
        FIRST_ESTIMATE_RATES["predictability"] * _params / _GPT2_PARAMS, 4)
FIRST_ESTIMATE_RATES["predictability:gpt2"] = FIRST_ESTIMATE_RATES["predictability"]

# One-time model loads (seconds), measured on the same run (warm disk cache),
# keyed by timing_history.load_key. Not per-unit; counted once per run.
FIRST_LOAD_SECONDS: dict[str, float] = {
    "predictability:gpt2": 7.8,
    "word2vec": 36.5,
    "bert": 0.7,
}

# Range multipliers around the point estimate: wide before this machine has
# history (built-in rates come from one other machine), narrow after.
FIRST_ESTIMATE_SPREAD: tuple[float, float] = (0.33, 3.0)
LEARNED_SPREAD: tuple[float, float] = (0.8, 1.25)
