"""Fixed constants behind the Run-tab estimates (disk, peak memory, time).

Measured values come from `python scripts/calibrate_estimates.py --scratch
<dir outside the repo>`, run 2026-10-26 (the "all" memory variant re-run the
same day with `--skip-full --memory-variants all`). "Validation corpus" below
means: validation/data, 10 paired stimuli, 301 whitespace words,
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
# Peak = RAM_GB["baseline"] + the SUM of the heavy items that will run. Word2Vec
# (_MODEL_CACHE), BERT (_BERT_CACHE) and the predictability backends
# (_BACKEND_CACHE, several models at once) stay cached for the whole process
# and main.py preloads them, so they add up. (MFA is a separate subprocess.)
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

# Llama (params, vocab_size, hidden_size, tie_word_embeddings).
# Params from the model cards:
#   llama32_3b  3.21B ("Params: 3B (3.21B)"), https://huggingface.co/meta-llama/Llama-3.2-3B
#   llama3_8b   8B (card and safetensors widget both say "8B"),
#               https://huggingface.co/meta-llama/Meta-Llama-3-8B
#   llama3_70b  71B (safetensors "Model size: 71B params"; the card table says
#               "70B" - the larger figure is used),
#               https://huggingface.co/meta-llama/Meta-Llama-3-70B
# vocab/hidden/tie: the official meta-llama config.json files are gated (HTTP
# 401 without a token), so they are read from public ungated mirrors of the
# same checkpoints:
#   https://huggingface.co/unsloth/Llama-3.2-3B/raw/main/config.json
#       (vocab 128256, hidden 3072, tie_word_embeddings true, bfloat16)
#   https://huggingface.co/unsloth/llama-3-8b/raw/main/config.json
#       (vocab 128256, hidden 4096, tie_word_embeddings false, bfloat16)
#   https://huggingface.co/NousResearch/Meta-Llama-3-70B/raw/main/config.json
#       (vocab 128256, hidden 8192, tie_word_embeddings false, bfloat16)
_LLAMA: dict[str, tuple[float, int, int, bool]] = {
    "llama32_3b": (3.21e9, 128256, 3072, True),
    "llama3_8b": (8e9, 128256, 4096, False),
    "llama3_70b": (71e9, 128256, 8192, False),
}


def _cpu_gib(spec: tuple[float, int, int, bool]) -> float:
    """CPU tier: from_pretrained(model_id) with no dtype. In transformers 5.13.0
    (the version pinned in conda-lock.yml) `dtype` "defaults to "auto"", which
    uses the config.json dtype (bfloat16 for all three) = 2 B/param:
    https://huggingface.co/docs/transformers/v5.13.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained
    """
    params = spec[0]
    return _gib(params * 2 * _OVERHEAD)


def _cuda_4bit_gib(spec: tuple[float, int, int, bool]) -> float:
    """@cuda tier: the extractor's first GPU tier, 4-bit bitsandbytes. Only
    nn.Linear layers become 4-bit (~0.5 B/param); "all other modules ... are
    converted to torch.float16", and transformers leaves lm_head unquantized,
    so embed_tokens + lm_head stay fp16 (2 B/param), counted once when
    tie_word_embeddings is true:
    https://huggingface.co/docs/transformers/v5.13.0/en/quantization/bitsandbytes
    If the 4-bit load fails the extractor falls back to fp16 on the GPU, which
    needs about the CPU figure instead.
    """
    params, vocab, hidden, tied = spec
    embed = vocab * hidden * (1 if tied else 2)
    return _gib(((params - embed) * 0.5 + embed * 2) * _OVERHEAD)

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
    # Llama entries are computed (gated models, not measured): see _cpu_gib /
    # _cuda_4bit_gib below for the rule and sources.
    "llama32_3b": _cpu_gib(_LLAMA["llama32_3b"]),
    "llama32_3b@cuda": _cuda_4bit_gib(_LLAMA["llama32_3b"]),
    "llama3_8b": _cpu_gib(_LLAMA["llama3_8b"]),
    "llama3_8b@cuda": _cuda_4bit_gib(_LLAMA["llama3_8b"]),
    "llama3_70b": _cpu_gib(_LLAMA["llama3_70b"]),
    "llama3_70b@cuda": _cuda_4bit_gib(_LLAMA["llama3_70b"]),
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
    # Not a model load: the first phoneme_surprisal stimulus took 5.88 s vs
    # 0.002-0.004 s for each later one (the cohort trie is built once per
    # process, extractors/phoneme_surprisal._TRIE_CACHE). Read from the
    # timing_history.json samples of the same run.
    "phoneme_surprisal": 5.9,
}

# Llama loads: NOT measured (gated models). Scaled from the measured GPT-2 load
# by weight bytes read: seconds per byte = 7.8 s / (124M params x 4 B, fp32 —
# GPT-2 loads full precision, see RAM_GB), times the Llama weight size =
# params (as in _LLAMA / RAM_GB) x 2 B (bfloat16, the config.json dtype cited
# above). Every device tier reads the same bf16 weights from disk. Like the
# GPT-2 figure this is a warm-disk-cache load; a cold first download/read is
# slower.
_GPT2_WEIGHT_BYTES = _GPT2_PARAMS * 4
for _slug, (_params, *_rest) in _LLAMA.items():
    FIRST_LOAD_SECONDS[f"predictability:{_slug}"] = round(
        FIRST_LOAD_SECONDS["predictability:gpt2"] * _params * 2
        / _GPT2_WEIGHT_BYTES, 1)

# Range multipliers around the point estimate: wide before this machine has
# history (built-in rates come from one other machine), narrow after.
FIRST_ESTIMATE_SPREAD: tuple[float, float] = (0.33, 3.0)
LEARNED_SPREAD: tuple[float, float] = (0.8, 1.25)
