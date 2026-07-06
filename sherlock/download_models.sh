#!/usr/bin/env bash
# Prefetch model weights into $HF_HOME. RUN ON A LOGIN NODE (needs internet).
# Compute nodes are offline, so the GPU job (run_extract.sbatch) reads from this
# cache with HF_HUB_OFFLINE=1.
#
# Requires an HF token with Llama-3 access already granted on huggingface.co:
#   HF_TOKEN=hf_xxx bash sherlock/download_models.sh
set -euo pipefail

# ------------------------- CONFIG (edit these) -------------------------
# Big model weights (Llama-3-8B ~16 GB, GPT-2) live on personal $SCRATCH by
# default — they're large and fully regenerable, so they don't belong in shared
# or backed-up space. NOTE: run_extract.sbatch must use the SAME HF_HOME, or the
# offline GPU job won't find the weights you download here.
export HF_HOME="${HF_HOME:-${SCRATCH:-$GROUP_HOME}/sensi/hf_cache}"
ENV_PREFIX="${ENV_PREFIX:-$GROUP_HOME/envs/sensi}"
# Which local Llama to cache. NOTE: real HF repo id (see README caveat) —
# the app's llm_cloze.py label 'meta-llama/Llama-3-8B' maps to this:
LLAMA_REPO="${LLAMA_REPO:-meta-llama/Meta-Llama-3-8B}"
# -----------------------------------------------------------------------

: "${HF_TOKEN:?Set HF_TOKEN (an HF access token with Llama-3 access granted)}"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

echo "[dl] HF_HOME     = $HF_HOME"
echo "[dl] LLAMA_REPO  = $LLAMA_REPO"
mkdir -p "$HF_HOME"

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"

# GPT-2 (legacy surprisal feature) — small, ungated
huggingface-cli download gpt2 --quiet

# Llama-3-8B — gated; requires the token + accepted license
huggingface-cli download "$LLAMA_REPO"

echo "[dl] DONE. Weights cached under $HF_HOME"
echo "[dl] Point run_extract.sbatch's HF_HOME at this path and pass"
echo "     --llm-model $LLAMA_REPO to main.py."
