#!/usr/bin/env bash
# SLURM GPU job: run the SensiExtractor batch CLI on Sherlock.
# Submit from a login node:   sbatch sherlock/run_extract.sbatch
#
# This is a Phase-2 SMOKE-TEST template: it runs the existing `main.py` word-level
# batch with LLM-cloze on, to prove the HF backend populates the LLM_* columns on a
# GPU node. Adjust the SBATCH directives to your account/partition before submitting.
#
# --- SLURM directives (placeholders — verify against `sh_part` / Sherlock docs) ---
#SBATCH --job-name=sensi-extract
#SBATCH --partition=gpu          # or an owners partition you belong to
#SBATCH --gpus=1                 # one GPU; add -C GPU_MEM:24GB to pin VRAM class
#SBATCH --cpus-per-task=8        # MFA/spaCy are CPU-bound — give them cores
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out    # combined stdout+stderr

set -euo pipefail

# ------------------------- CONFIG (edit these) -------------------------
ENV_PREFIX="${ENV_PREFIX:-$GROUP_HOME/envs/sensi}"
REPO="${REPO:-$GROUP_HOME/code/SensiExtractor}"
export HF_HOME="${HF_HOME:-$OAK/sensi/hf_cache}"
export MFA_ROOT_DIR="${MFA_ROOT_DIR:-$OAK/sensi/mfa_data}"

# Input/output for this run (a small sample for the smoke test):
TRANSCRIPTS_DIR="${TRANSCRIPTS_DIR:-$SCRATCH/sensi/sample/transcripts}"
AUDIO_DIR="${AUDIO_DIR:-$SCRATCH/sensi/sample/audio}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRATCH/sensi/sample/out}"

LLM_MODEL="${LLM_MODEL:-meta-llama/Meta-Llama-3-8B}"
# -----------------------------------------------------------------------

# Compute nodes are offline: force HF to read only the prefetched cache.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# Belt-and-suspenders (main.py setdefaults this too; harmless on Sherlock's Intel).
export MKL_THREADING_LAYER="${MKL_THREADING_LAYER:-sequential}"

echo "[job] node=$(hostname)  gpu=$CUDA_VISIBLE_DEVICES  jobid=$SLURM_JOB_ID"
nvidia-smi || echo "[job] WARNING: nvidia-smi failed — is this a GPU node?"

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"

# NOTE: LLM-cloze is not exposed as a CLI flag by name — it's ON by default in main.py
# unless --no-llm-cloze is passed. We pass --llm-model to force the exact repo id we
# cached. Everything else here uses main.py's batch mode as-is.
python "$REPO/main.py" \
  --transcripts-folder "$TRANSCRIPTS_DIR" \
  --audio-folder       "$AUDIO_DIR" \
  --output-folder      "$OUTPUT_DIR" \
  --analyzer           "sherlock-smoketest" \
  --level              word \
  --llm-model          "$LLM_MODEL"

echo "[job] DONE. Check $OUTPUT_DIR/word/*_features.csv for populated LLM_Cloze columns."
