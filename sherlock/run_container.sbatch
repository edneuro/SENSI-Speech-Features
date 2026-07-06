#!/usr/bin/env bash
# SLURM GPU job: run the SensiExtractor batch CLI on Sherlock INSIDE the Apptainer
# container (sensi.sif). This is the container-based counterpart to the conda-based
# run_extract.sbatch — same smoke test, but the environment comes from the .sif
# instead of a bare-metal conda env, so there is no $HOME/quota/module friction.
# Submit from a login node:   sbatch sherlock/run_container.sbatch
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
# Durable base ($OAK if present, else $GROUP_HOME).
STORAGE_BASE="${STORAGE_BASE:-${OAK:-$GROUP_HOME}}"
# The built container (one file). Copy it here with rsync after building in WSL.
SIF="${SIF:-$GROUP_HOME/sensi/sensi.sif}"
# App code — bind-mounted, NOT baked into the .sif, so you can `git pull` and
# re-run without rebuilding the container.
REPO="${REPO:-$GROUP_HOME/code/SensiExtractor}"
# HF weights on personal $SCRATCH — MUST match download_models.sh's HF_HOME.
export HF_HOME="${HF_HOME:-${SCRATCH:-$STORAGE_BASE}/sensi/hf_cache}"

# Input/output for this run (a small sample for the smoke test):
TRANSCRIPTS_DIR="${TRANSCRIPTS_DIR:-$SCRATCH/sensi/sample/transcripts}"
AUDIO_DIR="${AUDIO_DIR:-$SCRATCH/sensi/sample/audio}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRATCH/sensi/sample/out}"

LLM_MODEL="${LLM_MODEL:-meta-llama/Meta-Llama-3-8B}"
# -----------------------------------------------------------------------

echo "[job] node=$(hostname)  gpu=${CUDA_VISIBLE_DEVICES:-none}  jobid=${SLURM_JOB_ID:-?}"
nvidia-smi || echo "[job] WARNING: nvidia-smi failed — is this a GPU node?"

[ -f "$SIF" ]  || { echo "[job] ERROR: container not found at $SIF (build it in WSL, rsync it over)"; exit 1; }
[ -d "$REPO" ] || { echo "[job] ERROR: repo not found at $REPO"; exit 1; }
mkdir -p "$OUTPUT_DIR"

# Bind host paths INTO the container. Everything the job touches at run time —
# code, weights, data — lives outside the .sif and is mounted here. MFA models
# and the conda/pip env are already baked inside the .sif.
#   --nv                         : pass through the host NVIDIA driver (GPU)
#   --bind src:dst               : mount host dir at that path inside the container
#   --env HF_HUB_OFFLINE=1 ...   : compute nodes are offline; read cache only
#   --env HF_HOME=...            : point HF at the bind-mounted weight cache
apptainer exec --nv \
  --bind "$REPO:$REPO" \
  --bind "$HF_HOME:$HF_HOME" \
  --bind "$(dirname "$OUTPUT_DIR"):$(dirname "$OUTPUT_DIR")" \
  --env  HF_HUB_OFFLINE=1 \
  --env  TRANSFORMERS_OFFLINE=1 \
  --env  HF_HOME="$HF_HOME" \
  "$SIF" \
  python "$REPO/main.py" \
    --transcripts-folder "$TRANSCRIPTS_DIR" \
    --audio-folder       "$AUDIO_DIR" \
    --output-folder      "$OUTPUT_DIR" \
    --analyzer           "sherlock-container-smoketest" \
    --level              word \
    --llm-model          "$LLM_MODEL"

echo "[job] DONE. Check $OUTPUT_DIR/word/*_features.csv for populated LLM_Cloze columns."
