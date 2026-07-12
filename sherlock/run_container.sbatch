#!/usr/bin/env bash
# SLURM GPU job: run the SensiExtractor pipeline on Sherlock INSIDE the Apptainer
# container (sensiextractor.sif), driven by the two-file YAML config in
# config/ (features.yaml + run.yaml), via the top-level run.py. The
# environment comes from the .sif, so there is no $HOME/quota/module friction.
# Submit from a login node:   sbatch sherlock/run_container.sbatch
#
# --- SLURM directives (placeholders — verify against `sh_part` / Sherlock docs) ---
#SBATCH --job-name=sensi-extract
#SBATCH --partition=gpu          # or an owners partition you belong to (see `sh_part`)
#SBATCH --gpus=1                 # one GPU
##SBATCH -C GPU_MEM:16GB         # VRAM: Llama-3-8B in 4-bit needs ~6-8 GB; 16 GB is
                                 # comfortable. Uncomment to pin (use 24GB for the
                                 # fp16 fallback / headroom). Remove the extra '#' to enable.
#SBATCH --cpus-per-task=8        # MFA/spaCy are CPU-bound — give them cores
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out    # combined stdout+stderr

set -euo pipefail

# Raise the open-file limit to the node's hard max. MFA's import tree (sklearn/
# hdbscan/pandas/…) through the FUSE-mounted .sif needs many fds; a low default
# soft limit causes `OSError: [Errno 24] Too many open files`. Compute-node hard
# limit is ~131072 (login nodes cap at 256, which is why sanity checks must run
# on a compute node). Harmless if already high.
ulimit -n "$(ulimit -Hn)" 2>/dev/null || true

# ------------------------- CONFIG -------------------------------------
# These default to the folder layout in RUNBOOK.md step 3. If you put the repo
# or weights somewhere else, DON'T edit this file — override at submit time, e.g.
#   sbatch --export=ALL,REPO=/my/path/SensiExtractor,HF_HOME=/my/scratch/hf_cache \
#          -C GPU_MEM:16GB sherlock/run_container.sbatch
# (editing here works too, but a re-rsync of the repo would overwrite it).
# Durable base ($OAK if present, else $GROUP_HOME).
STORAGE_BASE="${STORAGE_BASE:-${OAK:-$GROUP_HOME}}"
# App code — bind-mounted, NOT baked into the .sif, so a re-rsync updates the
# code (and config/features.yaml) without rebuilding the container.
REPO="${REPO:-$GROUP_HOME/SensiExtractor}"
# The built container (one file), rsync'd into the repo's sherlock/ folder.
SIF="${SIF:-$REPO/sherlock/sensiextractor.sif}"
# HF weights on personal $SCRATCH — MUST match download_models.sh's HF_HOME.
export HF_HOME="${HF_HOME:-${SCRATCH:-$STORAGE_BASE}/sensi/hf_cache}"

# The run is driven by the two-file YAML config in config/ (see RUNBOOK.md
# step 2). features.yaml says WHAT to extract (the GUI's live config, rsync'd
# from your machine); run.yaml says WHERE on Sherlock (folders, analyzer name,
# token-file paths) — filled in directly on Sherlock, never rsync'd.
CONFIG_DIR="${CONFIG_DIR:-$REPO/config}"
# -----------------------------------------------------------------------

echo "[job] node=$(hostname)  gpu=${CUDA_VISIBLE_DEVICES:-none}  jobid=${SLURM_JOB_ID:-?}"
nvidia-smi || echo "[job] WARNING: nvidia-smi failed — is this a GPU node?"

# Fail fast with the exact wrong path + how to point it right (override REPO/SIF/
# HF_HOME/CONFIG_DIR via `sbatch --export=ALL,VAR=...`; see the CONFIG_DIR note above).
_hint="If this path is wrong, pass the right one, e.g. sbatch --export=ALL,REPO=/your/path ..."
[ -f "$SIF" ]    || { echo "[job] ERROR: container not found at $SIF"; echo "  $_hint"; exit 1; }
[ -d "$REPO" ]   || { echo "[job] ERROR: repo not found at $REPO"; echo "  $_hint"; exit 1; }
[ -f "$CONFIG_DIR/features.yaml" ] || { echo "[job] ERROR: features.yaml not found in $CONFIG_DIR (rsync the repo's config/ over, or edit one here)"; echo "  $_hint"; exit 1; }
[ -f "$CONFIG_DIR/run.yaml" ] || { echo "[job] ERROR: run.yaml not found in $CONFIG_DIR (create it on Sherlock — machine-specific folders/token paths)"; echo "  $_hint"; exit 1; }
[ -d "$HF_HOME" ] || echo "[job] WARNING: HF_HOME $HF_HOME not found — weights may be missing (see step 5). $_hint"

# Bind host paths INTO the container. Everything the job touches at run time —
# code, weights, data — lives outside the .sif and is mounted here. MFA models
# and the conda/pip env are already baked inside the .sif. $SCRATCH is bound
# wholesale because the config's input/output folders live under it.
#   --nv                         : pass through the host NVIDIA driver (GPU)
#   --bind src:dst               : mount host dir at that path inside the container
#   --env HF_HUB_OFFLINE=1 ...   : compute nodes are offline; read cache only
#   --env HF_HOME=...            : point HF at the bind-mounted weight cache
apptainer exec --nv \
  --bind "$REPO:$REPO" \
  --bind "$HF_HOME:$HF_HOME" \
  --bind "$SCRATCH:$SCRATCH" \
  --env  HF_HUB_OFFLINE=1 \
  --env  TRANSFORMERS_OFFLINE=1 \
  --env  HF_HOME="$HF_HOME" \
  "$SIF" \
  python "$REPO/run.py" --config-dir "$CONFIG_DIR"

echo "[job] DONE. Check the output_folder from your config for *_features.csv."
