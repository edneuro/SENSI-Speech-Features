#!/usr/bin/env bash
# Build the SensiExtractor conda env on Sherlock. RUN ON A LOGIN NODE (needs internet).
#
# This mirrors the real 4-step recipe (the checked-in environment.yml is incomplete):
#   1. conda create python + montreal-forced-aligner   (conda-forge)
#   2. pip install -r requirements.txt
#   3. python -m spacy download en_core_web_sm
#   4. mfa model download acoustic/dictionary english_us_arpa
#
# It does NOT modify the repo. Safe to re-run (conda/pip are idempotent-ish).
set -euo pipefail

# ------------------------- CONFIG (edit these) -------------------------
# Path-based ("prefix") env, mirroring the local ./env convention. Put it on
# $GROUP_HOME or $OAK — NOT $HOME (quota). Example:
ENV_PREFIX="${ENV_PREFIX:-$GROUP_HOME/envs/sensi}"
# Repo location on Sherlock (where you cloned/rsynced this project):
REPO="${REPO:-$GROUP_HOME/code/SensiExtractor}"
# Where MFA stores its acoustic/dictionary models:
export MFA_ROOT_DIR="${MFA_ROOT_DIR:-$OAK/sensi/mfa_data}"
# -----------------------------------------------------------------------

echo "[setup] ENV_PREFIX = $ENV_PREFIX"
echo "[setup] REPO       = $REPO"
echo "[setup] MFA_ROOT   = $MFA_ROOT_DIR"

command -v conda >/dev/null 2>&1 || {
  echo "[setup] ERROR: 'conda' not found. Install miniforge under \$GROUP_HOME or"
  echo "        load Sherlock's conda module first (check: ml spider miniconda)." >&2
  exit 1
}

# 1. conda layer: python + MFA (Kaldi comes in as a dependency)
if [ ! -d "$ENV_PREFIX" ]; then
  conda create -y -p "$ENV_PREFIX" -c conda-forge python=3.10 montreal-forced-aligner
else
  echo "[setup] env already exists at $ENV_PREFIX — skipping create"
fi

# Activate the prefix env for the pip/spacy/mfa steps
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"

# 2. pip layer: everything else the code imports (single source of truth = requirements.txt)
pip install -r "$REPO/requirements.txt"

# 3. spaCy English model
python -m spacy download en_core_web_sm

# 4. MFA acoustic + dictionary models (downloaded into $MFA_ROOT_DIR)
mkdir -p "$MFA_ROOT_DIR"
mfa model download acoustic   english_us_arpa
mfa model download dictionary english_us_arpa

echo "[setup] DONE. Smoke test (text-only, no GPU needed):"
echo "    conda activate $ENV_PREFIX"
echo "    python $REPO/main.py --transcript <one.txt> --no-align --output /tmp/smoke.csv"
