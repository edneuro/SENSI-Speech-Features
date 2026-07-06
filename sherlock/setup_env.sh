#!/usr/bin/env bash
# Build the SensiExtractor conda env on Sherlock.
#
# RUN ON AN sh_dev COMPUTE NODE, *not* a login node. Sherlock's docs warn that
# conda's huge file-count "will likely fail on a login node". Grab one with:
#     sh_dev -c 8 -m 32GB -t 02:00:00
# then run this script inside that session. (Needs outbound internet for the
# downloads; Sherlock's own advice to install conda on sh_dev implies compute
# nodes have it — verify. If not, download packages on a login node first.)
#
# WHY CONDA AT ALL: MFA/Kaldi is conda-forge-only (not pip-installable), so a
# plain venv can't build the full pipeline. Sherlock discourages conda, so:
#   - use MINIFORGE (conda-forge's minimal installer), NOT full Anaconda
#   - install it OUTSIDE $HOME (blows the quota) — under $GROUP_HOME / $OAK
# See sherlock/README.md "Conda on Sherlock" for the miniforge bootstrap.
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
# Repo root — auto-detected from this script's own location (sherlock/ sits one
# level below the repo root). Override by setting REPO explicitly if needed.
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(dirname "$_SCRIPT_DIR")}"
# Durable storage base: prefer $OAK if you have it, else fall back to $GROUP_HOME.
# (The ${OAK:-...} form is safe under `set -u`; a bare $OAK is not.)
STORAGE_BASE="${STORAGE_BASE:-${OAK:-$GROUP_HOME}}"
# Where MFA stores its acoustic/dictionary models:
export MFA_ROOT_DIR="${MFA_ROOT_DIR:-$STORAGE_BASE/sensi/mfa_data}"
# -----------------------------------------------------------------------

echo "[setup] ENV_PREFIX = $ENV_PREFIX"
echo "[setup] REPO       = $REPO"
echo "[setup] MFA_ROOT   = $MFA_ROOT_DIR"

command -v conda >/dev/null 2>&1 || {
  echo "[setup] ERROR: 'conda' not found. Bootstrap miniforge OUTSIDE \$HOME first," >&2
  echo "        on an sh_dev node (see sherlock/README.md 'Conda on Sherlock'):" >&2
  echo "          wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh" >&2
  echo "          bash Miniforge3-Linux-x86_64.sh -b -p \$GROUP_HOME/miniforge3" >&2
  echo "          source \$GROUP_HOME/miniforge3/etc/profile.d/conda.sh" >&2
  exit 1
}

# 1. conda layer: python + MFA (Kaldi comes in as a dependency)
if [ ! -d "$ENV_PREFIX" ]; then
  # `compilers` pulls a modern conda-forge gcc/g++ — Sherlock's system compiler
  # (gcc 4.8.5) is too old to build modern C/C++/Cython packages from source.
  conda create -y -p "$ENV_PREFIX" -c conda-forge python=3.10 montreal-forced-aligner compilers
else
  echo "[setup] env already exists at $ENV_PREFIX — skipping create"
fi

# Activate the prefix env for the pip/spacy/mfa steps
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"

# 1b. Compiled scientific packages from conda-forge (prebuilt binaries). pip
# otherwise tries to BUILD these from source and stalls/fails on Sherlock's old
# system compiler — so install them via conda first; pip then sees them as
# already-satisfied and skips them.
conda install -y -c conda-forge \
  pandas numpy scipy pyarrow gensim librosa soundfile praat-parselmouth

# 2. pip layer: everything else the code imports (single source of truth = requirements.txt)
# Modernize pip first so it prefers prebuilt wheels over compiling from source.
pip install --upgrade pip setuptools wheel
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
