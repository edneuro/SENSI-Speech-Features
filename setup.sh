#!/bin/bash
# SensiExtractor - First-time setup (macOS / Linux)
# Run from inside the SensiExtractor folder:
#   bash setup.sh

set -e

echo "============================================================"
echo " SensiExtractor - First-time setup"
echo "============================================================"
echo ""
echo "Everything will be downloaded into this folder:"
echo "  ./env/        Python environment (~2 GB)"
echo "  ./mfa_data/   MFA acoustic models (~1 GB)"
echo "  ./hf_cache/   GPT-2 weights (~500 MB, on first run)"
echo ""
read -p "Press Enter to continue (Ctrl+C to cancel)..."

echo ""
echo "[1/4] Creating conda environment in ./env ..."
# CPU Kaldi is pinned in environment.yml (kaldi=*=cpu_*), which keeps the
# ~3 GB CUDA/MAGMA stack out of the env — MFA alignment is CPU-only.
conda env create --prefix ./env -f environment.yml

echo ""
echo "[2/4] Installing Python dependencies from requirements.txt ..."
# environment.yml only bootstraps conda-level deps (Kaldi/MFA/python/pip);
# the full Python dependency set lives in requirements.txt.
conda run --prefix ./env pip install -r requirements.txt

echo ""
echo "[3/4] Downloading MFA models into ./mfa_data ..."
conda run --prefix ./env python utils/download_models.py

echo ""
echo "[4/4] Downloading spaCy model into ./env ..."
conda run --prefix ./env python -m spacy download en_core_web_sm

echo ""
echo "============================================================"
echo " Setup complete!"
echo ""
echo " To run the app:"
echo "   conda activate ./env"
echo "   python gui.py"
echo "============================================================"
