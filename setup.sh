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
echo "[1/3] Creating conda environment in ./env ..."
# Force CPU builds during the solve. Otherwise conda detects the GPU and pulls
# the CUDA build of Kaldi (MFA's engine) — ~3 GB of NVIDIA/MAGMA libraries that
# forced alignment never uses (it is CPU-only). Faking no-GPU picks CPU Kaldi.
export CONDA_OVERRIDE_CUDA=""
conda env create --prefix ./env -f environment.yml

echo ""
echo "[2/3] Downloading MFA models into ./mfa_data ..."
conda run --prefix ./env python download_models.py

echo ""
echo "[3/3] Downloading spaCy model into ./env ..."
conda run --prefix ./env python -m spacy download en_core_web_sm

echo ""
echo "============================================================"
echo " Setup complete!"
echo ""
echo " To run the app:"
echo "   conda activate ./env"
echo "   python gui.py"
echo "============================================================"
