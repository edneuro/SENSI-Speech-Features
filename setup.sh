#!/bin/bash
# SensiExtractor - First-time setup (macOS / Linux)
# Run from inside the SensiExtractor folder:  bash setup.sh
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
echo "[1/4] Installing conda-lock (deterministic installer) ..."
conda install -n base -c conda-forge conda-lock -y

echo ""
echo "[2/4] Creating ./env from conda-lock.yml (exact, pinned versions) ..."
# conda-lock.yml pins every package for this platform. CPU Kaldi is pinned in
# environment.yml; the lock captures it exactly.
conda-lock install -p ./env conda-lock.yml

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
