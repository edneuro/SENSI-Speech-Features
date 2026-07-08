@echo off
echo ============================================================
echo  SensiExtractor - First-time setup
echo ============================================================
echo.
echo Everything will be downloaded into this folder:
echo   .\env\        Python environment (~2 GB)
echo   .\mfa_data\   MFA acoustic models (~1 GB)
echo   .\hf_cache\   GPT-2 weights (~500 MB, on first run)
echo.
echo Run this from Anaconda Prompt. Press Ctrl+C to cancel.
echo.
pause

echo.
echo [1/3] Creating conda environment in .\env ...
rem Force CPU builds during the solve. Otherwise conda detects the GPU and pulls
rem the CUDA build of Kaldi (MFA's engine) — ~3 GB of NVIDIA/MAGMA libraries that
rem forced alignment never uses (it is CPU-only). Faking no-GPU picks CPU Kaldi.
set CONDA_OVERRIDE_CUDA=
call conda env create --prefix .\env -f environment.yml
if errorlevel 1 (
    echo.
    echo ERROR: conda env create failed.
    echo Make sure you are running this from Anaconda Prompt.
    pause
    exit /b 1
)

echo.
echo [2/3] Downloading MFA models into .\mfa_data ...
call conda run --prefix .\env python download_models.py

echo.
echo [3/3] Downloading spaCy model into .\env ...
call conda run --prefix .\env python -m spacy download en_core_web_sm

echo.
echo ============================================================
echo  Setup complete!
echo.
echo  To run the app:
echo    1. Open Anaconda Prompt
echo    2. cd to this folder
echo    3. conda activate .\env
echo    4. python gui.py
echo ============================================================
pause
