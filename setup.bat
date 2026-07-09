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
echo [1/4] Creating conda environment in .\env ...
rem CPU Kaldi is pinned in environment.yml (kaldi=*=cpu_*), which keeps the
rem ~3 GB CUDA/MAGMA stack out of the env — MFA alignment is CPU-only.
call conda env create --prefix .\env -f environment.yml
if errorlevel 1 (
    echo.
    echo ERROR: conda env create failed.
    echo Make sure you are running this from Anaconda Prompt.
    pause
    exit /b 1
)

echo.
echo [2/4] Installing Python dependencies from requirements.txt ...
rem environment.yml only bootstraps conda-level deps (Kaldi/MFA/python/pip);
rem the full Python dependency set lives in requirements.txt.
call conda run --prefix .\env pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo ERROR: pip install -r requirements.txt failed.
    pause
    exit /b 1
)

echo.
echo [3/4] Downloading MFA models into .\mfa_data ...
call conda run --prefix .\env python utils\download_models.py

echo.
echo [4/4] Downloading spaCy model into .\env ...
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
