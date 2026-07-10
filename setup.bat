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
echo [1/4] Installing conda-lock (deterministic installer) ...
call conda install -n base -c conda-forge conda-lock -y
if errorlevel 1 (
    echo.
    echo ERROR: conda install -n base -c conda-forge conda-lock failed.
    echo Make sure you are running this from Anaconda Prompt.
    pause
    exit /b 1
)

echo.
echo [2/4] Creating .\env from conda-lock.yml (exact, pinned versions) ...
rem conda-lock.yml pins every package for this platform. CPU Kaldi is pinned in
rem environment.yml; the lock captures it exactly.
call conda-lock install -p .\env conda-lock.yml
if errorlevel 1 (
    echo.
    echo ERROR: conda-lock install failed.
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
