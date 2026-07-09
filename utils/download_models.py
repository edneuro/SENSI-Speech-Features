"""
One-time setup: download MFA acoustic models into ./mfa_data/

Run this from the project root inside the activated conda environment:
    conda activate .\env          (Windows)
    conda activate ./env          (Mac)
    python utils/download_models.py
"""

import os
import subprocess
import sys
from pathlib import Path

# This script lives in utils/; the project root (where mfa_data/ lives) is one up.
_ROOT = Path(__file__).resolve().parent.parent
mfa_dir = _ROOT / 'mfa_data'
os.environ['MFA_ROOT_DIR'] = str(mfa_dir)

print(f'Downloading MFA models to: {mfa_dir}')

for model_type, model_name in [
    ('acoustic',    'english_us_arpa'),
    ('dictionary',  'english_us_arpa'),
]:
    print(f'  mfa model download {model_type} {model_name} ...')
    result = subprocess.run(
        ['mfa', 'model', 'download', model_type, model_name],
        text=True,
    )
    if result.returncode != 0:
        print(f'ERROR: failed to download {model_type} model.')
        sys.exit(1)

print('\nDone. MFA models are ready.')

print(
    '\n--- Word2Vec ---\n'
    'Google News 300d (~1.7 GB) downloads automatically on first use via gensim.\n'
    'No manual step needed. The model is cached in ~/gensim-data/ after the\n'
    'first download. BERT embeddings work independently and need no download.'
)
