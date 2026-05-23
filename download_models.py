"""
One-time setup: download MFA acoustic models into ./mfa_data/

Run this from inside the activated conda environment:
    conda activate .\env          (Windows)
    conda activate ./env          (Mac)
    python download_models.py
"""

import os
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).parent.resolve()
mfa_dir = _HERE / 'mfa_data'
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

print('\n--- Word2Vec (manual download required) ---')
print(
    'Google News 300d is ~1.7 GB compressed (~3.6 GB on disk).\n'
    'Automated download is not available. To use Word2Vec embeddings:\n'
    '  1. Download: https://code.google.com/archive/p/word2vec/\n'
    '     File: GoogleNews-vectors-negative300.bin.gz\n'
    '  2. Decompress: gunzip GoogleNews-vectors-negative300.bin.gz\n'
    '  3. Place at: hf_cache/word2vec/GoogleNews-vectors-negative300.bin\n'
    '  (BERT embeddings work without this file.)'
)
