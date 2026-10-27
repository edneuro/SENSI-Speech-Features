"""
One-time setup: download MFA acoustic models into ./data/mfa_data/

Run this from the project root inside the activated conda environment:
    conda activate .\\env          (Windows)
    conda activate ./env          (Mac)
    python scripts/download_models.py
"""

import os
import subprocess
import sys
from pathlib import Path

# This script lives in scripts/; the project root is one up.
_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    sys.path.insert(0, str(_ROOT))
    from sensi.utils.local_caches import move_legacy_model_dirs, use_project_caches
    move_legacy_model_dirs()
    use_project_caches()
    mfa_dir = Path(os.environ['MFA_ROOT_DIR'])

    print(f'Downloading MFA models to: {mfa_dir}')

    for model_type, model_name in [
        ('acoustic',    'english_us_arpa'),
        ('dictionary',  'english_us_arpa'),
        # g2p generates pronunciations for out-of-dictionary words (~11 MB).
        # Only used when the alignment.g2p_oov pref is on, but vendored here so
        # the toggle works offline for collaborators.
        ('g2p',         'english_us_arpa'),
    ]:
        print(f'  mfa model download {model_type} {model_name} ...')
        result = subprocess.run(
            ['mfa', 'model', 'download', model_type, model_name],
            text=True,
        )
        if result.returncode != 0:
            print(f'ERROR: failed to download {model_type} model.')
            return 1

    print('\nDone. MFA models are ready.')

    print(
        '\n--- Word2Vec ---\n'
        'Google News 300d (~1.7 GB) downloads automatically on first use via gensim.\n'
        'No manual step needed. The model is cached in data/gensim_data/ after the\n'
        'first download. BERT embeddings work independently and need no download.'
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
