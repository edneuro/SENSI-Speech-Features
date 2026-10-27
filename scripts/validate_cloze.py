"""One-shot validation of the configured LLM-cloze model against Provo Corpus.

Run after install, from the project root:
    python scripts/validate_cloze.py

Downloads Provo Corpus (~few MB) into data/provo/ on first run, computes
LLM_Cloze for each Provo word using whichever LLM the pipeline auto-selects
(or --llm-model override), and reports Pearson r against human cloze norms.

Exits 0 if correlation falls within the expected range for the model, else 1.
"""

import argparse
import csv
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import pearsonr

# This script lives in scripts/; put the project root on sys.path so `sensi`
# imports resolve when run as `python scripts/validate_cloze.py` (which otherwise
# only puts scripts/ on the path, not the root).
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sensi.extractors.predictability import add_predictability, select_model
from sensi.extractors.predictability_models import slug_for_hf_id

PROVO_CSV_URL = 'https://osf.io/download/eyfxq/'  # Provo Corpus item-level cloze data
PROVO_DIR = _ROOT / 'data' / 'provo'
PROVO_CSV = PROVO_DIR / 'provo_predictability.csv'

EXPECTED_RANGES: dict[str, tuple[float, float]] = {
    'gpt-4o-mini':                (0.78, 0.86),
    'gpt-4o':                     (0.80, 0.88),
    'meta-llama/Meta-Llama-3-70B': (0.80, 0.88),
    'meta-llama/Meta-Llama-3-8B':  (0.74, 0.83),
    'meta-llama/Llama-3.2-3B':     (0.70, 0.80),
}


def _ensure_provo(provo_csv_path: Path):
    provo_csv_path.parent.mkdir(parents=True, exist_ok=True)
    if provo_csv_path.exists():
        return
    import urllib.request
    print(f'Downloading Provo Corpus to {provo_csv_path}…')
    urllib.request.urlretrieve(PROVO_CSV_URL, provo_csv_path)


def main():
    parser = argparse.ArgumentParser(
        description='Validate LLM-cloze model against Provo Corpus human cloze norms.'
    )
    parser.add_argument('--llm-model', default=None,
                        help='LLM model ID to use (default: auto-select).')
    parser.add_argument('--provo-csv', default=str(PROVO_CSV),
                        help='Path to a local Provo predictability CSV (skip auto-download).')
    args = parser.parse_args()

    provo_csv_path = Path(args.provo_csv)
    _ensure_provo(provo_csv_path)

    df = pd.read_csv(provo_csv_path)
    # Provo columns include: Text_ID, Word_Number, Word, Word_Cloze (modal-prediction proportion)
    # Build per-paragraph token sequences and track Word_Cloze per word
    grouped = df.groupby('Text_ID')

    model_id = args.llm_model or select_model()
    slug = slug_for_hf_id(model_id)
    if slug is None:
        default_id = select_model()
        default_slug = slug_for_hf_id(default_id)
        print(f'Model {model_id!r} is not in the local roster (sensi/extractors/'
              f'predictability_models.py); falling back to the auto-selected '
              f'default {default_id!r}.')
        model_id, slug = default_id, default_slug
    print(f'Validation model: {model_id} (roster slug: {slug})')

    all_human, all_model = [], []
    for text_id, group in grouped:
        words = list(group['Word'].astype(str))
        text = ' '.join(words)
        token_df = pd.DataFrame({
            'token': words,
            'upos': ['NOUN'] * len(words),  # treat all as content for validation
        })
        scored = add_predictability(token_df, text, models=[slug], metrics=['cloze'])
        all_model.extend(scored[f'{slug}_Cloze'].tolist())
        all_human.extend(group['Word_Cloze'].tolist())

    pairs = [(h, m) for h, m in zip(all_human, all_model) if h is not None and m is not None and not (pd.isna(h) or pd.isna(m))]
    h_arr = [p[0] for p in pairs]
    m_arr = [p[1] for p in pairs]
    # scipy's PearsonRResult carries .statistic at runtime (verified against
    # the pinned scipy 1.15.2), but its stubs declare neither the attribute
    # nor a concrete element type for the legacy tuple-unpack.
    r = float(pearsonr(h_arr, m_arr).statistic)  # pyright: ignore[reportAttributeAccessIssue]

    lo, hi = EXPECTED_RANGES.get(model_id, (0.6, 0.95))
    status = 'PASS ✓' if lo <= r <= hi else 'FAIL ✗'
    print()
    print('Validation against Provo Corpus')
    print(f'Model:  {model_id}')
    print(f'N words tested: {len(pairs):,}')
    print(f'Pearson r vs human cloze: {r:.3f}')
    print(f'Expected range for this model: {lo:.2f}–{hi:.2f}')
    print(status)

    sys.exit(0 if lo <= r <= hi else 1)


if __name__ == '__main__':
    main()
