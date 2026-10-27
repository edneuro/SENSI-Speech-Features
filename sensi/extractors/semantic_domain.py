"""Semantic-domain category indicators with pluggable dictionary backends.

Two backends share the same interface:
    - Empath: open-source, pip-installed. No license required. The default
              (and only auto-selected) backend.
    - LIWC: DISABLED as a shipped/auto-selected backend. The LIWC-22 EULA
            (https://www.liwc.app/help/eula) forbids using LIWC dictionary
            material outside LIWC's own software. `_LiwcScorer` is retained
            so a license holder can still opt in explicitly (semantic.backend:
            liwc in features.yaml, with their own .dic in data/dictionaries/), but `detect_backend` never
            selects it automatically.

Backend selection: `detect_backend` always returns Empath. A stray .dic in
data/dictionaries/ no longer activates LIWC.

Output columns added to the token DataFrame:
    Sem_<Dict>_<category> ∈ {0, 1, NaN}     (one binary column per category)
    Sem_Categories                          (comma-separated list of matched categories
                                             for each token, "" if no match, NaN if function word)

POS filter: only NOUN, VERB, ADJ, ADV, PROPN (CONTENT_POS) are scored. Other tokens get NaN.
"""

from pathlib import Path
from typing import Iterable

from sensi.paths import DICT_DIR

import numpy as np
import pandas as pd


CONTENT_POS = {'NOUN', 'VERB', 'ADJ', 'ADV', 'PROPN'}

_DICTIONARIES_DIR = DICT_DIR


class BackendNotAvailable(Exception):
    pass


def detect_backend(dictionaries_dir: Path | None = None) -> tuple[str, Path | None]:
    """Return (backend_name, dict_path). Always Empath.

    LIWC is no longer auto-selected: the LIWC-22 EULA
    (https://www.liwc.app/help/eula) forbids using LIWC dictionary material
    outside LIWC's own software, so SensiExtractor does not ship or default
    to it. The `_LiwcScorer` path is retained only for a license holder who
    explicitly selects it (semantic.backend: liwc, with their own `.dic` in
    data/dictionaries/; handled by `sensi.pipeline._resolve_sem_backend`, not here). `dictionaries_dir` is accepted
    for backward compatibility but a stray `.dic` no longer activates LIWC.
    """
    return 'empath', None


def add_semantic_domain(
    token_df: pd.DataFrame,
    backend: str,
    categories: Iterable[str],
    dictionary_path: Path | str | None = None,
) -> pd.DataFrame:
    """Add Sem_<Dict>_<category> columns + Sem_Categories roll-up.

    Args:
        token_df: must contain 'lemma' and 'upos' columns.
        backend:  'liwc' or 'empath'.
        categories: list of category names to score.
        dictionary_path: required for LIWC; ignored for Empath.
    """
    backend = backend.lower()
    categories = list(categories)
    df = token_df.copy()

    if backend == 'liwc':
        scorer = _LiwcScorer(dictionary_path)
        dict_label = 'LIWC'
    elif backend == 'empath':
        scorer = _EmpathScorer()
        dict_label = 'Empath'
    else:
        raise BackendNotAvailable(f'Unknown backend: {backend}')

    # Score each token
    matches_per_row: list[set[str] | None] = []
    for _, row in df.iterrows():
        if row['upos'] not in CONTENT_POS:
            matches_per_row.append(None)  # sentinel for NaN
            continue
        matches = scorer.categories_for(row['lemma'], categories)
        matches_per_row.append(matches)

    # Emit one column per requested category
    for cat in categories:
        col = f'Sem_{dict_label}_{cat}'
        df[col] = [
            np.nan if m is None else (1 if cat in m else 0)
            for m in matches_per_row
        ]

    df['Sem_Categories'] = [
        np.nan if m is None else (','.join(sorted(m)) if m else '')
        for m in matches_per_row
    ]
    return df


# --------------------- backends ---------------------

class _EmpathScorer:
    def __init__(self):
        from empath import Empath  # imported here so it's only required when used
        self._lex = Empath()

    def categories_for(self, lemma: str, requested: list[str]) -> set[str]:
        out: set[str] = set()
        if not isinstance(lemma, str) or not lemma.strip():
            return out
        for cat in requested:
            if cat in self._lex.cats and lemma.lower() in self._lex.cats[cat]:
                out.add(cat)
        return out


class _LiwcScorer:
    def __init__(self, dictionary_path: Path | str | None):
        if dictionary_path is None:
            raise BackendNotAvailable('LIWC backend requires a .dic file in data/dictionaries/')
        path = Path(dictionary_path)
        if not path.exists():
            raise BackendNotAvailable(f'LIWC dictionary not found: {path}')
        try:
            import liwc
        except ImportError as exc:
            raise BackendNotAvailable(f'`liwc` package not installed: {exc}')
        parse, category_names = liwc.load_token_parser(str(path))
        self._parse = parse
        self._all_categories = set(category_names)

    def categories_for(self, lemma: str, requested: list[str]) -> set[str]:
        if not isinstance(lemma, str) or not lemma.strip():
            return set()
        matched = set(self._parse(lemma.lower()))
        return matched & set(requested)
