"""Educator's Word Frequency Guide (EWFG) grade-band frequency lookup.

Adds five columns to a token DataFrame:
    Edu_Zipf_K2, Edu_Zipf_3to5, Edu_Zipf_6to8, Edu_Zipf_9to12, Edu_Zipf_Overall

Reads from a lookup parquet file at `dictionaries/edu_zipf.parquet` (default)
or a user-supplied path. The lookup must have columns:
    word, edu_zipf_k2, edu_zipf_3to5, edu_zipf_6to8, edu_zipf_9to12, edu_zipf_overall

If the lookup file is missing, all five columns are emitted as NaN and a
warning is logged. This keeps the schema stable and unblocks downstream
analysis even when EWFG data acquisition is pending.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

EXPECTED_COLUMNS = [
    'Edu_Zipf_K2', 'Edu_Zipf_3to5', 'Edu_Zipf_6to8', 'Edu_Zipf_9to12', 'Edu_Zipf_Overall',
]

_LOOKUP_COLUMNS = [
    'edu_zipf_k2', 'edu_zipf_3to5', 'edu_zipf_6to8', 'edu_zipf_9to12', 'edu_zipf_overall',
]

_DEFAULT_LOOKUP_PATH = Path(__file__).resolve().parent.parent / 'dictionaries' / 'edu_zipf.parquet'

_lookup_cache: dict[Path, pd.DataFrame] = {}

# Maps band key → output column name
_BAND_MAP: dict[str, str] = {
    'k2':     'Edu_Zipf_K2',
    '3to5':   'Edu_Zipf_3to5',
    '6to8':   'Edu_Zipf_6to8',
    '9to12':  'Edu_Zipf_9to12',
    'overall': 'Edu_Zipf_Overall',
}
_ALL_BANDS = list(_BAND_MAP.keys())


def add_edu_freq(
    token_df: pd.DataFrame,
    lookup_path: Path | str | None = None,
    bands: list[str] | None = None,
) -> pd.DataFrame:
    """Add EWFG grade-band frequency columns to *token_df*.

    Parameters
    ----------
    token_df:
        DataFrame with at least ``lemma`` and ``upos`` columns.
    lookup_path:
        Path to the EWFG parquet lookup.  Defaults to
        ``dictionaries/edu_zipf.parquet`` relative to the project root.
    bands:
        Which grade-band columns to emit.  Accepted keys:
        ``"k2"``, ``"3to5"``, ``"6to8"``, ``"9to12"``, ``"overall"``.
        ``None`` (default) emits all five columns — preserving backward
        compatibility.
    """
    df = token_df.copy()
    path = Path(lookup_path) if lookup_path else _DEFAULT_LOOKUP_PATH

    active_bands = _ALL_BANDS if bands is None else bands
    unknown = [b for b in active_bands if b not in _BAND_MAP]
    if unknown:
        raise ValueError(
            f"Unknown EWFG band(s) {unknown!r}. "
            f"Accepted keys: {', '.join(_ALL_BANDS)}."
        )
    active_cols = [_BAND_MAP[b] for b in active_bands]
    active_lookup_cols = [_LOOKUP_COLUMNS[EXPECTED_COLUMNS.index(c)] for c in active_cols]

    lookup = _load_lookup(path)
    if lookup is None:
        for col in active_cols:
            df[col] = np.nan
        return df

    # Lower-case lemmas for lookup; punctuation gets NaN
    keys = df.apply(
        lambda r: r['lemma'].lower() if r['upos'] != 'PUNCT' else None,
        axis=1,
    )
    matched = lookup.reindex(keys.fillna('__missing__'))

    for out_col, in_col in zip(active_cols, active_lookup_cols):
        df[out_col] = matched[in_col].values
    return df


def _load_lookup(path: Path) -> pd.DataFrame | None:
    if path in _lookup_cache:
        return _lookup_cache[path]
    if not path.exists():
        logger.warning(
            'EWFG lookup not found at %s — Edu_Zipf_* columns will be NaN. '
            'See docs/features/edu_zipf.md ("Data acquisition" section) '
            'for acquisition steps.',
            path,
        )
        return None
    try:
        df = pd.read_parquet(path).set_index('word')
    except Exception as exc:
        logger.warning('Failed to load EWFG lookup from %s: %s', path, exc)
        return None
    # Drop duplicate word keys (reindex requires a unique index) — mirrors
    # extractors/lexical_norms.py; user-supplied tables do contain dupes.
    df = df[~df.index.duplicated(keep='first')]
    _lookup_cache[path] = df
    return df
