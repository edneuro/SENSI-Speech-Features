"""Per-word psycholinguistic norm lookups (AoA, concreteness, imageability,
familiarity, valence/arousal/dominance).

Adds these columns to a token DataFrame:
    AoA_Mean, AoA_SD            (Kuperman et al. 2012)
    Concreteness               (Brysbaert et al. 2014)
    Imageability, Familiarity  (MRC Psycholinguistic Database)
    Valence, Arousal, Dominance (Warriner et al. 2013)

Norm data is **never shipped** with SensiExtractor (house-use / BYO-files —
see the data-distribution policy). Each source is a user-supplied parquet at
a configurable local path under ``dictionaries/``. If a source's file is
missing, that source's columns are emitted as NaN and a warning is logged;
the run never hard-fails.

Lookup is keyed on ``lemma.lower()`` with a fallback to the surface
``token.lower()``; PUNCT rows and out-of-vocabulary words get NaN (never
imputed). The expected parquet schema for each source is documented in
``dictionaries/README.md``.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_DICT_DIR = Path(__file__).resolve().parent.parent / 'dictionaries'

# source -> (default filename, {lookup_column: Output_Column})
_SOURCES: dict[str, tuple[str, dict[str, str]]] = {
    'aoa':          ('aoa_kuperman.parquet',
                     {'aoa_mean': 'AoA_Mean', 'aoa_sd': 'AoA_SD'}),
    'concreteness': ('concreteness_brysbaert.parquet',
                     {'concreteness': 'Concreteness'}),
    'mrc':          ('mrc.parquet',
                     {'imageability': 'Imageability', 'familiarity': 'Familiarity'}),
    'warriner':     ('warriner_vad.parquet',
                     {'valence': 'Valence', 'arousal': 'Arousal',
                      'dominance': 'Dominance'}),
}

# MRC fields where a stored 0 means "missing", not a real rating.
_MRC_ZERO_AS_MISSING = {'imageability', 'familiarity'}

ALL_OUTPUT_COLUMNS: list[str] = [
    out for _, colmap in _SOURCES.values() for out in colmap.values()
]

# Cache by path; value is the indexed lookup DataFrame or None (missing/failed).
_lookup_cache: dict[Path, pd.DataFrame | None] = {}


def add_lexical_norms(
    token_df: pd.DataFrame,
    *,
    paths: dict[str, str | Path] | None = None,
) -> pd.DataFrame:
    """Add the lexical-norm columns to *token_df*.

    Args:
        token_df: must have ``token``, ``lemma``, ``upos`` columns.
        paths: optional ``{source: path}`` overrides (sources: ``aoa``,
            ``concreteness``, ``mrc``, ``warriner``). Unspecified sources use
            their default ``dictionaries/<file>.parquet`` path.
    """
    df = token_df.copy()
    paths = {k: Path(v) for k, v in (paths or {}).items()}

    upos = df['upos'] if 'upos' in df.columns else pd.Series([''] * len(df), index=df.index)
    is_word = upos.ne('PUNCT')
    lemma_keys = df['lemma'].astype(str).str.lower().where(is_word)
    token_col = df['token'] if 'token' in df.columns else df['lemma']
    token_keys = token_col.astype(str).str.lower().where(is_word)

    for source, (default_name, colmap) in _SOURCES.items():
        path = paths.get(source) or (_DICT_DIR / default_name)
        lookup = _load_lookup(path, source)
        for lookup_col, out_col in colmap.items():
            if lookup is None or lookup_col not in lookup.columns:
                df[out_col] = np.nan
            else:
                df[out_col] = _matched(lookup[lookup_col], lemma_keys, token_keys)
    return df


def _matched(col: pd.Series, lemma_keys: pd.Series, token_keys: pd.Series) -> np.ndarray:
    """Look up `col` by lemma, falling back to surface token where lemma misses."""
    primary = col.reindex(lemma_keys.to_numpy()).to_numpy(dtype=float)
    fallback = col.reindex(token_keys.to_numpy()).to_numpy(dtype=float)
    return np.where(np.isnan(primary), fallback, primary)


def _load_lookup(path: Path, source: str) -> pd.DataFrame | None:
    if path in _lookup_cache:
        return _lookup_cache[path]
    if not path.exists():
        logger.warning(
            'Lexical-norms lookup for %r not found at %s - its columns will be '
            'NaN. Build the norm tables with download_lexical_norms.py (or see '
            'dictionaries/README.md for sources and the expected schema).',
            source, path,
        )
        _lookup_cache[path] = None
        return None
    try:
        df = pd.read_parquet(path).set_index('word')
    except Exception as exc:
        logger.warning('Failed to load lexical-norms lookup %s: %s', path, exc)
        _lookup_cache[path] = None
        return None
    # Drop duplicate word keys (reindex requires a unique index).
    df = df[~df.index.duplicated(keep='first')]
    # MRC gotcha: a stored 0 means "missing".
    if source == 'mrc':
        for c in _MRC_ZERO_AS_MISSING:
            if c in df.columns:
                df[c] = df[c].replace(0, np.nan)
    _lookup_cache[path] = df
    return df
