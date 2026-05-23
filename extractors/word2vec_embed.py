"""Word2Vec embedding extractor (Google News 300d via gensim).

Expected model: hf_cache/word2vec/GoogleNews-vectors-negative300.bin
Download from: https://code.google.com/archive/p/word2vec/
               (GoogleNews-vectors-negative300.bin.gz, ~1.7 GB compressed)

OOV words receive a zero vector (300 float32 zeros).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any

_DEFAULT_MODEL_PATH = (
    Path(__file__).parent.parent / "hf_cache" / "word2vec"
    / "GoogleNews-vectors-negative300.bin"
)


class Word2VecUnavailable(RuntimeError):
    """Raised when gensim is missing or the model file is not found."""


def _load_model(model_path: str | Path) -> Any:
    p = Path(model_path)
    if not p.exists():
        raise Word2VecUnavailable(
            f"Word2Vec model not found at {p}. "
            "Download GoogleNews-vectors-negative300.bin from "
            "https://code.google.com/archive/p/word2vec/ and place it in "
            "hf_cache/word2vec/"
        )
    try:
        from gensim.models import KeyedVectors  # type: ignore
    except ImportError:
        raise Word2VecUnavailable(
            "gensim is not installed. Run: pip install gensim"
        )
    return KeyedVectors.load_word2vec_format(str(p), binary=True)


def embed_word2vec(
    word_df: pd.DataFrame,
    *,
    model_path: str | Path | None = None,
) -> np.ndarray:
    """Return (n_words, 300) float32 matrix.

    Row i corresponds to word_df.iloc[i]. OOV rows are zero vectors.
    """
    mp = Path(model_path) if model_path is not None else _DEFAULT_MODEL_PATH
    model = _load_model(mp)

    tokens = word_df["token"].tolist()
    n = len(tokens)
    mat = np.zeros((n, 300), dtype=np.float32)
    for i, tok in enumerate(tokens):
        if tok in model:
            mat[i] = model[tok]
        elif tok.lower() in model:
            mat[i] = model[tok.lower()]
    return mat
