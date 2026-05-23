"""Word2Vec embedding extractor (Google News 300d via gensim).

If the model binary is not found at the default path, it is downloaded
automatically from the gensim model repository on first use (~1.7 GB,
cached by gensim in ~/gensim-data/ for subsequent runs).

Optional local override: place GoogleNews-vectors-negative300.bin at
hf_cache/word2vec/GoogleNews-vectors-negative300.bin to skip the download.

OOV words receive a zero vector (300 float32 zeros).
"""
from __future__ import annotations

import sys
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
    try:
        from gensim.models import KeyedVectors  # type: ignore
        import gensim.downloader as gensim_dl   # type: ignore
    except ImportError:
        raise Word2VecUnavailable(
            "gensim is not installed. Run: pip install gensim"
        )
    p = Path(model_path)
    if p.exists():
        return KeyedVectors.load_word2vec_format(str(p), binary=True)
    # Auto-download on first use; gensim caches in ~/gensim-data/.
    print(
        "[word2vec] Model not found locally — downloading Google News 300d "
        "(~1.7 GB, one-time). Subsequent runs use gensim's local cache.",
        file=sys.stderr,
    )
    try:
        return gensim_dl.load("word2vec-google-news-300")
    except Exception as exc:
        raise Word2VecUnavailable(
            f"Word2Vec auto-download failed: {exc}. "
            "Check your internet connection or place "
            "GoogleNews-vectors-negative300.bin in hf_cache/word2vec/ manually."
        ) from exc


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
