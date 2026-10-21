"""Word2Vec embedding extractor (Google News 300d via gensim).

If the model binary is not found at the default path, it is downloaded
automatically from the gensim model repository on first use (~1.7 GB,
cached by gensim in ~/gensim-data/ for subsequent runs).

Optional local override: place GoogleNews-vectors-negative300.bin at
hf_cache/word2vec/GoogleNews-vectors-negative300.bin to skip the download.

PUNCT and OOV rows receive a zero vector and is_valid=False.
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
    cached = Path(gensim_dl.base_dir, "word2vec-google-news-300").exists()
    print(
        "[word2vec] Loading Google News 300d from gensim's cache…" if cached
        else "[word2vec] Model not found locally — downloading Google News "
             "300d (~1.7 GB, one-time). Subsequent runs use gensim's local cache.",
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
) -> tuple[np.ndarray, np.ndarray]:
    """Return (matrix, is_valid).

    matrix:   (n_words, 300) float32, row-aligned with word_df.
    is_valid: (n_words,) bool. True iff the row is a non-PUNCT token that was
              found in vocab (exact or lowercase). PUNCT and OOV rows are zero
              vectors with is_valid False.
    """
    mp = Path(model_path) if model_path is not None else _DEFAULT_MODEL_PATH
    model = _load_model(mp)

    tokens = word_df["token"].tolist()
    if "upos" in word_df.columns:
        upos = word_df["upos"].tolist()
    else:
        upos = [None] * len(tokens)  # None != "PUNCT", so treated as content words

    n = len(tokens)
    mat = np.zeros((n, 300), dtype=np.float32)
    is_valid = np.zeros(n, dtype=bool)
    for i, (tok, pos) in enumerate(zip(tokens, upos)):
        if pos == "PUNCT":
            # PUNCT: forced zero, is_valid stays False (consistent with BERT's PUNCT exclusion)
            continue
        if tok in model:
            mat[i] = model[tok]
            is_valid[i] = True
        elif tok.lower() in model:
            mat[i] = model[tok.lower()]
            is_valid[i] = True
        # else OOV: row stays zero, is_valid stays False
    return mat, is_valid
