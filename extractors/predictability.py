"""Unified multi-model predictability extractor (surprisal / cloze / top-match).

Replaces extractors/surprisal.py and the old extractors/llm_cloze.py. Each
selected roster model is scored once; only the requested metrics are written,
prefixed by the model slug. See the design spec §4.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from extractors.predictability_models import spec_for

_METRIC_COLS = {"surprisal": "Surprisal", "cloze": "Cloze", "topmatch": "TopMatch"}

def add_predictability(df, text, *, models, metrics, backends=None):
    metrics = [m for m in metrics if m in _METRIC_COLS]
    if not models or not metrics:
        return df.copy()
    for slug in models:
        spec_for(slug)          # validate; KeyError on unknown
    out = df.copy()
    upos = out["upos"].tolist()
    tokens = [str(t) for t in out["token"].tolist()]
    for slug in models:
        backend = (backends or {}).get(slug) or _get_backend(slug)
        scores = backend.score(text, tokens)
        _write_metric_columns(out, slug, metrics, upos, scores)
    return out

def _write_metric_columns(out, slug, metrics, upos, scores):
    cols = {m: [] for m in metrics}
    for pos_tag, score in zip(upos, scores):
        for m in metrics:
            if pos_tag == "PUNCT":
                cols[m].append(np.nan); continue
            if m == "surprisal":
                cols[m].append(round(score["surprisal"], 4)
                               if score["surprisal"] == score["surprisal"] else np.nan)
            elif m == "cloze":
                cols[m].append(round(score["p"], 6)
                               if score["p"] == score["p"] else np.nan)
            else:  # topmatch
                tm = score["top_match"]
                cols[m].append(np.nan if tm is None else int(tm))
    for m in metrics:
        out[f"{slug}_{_METRIC_COLS[m]}"] = cols[m]

def _get_backend(slug):   # real backend wired in Task 2b
    raise NotImplementedError("real backend added in Task 2b")
