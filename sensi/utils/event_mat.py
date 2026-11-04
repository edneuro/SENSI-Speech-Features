"""Pure builders for the per-stimulus event .mat files (word / syllable).

One entry per spoken word (or syllable) with onset/offset/duration/token and
every kept numeric feature, for stem plots in MATLAB. No I/O here: the
functions return a dict of variables that the caller hands to
scipy.io.savemat(..., oned_as="column"). Spec:
dev/superpowers/specs/2026-11-03-event-mat-design.md section 3.
"""
from __future__ import annotations

import re
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from sensi.extractors.lexical_norms import ALL_OUTPUT_COLUMNS

# MATLAB identifier rule; names that break it raise instead of being renamed.
VALID_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,62}$")

_SURPRISAL_RE = re.compile(r"^[a-z0-9_]+_Surprisal$")
_ID_COLS = {"token_id", "sentence_id", "doc_id", "paragraph_id",
            "token_in_sentence_id", "head_token_id", "start", "end"}
_STRING_COLS = {"token", "lemma", "upos", "xpos", "feats", "dep_rel",
                "sentence", "Sem_Categories", "Transcript"}
_LOOKUP_COLS = {"Zipf_Freq"} | set(ALL_OUTPUT_COLUMNS)

# Already covered by onset/offset/duration or positional ids not repeated.
_WORD_EXCLUDE = {"Onset", "End", "Word_Duration", "doc_id", "paragraph_id",
                 "token_in_sentence_id", "head_token_id", "start", "end"}
_SYLL_EXCLUDE = {"Onset", "End", "Duration", "P_Center", "doc_id",
                 "token_in_sentence_id", "ARPABET", "IPA", "Label", "token"}


def merge_rule(column: str) -> str | None:
    """How a column combines when several tokens form one spoken word.

    Returns "sum" / "product" / "all" / "any" / "first" / "lookup", or None
    for an unmapped column (a test fails on that, so a new feature cannot
    silently fall into "first row"). `End` (last row's value) and
    `Word_Duration` (recomputed End - Onset) are reported as "first" and
    handled specially in merge_spoken_words.
    """
    if _SURPRISAL_RE.match(column):
        return "sum"
    if column.endswith("_Cloze"):
        return "product"
    if column.endswith("_TopMatch"):
        return "all"
    if column == "Sem_Categories":
        return "first"
    if column.startswith("Sem_"):
        return "any"
    if column in _LOOKUP_COLS or column.startswith("Edu_Zipf_"):
        return "lookup"
    if (column in _ID_COLS or column in _STRING_COLS
            or column in ("Onset", "End", "Word_Duration", "Word_Gap", "Word_Gap_After")
            or column.startswith(("Pitch_", "Volume_"))):
        return "first"
    return None


def _num(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)


def _combine(rule: str | None, values: pd.Series):
    if rule in (None, "first"):
        return values.iloc[0]
    v = _num(values)
    if np.isnan(v).any():          # a missing part makes the whole unknown
        return np.nan
    if rule == "sum":
        return float(v.sum())
    if rule == "product":
        return float(v.prod())
    if rule == "all":
        return float(bool(np.all(v != 0)))
    if rule == "any":
        return float(bool(np.any(v != 0)))
    return values.iloc[0]


def merge_spoken_words(df: pd.DataFrame,
                       relookup: Callable[[str], dict[str, float]]) -> pd.DataFrame:
    """Merge consecutive non-PUNCT rows sharing (Onset, End, Transcript).

    Groups of one row pass through unchanged. Rows with NaN Onset or PUNCT
    are never merged. A merged row's `token` is the original text span (the
    group's Transcript when no sentence offsets are available).
    """
    need = {"Onset", "End", "Transcript"}
    if df.empty or not need.issubset(df.columns):
        return df.copy()
    upos = df["upos"] if "upos" in df.columns else pd.Series("", index=df.index)

    groups: list[list[int]] = []
    prev = None
    for i, (on, en, tr, up) in enumerate(zip(df["Onset"], df["End"],
                                             df["Transcript"], upos)):
        mergeable = not (pd.isna(on) or pd.isna(tr) or up == "PUNCT")
        key = (on, en, tr) if mergeable else None
        if key is not None and key == prev:
            groups[-1].append(i)
        else:
            groups.append([i])
        prev = key

    rows = []
    for g in groups:
        if len(g) == 1:
            rows.append(df.iloc[g[0]].to_dict())
            continue
        part = df.iloc[g]
        word = part["Transcript"].iloc[0]
        looked = relookup(word) or {}
        shown = word            # start/end are 1-indexed inclusive
        if {"sentence", "start", "end"}.issubset(df.columns):
            try:
                shown = str(part["sentence"].iloc[0])[
                    int(part["start"].iloc[0]) - 1:int(part["end"].iloc[-1])]
            except (TypeError, ValueError):
                shown = word
        row = {}
        for col in df.columns:
            if col == "token":
                row[col] = shown
            elif col == "End":
                row[col] = part[col].iloc[-1]
            elif col == "Word_Duration":
                row[col] = float(part["End"].iloc[-1] - part["Onset"].iloc[0])
            elif merge_rule(col) == "lookup":
                row[col] = float(looked.get(col, np.nan))
            else:
                row[col] = _combine(merge_rule(col), part[col])
        rows.append(row)
    return pd.DataFrame(rows, columns=df.columns)


def _cellstr(items) -> np.ndarray:
    arr = np.empty(len(items), dtype=object)
    for i, s in enumerate(items):
        arr[i] = str(s)
    return arr


def _feature_columns(df, keep, exclude) -> list[str]:
    keep = set(keep)
    out = []
    for col in df.columns:
        if col not in keep or col in exclude:
            continue
        src = df[col]
        if col in _STRING_COLS:
            continue
        num = pd.to_numeric(src, errors="coerce")
        if num.isna().all() and src.notna().any():
            continue                      # string column
        out.append(col)
    return out


def _check_names(names: Iterable[str]) -> None:
    for n in names:
        if not VALID_NAME.match(n):
            raise ValueError(f"'{n}' is not a valid MATLAB variable name")


def _finish(vars_: dict, cols: list[str], df: pd.DataFrame,
            units: dict[str, str]) -> dict:
    _check_names(cols)
    for c in cols:
        vars_[c] = _num(df[c])
    vars_["columns"] = _cellstr(cols)
    vars_["units"] = _cellstr([units.get(c, "") for c in cols])
    return vars_


def word_event_vars(df: pd.DataFrame, keep_columns: Iterable[str], text: str,
                    units: dict[str, str],
                    relookup: Callable[[str], dict[str, float]]) -> dict | None:
    """Variables for word/<stim>_word_events.mat, or None if nothing aligned."""
    if "Onset" not in df.columns or "End" not in df.columns:
        return None
    keep = list(keep_columns)
    d = df
    if "upos" in d.columns:
        d = d[d["upos"] != "PUNCT"]
    d = d[d["Onset"].notna() & d["End"].notna()].reset_index(drop=True)
    if d.empty:
        return None
    d = merge_spoken_words(d, relookup)

    onset, offset = _num(d["Onset"]), _num(d["End"])
    tok = d["token"] if "token" in d.columns else d.get("Transcript")
    vars_ = {
        "text": text,
        "token": _cellstr(list(tok)),
        "onset": onset,
        "offset": offset,
        "duration": offset - onset,
    }
    for idc in ("sentence_id", "token_id"):
        if idc in d.columns:
            vars_[idc] = _num(d[idc])
    cols = [c for c in _feature_columns(d, keep, _WORD_EXCLUDE)
            if c not in vars_]
    return _finish(vars_, cols, d, units)


def syllable_event_vars(syll_df: pd.DataFrame, keep_columns: Iterable[str],
                        text: str, units: dict[str, str]) -> dict | None:
    """Variables for syllable/<stim>_syllable_events.mat, or None if empty."""
    if "Onset" not in syll_df.columns or "End" not in syll_df.columns:
        return None
    keep = list(keep_columns)
    d = syll_df[syll_df["Onset"].notna() & syll_df["End"].notna()
                ].reset_index(drop=True)
    if d.empty:
        return None
    onset, offset = _num(d["Onset"]), _num(d["End"])
    vars_ = {
        "text": text,
        "label": _cellstr(list(d["Label"])) if "Label" in d.columns
        else _cellstr([""] * len(d)),
        "token": _cellstr(list(d["token"])) if "token" in d.columns
        else _cellstr([""] * len(d)),
        "onset": onset,
        "offset": offset,
        "duration": _num(d["Duration"]) if "Duration" in d.columns
        else offset - onset,
    }
    if "P_Center" in keep and "P_Center" in d.columns:
        vars_["pcenter"] = _num(d["P_Center"])
    for idc in ("sentence_id", "syllable_id"):
        if idc in d.columns:
            vars_[idc] = _num(d[idc])
    cols = [c for c in _feature_columns(d, keep, _SYLL_EXCLUDE)
            if c not in vars_]
    return _finish(vars_, cols, d, units)
