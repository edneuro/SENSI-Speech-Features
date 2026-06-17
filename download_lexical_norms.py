"""Download and build the lexical-norm lookup tables for the lexical-norms
feature (`extractors/lexical_norms.py`).

Run once from the project root:

    python download_lexical_norms.py

It fetches the published norm sets from public mirrors and writes the parquet
files SensiExtractor expects into `dictionaries/` (plus a `.csv` mirror of each
for human-readable inspection — the extractor uses the parquet):

    dictionaries/aoa_kuperman.parquet          word, aoa_mean, aoa_sd
    dictionaries/concreteness_brysbaert.parquet word, concreteness
    dictionaries/warriner_vad.parquet          word, valence, arousal, dominance
    dictionaries/mrc.parquet                   word, imageability, familiarity

Each source is independent; if one URL is unreachable the others still build.

These norms are free for research use. URLs point at stable public mirrors;
the authoritative sources and column details are in dictionaries/README.md.
"""

from pathlib import Path

import numpy as np
import pandas as pd

_DICT_DIR = Path(__file__).resolve().parent / "dictionaries"


def _clean_mrc(df: pd.DataFrame) -> pd.DataFrame:
    """In MRC, a `0` means 'not rated'. Convert those to NaN and drop rows that
    have neither an imageability nor a familiarity rating (most of MRC's ~150k
    words), leaving only the ~9k words that carry a real rating."""
    for col in ("imageability", "familiarity"):
        df[col] = df[col].replace(0, np.nan)
    return df.dropna(subset=["imageability", "familiarity"], how="all")


# Each entry: (output filename, source URL, read kwargs, {src_col: out_col}, clean_fn|None)
_SOURCES = [
    (
        "aoa_kuperman.parquet",
        "https://raw.githubusercontent.com/wehlutyk/brainscopypaste/master/data/AoA/Kuperman-BRM-data-2012.csv",
        {},
        {"Word": "word", "Rating.Mean": "aoa_mean", "Rating.SD": "aoa_sd"},
        None,
    ),
    (
        "concreteness_brysbaert.parquet",
        "https://raw.githubusercontent.com/ArtsEngine/concreteness/master/Concreteness_ratings_Brysbaert_et_al_BRM.txt",
        {"sep": "\t"},
        {"Word": "word", "Conc.M": "concreteness"},
        None,
    ),
    (
        "warriner_vad.parquet",
        "https://raw.githubusercontent.com/JULIELab/XANEW/master/Ratings_Warriner_et_al.csv",
        {},
        {"Word": "word", "V.Mean.Sum": "valence",
         "A.Mean.Sum": "arousal", "D.Mean.Sum": "dominance"},
        None,
    ),
    (
        # Parsed MRC mirror. Most of MRC's ~150k words have 0 (= "not rated")
        # for imageability/familiarity; _clean_mrc drops those and blanks the
        # zeros, leaving only words with a real rating.
        "mrc.parquet",
        "https://huggingface.co/datasets/StephanAkkerman/MRC-psycholinguistic-database/resolve/main/mrc_psycholinguistic_database.csv",
        {},
        {"Word": "word", "Imageability": "imageability", "Familiarity": "familiarity"},
        _clean_mrc,
    ),
]


def _build_one(filename: str, url: str, read_kwargs: dict, colmap: dict, clean) -> None:
    out_path = _DICT_DIR / filename
    print(f"[norms] {filename} <- {url}")
    try:
        raw = pd.read_csv(url, **read_kwargs)
    except Exception as exc:
        print(f"        SKIPPED - could not download/read: {exc}")
        return
    missing = [c for c in colmap if c not in raw.columns]
    if missing:
        print(f"        SKIPPED - source columns changed; missing {missing}. "
              f"Found: {list(raw.columns)[:10]}")
        return
    df = raw[list(colmap)].rename(columns=colmap)
    df["word"] = df["word"].astype(str).str.lower().str.strip()
    df = df[df["word"] != ""]
    # Clean BEFORE de-duplicating so a rated entry isn't shadowed by an
    # unrated same-word duplicate (MRC stores multiple POS rows per word).
    if clean is not None:
        df = clean(df)
    df = df.drop_duplicates(subset="word", keep="first")
    _DICT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    # Human-readable mirror next to the parquet (the extractor uses the
    # parquet; the CSV is just for eyeballing the contents).
    csv_path = out_path.with_suffix(".csv")
    df.to_csv(csv_path, index=False)
    print(f"        wrote {out_path.name} + {csv_path.name}  ({len(df):,} rows)")


def main() -> None:
    for filename, url, read_kwargs, colmap, clean in _SOURCES:
        _build_one(filename, url, read_kwargs, colmap, clean)
    print("\nDone.")


if __name__ == "__main__":
    main()
