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

MRC (imageability, familiarity) is NOT auto-built — see dictionaries/README.md
(it needs the MRC database / a parser). Each source is independent; if one URL
is unreachable the others still build.

These norms are free for research use. URLs point at stable public mirrors;
the authoritative sources and column details are in dictionaries/README.md.
"""

from pathlib import Path

import pandas as pd

_DICT_DIR = Path(__file__).resolve().parent / "dictionaries"

# Each entry: (output filename, source URL, pandas read kwargs, {src_col: out_col})
_SOURCES = [
    (
        "aoa_kuperman.parquet",
        "https://raw.githubusercontent.com/wehlutyk/brainscopypaste/master/data/AoA/Kuperman-BRM-data-2012.csv",
        {},
        {"Word": "word", "Rating.Mean": "aoa_mean", "Rating.SD": "aoa_sd"},
    ),
    (
        "concreteness_brysbaert.parquet",
        "https://raw.githubusercontent.com/ArtsEngine/concreteness/master/Concreteness_ratings_Brysbaert_et_al_BRM.txt",
        {"sep": "\t"},
        {"Word": "word", "Conc.M": "concreteness"},
    ),
    (
        "warriner_vad.parquet",
        "https://raw.githubusercontent.com/JULIELab/XANEW/master/Ratings_Warriner_et_al.csv",
        {},
        {"Word": "word", "V.Mean.Sum": "valence",
         "A.Mean.Sum": "arousal", "D.Mean.Sum": "dominance"},
    ),
]


def _build_one(filename: str, url: str, read_kwargs: dict, colmap: dict) -> None:
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
    df = df[df["word"] != ""].drop_duplicates(subset="word", keep="first")
    _DICT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    # Human-readable mirror next to the parquet (the extractor uses the
    # parquet; the CSV is just for eyeballing the contents).
    csv_path = out_path.with_suffix(".csv")
    df.to_csv(csv_path, index=False)
    print(f"        wrote {out_path.name} + {csv_path.name}  ({len(df):,} rows)")


def main() -> None:
    for filename, url, read_kwargs, colmap in _SOURCES:
        _build_one(filename, url, read_kwargs, colmap)
    print("\nDone. MRC (imageability/familiarity) is not auto-built - see "
          "dictionaries/README.md.")


if __name__ == "__main__":
    main()
