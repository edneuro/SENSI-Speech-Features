# `data/dictionaries/` — lexical-norm lookup tables

The lexical-norms feature (`sensi/extractors/lexical_norms.py`) reads per-word norm
tables from this folder as **parquet** files. If a file is absent, its columns
are emitted as `NaN` with a warning — the run still completes.

## Easiest way: run the build script
From the project root:

```bash
python scripts/download_lexical_norms.py
```

This downloads all four published norm sets and writes
`aoa_kuperman.parquet`, `concreteness_brysbaert.parquet`,
`warriner_vad.parquet`, and `mrc.parquet` here with the right schema. A
matching `.csv` mirror is written next to each parquet for easy reading (the
extractor uses the parquet; the CSV is just for eyeballing — parquet is a
compressed binary format).

## Sources, files, and schemas
Every table has a lowercased `word` column (citation form / lemma) plus the
value columns. Lookup is by `lemma.lower()`, falling back to `token.lower()`.

### AoA → `aoa_kuperman.parquet` (`word`, `aoa_mean`, `aoa_sd`)
Kuperman, Stadthagen-Gonzalez & Brysbaert (2012), ~30k words.
- Download (CSV): <https://raw.githubusercontent.com/wehlutyk/brainscopypaste/master/data/AoA/Kuperman-BRM-data-2012.csv>
- Authoritative copy (OSF): <https://osf.io/d7x6q/>
- Build mapping: `Word → word`, `Rating.Mean → aoa_mean`, `Rating.SD → aoa_sd`.

### Concreteness → `concreteness_brysbaert.parquet` (`word`, `concreteness`)
Brysbaert, Warriner & Kuperman (2014), ~40k words, scale 1–5.
- Download (tab-separated `.txt`): <https://raw.githubusercontent.com/ArtsEngine/concreteness/master/Concreteness_ratings_Brysbaert_et_al_BRM.txt>
- Build mapping: `Word → word`, `Conc.M → concreteness` (the file is
  **tab-delimited**; `Bigram=1` rows are two-word expressions).

### Valence / Arousal / Dominance → `warriner_vad.parquet` (`word`, `valence`, `arousal`, `dominance`)
Warriner, Kuperman & Brysbaert (2013), ~14k words, scale 1–9.
- Download (CSV): <https://raw.githubusercontent.com/JULIELab/XANEW/master/Ratings_Warriner_et_al.csv>
- Build mapping: `Word → word`, `V.Mean.Sum → valence`,
  `A.Mean.Sum → arousal`, `D.Mean.Sum → dominance`.

### Imageability / Familiarity → `mrc.parquet` (`word`, `imageability`, `familiarity`)
MRC Psycholinguistic Database.
- Download (parsed CSV mirror): <https://huggingface.co/datasets/StephanAkkerman/MRC-psycholinguistic-database>
  (`mrc_psycholinguistic_database.csv`). Authoritative source: UWA MRC database
  <https://websites.psychology.uwa.edu.au/school/MRCDatabase/uwa_mrc.htm> (raw
  `mrc2.dct` is a fixed-width record format; the build script uses the parsed
  mirror instead).
- Build mapping: `Word → word`, `Imageability → imageability`,
  `Familiarity → familiarity`. **MRC `0` means "missing"** — the zeros are left
  as-is; the extractor converts a stored `0` in these fields to `NaN` on load.
  (MRC's native integer scale is 100–700; the extractor does not rescale.)

See [`docs/features/lexical_norms.md`](../../docs/features/lexical_norms.md) for
meaning, EEG rationale, and the full reference list.
