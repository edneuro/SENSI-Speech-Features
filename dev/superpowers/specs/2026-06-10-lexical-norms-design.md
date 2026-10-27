# Lexical norms batch — AoA, Concreteness, Imageability, Familiarity

**Date:** 2026-06-10
**Status:** spec finalized (no code yet) — ready to implement on go-ahead

## Why
Per-word psycholinguistic norms are staple naturalistic-EEG regressors and
are currently missing. The literature is already cited in our docs
(`sem_indicators.md` → Barber 2013 concreteness/N400; `edu_zipf.md` →
Kuperman 2012 AoA). This batch adds them as **lookup features**, reusing
the exact pattern SUBTLEX (`word_freq.py`) and the now-disabled EWFG
(`educator_freq.py`) already use: a lemma-keyed table join with
NaN-on-miss.

Origin: the user proposal in `docs/AoA&MRC.md`. This spec narrows it after
review (below).

## Scope decisions (from review of `docs/AoA&MRC.md`)
**Add** (these are genuinely new):
- **AoA** — Kuperman, Stadthagen-Gonzalez & Brysbaert (2012), ~30k words.
- **Concreteness** — **Brysbaert et al. (2014), ~40k words** as the primary
  source (≈5× the coverage of MRC concreteness ~8k).
- **Imageability, Familiarity** — MRC Psycholinguistic Database (these are
  MRC's unique contribution; Brysbaert has neither).
- **Valence, Arousal, Dominance** — Warriner, Kuperman & Brysbaert (2013),
  ~14k words (decision #3 below). Same join; high-value emotion-ERP
  regressors.

**Drop from the original proposal** (already covered, better, by existing
machinery — do **not** re-import from MRC):
- `mrc_wtype` / content-vs-function → already from spaCy `upos` + the
  pipeline's existing `CONTENT_POS` set; spaCy's POS is contextual.
- `mrc_nphon` → the per-stimulus phoneme sidecar already gives the *actual
  pronounced* phoneme count.
- `mrc_nsyl` → the syllable sidecar / `pyphen` already give syllable count.
- `mrc_nlet` → trivially `len(token)`.

## Columns

### Word level → word CSV (TitleCase, matching `Zipf_Freq` / `Surprisal`)
| Column | Source | Units / scale |
|---|---|---|
| `AoA_Mean` | Kuperman | years |
| `AoA_SD` | Kuperman | years (rating dispersion; quality/weight) |
| `Concreteness` | Brysbaert 2014 | 1–5 |
| `Imageability` | MRC | MRC scale (see scaling gotcha) |
| `Familiarity` | MRC | MRC scale |
| `Valence` | Warriner 2013 | 1–9 |
| `Arousal` | Warriner 2013 | 1–9 |
| `Dominance` | Warriner 2013 | 1–9 |

Unmatched tokens → `NaN` (never imputed). `NaN` already signals "no match",
so per-row match-flag columns are **omitted** (decision #2); match rate is
reported per document as a coverage stat instead.

### Document level → `documents_summary` (snake_case, matching `mean_zipf_freq`)
Computed over **content words** (`CONTENT_POS`), documenting the rule:
- `mean_aoa`, `median_aoa`, `max_aoa`, `sd_aoa`
- `prop_late_acquired` — fraction of content words with
  `AoA_Mean > late_acquired_threshold` (default 10 yrs, configurable —
  decision #4)
- `mean_concreteness`, `mean_imageability`, `mean_familiarity`
- `mean_valence`, `mean_arousal`, `mean_dominance`
- coverage stats (fraction of content tokens that matched each norm):
  `aoa_coverage`, `concreteness_coverage`, `imageability_coverage`,
  `familiarity_coverage`, `valence_coverage`, `arousal_coverage`,
  `dominance_coverage` — also printed to stdout.

## Sources & files — user-supplied, never distributed (decision #1)
Same policy as EWFG: **SensiExtractor ships no norm data.** Each source is a
user-supplied file at a **configurable local path** under `dictionaries/`
(house-use; collaborators build/obtain their own):
`aoa_kuperman.parquet`, `concreteness_brysbaert.parquet`, `mrc.parquet`
(imageability + familiarity), `warriner_vad.parquet`. If a file is absent:
log a warning, emit that source's columns as all-`NaN`, and continue —
never hard-fail (mirrors `educator_freq.py`). This sidesteps redistribution
licensing entirely. A `dictionaries/README.md` documents the expected
schema for each table so users can build their own.

## Wiring (one edit each, per the registry pattern)
- New extractor `extractors/lexical_norms.py` — `add_lexical_norms(df, *,
  sources=..., paths=...)`; lemma-keyed `reindex` lookups (vectorized, like
  `educator_freq`). Lemma fallback: try surface token, then `lemma.lower()`.
- `gui/planner/registry.py` — Stages `aoa_norms`, `concreteness_norms`,
  `mrc_norms` (or one `lexical_norms` stage producing all); word-level
  cluster(s) + a doc-aggregate cluster. Reuse the `doc_aggregates` marker
  pattern for the means.
- `utils/preferences.py` — additive bump (vNN): per-column word toggles +
  a `master_lexical_norms`, doc-aggregate output toggles. (Mirrors how the
  prosody/embeddings batches added keys.)
- `gui/section_word_features.py` — a "Lexical norms" card (word level);
  `gui/section_doc_features.py` — aggregate toggles in the Aggregates card.
- `utils/doc_summary.py` — doc aggregates + coverage stats.
- Per-feature docs: `docs/features/lexical_norms_aoa.md`,
  `lexical_norms_concreteness.md`, etc. (or one shared doc via
  `_SHARED_DOC_MAP`), authored to the usual standard with citations.

## Defensive details (carry over from the proposal — all agreed)
- **No imputation** for OOV — leave `NaN`, let downstream decide.
- **MRC `0` = missing** → convert `0 → NaN` on import for MRC numeric fields.
- **MRC scaling** — verify the stored scale against the chosen MRC parser
  before assuming a rescale; lock whatever it is with a known-value test
  (the proposal's "÷100" is treated as *to-verify*, not fact).
- **Coverage reporting** to stdout + into the docs.

## Docs: data dictionary + modeling notes
- Data dictionary: every column, source + citation, units/scale, coverage,
  word- vs doc-level.
- Modeling notes: concreteness↔imageability collinearity (~0.8 — warn
  against both in one model without residualizing/PCA); frequency/AoA/
  length/concreteness are intercorrelated (suggest checking VIF);
  content-vs-function distinction.
- **Relative AoA** documented as an **analysis-time derived variable, not a
  stored column**: `relative_aoa = AoA_Mean − child_age`. SensiExtractor
  never sees participant age, so this lives only as guidance (join stimulus
  AoA with participant metadata downstream; uses: AoA×age interaction,
  above/below-age contrast, doc-level proportion-above-age; caveat: AoA is
  an adult-rated population norm, not the individual child's vocabulary).

## Tests (TDD on the extractor math)
Lemma fallback; MRC `0→NaN`; MRC scaling (locked to verified value);
no-impute on OOV; coverage computation; a couple of known-word value
checks (e.g. a concrete vs. abstract word); missing-file → all-NaN + warning.

## Resolved decisions
1. **No distribution of norm data.** User-supplied files at configurable
   paths; warn + all-`NaN` if absent (house-use, BYO-dictionary — same as
   EWFG). No licensed data in the repo.
2. **No per-token match-flag columns.** `NaN` signals "no match"; match rate
   is surfaced as per-document coverage stats.
3. **Include Warriner VAD now** (`Valence`, `Arousal`, `Dominance` + doc
   means/coverage) — same join, high value.
4. **`prop_late_acquired` default AoA > 10 yrs, configurable** via a pref
   (`lexical_norms.late_acquired_threshold`), documented as a convention.
