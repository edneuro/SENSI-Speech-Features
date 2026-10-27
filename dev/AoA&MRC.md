# Extend the speech feature extractor with AoA and MRC psycholinguistic features

## Context
This project extracts word- and document-level features from speech stimuli for
EEG analysis (TRF-style modeling). Word frequency is already implemented via
SUBTLEX-US. Before writing anything, review the existing extractor and the
SUBTLEX implementation, and match its conventions (lemmatization, tokenization,
table schema, config, tests). Reuse the existing token/lemma pipeline — do NOT
introduce a second one.

## Norm data sources (consume local files; make paths configurable)
- AoA: Kuperman, Stadthagen-Gonzalez & Brysbaert (2012), ~30k English words.
- MRC Psycholinguistic Database (samzhang111 parser or the raw dict file).
- (Optional, higher coverage — wire in if files are present, else skip cleanly:
  Brysbaert et al. 2014 concreteness ~40k; Glasgow Norms ~5.5k.)
If a norm file is missing, log a warning and continue with that feature null —
never hard-fail the whole extractor.

## Word-level features (time-resolved; one row per token, aligned to word onset)
From Kuperman:
- aoa_mean        — mean age of acquisition (years)
- aoa_sd          — rating dispersion (carry it; useful as a quality flag/weight)
- aoa_matched     — bool, whether the token matched (after lemma fallback)

From MRC (only these — skip MRC frequency and MRC AoA):
- mrc_conc        — concreteness
- mrc_imag        — imageability
- mrc_fam         — familiarity
- mrc_nphon       — number of phonemes
- mrc_nsyl        — number of syllables
- mrc_nlet        — number of letters
- mrc_wtype       — word class (also derive content_vs_function as a clean flag)

## Document-level features (one row per passage; aggregates of the above)
- aoa_mean_doc, aoa_median_doc, aoa_max_doc, aoa_sd_doc
- prop_late_acquired  — fraction of content words with AoA > 10
- aoa_coverage        — fraction of tokens that matched a norm
(Compute these over content words; document the content/function rule used.)

## Things to keep in mind (implement defensively)
- OOV handling: lemmatize, fall back to lemma value, then record the match flag.
  Do NOT impute 0 or the mean for unmatched tokens — leave them null and let
  downstream code decide. Silent imputation corrupts regressors.
- MRC gotcha: in MRC, a field value of 0 means "missing", not a real zero.
  Convert 0 → null on import for all MRC numeric fields.
- MRC scaling gotcha: CONC/IMAG/FAM are stored multiplied by 100 — rescale.
- Concreteness and imageability correlate ~0.8; keep both columns but add a doc
  note warning against putting both in one model without residualizing/PCA.
- Phoneme/syllable/letter counts are correlates of word duration — label them in
  the docs as acoustic-length CONTROL regressors, not effects of interest.
- Report coverage stats (% matched per feature) to stdout and to the docs.

## Documentation to add/update
- A data dictionary: every feature, its source + citation, units, scale, and
  coverage, and whether it's word-level (TRF regressor) or document-level
  (passage covariate).
- A "modeling notes" section covering: the concreteness/imageability collinearity
  warning; that frequency, AoA, length and concreteness are intercorrelated
  (suggest checking VIF before trusting any single TRF weight); and the
  content-vs-function distinction.
- A "relative AoA" subsection describing a DERIVED ANALYSIS-TIME variable
  (not a stored feature): relative_aoa = word_AoA − child_age, in years.
  Explain it indexes a word's difficulty relative to the listener's level;
  that it's computed by joining word AoA (stimulus) with participant age
  (metadata) at analysis time; suggested uses (AoA × age interaction term,
  above/below-age word contrast, document-level proportion-above-age covariate);
  and the caveat that AoA is an adult-rated population norm, not a measure of the
  individual child's vocabulary.

## Deliverables
- Extended extractor producing the columns above for word- and document-level.
- Unit tests for: lemma fallback, MRC 0→null, MRC ×100 rescaling, coverage
  reporting, and a couple of known-word value checks.
- Updated docs as specified.
Keep word-level and document-level outputs as separate tables/files.