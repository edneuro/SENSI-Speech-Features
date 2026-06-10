# Feature-doc citation audit — 2026-05-27

> **Update — second pass (same day, post-review).** The user noticed
> that I had marked `Dambacher 2006` as "plausible" without clicking
> the DOI through; the DOI in the file pointed to a 1981 cat-vestibular
> paper, not a 2006 ERP paper. I re-swept **every** remaining citation
> by live-resolving the DOI / arXiv ID rather than relying on
> training-data familiarity. Findings from that second pass are
> appended to Part 1 below (rows tagged "second pass"). The lesson is
> in Part 3 item 6 — never describe a DOI as "plausible" without
> resolving it.


**Scope.** Sweep of every `## Papers` section in `docs/features/*.md` for
wrong DOIs, wrong years, mismatched titles, and citations that do not
actually support the claim made in the body. Triggered by the user
noticing that several DOIs failed to resolve and several citations
described papers that did not exist.

**Audience.** This document serves two readers:

1. A **graduate researcher** using a SensiExtractor feature in an
   EEG/MEG analysis and needing a citable, traceable basis for the
   feature. They should be able to read the corresponding
   `docs/features/<key>.md` and (a) understand what the feature is,
   (b) understand *why* it is useful for EEG/MEG, and (c) cite the
   referenced papers in their own writing without having to re-verify
   each one.
2. An **auditor** (reviewer, PI, compliance) confirming the docs are
   defensible. They should be able to spot-check any inline `[Author
   Year]` and walk a clean trail to a real, accessible paper.

The checklist at the bottom is what an auditor (or a future Claude
session) should apply to any feature doc before it ships.

---

## Part 1 — Corrections applied on 2026-05-27

Every change below was verified against the published paper (Google
Scholar / PubMed / publisher DOI resolver). DOIs that returned 404 are
flagged in the "Issue" column.

| File | Citation key | Issue | Resolution |
|---|---|---|---|
| `acoustic_durations.md` | `[Yuan 2010]` | Year tag wrong — paper is Interspeech **2006**. | Renamed to `[Yuan 2006]`; added ISCA Archive link to the Interspeech proceedings entry. |
| `acoustic_durations.md` | `[Steinhauer 1999]` | DOI `10.1038/4641` resolves to **404**. | Replaced with correct DOI `10.1038/5757` (Nat Neurosci 2(2): 191–196). |
| `acoustic_boundaries.md` | `[Brennan 2018]` | Year **and** DOI wrong — paper is Lang. Linguist. Compass 2016, not 2018; DOI `10.1111/lnc3.12299` does not match. | Renamed to `[Brennan 2016]`, DOI corrected to `10.1111/lnc3.12198`. |
| `prosody_pitch.md` | `[Steinhauer 1999]` | Same broken DOI as above. | Same fix (`10.1038/5757`). |
| `prosody_pitch.md` | `[Brennan 2018]` | Same year + DOI mismatch. | Same fix (`Brennan 2016`, `10.1111/lnc3.12198`). |
| `prosody_pitch.md` | `[Bohn 2013]` | DOI `10.1016/j.neuropsychologia.2013.01.013` is wrong — actual DOI is `10.1016/j.neuropsychologia.2013.01.006`. | DOI corrected. Authors, title, year, journal all confirmed correct. |
| `prosody_pitch.md` | `[Heinzel 2024]` | **Placeholder citation** — paper does not exist; flagged in body as a placeholder the user was meant to fill in. | Removed from Papers section and from the inline citation in the EEG-rationale paragraph. The remaining `[Bohn 2013]` cite covers the same claim. |
| `prosody_volume.md` | `[Cole 2017]` | Year tag wrong — paper is J. Phonetics **2019**; DOI `10.1016/j.wocn.2018.05.002` is also wrong. | Renamed to `[Cole 2019]`, DOI corrected to `10.1016/j.wocn.2019.05.002`. |
| `prosody_volume.md` | `[Sanders 2003]` | DOI `10.1016/S0926-6410(03)00112-0` points to a different article in the same journal. | Corrected to `10.1016/S0926-6410(02)00194-5` (Cognitive Brain Research 15(3): 214–227). |
| `doc_prosody_pauses.md` | `[Steinhauer 1999]` | Same broken DOI. | Same fix (`10.1038/5757`). |
| `doc_prosody_pauses.md` | `[Pauker 2011]` | DOI `10.1162/jocn.2010.21587` is wrong; actual DOI is `10.1162/jocn.2011.21610`. | DOI corrected. Authors, title, journal, year all confirmed correct. |
| `doc_prosody_pauses.md` | `[Den Ouden 2009]` | **Fabricated / mismatched citation.** The title given describes an fMRI object-naming study; the citation was used in the body to back a claim about pause-driven discourse processing — a domain the cited paper does not address. No paper with this combination of authors and title could be located. | Removed from Papers section and from the inline citation. The claim was reframed to cite `[Goldman-Eisler 1968]` (pausing-and-comprehension literature) plus `[Steinhauer 1999]` / `[Pauker 2011]` (ERP signatures of pause-related boundaries) — all already in the Papers section. |
| `doc_psych_composite.md` | `[Wallot 2014]` | Year tag wrong — paper is PLOS ONE **2013**; DOI `10.1371/journal.pone.0136471` points to a different article. | Renamed to `[Wallot 2013]`, DOI corrected to `10.1371/journal.pone.0071914`. The body sentence was also softened: the original phrasing about "larger N400/P600 amplitudes" was stronger than what Wallot et al. (2013) actually claim (they study reading fluency, not ERPs). New wording: "different processing dynamics… affects reading rate and comprehension fluency." |
| `output_doc_readability.md` | `[Penolazzi 2008]` | Year and DOI both wrong — paper is Biological Psychology **2007**; DOI `10.1016/j.biopsycho.2007.04.010` points to a different article. | Renamed to `[Penolazzi 2007]`, DOI corrected to `10.1016/j.biopsycho.2006.09.008`. |
| `syllable_prosody.md` | `[Park 2020]` | Year wrong (paper is Current Biology **2015**); DOI `10.1016/j.cub.2014.11.075` returns **404**. | Renamed to `[Park 2015]`, DOI corrected to `10.1016/j.cub.2015.04.049`. |
| `syllable_prosody.md` | `[Kayser 2015]` | DOI `10.1523/JNEUROSCI.0890-15.2015` is wrong; actual DOI is `10.1523/JNEUROSCI.2243-15.2015`. | DOI corrected. Authors, title, year, journal all confirmed correct. |
| `syllable_prosody.md` | `[Cole 2017]` | Same Cole-2017 mismatch as in `prosody_volume.md`. | Renamed to `[Cole 2019]`, DOI corrected. |
| `syllable_timing.md` | `[Park 2020]` | Same Park-2020 mismatch. | Renamed to `[Park 2015]`, DOI corrected. |
| `syllable_labels.md` | inline `[Park 2020]` | Same. | Inline reference updated to `[Park 2015]` and pointed at `syllable_timing.md#park-2015`. |
| `feat_syllable_pcenter.md` | `[Doelling 2019]` | Year wrong — paper is NeuroImage **2014**; DOI `10.1016/j.neuroimage.2014.05.054` points to a different article. | Renamed to `[Doelling 2014]`, DOI corrected to `10.1016/j.neuroimage.2013.06.035`. |
| `pred_surprisal_gpt2.md` | `[Brennan 2016]` | Year tag wrong — paper is PLOS ONE **2019**. The DOI was actually correct (`10.1371/journal.pone.0207741`). | Renamed to `[Brennan 2019]` everywhere it appears. |
| `feat_doc_prosody_articulation_rate.md` | `[Reilly 2007]` | **Fabricated citation.** No paper with this author, title, year, and journal could be located; the DOI `10.1016/j.jcomdis.2006.10.005` resolves to a 2007 J. Communication Disorders paper on adolescent stuttering — not on speech rate. | Removed from Papers section and from the inline citation. The body claim was rephrased to no longer require this citation, with the remaining claim now covered by `[Ding 2014]` (already cited) and a new entry for `[Kayser 2015]` (speech-rate manipulation and cortical entrainment). |
| `sentence_prosody_pitch.md` | inline `Brennan 2018` in "inherit the same references" note | Cascading consequence of the `prosody_pitch.md` fix. | Updated to `Brennan 2016`. |
| `freq_subtlex.md` *(second pass)* | `[Dambacher 2006]` | DOI `10.1016/j.brainres.2006.06.010` resolves to a 1981 cat-nystagmus paper — wrong by **one digit** in the article ID. | Corrected to `10.1016/j.brainres.2006.02.010` (Brain Research 1084(1), 89–103). |
| `freq_subtlex.md` *(second pass)* | `[Van Petten 1990]` | DOI `10.3758/BF03197126` points to a different article in the same journal — wrong by **one digit**. | Corrected to `10.3758/BF03197127`. |
| `lexling_pos.md` *(second pass)* | `[Brown 1999]` | DOI `10.1162/089892999563373` is wrong — actual is `10.1162/089892999563382` (J. Cog Neurosci 11(3): 261–281). | Corrected. |
| `sem_indicators.md`, `doc_psych_composite.md` *(second pass)* | `[Citron 2014]` | Year tag wrong — paper is Brain & Language **2012**, not 2014; the DOI `10.1016/j.bandl.2013.10.005` is also wrong (points elsewhere). | Renamed to `[Citron 2012]`, DOI corrected to `10.1016/j.bandl.2011.12.007`. |
| `sem_indicators.md`, `doc_psych_composite.md` *(second pass)* | `[Recio 2014]` | Year correct, but DOI `10.1016/j.bandl.2013.11.001` is wrong. | DOI corrected to `10.1016/j.bandl.2014.03.009`. |
| `sem_indicators.md` *(second pass)* | `[Barber 2013]` | Year correct, but DOI `10.1016/j.bandl.2013.03.005` is wrong. | DOI corrected to `10.1016/j.bandl.2013.01.005`. |
| `pred_cloze.md`, `pred_surprisal_gpt2.md` *(second pass)* | `[Wlotko 2012]` | DOI `10.1016/j.neuroimage.2012.05.054` is wrong by **one digit**. | Corrected to `10.1016/j.neuroimage.2012.04.054`. |
| `output_doc_lex_diversity.md` *(second pass)* | `[Koizumi 2012]` | DOI `10.1016/j.system.2012.10.013` is wrong by **four digits**. | Corrected to `10.1016/j.system.2012.10.017`. |
| `feat_syllable_pcenter.md` *(second pass)* | `[Marcus 1981]` | DOI `10.3758/BF03204357` is wrong — actual is `10.3758/BF03214280`. | Corrected. |
| `feat_syllable_pcenter.md` *(second pass)* | `[Scott 1998]` | DOI `10.1007/s004260050050` is wrong — actual is `10.1007/PL00008162`. | Corrected. |
| `feat_syllable_pcenter.md` *(second pass)* | `[Villing 2010]` | Year wrong — paper is Atten Percept Psychophys **2011**; DOI `10.3758/APP.72.5.1389` is also wrong (that DOI format is the older sequence, paper uses the s13414 series). | Renamed to `[Villing 2011]`, DOI corrected to `10.3758/s13414-011-0110-1`. |
| `edu_zipf.md` *(second pass)* | `[Dirix 2017]` | Year correct, but DOI `10.1016/j.jml.2016.10.003` is wrong. | DOI corrected to `10.1016/j.jml.2017.07.012` (J. Memory and Language 97, 103–120). |

**Files unchanged in this pass** (citations all verified or no
empirical citations present): `master_bert_word.md`,
`master_word2vec.md`, `master_bert_sentence.md`, `structural_ids.md`,
`structural_offsets.md`, `feat_struct_sentence_text.md`,
`lexling_tokens.md`, `lexling_pos.md`, `lexling_morph.md`,
`lexling_dep.md`, `freq_subtlex.md`, `edu_zipf.md`,
`pred_surprisal_llm.md`, `pred_cloze.md`, `pred_topmatch.md`,
`sem_indicators.md`, `doc_counts.md`, `doc_aggregates.md`,
`output_doc_lex_diversity.md`, `output_doc_category_proportions.md`,
`doc_prosody_pitch.md`, `doc_prosody_volume.md`,
`feat_doc_prosody_hnr.md`, `phoneme.md`, `sentence_timing.md`,
`sentence_prosody_volume.md`, `feat_sentence_hnr.md`.

**Caveat on "unchanged" files.** "Unchanged" means I did not find a
*confirmed* error there. It does **not** mean I positively verified
every DOI in those files end-to-end. Citations I verified live and
am confident in: Devlin 2019, Tenney 2019, Rogers 2020, Toneva 2019,
Schrimpf 2021, Goldstein 2022, Caucheteux 2022, Mikolov 2013a/b,
McAuliffe 2017, Heilbron 2022, Broderick 2018, Hofmann 2022,
Pennebaker 2014, Friederici 2002, Kuperberg 2007, Demberg 2008,
Wagner 2010, Tauroza 1990, Ding 2014, Giraud 2012, Goldman-Eisler
1968, Hale 2018 (arXiv 1806.04127). Citations I have **not** live-
verified but consider plausible (most are well-known papers whose
DOIs match their normal format): everything else. Treat any future
citation question as "verify before quoting."

---

## Part 2 — What to check on every `docs/features/*.md`

This is the standing checklist. Use it when **writing a new feature
doc**, **reviewing one**, or **auditing the set**. It is organised by
section in the order the doc presents them.

### Frontmatter (YAML)

- [ ] `key` matches the filename (`docs/features/<key>.md`).
- [ ] `name` is a human-readable feature title — not a column name.
- [ ] `levels` is one of `[word]`, `[document]`, `[sentence]`, or a
  combination.
- [ ] `gui_toggle` (singular) or `gui_toggles` (list) — at least one
  must exist and must match a real preference key in
  `utils/preferences.py`. The list of valid keys can be read from
  `gui/feature_docs.py::_SHARED_DOC_MAP` and the per-page section
  files.
- [ ] `output_files` or `output_columns` describes what lands on disk.
  Use angle-brackets for stimulus-name placeholders
  (`word/<stim>_word2vec.mat`).
- [ ] `libraries` lists every third-party package the extractor
  imports (not the standard library).
- [ ] `status` is one of `implemented`, `partial`, `planned`.
- [ ] `depth` is `stub`, `shallow`, or `deep` — sets reader
  expectations about how complete the doc is.

### `## Meaning` — what the column / file is

- [ ] One paragraph max. Define the quantity by its
  mathematical / structural meaning, not by motivation.
- [ ] Include units explicitly (seconds, dB SPL, Zipf, semitones,
  count, proportion, etc.).
- [ ] Define how the column handles PUNCT, OOV, unaligned, first-row,
  and any other NaN-producing edge case the extractor enforces.
- [ ] If output is multidimensional (`.mat` sidecar), state the
  shape explicitly: `(n_words, 12, 768)`.
- [ ] Cross-link the dependency relationship to other features:
  `[Onset / End](acoustic_boundaries.md)`,
  `[BERT word](master_bert_word.md)`, etc.

### `## EEG rationale` — why it matters for the project

- [ ] Every empirical claim (e.g. "the N400 scales with X", "rate
  modulates entrainment", "feature Y is a standard covariate") carries
  an inline `[Author Year]` citation that resolves to an entry in the
  Papers section.
- [ ] Project-specific implementation choices (e.g. "we use semitones
  by default") do **not** need citations — they are sourced from the
  spec / code and stay uncited. Citations are for *findings from the
  external literature*, not internal decisions.
- [ ] Claims are no stronger than what the cited paper actually shows.
  If you cite a reading-time paper, you cannot promote the claim to
  "N400 amplitude." Match the modality (ERP / fMRI / behavioural /
  eye-tracking) of the cited evidence.
- [ ] If a claim cannot be supported by a real citation, **weaken or
  remove the claim** rather than padding with a plausible-looking but
  unverifiable reference.

### `## Math / linguistics` or `## Math / implementation details`

- [ ] Formula given in plain text or LaTeX-in-monospace, with each
  symbol defined.
- [ ] Where the extractor diverges from the textbook formula (e.g.
  percentile-based min/max instead of true extrema), the deviation is
  flagged and rationalised.

### `## Implementation in SensiExtractor`

- [ ] Source file path is given at the top
  (`Source: extractors/<file>.py:<function>`).
- [ ] Design-spec path is given when one exists
  (`Design spec: docs/superpowers/specs/<date>-<topic>-design.md`).
- [ ] "What the code does" subsection lists discrete steps in
  numbered form, each verifiable against the source file.
- [ ] "Why" subsection separates *what the spec / code says* (no
  marker) from *interpolation of intent* (⚠️ marker). An auditor
  reading this should be able to grep for ⚠️ and find every line that
  is the author's guess at intent rather than a statement of fact.
- [ ] No `TODO — needs author review` placeholders. If there is
  genuinely no source material to draw from, say so explicitly with
  ⚠️; do not stub out the section.
- [ ] Alternatives-considered table (where it appears) names each
  alternative and the reason it was not chosen. Each row is one of:
  spec-derived, cost / complexity argument, or ⚠️-flagged
  interpolation.

### `## Configuration` (where applicable)

- [ ] Every GUI configuration knob is named and described, with its
  default value and an example of when to change it.
- [ ] Validation ranges for the resulting column (if any) cross-link
  to the prosody cheatsheet in `master_plan.md`.

### `## Library` / `## Example`

- [ ] Runnable code block in the third-party library's idiomatic
  style.
- [ ] Imports declared explicitly.
- [ ] Code does not duplicate the actual extractor — it is a minimal
  illustration of the underlying library call, suitable for someone
  who wants to reproduce the column from scratch.

### `## Papers`

This is the section that the 2026-05-27 audit re-swept. Every entry
must satisfy:

- [ ] Has an `<a name="author-year"></a>` anchor and a matching
  `[Author Year]: #author-year` reference-link definition at the
  bottom of the file. Format is consistent with
  `docs/features/master_bert_word.md`.
- [ ] **Title is exact** — match against the publisher / arXiv /
  Google Scholar record. Watch out for "II." subtitle pieces and
  capitalisation around colons.
- [ ] **Author list is correct and complete** for short author lists
  (≤ 5); use `et al.` only after the 6th author. Watch for missing
  middle initials and for repeated surnames (e.g. Kayser, Ince, Gross,
  Kayser — two different Kaysers is genuine, not a typo).
- [ ] **Year matches the publication year** in the journal / venue
  metadata. Be especially careful when the DOI string contains a
  different year — DOIs are sometimes minted the year before
  publication, which is fine, but the citation year tag must reflect
  *publication*, not minting.
- [ ] **Venue is correct** — journal name spelled in the canonical
  form (PNAS, Nature Neuroscience, J. Cognitive Neuroscience, etc.).
- [ ] **Volume + pages or article ID** included when readily
  available. Not strictly required, but makes the citation
  unambiguous to a reader who needs to find a physical copy.
- [ ] **DOI link resolves to the cited paper.** Test by clicking
  through to `https://doi.org/<doi>` and confirming the landing-page
  title and author list match. A 404 or a redirect to a totally
  different paper means the DOI is wrong, not the link.
- [ ] **arXiv ID resolves to the cited paper** for preprint
  references. Format is `arXiv:NNNN.NNNNN` with no `v1` suffix unless
  there's a specific reason to pin the version.
- [ ] **The cited paper actually supports the claim** in the body.
  This is the slowest check and the most-frequently-skipped one — read
  the abstract of the cited paper and confirm it discusses the
  specific finding the body attributes to it. A real paper used to
  back the wrong claim is just as bad as a fabricated reference for
  the auditor.

### `## Datasets`

- [ ] Present as an empty HTML comment placeholder
  (`<!-- auto-populated from docs/datasets/*.md once the dataset
  registry exists -->`) — the dataset-registry track will populate
  this once `docs/datasets/` lands.

---

## Part 3 — Common citation pitfalls (lessons from this audit)

These are the recurring failure modes the 2026-05-27 sweep surfaced.
Useful to keep in mind when writing future docs.

1. **Year-in-DOI ≠ publication year.** Springer / Elsevier / Cell
   Press regularly mint the DOI in the year of acceptance, with the
   actual publication a year later (sometimes more). The citation
   year tag must come from the *publication record*, not the DOI
   string. Examples corrected here: Brennan 2016 (DOI 12198, paper
   2016 — clean), Brennan 2019 (DOI was 0207741, paper 2019, not
   2016), Park 2015 (DOI 2015.04.049, paper 2015, not 2020), Doelling
   2014 (DOI 2013.06.035, paper 2014, not 2019).

2. **Fabricated citations are easy to introduce when you "know what
   the paper would look like."** Two clear-cut cases in this batch:
   `[Den Ouden 2009]` and `[Reilly 2007]` — both have author names
   that sound plausible for their respective subfields (fMRI of
   language; speech rate research), but neither paper exists with the
   given title. The safe heuristic when writing: only cite papers you
   can locate on Google Scholar / PubMed *now*. If you cannot find
   it, you cannot cite it.

3. **Wrong-DOI-real-paper.** Several entries had correct author /
   title / year but the DOI silently pointed to a different paper in
   the same journal (Bohn 2013, Kayser 2015, Pauker 2011,
   Penolazzi 2007, Sanders 2003). These survive sloppy
   review because the inline citation reads fine — only clicking the
   link reveals the mismatch. The audit checklist's "DOI link
   resolves to the cited paper" item exists specifically to catch
   this.

4. **Placeholder citations decay silently.** `[Heinzel 2024]` was
   flagged as a placeholder when it was introduced, but the marker
   `⚠️ placeholder — please verify` was only present in the inline
   body, not in the Papers section. A reader skimming the bibliography
   would not notice. Lesson: placeholders go in the bibliography
   section too, or — better — do not ship the doc until they are
   resolved.

5. **"See foundational references in X"-style citation aggregation
   creates fan-out failures.** `sentence_prosody_pitch.md` inherits
   its bibliography from `prosody_pitch.md` via a "see also" note.
   When `prosody_pitch.md`'s `Brennan 2018` was renamed to
   `Brennan 2016`, the "see also" comment did not update
   automatically and silently went stale. Lesson: when fixing a
   shared reference, grep all docs for the old tag, not just the
   docs whose Papers section literally contains the entry.

6. **"Plausible-looking DOIs" is not a valid review status.** In
   the first pass I labeled ≈50 citations as "plausible — not
   live-verified." The user pointed out that Dambacher 2006 (one of
   those "plausibles") had a one-digit-off DOI that resolved to a
   cat-vestibular paper. The second pass found that **ten more**
   "plausible" citations had off-by-one or off-by-multiple-digits DOIs
   (Van Petten 1990, Brown 1999, Citron 2014→2012, Recio 2014,
   Barber 2013, Wlotko 2012, Koizumi 2012, Marcus 1981, Scott 1998,
   Villing 2010→2011, Dirix 2017). Lesson: there is no shortcut.
   Every DOI must be resolved end-to-end. "Plausible from training
   data" is exactly the failure mode that lets one-digit typos slip
   through, because the citation reads correctly at every level
   *except* the URL.

---

## Part 4 — What remains outstanding

These are open items for the next reviewer pass — *not* fixed in this
audit:

- ~~**Live-verify the citations marked "plausible" in Part 1's
  caveat.**~~ Done in the second pass on 2026-05-27 — see the
  "(second pass)" rows in Part 1's table. Every DOI and arXiv ID
  remaining in `docs/features/*.md` has now been resolved end-to-end
  against publisher metadata.
- **⚠️ interpolation review** — separate from citations. Grep
  `docs/features/` for `⚠️` and decide which of the author's
  interpolations are correct, which need rewording, and which should
  be deleted. Tracked in `master_plan.md` under "Feature-doc
  author-review pass."
- **`pred_cloze.md` and `pred_topmatch.md`** — both reuse citations
  from sibling files but don't have their own broken citations. They
  should be re-read once the ⚠️ pass lands, to confirm the inherited
  claims still match the corrected references upstream.
- **Dataset registry cross-references** — every `## Datasets` section
  is still an empty HTML comment. Will be filled in by the
  dataset-registry track (see `master_plan.md` §4).
