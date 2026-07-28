# Phoneme label split, IPA, and phoneme-coverage correctness — design spec

**Date:** 2026-07-27
**Status:** design approved, pending spec review → plan → implement
**Level:** Sub-word (phoneme, syllable) + word (structural)
**Deps:** MFA alignment (already required for sub-word output); optional MFA `english_us_arpa` G2P model (11 MB, new vendored asset). No new Python deps.

---

## 1. What we're doing and why

Three user-reported items, plus the coupled correctness work each one forces:

1. **Split the fused phoneme label.** The phoneme CSV's `Label` column packs the ARPABET
   segment and its stress digit into one string (`EY2`), which is not analysable without
   string surgery. It becomes three columns: `ARPABET`, `Stress`, `IPA`.
2. **Add IPA.** More researchers read IPA than ARPABET. New `IPA` column on the phoneme CSV
   and a dash-joined `IPA` on the syllable CSV.
3. **Fix phoneme coverage.** The reported symptom ("some words had no phonemes") is real and
   has **two independent causes**, both confirmed against real aligned data in `validation/`
   (§2). Neither was previously visible: one silently deletes words, the other silently
   duplicates them.

A fourth item follows from the investigation: newline tokens occupy word-CSV rows and shift
the id columns (§2.3). Paragraph membership is already recorded by the existing `paragraph_id`
column, so removing those rows loses nothing (§7).

## 2. Investigation findings (evidence, not conjecture)

All three findings were reproduced against the ten real stimuli in `validation/output/` and,
for the OOV case, against a fresh MFA alignment.

### 2.1 OOV words are silently deleted

MFA aligns any word absent from `english_us_arpa.dict` using its `<unk>` model, whose single
phone is `spn`. `utils/textgrid_parser.py:56` filters `spn` as silence, so the word reaches
`syllabify_pairs` with an empty phone list and `extractors/syllabify.py:201` skips it —
producing **zero phoneme rows and zero syllable rows, with no warning**.

Reproduced by re-aligning `validation/data/audio/1_M.wav` with `Bella` replaced by the
nonsense word `Zorblax`:

```
words tier:   ('zorblax', 0.00, 0.33)     ← the word is aligned
phones tier:  ('spn',     0.00, 0.33)     ← filtered out as silence
result:       WORD WITH ZERO PHONES: zorblax
```

`spn` means **out-of-dictionary**, not *not spoken*. Any stimulus containing unusual proper
nouns loses those words from every sub-word output. The ten validation stimuli happen not to
trigger it (`Adrianna` is in-dictionary), which is why it went unnoticed.

### 2.2 Contractions and possessives duplicate every phoneme

spaCy splits `wasn't` into `was` + `n't`; MFA has one word `wasn't`. The *join* repair rule
in `extractors/align_merge.py:150` assigns the **same** `[Onset, End]` to both token rows, and
`_phones_for_word` (`syllabify.py:174`) selects by interval containment — so each token
receives the complete phone set:

```
was  1 1  W   4.50 4.60      n't  1 1  W   4.50 4.60
was  1 2  AH1 4.60 4.68      n't  1 2  AH1 4.60 4.68     ← the same acoustic
was  2 1  Z   4.68 4.75      n't  2 1  Z   4.68 4.75        phonemes, emitted
was  2 2  AH0 4.75 4.81      n't  2 2  AH0 4.75 4.81        twice
was  2 3  N   4.81 4.84      n't  2 3  N   4.81 4.84
was  2 4  T   4.84 4.87      n't  2 4  T   4.84 4.87
```

Confirmed in `3_M` (12 rows, `was`+`n't`) and `LJ003-0007` (10 rows, `debtor`+`'s`).
Consequences beyond the CSV: syllable rows duplicate identically; and
`phoneme_surprisal_mat_vars` accumulates with `surp[idx] += ...`
(`phoneme_surprisal.py:241`), so **colliding duplicates sum** — the `.mat` surprisal and
entropy trains carry doubled values, and `word_onset` fires twice at one instant.

### 2.3 Newline tokens occupy rows and shift the id columns

spaCy emits a literal newline as a token (`upos == 'SPACE'`, text `\n`). `text_parser.py`
iterates every token, so it becomes an all-NaN word-CSV row that consumes a `token_id` **and**
a `token_in_sentence_id`:

```
token_id=10  token_in_sentence_id=10  token=\n     ← consumes position 10
token_id=11  token_in_sentence_id=11  token=were
```

Every word after a line break is therefore off by one in both id columns.
`master_plan.md` §4 claims `token_in_sentence_id` supplies Brain Treebank's
`Index_in_Sentence`; for newline-containing transcripts that claim is currently false.
Affects 3 of the 10 validation stimuli (the `LJ*` files); the single-line ISC stimuli have no
SPACE tokens.

**Not paragraph breaks.** The newlines in these transcripts are hard line *wraps*, and at
least one falls mid-sentence — in `LJ002-0321`, `'…time of distribution\nwere often 48
hours…'` is one sentence (spaCy assigns both halves `sentence_id=1`). No transcript in the
set contains a blank line. So a paragraph counter incremented per `\n` would split a single
sentence across two paragraphs. Paragraphs must be defined by blank lines — which is already
how the shipped `paragraph_id` column works (§7), so no change is needed there.

### 2.4 Out of scope: "was this word actually spoken?"

MFA is a *forced* aligner: it assumes the transcript is what was said and must place every
word somewhere. A word present in the transcript but absent from the audio gets squeezed into
a sliver, stealing time from its neighbours — **already true today for every in-dictionary
word**, and not something this batch introduces or fixes.

The G2P option in §5 does not detect absence either. Verified: because `1_M.wav` actually says
"Bella", the G2P run confidently aligned `Z AO1 R B L AE2 K S` across the interval where
"Bella" was spoken. G2P recovers pronunciations for *unusual words that were spoken*; it does
not adjudicate whether they were.

Detecting genuine transcript/audio mismatch needs per-word alignment likelihood. MFA 3.4.0's
`align` CLI exposes no such output (checked: no likelihood/confidence flag, and no
`alignment_analysis.csv` is produced). That is the parked **Correctness scoring** roadmap item
and stays parked. G2P therefore defaults **off** (§8), so no run silently gains modeled
pronunciations.

## 3. Phoneme CSV — new column layout

`Label` is **replaced** (not kept alongside; it is fully derivable from the split):

```
doc_id, sentence_id, token_in_sentence_id, token, syllable_id, phoneme_id,
ARPABET, Stress, IPA, Pron_Source, Onset, End, Duration,
phoneme_surprisal, cohort_entropy
```

| Column | Type | Contents |
|---|---|---|
| `ARPABET` | str | Bare segment, stress digit removed: `EY`, `D`, `IY`. For an unresolved OOV word, `spn`. |
| `Stress` | nullable `Int64` | `0`/`1`/`2` on vowels; **empty** on consonants. Empty, not `0` — `0` means "unstressed vowel", a different claim from "stress does not apply". |
| `IPA` | str | Single IPA segment (§4). Empty for `spn`. |
| `Pron_Source` | str | `dictionary` \| `g2p` \| `oov_unresolved` (§5). |

Example (`Adrianna`, first two syllables):

```
token     syllable_id phoneme_id ARPABET Stress IPA Pron_Source Onset End
Adrianna       1          1        EY      2    eɪ  dictionary  0.00  0.12
Adrianna       2          1        D            d   dictionary  0.12  0.20
Adrianna       2          2        R            ɹ   dictionary  0.20  0.25
Adrianna       2          3        IY      0    i   dictionary  0.25  0.34
```

## 4. Syllable CSV — additive only

`Label` and `ARPABET` are **unchanged** (`ARPABET` keeps its stress digits, e.g. `D-R-IY0`).
One new column:

| Column | Type | Contents |
|---|---|---|
| `IPA` | str | Dash-joined IPA over the syllable's phonemes, e.g. `d-ɹ-i`. No `ˈ`/`ˌ` stress marks — the syllable's stress remains encoded in `ARPABET`. |

Deliberately no syllable-level `Stress` column: the user chose additive-only here to avoid
breaking the existing `ARPABET` string.

## 5. New module — `extractors/arpabet_ipa.py`

Pure, dependency-free, no I/O. Per `feedback-tdd-scope` this is `extractors/` math, so it is
written test-first.

```python
ARPABET_TO_IPA: dict[str, str]                       # 39-phone table, bare codes
def to_ipa(arpabet: str, stress: int | None) -> str  # single segment
def syllable_to_ipa(pairs) -> str                    # dash-joined
```

**Stress-aware**, because two ARPABET vowels are two distinct IPA vowels depending on stress:

| ARPABET | Stress | IPA | Note |
|---|---|---|---|
| `AH` | `0` | `ə` | schwa — English's commonest vowel |
| `AH` | `1`/`2` | `ʌ` | |
| `ER` | `0` | `ɚ` | |
| `ER` | `1`/`2` | `ɝ` | |

Remaining mappings are one-to-one and unambiguous: `AA`→`ɑ`, `AO`→`ɔ`, `EY`→`eɪ`, `OW`→`oʊ`,
`AY`→`aɪ`, `AW`→`aʊ`, `OY`→`ɔɪ`, `R`→`ɹ`, `TH`→`θ`, `DH`→`ð`, `ZH`→`ʒ`, `NG`→`ŋ`, etc. The
full table lives in the module, documented once, and is the single source of truth. `spn` and
any unrecognised code map to `""` (never a guess).

Consequence to state plainly: `IPA` is **not** a pure function of the `ARPABET` column alone —
it reads `Stress` too. That is intentional and is documented in `docs/features/`.

## 6. Phoneme-coverage fixes

### 6.1 Stop filtering `spn`

`utils/textgrid_parser.py::parse_phones` drops `spn` from its silence set, keeping
`{'', 'sil', 'SIL', 'sp'}`. **This is the change that ends the silent deletion** — the OOV
word now reaches `syllabify_pairs` with a phone to attach. It yields one syllable and one
phoneme row carrying `ARPABET='spn'`, `Stress`/`IPA` empty, `Pron_Source='oov_unresolved'`,
and the word's real MFA timing.

`syllabify.Phone.is_vowel` returns `False` for `spn` (no trailing digit), so
`syllabify_word` treats it as the vowel-less case and emits a single syllable — the existing
`hmm` path, no new branch.

`add_phoneme_surprisal` must emit `NaN` (not the OOV cap) for an `spn` row: there is no
phoneme identity to score. Handled by skipping groups whose phones are `spn`-only.

### 6.2 Optional G2P recovery

`extractors/aligner.py::_run_mfa` passes `--g2p_model_path <mfa_data/pretrained_models/g2p/
english_us_arpa.zip>` when the `alignment.g2p_oov` pref is on **and** the model file exists.
When on, MFA generates a pronunciation for OOV items and aligns it with real boundaries.

Verified on the `Zorblax` corpus: `zorblax` received `Z AO1 R B L AE2 K S`, zero words were
left phone-less, and — importantly — **in-dictionary phones were untouched**. Diffing the two
phone tiers, the only change outside the OOV word was its shared boundary with the next word
moving 0.33 → 0.35 s. Every other interval was bit-identical.

Cache invalidation: `_freshness_dict` gains an `mfa_g2p` key and `SIDECAR_VERSION` goes
1→2, so `_is_fresh` re-aligns existing TextGrids exactly once, and toggling the pref
re-aligns thereafter.

`utils/download_models.py` also fetches the g2p model (`mfa model download g2p
english_us_arpa`, 11 MB into the already-vendored, git-ignored `mfa_data/`).

### 6.3 Determining `Pron_Source`

MFA does not mark G2P-derived words in the TextGrid, so provenance is derived: a words-tier
form absent from `english_us_arpa.dict` had its pronunciation generated. The dictionary is
already parsed and memoized by `phoneme_surprisal.load_mfa_lexicon`, so this reuses that
parse (a `frozenset` of keys built alongside the existing trie cache) rather than adding a
second read of a 200k-line file.

- word in dictionary → `dictionary`
- word not in dictionary, real phones present → `g2p`
- word not in dictionary, only `spn` → `oov_unresolved`

### 6.4 Zero-phoneme reporting

`syllabify_pairs` collects words that end with no phoneme rows and reports them to stdout and
the run log:

```
[phonemes] WARNING: 2 words had no phonemes (likely OOV): zorblax, mciver
```

This is the belt-and-braces guard: it fires if the G2P model is missing on a collaborator's
machine, and for any future cause of zero-phoneme words. Punctuation and shared-alignment
tokens (§6.5) are excluded — their absence is expected, not anomalous.

### 6.5 Deduplicate shared alignments

`syllabify_pairs` tracks emitted phone intervals per document. When consecutive token rows
carry an identical `[Onset, End]` — the `align_merge` *join* case — only the **first** token
gets syllable and phoneme rows. The word CSV's existing `Transcript` column already shows
both tokens aligned to `wasn't`, so the second token's absence is self-explanatory and needs
no new column.

Fixes, in one change: inflated phoneme/syllable row counts, duplicate syllable rows, the
doubled `.mat` surprisal/entropy values, and the double `word_onset` impulse.

Not chosen: rule-based clitic splitting (`n't`→`N T`, `'s`→`Z`, …). It would need a
hand-maintained table that mis-splits silently on edge cases, and it makes syllable
attribution ambiguous (in `wasn't`, `AH0` belongs cleanly to neither token).

## 7. SPACE-row removal (paragraph tracking already exists)

**Correction to an earlier draft of this spec:** paragraph tracking is already implemented and
fully wired. `extractors/text_parser.py:32,47-75,104` computes paragraph spans from blank
lines (`_PARAGRAPH_SEP_RE = r'\n[ \t]*\n+'`) — exactly the §2.3 rule — and emits
`paragraph_id`. It already has a `feat_struct_paragraph_id` pref
(`preferences.py:232`), a `_WORD_FEATURE_COLUMN_MAP` entry (`main.py:481`), a GUI toggle
(`section_word_features.py:54`), planner `emits_columns` (`registry.py:241`), and
`docs/features/structural_ids.md`. The doc summary already emits `n_paragraphs`
(`out_count_paragraphs`). The word CSV has carried `paragraph_id` all along.

So the requirement "we must still know which tokens belong to which paragraph" is **already
satisfied**, and removing the newline rows cannot lose it. No new column, pref, toggle, doc, or
migration entry is needed for paragraphs.

The only actual work is in `extractors/text_parser.py::parse`:

1. **Skip `token.is_space`** tokens — they never become rows.
2. **Assign `token_id` over surviving tokens**, building a `{token.i → token_id}` map, and
   resolve `head_token_id` **through that map** (a root maps to its own `token_id`).
3. `token_in_sentence_id` counts surviving tokens, becoming a true sentence position.

> ⚠️ Step 2 is mandatory, not optional. `text_parser.py:121` currently computes
> `head_token_id = token.head.i + 1` using spaCy's doc index, which counts SPACE tokens.
> Today `token_id` also counts them, so the two agree **by accident**. Skipping SPACE rows
> without the remap makes every `head_token_id` past the first newline point at the wrong
> word, with no error raised.

A sentence-level `paragraph_id` column is **not** added — out of scope, not requested, and
derivable by joining the word CSV on `sentence_id`.

## 8. Prefs — v19 → v20 (removal-migration)

Precedent: v14 (struct-id rename) and v18 (predictability unification), both handled in the
finalize block.

**Removed:** `feat_phoneme_label`.

**Added:**

| Key | Default | Notes |
|---|---|---|
| `feat_phoneme_arpabet` | inherits old `feat_phoneme_label` | |
| `feat_phoneme_stress` | inherits old `feat_phoneme_label` | |
| `feat_phoneme_ipa` | inherits old `feat_phoneme_label` | |
| `feat_phoneme_pron_source` | inherits old `feat_phoneme_label` | |
| `feat_syllable_ipa` | `False` | Opt-in, matching `feat_syllable_arpabet`'s verbose-column precedent |
| `alignment.g2p_oov` | `False` | Opt-in per §2.4 — never model a pronunciation unasked |

No paragraph key is added — `feat_struct_paragraph_id` already exists (§7).

Migration carries the old `feat_phoneme_label` value forward to all four replacements, so a
user who had it off keeps all four off.

`scripts/export_default_config.py` is re-run; the drift-guard test enforces it.

## 9. Coupled layers — the full checklist

Per `CLAUDE.md`'s "when you touch X, also touch Y". **The two entries marked ⚠️ are silent
breakages**: they read the stress digit out of the `Label` string and fail without raising.

| File | Change |
|---|---|
| ⚠️ `extractors/pcenter.py:21` | `group['Label'].str[-1:].str.isdigit()` → `Stress.notna()`. Otherwise **no vowel is ever detected** and every `P_Center` silently falls back to the syllable onset. No exception. |
| ⚠️ `extractors/phoneme_surprisal.py:196` | Read `ARPABET` (already stress-free) instead of `Label`; for the `strip_stress=False` path, recombine `ARPABET + Stress`. Otherwise `KeyError`, or wrong values. |
| `extractors/arpabet_ipa.py` | **New** (§5) |
| `extractors/syllabify.py` | `_PHON_COLUMNS`/`_SYLL_COLUMNS`; emit the split columns + `IPA` + `Pron_Source`; dedup (§6.5); zero-phoneme collection (§6.4) |
| `extractors/text_parser.py` | SPACE skip + `token_id`/`token_in_sentence_id`/head remap (§7). Paragraph logic already present — do not touch it. |
| `extractors/aligner.py` | `--g2p_model_path`; `mfa_g2p` sidecar key; `SIDECAR_VERSION` 1→2 |
| `utils/textgrid_parser.py` | Stop filtering `spn` (§6.1) |
| `utils/download_models.py` | Fetch the g2p model |
| `utils/preferences.py` | v20 migration (§8) |
| `utils/session_units.py` | **No change.** `_STATIC_UNITS` maps only physical-unit columns (seconds/dB/count) and has no categorical entries; the new label columns are correctly absent. |
| `main.py` | `_PHONEME_COLUMN_MAP` (4 new keys, drop `feat_phoneme_label`); `_SYLLABLE_COLUMN_MAP` (`feat_syllable_ipa`); `--g2p-oov` CLI flag; thread the pref to `_prealign_batch`. `_WORD_FEATURE_COLUMN_MAP` needs no change. |
| `gui/planner/registry.py` | `emits_columns` on the phoneme + syllable-label stages; G2P pref plumbing |
| `gui/planner/emit.py` | `g2p_oov` into the Namespace |
| `gui/section_subword_features.py` | One phoneme-label toggle → four; syllable `IPA` row; G2P toggle |
| `gui/main_window.py` | `settings_changed → refresh` wiring if a page gains a card |
| `run.py` | Honour `alignment.g2p_oov` |
| `docs/features/` | Rewrite `phoneme.md`; extend `syllable_labels.md`; new `feat_phoneme_ipa.md`, `g2p_oov.md`. Add the SPACE-row/id note to the existing `structural_ids.md`. |
| `gui/feature_docs.py` | Register the new toggle keys in `_SHARED_DOC_MAP` |
| `docs/master_plan.md` | §3 row, §6 prefs-version note, and correct the §4 Brain Treebank parity claim (§2.3) |
| `docs/changelog.md` | Full batch write-up |

Doc-level aggregation needs no change — `utils/doc_summary.py` references no syllable or
phoneme columns (verified).

## 10. Tests

**Unit (test-first, `extractors/`):**

- `arpabet_ipa`: full-table coverage; `AH0`→`ə` vs `AH1`→`ʌ`; `ER0`→`ɚ` vs `ER1`→`ɝ`;
  unknown code and `spn` → `""`; dash-joined syllable form.
- `syllabify`: split columns present and correct; `Stress` empty on consonants and `0` on
  unstressed vowels (distinct assertions — this is the trap the nullable dtype guards);
  dedup emits one row set for two tokens sharing an interval; `spn`-only word yields exactly
  one syllable + one phoneme row with `Pron_Source='oov_unresolved'`.
- `pcenter`: a synthetic frame using `Stress`, asserting the nucleus is still found — the
  direct regression test for the ⚠️ breakage.
- `phoneme_surprisal`: reads `ARPABET`; `spn` rows are `NaN`; `strip_stress=False`
  recombination.
- `text_parser`: newline-containing input yields no SPACE rows, contiguous ids, correct
  `head_token_id` after remap, `paragraph_id` from blank lines, and a mid-sentence `\n` does
  **not** increment `paragraph_id` (the §2.3 case).

**Integration / schema:** planner `emits_columns` drift, prefs v19→v20 round-trip and
`feat_phoneme_label` inheritance, defaults-template drift guard.

**Real-stimulus validation** — required, because the repo fixtures are empty placeholders and
a green suite does not prove a real run works (`CLAUDE.md`; `project-mkl-threading-entrypoints`).
Re-run all ten `validation/` stimuli and assert:

1. `P_Center` values **unchanged** vs. the current CSVs — the regression that would otherwise
   pass silently.
2. Row counts drop by exactly the duplicate copies and by zero elsewhere (measured on the
   current outputs, so these are exact expected values, not estimates):

   | Stimulus | Phoneme rows | Syllable rows |
   |---|---|---|
   | `3_M` | 111 → **105** (−6) | 41 → **39** (−2) |
   | `LJ003-0007` | 124 → **119** (−5) | 48 → **46** (−2) |
   | all other 8 | unchanged | unchanged |

   Note the removed count is *half* the rows flagged as sharing an interval (12 flagged in
   `3_M` = 6 kept + 6 removed).
3. `Stress` is empty on every consonant row and populated on every vowel row.
4. `IPA` is non-empty for every non-`spn` row.
5. `paragraph_id == 1` throughout; zero SPACE rows; `head_token_id` resolves to a real
   `token_id` on the three `LJ*` stimuli.
6. With `g2p_oov` off, the `Zorblax` corpus yields one `oov_unresolved` row; with it on, real
   phones marked `g2p`.
7. `phoneme_surprisal.mat` trains no longer double-count at duplicated onsets.

## 11. Non-goals

- Per-word alignment confidence / transcript-audio mismatch detection (§2.4) — stays parked
  as the roadmap's *Correctness scoring* item.
- Rule-based clitic phoneme splitting (§6.5).
- Syllable-level `Stress` column (§4).
- Sentence-level `paragraph_id` column (§7). Doc-level `n_paragraphs` and word-level
  `paragraph_id` already exist and are not touched.
- Any change to word-level `Surprisal`, prosody, or acoustic output.
