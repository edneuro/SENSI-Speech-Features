# Word- and syllable-level event `.mat` files — design

Status: draft for owner approval (2026-11-03). Branch: new, from `main`.

## 1. Intent

Researchers want to load one file per stimulus in MATLAB and **stem-plot a
word feature (surprisal, cloze, frequency, …) at each word's onset time,
optionally labelled with the token**:

```matlab
S = load('word/LJ002-0321_word_events.mat');
stem(S.onset, S.gpt2_Surprisal); hold on
text(S.onset, S.gpt2_Surprisal, S.token, 'Rotation', 45)
```

Today they must parse the features CSV in MATLAB (mixed string/numeric columns,
empty cells) to do that. The same per-event timings also let whoever runs a TRF
build regressors on *their own* sampling grid
(`x = zeros(N,1); x(round(S.onset*fs)+1) = S.gpt2_Surprisal;`), so the file
stores **events, not impulse trains** (owner: "the person who will do the TRF can
just fix to their grid").

Same idea at the syllable level, where the perceptual centre (`P_Center`) is an
alternative time anchor.

## 2. Goals / non-goals / anti-goals

Goals
- One `word/<stim>_word_events.mat` and one `syllable/<stim>_syllable_events.mat`
  per stimulus, opt-in.
- Every variable is directly plottable: equal-length column vectors, one entry
  per event, plus a `token`/`label` cellstr for `text()`.
- Feature doc + manual page with a copy-paste MATLAB snippet (load, stem, label,
  put-on-your-grid one-liner).

Non-goals
- No impulse trains, no fixed analysis grid, no resampling (the phoneme
  surprisal `.mat` keeps doing that; untouched).
- No sentence- or document-level event file (not asked for).
- No change to any existing CSV or `.mat`.

Anti-goals (a failure even if it "works")
- The `.mat` and the CSV disagreeing on a value or on which features exist,
  except for the documented merge of multi-token spoken words (§3.1).
- Rows without a real onset in the file (a NaN onset breaks `stem`/the grid
  one-liner silently).
- MATLAB variable names that differ from the CSV column names the docs
  describe (users must not learn a second vocabulary).

## 3. File contents

### 3.1 Word file — `word/<stim>_word_events.mat`

Rows = the word CSV's rows, **minus** punctuation (`upos == "PUNCT"`) and rows
whose `Onset` or `End` is NaN (unaligned). Row order = CSV order. If no row
survives (text-only run or nothing aligned), the file is not written (§5).

Variables (all `n × 1`):

| Variable | Source | Type |
|---|---|---|
| `text` | the transcript text as read (`_transcript_words`/raw file, whitespace-normalised) | char row |
| `token` | `token` | cellstr |
| `onset`, `offset` | `Onset`, `End` (s from audio start) | double |
| `duration` | `End − Onset` | double |
| `sentence_id`, `token_id` | same-named CSV columns | double |
| one variable per **numeric** column the word CSV keeps after the per-column drop map | verbatim column name, e.g. `gpt2_Surprisal`, `Zipf_Freq`, `AoA_Mean`, `Pitch_Mean`, `Sem_Empath_body` | double, `NaN` where the CSV cell is empty |
| `columns` | names of the feature variables above, in CSV order | cellstr |
| `units` | units of those, aligned with `columns`, from `sensi/utils/session_units.py` (empty string where none) | cellstr |

`onset`/`offset`/`duration`/`token`/`text` are always present when the file is
written, even if the user dropped `Onset`/`End` from the CSV (they are the
file's reason to exist). Columns already covered by those (`Onset`, `End`,
`Word_Duration`) and positional ids other than `sentence_id`/`token_id`
(`doc_id`, `paragraph_id`, `token_in_sentence_id`, `head_token_id`, `start`,
`end`) are not repeated. String columns (`lemma`, `upos`, `Sem_Categories`,
`Transcript`, …) are left out — the CSV has them.

No `is_valid` (owner question): every row has an onset by construction, and a
missing feature value is `NaN`, which `stem`/`text` skip.

**One row per spoken word (owner decision, 2026-11-03).** `align_merge` gives
every spaCy token joined to one MFA word ("do" + "n't" → "don't", "John" +
"'s", "can" + "not") that word's `Onset`/`End`. The word file merges such runs
(consecutive rows with identical `Onset`/`End` and the same MFA `Transcript`)
into one row. Why: the listener hears one acoustic word; and two impulses on one
sample are silently overwritten by the natural `x(idx) = v` grid one-liner.
`token` = the aligned word (`Transcript`). Columns combine as:

| Column kind | Rule | Why |
|---|---|---|
| `*_Surprisal` | sum | chain rule: −log P(do) − log P(n't\|do) = −log P(don't) |
| `*_Cloze` | product | joint probability of the whole word |
| `*_TopMatch` | all parts (AND) | the whole word was top-1 only if every part was |
| `Sem_*` 0/1 indicators | any part (OR) | |
| prosody, timing (`Pitch_*`, `Volume_*`, `Word_Gap`, …) | first row | identical: same Onset–End window |
| frequency, lexical norms (`Zipf_Freq`, `AoA_*`, `Concreteness`, …) | re-look-up the joined form via the **same** extractor lookup the CSV uses; `NaN` if absent | no exact combination rule; never invented |
| `token_id` / `sentence_id` | first row | |

Each new numeric column must be assigned a rule (a test fails on an unmapped
column, so a future feature cannot silently fall into "first row"). The `.mat`
therefore differs from the CSV on merged words — the feature doc says so and
lists the rules. The syllable file is unaffected (syllables already come from
the MFA word).

Numeric columns (my call): every column `pd.to_numeric` accepts, booleans and
0/1 indicators included (`Sem_Empath_*` → double); object columns that don't
convert are left out.

`text` (my call): the text spaCy annotated — the raw transcript file passed
through `transcript_prep.normalize_whitespace` (spaCy parses the raw text;
numbers are spelled out only for the aligner), so every `token` is a
substring of `text`. Doc note: use `S = load(...)`; a bare `load` puts a variable named
`text` in the workspace, shadowing MATLAB's `text()`.

### 3.2 Syllable file — `syllable/<stim>_syllable_events.mat`

Rows = syllable CSV rows with non-NaN `Onset`/`End` (PUNCT rows have none).

| Variable | Source |
|---|---|
| `text` | as above |
| `label` | `Label` (cellstr) |
| `token` | `token` of the parent word (cellstr) |
| `onset`, `offset`, `duration` | `Onset`, `End`, `Duration` |
| `pcenter` | `P_Center` — present only when P-center ran |
| `sentence_id`, `syllable_id` | same-named columns |
| one variable per remaining numeric column kept in the syllable CSV (`Pitch_Mean`, `Volume_Mean`, `Pitch_Delta`, `Volume_Delta`, …) | verbatim names |
| `columns`, `units` | as above |

`ARPABET`/`IPA` (strings) are left out.

### 3.3 Name rule

All current column names are already valid MATLAB identifiers (letters,
digits, `_`; predictability slugs are `gpt2`, `llama32_3b`, …). The writer
asserts `^[A-Za-z][A-Za-z0-9_]{0,62}$` for every variable name and raises on a
violation (a test pins it), rather than silently renaming.

## 4. Switches (owner: "yeah, that is fine")

My call, following the v21 pattern (`feat_phoneme_surprisal_mat` on the Sub-word
phoneme-surprisal card): per-file toggles on the level pages, not the Run page.

- `features.feat_word_events_mat` (default **False**) — a toggle on the Word
  features page, in a small "MATLAB events file" card (no master of its own;
  gated by `output.level_word`).
- `features.feat_syllable_events_mat` (default **False**) — a child toggle on
  the Sub-word syllable-timing card (gated by `output.level_subword` + that
  card's master).
- Prefs: `SCHEMA_VERSION` 21 → 22, additive migration (both keys False);
  `scripts/export_default_config.py` rerun.
- Planner: one `FeatureCluster` each (`output_csv="word_events_mat"` /
  `"syllable_events_mat"`, `emits_columns=()`), added to `LEVEL_ARTIFACTS`
  under `level_word` / `level_subword`; `emit.py` sets
  `ns.no_word_events_mat` / `ns.no_syllable_events_mat`. No new stages: the
  file is written from DataFrames the run already has; it requires
  `mfa_align` (and `syllabify_pairs` for syllables) so a plan with the toggle
  on always aligns.
- Run page: the two new files appear in the level's file list
  (`section_run.py` artifact→path map).

## 5. Pipeline

- A pure writer per level in a new module `sensi/utils/event_mat.py`:
  `word_event_vars(df, text, units) -> dict | None` and
  `syllable_event_vars(syll_df, text, units) -> dict | None` (None when no row
  survives). TDD (row filtering, NaN handling, name check, column selection).
- `pipeline.py` calls them **after** the per-column drop (so §2's "agree with
  the CSV" holds) and writes with `scipy.io.savemat(..., oned_as="column")`;
  object arrays for cellstrs, as `_word_embedding_vars` does.
  `Onset`/`End` must be read from the frame **before** the drop (it may remove
  them) — keep a reference to those two columns.
- Written in the **word pass only** (`run_word_level_batch`), never by the
  document pass, even when both levels are on.
- Independent of whether the CSV itself is written (my call): the toggle needs
  only the frame, which the word pass always builds; the syllable toggle makes
  `syllabify_pairs` run like `_want_psr` does for the phoneme `.mat`. "Agrees
  with the CSV" means: the same per-column toggles decide the columns.
- Units (my call): `emit.py` puts a `{(artifact, column): unit}` map on the
  namespace (`ns.event_units`), built once per dispatch from the same plan, so
  the writer never re-reads prefs. The unit lookup must cover the event files'
  columns even when the CSV artifact is off.
- Text-only stimuli (no `Onset` column, or all NaN) or zero surviving rows: no file, one stderr line
  (`[word_events] skipped for <stim>: no aligned words`), run continues.
- MFA is not re-run; no new expensive work.

## 6. Docs and manual

- `docs/features/word_events_mat.md` and `syllable_events_mat.md`: Meaning,
  what's in the file (table above), the MATLAB snippet (load → `stem` →
  `text` → put-on-grid one-liner, and the syllable one using `pcenter`),
  "In the code", Configuration. Linked from the ⓘ on both toggles.
- `dev/manual/features.yaml`: place both docs on their pages; output shown as
  a variable listing of the real example `.mat` (as the phoneme `.mat` page
  does); figure = a stem plot of `gpt2_Surprisal` with token labels at
  `onset`, drawn from the real example file. The example prefs are built in
  `scripts/manual/examples.py` (written via `_write_config`); turn both
  toggles on there, then `--refresh-examples`.
- Rebuild the manual; commit `docs/user-manual/`.

## 7. Tests

- Unit (TDD): merge rules on a hand-built "do"+"n't" frame (surprisal sum,
  cloze product, AND/OR, joined-form lookup, NaN when absent, every column
  mapped); writer filtering (PUNCT, NaN onset dropped; NaN feature kept),
  equal lengths, CSV-agreement after a drop map, name check raises, None on
  empty, syllable `pcenter` present/absent.
- Round-trip: `savemat` → `loadmat` gives cellstr `token` and `n×1` doubles.
- Planner: toggle on ⇒ artifact in `plan.output_csvs`, `mfa_align` planned;
  level off ⇒ not.
- Prefs: v21 → v22 migration; schema-drift + default-config drift tests.
- Real path: `run.py` on a scratch copy of `dev/validation/config` with both
  toggles on; open the files in MATLAB (owner) or `scipy.io.loadmat`.

## 8. Left to the builder

Card wording on the Word page; exact stderr text; manual figure styling (follow
`scripts/manual/figures.py` conventions); where the `text` string is captured
in `pipeline.py`.
