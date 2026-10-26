# Output switches redesign (section switches + Run-tab levels) — design

Date: 2026-10-26 · Owner: Amilcar Malave · Status: draft for owner approval

## Intent

Every feature page is built from **section cards** (Predictability, Frequency, Counts, …).
Many cards carry a "master" toggle labelled "Enable all counts", "Enable all predictability
columns", "Emit syllable label columns", and so on. Users read "Enable all …" as **select
every feature in this section**, when it actually means **this section may be included at
all; you pick which features**. The master also looks exactly like the per-feature
checkboxes, so it reads as one more feature. And it greys its section's controls only
sometimes (Predictability's models and metrics stay clickable), which makes the behaviour
look unpredictable.

The switch exists so a user can **quickly leave a whole section out of the outputs**
without unticking many boxes, and bring it back later unchanged. It is about what ends up
in the output files, not a "select all".

## Goals

0. **Three layers, one visual language:** a Run-tab switch per **level** (one per feature
   page) → a **section** switch per card → the per-**feature** checkboxes. Levels and sections
   use the same on/off switch; features keep their checkboxes.
1. One section switch per card, with the same look and behaviour everywhere.
2. Wording that says what it does: the section is **Included** or **Left out of outputs**.
3. A control that looks unlike the feature checkboxes, placed right after the section title.
4. Switching a section off always greys **every** control inside it; switching it on
   restores them exactly as they were (no ticks are changed by the switch).
5. Quick "All · None" links above each group of feature checkboxes.

## Non-goals

- A fully read-only Run tab with no switches (one reading of the parked July "Gating
  simplification" item). The owner chose one switch per level instead (see "Run tab").
- Changing what any feature computes.
- The size/time estimates, the app icon, and the repo name (separate follow-ups).

## Anti-goals (would make this a failure even if it works)

- A switch that changes the user's feature ticks (on or off).
- Any card where "off" leaves some control looking clickable, or where the run still
  writes that section's columns.
- A new switch that defaults to "off" on an existing user's config and silently removes
  output they were getting.

## Behaviour (decided with the owner)

- **Off = left out.** The section's columns are not written, and its features are not
  computed — unless another *included* section needs that data (the planner's dependency
  closure already handles this, e.g. document prosody still gets word timing). The owner
  confirmed: with Predictability off, surprisal is not calculated even if its tick is on
  (today: `gui/planner/plan.py:_predictability_active`).
- **On = the section runs with whatever features are ticked.** Turning it on ticks nothing.
- **Data needed by an included section is still computed** even when its own section is off.
  Owner decision (2026-10-26): if Word **Predictability** is left out but the Document
  section's "predictability means" is included and ticked, the means **are** calculated,
  using the models and metrics picked in the (greyed) Predictability card. Today that case
  produces nothing, because `_predictability_active` requires the word CSV; fix it in the
  planner.
- The switch state persists like every other setting.

## Look (decided with the owner: mockup option C, switch moved next to the title)

Header row, left to right: **section title → on/off switch → status tag**.

- Switch: a phone-style pill (cyan `#00B2D0` when on, grey when off), visually distinct
  from the square checkboxes.
- Tag: "Included" (cyan tint) / "Left out of outputs" (grey).
- Card edge: a 4 px left border, cyan when included, grey when left out; the header gets a
  light grey background when left out.
- When left out, the card body (description, group labels, all controls) is greyed.
- "All · None" links sit beside each group label ("Models", "Metrics", …) and are greyed and
  inert while the section is left out.

Reference mockup: `.superpowers/brainstorm/8246-1793019483/content/section-switch-v2.html`
(local, git-excluded).

## Scope of cards (owner approved "every card gets the switch")

Cards that already have a master key (keep the key, change the presentation):
`master_structural`, `master_lexling`, `master_frequency`, `master_predictability`,
`master_lexical_norms`, `master_acoustic`, `master_word_prosody` (Word page);
`master_doc_counts` (in `output`), `master_psych_composite`, `master_doc_prosody`
(in `output`) (Document page); `master_syllable_intervals`, `master_syllable_labels`,
`master_syllable_prosody`, `master_phoneme_intervals`, `master_phoneme_surprisal`
(Sub-word page); `master_sentence_timing`, `master_sentence_prosody` (Sentence page).

Cards with no master today, which gain one: Word page — Semantic indicators, Semantic
similarity; Document page — Aggregates, Lexical diversity, Readability, Category
proportions; Sub-word page — P-center; Sentence page — Sentence similarity.

**Exceptions (owner-approved calls):**
- **G2P ("Out-of-dictionary words") gets no section switch.** It writes no columns of its
  own; it changes how MFA aligns every word, so "left out of outputs" has no meaning there.
  It stays a plain on/off setting.
- **The Acoustic page has a single card,** so its section switch and the Run-tab
  **Acoustic level** switch are the same setting (`acoustic_ts_mat` today).

**Shared settings are never greyed (owner-approved call).** Settings inside a card that
other sections also use stay editable when the card is left out, and are set apart with a
label saying so. Known case: the Word Prosody card's pitch units / floor / ceiling (also
used by syllable, sentence and document prosody), e.g. "Pitch settings (used by all
prosody sections)". This is the only cross-section setting found in the 2026-10-26
inventory; section-only settings (pause threshold, P-center k, phoneme analysis rate,
late-AoA threshold, acoustic parameters, predictability model/metric/context pickers) grey
out with their card. The Predictability models and metrics stay greyed even though document
means use them (owner decision above); the HuggingFace token box is a credential, not a
feature control, and stays editable.

**Tag states (owner-approved call):** "Included" when at least one feature in the card is
ticked; "Included — nothing selected" when none is (common under the opt-in defaults);
"Left out of outputs" when the switch is off. The tag reflects the card only, not the
Run-tab level.

Some of these cards hold a single toggle that is itself a master-style key today
(`master_word2vec`, `master_bert_word`, `master_bert_sentence`, `master_syllable_pcenter`,
`master_acoustic_ts`). Whether such a card gets a separate section key or its existing key
becomes the section switch is left to the builder; the visible result must match Goals 1–4.

## Run tab: one switch per level (owner decision 2026-10-26)

Today the Run tab mixes whole output files (word CSV, document CSV, syllable CSV, phoneme
CSV, sentence CSV, acoustic .mat) with one feature's file (phoneme surprisal .mat), while
Word2Vec/BERT .mat files have no Run-tab control at all, and names like "Word-level CSVs
(one per stimulus)" describe one file, not the level. Replace the seven output toggles with
**one switch per level**, named after the feature page:

| Level switch | Covers (files it can write) |
|---|---|
| Word level | `word/<stim>_features.csv`, `_word2vec.mat`, `_bert_word.mat` |
| Document level | `document/documents_summary_<ts>.csv` (one file per run) |
| Sub-word level | `syllable/<stim>_syllables.csv`, `phoneme/<stim>_phonemes.csv`, `phoneme/<stim>_phoneme_surprisal.mat` |
| Sentence level | `sentence/<stim>_sentences.csv`, `_bert_sentence.mat` |
| Acoustic level | `acoustic/<stim>_acoustic.mat` (audio only) |

(The owner listed word, sentence, sub-word and acoustic; Document is included as the fifth
because it has its own page and file — confirmed with the mockup.)

Each level row shows: the switch, the level name, a count tag ("3 files per stimulus",
"1 file for the whole run", "Left out"), one plain line ("Everything ticked on the Word
features page"), and the files it will actually write with the current ticks; a file type
in the level that nothing ticked will produce is shown struck through. Readiness stays
below. Reference mockup: `.superpowers/brainstorm/8246-1793019483/content/run-tab-levels.html`.

Consequences:
- **Phoneme surprisal .mat** stops being a Run-tab toggle and becomes an ordinary feature of
  the Sub-word page's phoneme-surprisal section (written when that section is included and
  its .mat option ticked).
- **Word2Vec / BERT** stay controlled from their cards; they simply appear under their level.
- **Migration must keep every existing config's outputs identical.** Fold the old keys
  (`word_level_csv`, `document_summary`, `syllables_csv`, `phonemes_csv`, `sentences_csv`,
  `acoustic_ts_mat`, `phoneme_surprisal_mat`) into the five level keys, and map anything
  finer than a level onto the section switches: e.g. old `syllables_csv: true,
  phonemes_csv: false` → Sub-word level on, phoneme sections left out. Defaults match
  today's: Word level on, the rest off.
- The planner's `_OUTPUT_CSV_MAP` / `output_csvs` and every consumer (`gui/planner/emit.py`,
  `run.py`, `main.py` batch runners, preflight, the Run-tab readiness and "Will produce"
  list, `utils/session_units.py`) must read the new keys.

## Constraints

- **Golden rule (CLAUDE.md):** the GUI decides nothing about execution. Any new section key
  must be honoured by `gui/planner/` (cluster `master_toggle` in `registry.py`), not by
  `section_*.py` logic.
- **New settings keys** need a `SCHEMA_VERSION` bump + migration in `utils/preferences.py`,
  `python scripts/export_default_config.py`, and the schema-drift tests. New keys default to
  **on (included)** so existing configs keep producing the same outputs.
- Keep existing master keys' names, so current config files stay valid.
- Greying must reach nested layouts: the cause of today's inconsistency is
  `Card.add_child_layout` not recursing into nested rows, and Predictability adding its
  rows with `add_widget` (ungated). Fix it in the card widget so every card gets it.
- Per-feature docs (`docs/features/*.md`) and the ⓘ popups don't document masters; no doc
  change expected beyond any user-facing README/GUI text that quotes "Enable all".
- Verify in the real GUI (`python gui.py`), not only tests: flip each card's switch and
  check greying, persistence across restart, and that a run leaves the section's columns out.

## Left to the builder

- Widget implementation (a custom painted switch vs. a styled QCheckBox), QSS details, and
  exact spacing, as long as it matches the look above.
- How "All · None" is wired per group (it only sets the group's checkboxes; it never
  changes the section switch).
- Naming of any new section and level keys (prefer `master_<card>` / `level_<name>`).
- Exact wording of the level descriptions and count tags.
- Test design (test-after for GUI/prefs per the project's TDD scope).
