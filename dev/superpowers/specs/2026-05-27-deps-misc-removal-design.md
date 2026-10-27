# Design — Drop `deps` and `misc` columns from word-level output

**Date.** 2026-05-27
**Status.** Approved (option 1 of 3 in brainstorming session).
**Driver.** User feedback in `docs/in_app_documentation_needs_fix.md`:
the columns were emitted as constant `NA` (`deps`) or near-constant
(`misc` carrying only `SpaceAfter=No`), which is misleading in the
output CSV. After brainstorming the three options below, the user
chose to drop both columns.

## Options considered

1. **Drop both columns** — chosen.
2. Add Stanza as an optional extractor for enhanced UDs; toggle in GUI.
3. Repurpose `misc` to carry EEG-relevant non-UD tags
   (ClauseType, IsConjunct, etc.); still drop `deps`.

Rationale for option 1: the user is targeting "best implementation for
EEG analysis," not UD-standard compliance. The EEG literature using
dependency structure overwhelmingly uses **basic** dependency length /
role from the primary tree, both of which are already covered by the
existing `dep_rel` and `head_token_id` columns. Enhanced UDs are a
computational-linguistics convenience without demonstrated EEG payoff
that justifies adding Stanza (~500 MB model, +5-15 s per document) as a
dependency. The MISC field for English transcripts carries only
`SpaceAfter=No`, which is too thin to justify its own output column.

## Change set

- `extractors/text_parser.py`: stop emitting `deps` and `misc` in the
  per-token row dict; remove the `misc` computation; update the
  module docstring.
- `main.py`: drop `'deps'` and `'misc'` from `_COLUMN_ORDER`; update
  the module docstring's column listing.
- `gui/planner/registry.py`: drop `"deps"` and `"misc"` from the
  `lex-ling` cluster's `emits_columns` tuple.
- `gui/section_word_features.py`: update the `lexling_dep` row label
  to read "Dependency parse (dep_rel, head_token_id)".
- `docs/features/lexling_dep.md`: remove `deps` and `misc` from
  `output_columns` frontmatter, the Meaning table, and the
  Implementation section.
- `docs/feature_dependencies.md`: drop `deps`, `misc` from the two
  lines that list spaCy-derived columns.

## Verification

- Existing `tests/extractors/test_text_parser.py` (12 tests) continues
  to pass; no test currently asserts the presence of `deps` or `misc`,
  so no test updates are required.
- After the change, a fresh CLI run should produce a `features.csv`
  whose header has no `deps` or `misc` columns.

## Out of scope

- Future enhanced-UD support (option 2 above) is *not* prohibited by
  this change — if a concrete EEG use case for enhanced UDs surfaces
  later, a Stanza-based extractor can be added under a new toggle. The
  drop here is a "until we have a real use case" decision, not a "never
  again" decision.
