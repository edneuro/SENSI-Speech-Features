# Batch 3 — Word/Doc Separation, Modal Category Pickers, Per-Column Toggles

**Date:** 2026-05-11
**Author:** Amilcar Malave (with Claude)
**Status:** Draft for review
**Builds on:** `2026-05-11-gui-redesign-design.md` (batch 2)

---

## Motivation

Batch 2 shipped the PySide6 redesign but bundled word-level and document-level features onto a single Features page. After testing, the user reported:

1. **Word and document-level features feel commingled** when they should be clearly separated workflows.
2. **The inline category picker is cramped and hard to use** — categories are easier to browse in their own window.
3. **Word-level and document-level should each have their own category selection**, not share one list.
4. **Several "single toggle per group" feature blocks should be per-column checklists** so users can include/exclude individual output columns. Specifically: Structural (8 cols), Acoustic (4 cols), Composite psychological scores (4 cols), and Document-level Counts (4 cols).
5. **The visual style still feels like a document rather than an app** — sections need to "pop" more.
6. **Lab branding** — the SENSI lab uses cyan `#00B2D0`, not Stanford Cardinal. The supplied `assets/icon.svg` already lives in the title bar (landed in a separate commit on this branch).

Batch 3 addresses items 1–5 architecturally and tightens the visual style.

---

## Out of scope

- New extractors / new features
- CLI changes (`main.py` flags stay backward compatible)
- Pricing or cost-estimate changes
- The "logo PNG" — wire it when it's placed (currently only `icon.svg` is in `assets/`)

---

## High-level decisions

| Decision | Choice |
|---|---|
| Accent color | **SENSI cyan `#00B2D0`** (already landed) |
| Sidebar | **5 items**: Inputs, Word-level features, Document-level features, Run, Logs (split from batch 2's Features) |
| Category selection | **Modal dialog** — one button per level opens its own modal |
| Word/doc categories | **Independent**: separate prefs lists per level × per backend (4 lists total) |
| Per-column toggles | **Structural (8), Acoustic (4), Composite psych (4), Doc Counts (4)** become checklists |
| Composite psych scores location | **Document-level page** (they're doc-level outputs) |
| Visual style | **Card containers** around feature subsections, tighter type hierarchy |
| Schema | **v4 migration** from v3 |

---

## Architecture

### New sidebar layout

```
┌──────────────┬───────────────────────────────────────────┐
│ 📁 Inputs    │                                           │
│ ✦ Word level │   Active page                             │
│ ▦ Doc level  │                                           │
│ ▶ Run        │                                           │
│ 📜 Logs      │                                           │
└──────────────┴───────────────────────────────────────────┘
```

`gui/main_window.py` instantiates and registers five pages. `FeaturesSection` is split into two new modules: `gui/section_word_features.py` and `gui/section_doc_features.py`. The old `gui/section_features.py` is deleted.

### File structure changes

```
gui/
├── section_word_features.py   # NEW — was the word-level half of section_features.py
├── section_doc_features.py    # NEW — was the doc-level half
├── category_dialog.py         # NEW — modal QDialog wrapping CategoryPicker
├── section_features.py        # DELETED
└── category_picker.py         # KEPT — now embedded in the dialog instead of inline
```

### Schema v4 — semantic categories per level + per-column flags

`utils/preferences.py` introduces:

```json
{
  "schema_version": 4,
  "shared":  { ...unchanged from v3... },
  "predictability": { ...unchanged... },
  "semantic": {
    "backend": "auto",
    "word_categories_liwc":  [...],
    "word_categories_empath": [...],
    "doc_categories_liwc":   [...],
    "doc_categories_empath":  [...]
  },
  "features": {
    /* Lex-ling, Frequency, Predictability, Semantic, Acoustic groups —
       same flat keys as v3 except: */
    "structural": REMOVED,
    "feat_struct_doc_id": true,
    "feat_struct_paragraph_id": true,
    "feat_struct_sentence_id": true,
    "feat_struct_sentence_text": true,
    "feat_struct_char_start": true,
    "feat_struct_char_end": true,
    "feat_struct_term_id": true,
    "feat_struct_token_id": true,

    "acoustic_boundaries": REMOVED,
    "feat_acoustic_onset": true,
    "feat_acoustic_end": true,
    "feat_acoustic_word_duration": true,
    "feat_acoustic_word_gap": true,

    "sem_composite_psych": REMOVED,
    "feat_psych_authentic": true,
    "feat_psych_analytic": true,
    "feat_psych_tone": true,
    "feat_psych_clout": true
  },
  "output": {
    /* same flat keys as v3 except: */
    "doc_counts": REMOVED,
    "out_count_tokens": true,
    "out_count_content_tokens": true,
    "out_count_sentences": true,
    "out_count_paragraphs": true
  }
}
```

### v3 → v4 migration

1. Copy all `shared`, `predictability` fields verbatim.
2. **Semantic categories:** duplicate v3's `categories_liwc` and `categories_empath` into both word- and doc-level lists. Users retain their existing selections; word and doc start out aligned and can diverge afterward.
3. **Per-column expansion:**
   - v3 `features.structural` (bool) → all 8 `feat_struct_*` set to the same value
   - v3 `features.acoustic_boundaries` (bool) → all 4 `feat_acoustic_*` set to the same value
   - v3 `features.sem_composite_psych` (bool) → all 4 `feat_psych_*` set to the same value
   - v3 `output.doc_counts` (bool) → all 4 `out_count_*` set to the same value
4. All other v3 keys (`lexling_*`, `freq_*`, `pred_*`, `sem_indicators`, `output.*` minus the renamed ones) are carried over unchanged.
5. Rewrite the file with `schema_version: 4`.
6. v2 files chain through v3 → v4 on load (the existing v2-to-v3 path stays; v3-to-v4 runs after).

### Word-level features page (`gui/section_word_features.py`)

Sequence top to bottom:

```
[Section: Output]
  ☑ Word-level CSV

[Card: Structural]
  Document position and offsets — from spaCy.
  ☑ doc_id            ☑ paragraph_id
  ☑ sentence_id       ☑ sentence
  ☑ start             ☑ end
  ☑ term_id           ☑ token_id

[Card: Lexical-linguistic]
  ☑ Tokenization & lemmatization (token, lemma)
  ☑ Part-of-speech tags (upos, xpos)
  ☑ Morphological features (feats)
  ☑ Dependency parse (dep_rel, head_token_id, deps, misc)

[Card: Frequency]
  ☑ General-corpus frequency (SUBTLEX-US, Zipf_Freq)
  Grade-banded (EWFG):
    ☑ K–2     ☑ 3–5     ☑ 6–8     ☑ 9–12     ☑ Overall

[Card: Predictability]
  Surprisal:
    ☑ From GPT-2 → Surprisal
    ☑ From LLM → LLM_Surprisal
  Cloze features:
    ☑ Cloze probability → LLM_Cloze
    ☑ Top-1 match → LLM_TopMatch
  LLM model: [gpt-4o-mini ▾]
  API key: [paste — never saved]

[Card: Semantic indicators]
  Backend: ● Auto  ○ LIWC  ○ Empath
  ☑ Per-word category indicators
  Categories selected for word level: 15 of 73   [Configure word-level categories…]

[Card: Acoustic]
  True acoustic word boundaries from audio (MFA).
  ☑ Onset            ☑ End
  ☑ Word_Duration    ☑ Word_Gap
```

### Document-level features page (`gui/section_doc_features.py`)

```
[Section: Output]
  ☑ Document-level summary

[Card: Counts]
  ☑ n_tokens         ☑ n_content_tokens
  ☑ n_sentences      ☑ n_paragraphs

[Card: Aggregates]
  ☑ Means of word-level metrics (mean_zipf, mean_edu_zipf_*, mean_surprisal, mean_llm_cloze, mean_llm_surprisal)

[Card: Lexical diversity]
  ☑ Type-token ratio (TTR)
  ☑ MTLD

[Card: Readability]
  ☑ Flesch reading ease
  ☑ Flesch-Kincaid grade level

[Card: Category proportions]
  ☑ Compute per-category proportions (count_<cat>, prop_<cat>)
  Categories selected for document level: 20 of 73   [Configure document-level categories…]

[Card: Composite psychological scores]
  Pennebaker (2014) formulas applied to the loaded LIWC-style dictionary.
  These are approximations, not official LIWC-22 scores.
  ☑ psych_authentic    ☑ psych_analytic
  ☑ psych_tone         ☑ psych_clout
```

### Category dialog (`gui/category_dialog.py`)

A `QDialog` subclass that:
- Takes `(backend, level)` constructor args, where `level ∈ {"word", "document"}`.
- Embeds a `CategoryPicker` configured for that level/backend.
- Title bar reads `"Configure word-level categories"` or `"Configure document-level categories"`.
- Window size 640×600, resizable.
- Modal (blocks the main window while open).
- Bottom row: `[Reset to SOP default]` `[Cancel]` `[Done]`.
- Selection auto-saves on every checkbox toggle to the **level-specific** prefs key — e.g. `word_categories_liwc` or `doc_categories_empath`. Cancel doesn't roll back (consistent with current auto-save behavior); Done just closes.

`CategoryPicker` accepts a new `level: str` parameter and uses it to read/write the right prefs key:

```python
key = f"{level}_categories_{backend}"   # e.g., "word_categories_liwc"
```

The previous inline `CategoryPicker` instantiation in `FeaturesSection` is removed; both Word-level and Document-level pages have a count label and a button that opens the dialog.

### Visual polish (cards + density)

Each feature subsection is wrapped in `QFrame#Card`:

- White background (`#FFFFFF`)
- 1px `#D8DCE0` border
- 8px corner radius
- 16px internal padding
- 12px gap between cards (was 20px between sections in batch 2)

`SectionHeader` style stays. `QLabel#SectionHeader` font weight bumps from 700 → 600 (less heavy) and size from 14 → 13px since the card itself provides visual separation.

Per-column checklists use 2-column `QGridLayout` to keep cards compact. EWFG bands and Acoustic columns already fit this pattern; Structural's 8 toggles use a 4×2 grid; Composite psych and Counts use 2×2.

Spacing rules:
- 12px between cards
- 8px between rows inside a card
- 4px between a checkbox label and its inline column-name hint

### Module + import summary

| File | Status | Purpose |
|---|---|---|
| `gui/section_word_features.py` | NEW | Builds the word-level page |
| `gui/section_doc_features.py` | NEW | Builds the doc-level page |
| `gui/category_dialog.py` | NEW | Modal dialog wrapping `CategoryPicker` |
| `gui/section_features.py` | DELETED | Functionality moved to two pages |
| `gui/category_picker.py` | MODIFIED | Accepts `level: str` param; reads/writes level-specific prefs key |
| `gui/main_window.py` | MODIFIED | 5-item sidebar, instantiates the two new sections |
| `utils/preferences.py` | MODIFIED | Schema v4 + v3→v4 migration |

`gui/preflight.py`, `gui/run_dispatch.py`, `gui/section_run.py`, `gui/section_logs.py`, `gui/section_inputs.py`, `gui/widgets.py`, `gui/status_strip.py` are unaffected.

### Behavior preserved

- Run dispatch logic and pipeline calls unchanged.
- CLI in `main.py` unchanged.
- Output CSV column names unchanged (the new per-column toggles only control whether each named column appears in the output; the names themselves are stable).

### Behavior changes (intentional)

- **Word-level vs Document-level configuration is now on separate sidebar pages.** Toggling between them is a one-click switch, no scrolling past unrelated subsections.
- **Categories are independent per level.** Existing v3 users get identical word and doc selections on first launch after migration; from then on they can diverge.
- **Per-column toggles** in 4 subsections give finer-grained CSV output control. Default value preserved via migration.

---

## Open questions (defer to implementation if no answer needed now)

1. **Logo PNG placement** — `assets/logo.png` not yet provided. When it lands, drop it in the sidebar header above the section list (about 200px wide). Until then the sidebar stays as-is.
2. **Composite-psych column-name naming** — the v3 names are `psych_authentic`, `psych_analytic`, `psych_tone`, `psych_clout`. Keep these names for the per-column toggles to avoid breaking downstream consumers (the toggles control whether the column appears; they don't rename).
3. **Modal Cancel semantics** — should Cancel revert all changes made in the dialog session, or close-and-keep (consistent with current auto-save)? Plan assumes close-and-keep. Easy to change to revert-on-cancel later if needed.

---

## Acceptance criteria

1. Launching `python gui.py` shows a 5-item sidebar: Inputs / Word-level features / Document-level features / Run / Logs.
2. The Word-level page shows 6 cards (Output, Structural with 8 column toggles, Lex-ling with 4, Frequency with SUBTLEX + 5 EWFG bands, Predictability with 4 toggles + model + API key, Semantic indicators with backend radios + Configure button, Acoustic with 4 column toggles).
3. The Document-level page shows 6 cards (Output, Counts with 4 column toggles, Aggregates, Lex diversity, Readability, Category proportions with Configure button, Composite psych with 4 column toggles).
4. Clicking "Configure word-level categories…" opens a modal at 640×600 with the active backend's full category list and word-level selections highlighted.
5. Closing the dialog returns focus to the main window; selections auto-saved to `word_categories_liwc` (or `_empath`).
6. Same for the document-level button → `doc_categories_*`.
7. Word and doc category selections can be different after editing one without affecting the other.
8. Per-column toggles in Structural / Acoustic / Composite psych / Counts each independently filter which columns appear in the output CSVs.
9. Loading a v3 `preferences.json` auto-migrates to v4 with no lost data; the 4 expanded toggles all match the prior parent value.
10. Cyan accent (`#00B2D0`) appears on: active sidebar item, primary Run button, checked checkboxes, focus border on inputs, progress bar. Stanford Cardinal is fully gone.
11. The `assets/icon.svg` shows in the title bar and Windows taskbar.
12. Cards are visually distinct from page background — white panels with subtle borders, 12px gaps between them.
