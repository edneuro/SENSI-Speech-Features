# SensiExtractor GUI Redesign — Design Spec

**Date:** 2026-05-11
**Author:** Amilcar Malave (with Claude)
**Status:** Draft for review

---

## Motivation

The current `gui.py` is functional but exposes implementation details where it should expose features, stacks all controls vertically, hides content when the window is small, and uses tkinter's default 1990s widget styling. Concretely, the following complaints drove this redesign:

1. **Dictionary backend lives in "Inputs"**, conceptually unrelated to inputs — it belongs next to the Semantic-domain feature it actually controls.
2. **Category dialog is hidden behind a modal**, making it look as if there are only ~15 categories when LIWC ships ~70 and Empath ~200.
3. **Content gets clipped** when the window isn't maximized — the "Run document-level extraction" button is invisible at typical window sizes.
4. **Everything stacks vertically**, with no 2D structure or visual rhythm.
5. **Feature sections leak the implementation** (`spaCy parse`, `GPT-2 surprisal`, `LLM-cloze`, `MFA + timing`) instead of naming the feature being extracted (Structural, Surprisal, Cloze probability, Acoustic boundaries).
6. **No grouping** of related features into the project's stated taxonomy (Structural / Lexical-linguistic / Statistical / Acoustic, per `Feature_Extraction_Guide.md`).
7. **Visual style is dated** — flat ttk widgets, no theme, no spacing rhythm.

This redesign addresses all seven points without changing the underlying extractors, CLI, or output formats.

---

## Out of scope

- No changes to extractor logic, the spaCy pipeline, MFA invocation, LLM cloze computation, or output CSV column names.
- No change to `main.py` CLI flags or behavior.
- No new extractors or features beyond what already exists.
- No change to where models, dictionaries, or caches live on disk.

---

## High-level decisions (all confirmed during brainstorming)

| Decision | Choice |
|---|---|
| GUI framework | **PySide6 (Qt)** — native widgets, real layout managers, QSS theming, HiDPI-correct |
| Layout | **Sidebar + content** (`QStackedWidget` switched by sidebar selection) |
| Run model | **Unified Run** with two truly independent output checkboxes (Word-level CSV, Document-level summary) |
| Word/doc independence | Document-level features compute their own intermediates internally — no dependency on word-level toggles |
| Feature toggles | **One toggle per feature group**, with sub-toggles inside Lexical-linguistic and Frequency-band for column filtering |
| Surprisal methods | **Allow both** GPT-2 and LLM in parallel — keeps current behavior |
| OpenAI API key | **In-session paste only**, never persisted, no env-var requirement |
| Category picker | **Inline, not modal**; backend-specific SOP defaults flagged with a badge |
| Category save | **Auto-save on every change** |
| Validation | **Continuous and inline** — folder paths, analyzer name validated on input; pre-flight warnings inline next to offending toggles |
| Theme | Light / Dark / **System (default)**, single fixed accent color |

---

## Architecture

### File structure

```
SensiExtractor/
├── gui.py                    # Entry point: instantiates QApplication, runs MainWindow
├── gui/                      # NEW package — split from current 5 flat files
│   ├── __init__.py
│   ├── main_window.py        # QMainWindow subclass; sidebar + QStackedWidget routing
│   ├── theme.py              # QSS stylesheets (light/dark), palette helpers, font/spacing tokens
│   ├── widgets.py            # Reusable Qt widgets: SectionHeader, Card, ToggleRow, FolderRow,
│   │                         #   AmberDotIndicator (for inline warnings)
│   ├── section_inputs.py     # 📁 Inputs page
│   ├── section_features.py   # ✨ Features page (Structural, Lex-ling, Frequency,
│   │                         #                   Predictability, Semantic, Acoustic,
│   │                         #                   Document-level summaries)
│   ├── section_run.py        # ▶ Run page + pre-flight summary + cost estimate
│   ├── section_logs.py       # 📜 Logs page (current + past runs, QSyntaxHighlighter colored)
│   ├── status_strip.py       # Persistent bottom status strip
│   ├── category_picker.py    # Inline category picker (replaces gui_categories_dialog)
│   ├── preflight.py          # Continuous pre-flight check engine (run on relevant signal changes)
│   └── run_dispatch.py       # Background-thread dispatch to main.run_word_level_batch /
│                             #   run_doc_level_batch (replaces gui_run.py orchestration)
├── extractors/               # Unchanged
├── utils/preferences.py      # Updated for schema v3 + v2→v3 migration
└── config/openai_pricing.json # NEW — token rates by model, editable without code change
```

The current files `gui_word_tab.py`, `gui_doc_tab.py`, `gui_categories_dialog.py`, and `gui_run.py` are replaced by the modules under `gui/`.

### Why PySide6 (and not CustomTkinter)

The first draft of this spec recommended CustomTkinter for minimum-disruption migration. On reflection, PySide6 is materially better for this tool's lifespan: native widgets render correctly on HiDPI Windows monitors, `QStackedWidget` + `QListWidget` give a real sidebar/content router instead of mimicking one in tkinter, `QTableView` lets the pairing preview be a sortable table, `QSyntaxHighlighter` color-codes the log viewer, QSS gives us full visual control, and Qt's signal/slot model handles the "continuous pre-flight" reactivity cleanly. The migration is ~2-3× more work than CustomTkinter (Qt has a different widget API), but the result is a tool that will not need another redesign for visual reasons.

### Window shell

- **Min size:** 1000 × 700. Opens at 1200 × 800.
- **Top bar (36px):** title left, theme toggle (☀/🌙/💻) and Help button right.
- **Sidebar (220px fixed):** `QListWidget` with custom item delegates — navigation items (Inputs, Features, Run, Logs) styled as buttons; active item highlighted with the accent color. Persistent footer shows analyzer name and active dictionary backend.
- **Content area:** `QStackedWidget` switching between section pages, each wrapped in a `QScrollArea` so content is **always reachable** regardless of window size. This is the root fix for the "options disappear unless I maximize" complaint.
- **Persistent status strip (32px):** custom `QFrame` docked at the bottom via the main `QVBoxLayout`, visible across all sections. Shows job state (Idle / Running / Done / Errored), determinate `QProgressBar`, current file, and Cancel button.

```
┌──────────────────────────────────────────────────────────────────────────┐
│ SensiExtractor                                       [☀/🌙/💻]  [Help]   │
├──────────────┬───────────────────────────────────────────────────────────┤
│ 📁 Inputs    │                                                           │
│ ✨ Features  │           Active section content                          │
│ ▶  Run       │           (CTkScrollableFrame)                            │
│ 📜 Logs      │                                                           │
│              │                                                           │
│              ├───────────────────────────────────────────────────────────┤
│ Analyzer:    │ ●  Running  ▮▮▮▮▮▮▮░░░░  7/14  story_07.wav         [✕]   │
│ Amilcar M.   │                                                           │
│ Backend: LIWC│                                                           │
└──────────────┴───────────────────────────────────────────────────────────┘
```

### Responsiveness

| Width | Behavior |
|---|---|
| ≥ 1100px | Two-column where designed (e.g. category picker shows 2 columns) |
| 800–1100px | Single-column fallback; all content remains accessible via scrolling |
| < 1000px | Window hits min size; inner scrollbars take over |

### Theming

- Two QSS stylesheets (`theme_light.qss`, `theme_dark.qss`) loaded at startup based on the user's setting or the OS palette (detect via `QGuiApplication.styleHints().colorScheme()` on Qt ≥ 6.5). Setting: `light / dark / system`. Default: `system`.
- **Single accent color** — chosen and committed once, not user-configurable. Tentative: indigo `#5B5BD6` (works on both light and dark backgrounds, distinct from typical Windows blues).
- **Fonts:** Segoe UI 10pt body (Windows), SF Pro 13pt (macOS), Inter 10pt fallback (Linux). Section headers 12pt semibold with a 1px divider below. Monospace JetBrains Mono → Consolas → Menlo for log viewer.
- **Spacing:** 8px grid. 24px between section blocks. 12px between form rows. Enforced via QSS margin/padding rules.
- **Corner radius:** cards 8px, buttons 6px, inputs 4px (QSS `border-radius`).
- **Icons:** Qt has good vector-icon support via `QIcon` and SVG. Bundle a minimal set of lucide-style SVGs for sidebar (folder, sparkles, play, scroll) — `Pillow` not needed since Qt loads SVG natively.

---

## Sections

### 📁 Inputs

| Element | Behavior |
|---|---|
| Analyzer name field | Persisted to `preferences.json`. Required for run. **Validated on every keystroke** — empty value shows an amber border and "Required" helper text. |
| Folder rows (Transcripts, Audio, Output) | Each row shows file count next to the path when a valid folder is selected. Browse button opens `QFileDialog.getExistingDirectory`. **Validated on selection and on text change** — non-existent path shows a red border with "Folder does not exist" helper text. Empty audio folder with audio-dependent features enabled shows an amber warning. |
| Pairing preview | Computes matched/unmatched pairs using the same logic as the batch runner, in real time as folders are picked. Shows a count summary and a `QTableView` of paired/unmatched files (sortable). |

The dictionary backend selector **does not live here.** It moved to the Semantic-domain subsection. The sidebar footer still displays the active backend at all times.

### ✨ Features

A single scrollable page with subsections in the project's documented taxonomy. Each subsection has a header, a one-line subtitle, the feature toggle(s), and any method picker as a secondary control attached to that feature.

#### Output (top of page)

```
☑ Word-level CSV               ☑ Document-level summary
```

Toggling Document-level summary off hides the "Document-level summaries" subsection at the bottom of the page.

#### Structural

One toggle, from spaCy. Subtitle lists column names: `doc_id, paragraph_id, sentence_id, sentence, start, end, term_id, token_id`.

#### Lexical–linguistic

Four sub-toggles (column filtering — spaCy still runs once per stimulus regardless):

- **Tokenization & lemmatization** → `token`, `lemma`
- **Part-of-speech tags** → `upos`, `xpos`
- **Morphological features** → `feats`
- **Dependency parse** → `dep_rel`, `head_token_id`, `deps`, `misc`

#### Frequency

- **General-corpus frequency** (single toggle) — Source: SUBTLEX-US (Zipf scale). Column: `Zipf_Freq`.
- **Grade-banded frequency** — Source: EWFG (Zeno et al. 1995). Each band is its own checkbox; pick the ones you want as columns:
  - ☑ K–2     → `Edu_Zipf_K2`
  - ☑ 3–5     → `Edu_Zipf_3to5`
  - ☑ 6–8     → `Edu_Zipf_6to8`
  - ☑ 9–12    → `Edu_Zipf_9to12`
  - ☑ Overall → `Edu_Zipf_Overall`

  Default: all bands on. The band checkboxes are disabled (greyed) and show an inline amber warning if `dictionaries/edu_zipf.parquet` is missing — "EWFG lookup not bundled (license-bound). See README for acquisition." This is the continuous-pre-flight pattern in action.

#### Predictability

```
Surprisal — -log₂ P(word | preceding context), in bits.
  ☑ From GPT-2  → column: Surprisal (legacy baseline)
  ☑ From LLM    → column: LLM_Surprisal

Cloze features (single LLM forward pass)
  ☑ Cloze probability    → LLM_Cloze
  ☑ Top-1 match          → LLM_TopMatch

LLM model:  [gpt-4o-mini ▾]   (applies to all LLM features above)
API key:    [••••••••••••••••] ✓ from $OPENAI_API_KEY
            (row only shown when an OpenAI model is selected)
```

Both surprisal methods can run in parallel — `Surprisal` (GPT-2) and `LLM_Surprisal` are kept as separate columns, preserving current behavior documented in `Feature_Extraction_Guide.md`.

The LLM model picker applies to all three LLM features. When an OpenAI model is selected, an API-key row appears below it:

- **In-session paste only.** Empty by default each launch. User pastes a key, it's held in memory for the duration of the run, never written to disk.
- Field is masked (`••••••••`); a "Show" button briefly reveals the value if the user needs to verify the paste.
- **Never written to `preferences.json`** and **does not read `$OPENAI_API_KEY`**. Pasting is the only ingress. Reasoning: makes the GUI safe on shared lab machines with no setup ceremony; the user is responsible for storing their key somewhere outside the app (password manager, lab vault).
- Local HF models (Llama variants) hide the API-key row entirely.

#### Semantic domain

```
Backend: ● LIWC (dictionaries/liwc-2022.dic)
         ○ Empath (open-source fallback)
         ○ Auto

☑ Category indicators (per-word, content POS only)
  Categories — 15 of 73 selected   [SOP default]  [All]  [None]
  [🔍 Search categories…]

  ┌──────────────────────────────────────────┐
  │ ☑ affect    ⓘ ★ SOP   ☑ cogproc  ⓘ ★ SOP │
  │ ☑ emo_pos   ⓘ ★ SOP   ☑ emo_neg  ⓘ ★ SOP │
  │ ... (full backend list, scrollable)      │
  └──────────────────────────────────────────┘

☑ Composite psychological scores (doc-level, dictionary backend only)
  → psych_authentic, psych_analytic, psych_tone, psych_clout
  Computed from Pennebaker et al. (2014) formulas applied to the
  loaded LIWC-style dictionary. These are approximations — for the
  official LIWC-22 numbers, run the LIWC desktop software (liwc.app).
  Used dictionary: dictionaries/liwc-2022.dic  (shown for reproducibility)
```

**Backend radio** switches active backend. Switching reloads the category list, recomputes "X of N selected," and shows that backend's SOP defaults as flagged.

**Category picker** is inline (no modal). Two columns at ≥1100px width, one column below that. Each row shows:
- Checkbox
- Category name
- ⓘ info icon — hover/click reveals description + 5–10 example words pulled from the active dictionary (LIWC: words tagged with that category in the .dic file; Empath: seed words from `Empath().cats[name]`)
- `★ SOP` badge if this category is in the backend's SOP default list

**SOP default lists** (stored in `utils/preferences.py`):

- LIWC: `affect, emo_pos, emo_neg, cogproc, social, body, motion, percept, time, space, money, work, family, relig, focuspresent` *(exact list to be finalized against LIWC-22 category names during implementation)*
- Empath: `social, cognition, emotion_positive, emotion_negative, body, motion, perception, time, space, math, science, work, family, communication, money` *(current 15)*

The two lists are deliberately backend-native — no cross-backend name mapping — because LIWC's `emo_pos` and Empath's `emotion_positive` are not strictly equivalent.

**Auto-save**: every checkbox toggle writes to `preferences.json`. `[SOP default]`, `[All]`, `[None]` are one click away if the user wants to reset.

#### Acoustic

One toggle, from MFA:

- **Word boundaries** — Columns: `Onset, End, Word_Duration, Word_Gap`. Method label: "MFA (Montreal Forced Aligner)".

If MFA is not installed (conda env missing or not activated), the toggle is shown disabled with a tooltip explaining how to enable it. Reported in the Run pre-flight check.

#### Document-level summaries

Shown only when "Document-level summary" output checkbox is checked. Each toggle controls a column group in the doc-level master CSV:

- ☑ **Counts** → `n_tokens, n_content_tokens, n_sentences, n_paragraphs`
- ☑ **Aggregates** → means of the word-level metrics above
- ☑ **Lexical diversity** → `ttr, mtld`
- ☑ **Readability** → `flesch_reading_ease, flesch_kincaid_grade`
- ☑ **Category proportions** → `count_<cat>, prop_<cat>` per active semantic category

### ▶ Run

```
Output (mirror of Features page; can be edited here too)
  ☑ Word-level CSVs       (one per stimulus, ~14 files)
  ☑ Document-level summary  (single multi-row CSV)

── READINESS ──
  ✓ Ready to run
  (or: ⚠ 2 warnings — review Features page)

  When warnings exist, they're listed compactly with a "Jump to setting"
  link that navigates straight to the offending toggle. Hard errors
  (missing folder, missing analyzer name) disable the Run button entirely.

── COST ESTIMATE ──  (only shown if any OpenAI model is selected)
  Model:           gpt-4o-mini
  Input tokens:    ~ 47,200      (counted via tiktoken)
  Output tokens:   ~ 14,300      (≈ 1 token per word target)
  Rates (last updated 2026-05-11, from config/openai_pricing.json):
                   $0.15 / 1M input    $0.60 / 1M output
  Estimated:       ≈ $0.016
  ⚠ OpenAI pricing changes — verify at openai.com/api/pricing

           [ ▶  Run extraction ]
```

**Continuous pre-flight, not Run-page-only.** Validation runs reactively whenever a relevant setting changes (folder paths, dictionary backend, feature toggles, LLM model, API key field). Warnings appear **inline** next to the offending toggle in Features as a small amber dot with a tooltip, e.g.:

- `☑ Grade-banded frequency  🟡 EWFG parquet not found — columns will be NaN`
- `☑ Surprisal from GPT-2    🟡 hf_cache/gpt2 missing — first run will download (~500 MB)`
- `☑ Cloze probability       🟡 OpenAI model selected but no API key pasted`
- `☑ Word boundaries (MFA)   🔴 MFA binary not found — feature disabled`

The Run page just summarises: "Ready" or "N warnings." This pattern means the Features page is the single source of truth for what's wrong, and you fix issues where they actually live.

**Cost estimate** uses `tiktoken` for accurate per-stimulus input-token counts (no chars×0.25 heuristic). Output tokens estimated at 1 per target word (LLM cloze emits a single next-token logit per word). Rates are loaded from `config/openai_pricing.json` (editable without code change); the file ships with current rates and a "last updated" timestamp surfaced in the UI. Local models show "no API cost." `tiktoken` is added as a dependency.

**Run button** dispatches to the batch runner in a background thread (the current `_kick_off` and `_run_*_batch` logic in `gui_run.py` moves into `gui/run_dispatch.py`). The GUI calls the pipeline functions in `main.py` directly — it does not shell out via CLI. The two output checkboxes act independently: each turns on whether `run_word_level_batch` and/or `run_doc_level_batch` is invoked for this run. **Document-level dispatch is fully self-contained** — it computes every intermediate it needs internally, regardless of which word-level toggles are on. The word-level toggles only control what columns appear in the word-level CSV, never what document-level can compute.

### 📜 Logs

- **Current-run log:** mirrors what was previously in the bottom `scrolledtext` widget, with **Info / Warn / Error** filter chips and a text search box. Auto-scrolls unless the user scrolls up (standard log-viewer behavior).
- **Past runs list:** populated from `run_<ts>.log` files in the configured Output folder. Shows timestamp, status (Running / Done / Errored), file count, runtime, and a [View] button that loads that file into the viewer.
- **[Copy]** and **[Open log file…]** buttons at the top of the current-run log.

---

## Persistent status strip

Anchored at the bottom of the content pane, visible regardless of active section. States:

| State | Display |
|---|---|
| Idle | `○  Idle   Ready to run.` |
| Running | `●  Running  [▮▮▮░░░] 7/14   story_07.wav   [✕]` |
| Done | `✓  Done — 14/14 files in 3m 42s.   [Open output folder]` |
| Errored | `⚠  Errored on 2 files — see Logs.   [Go to Logs]` |

---

## Preferences schema (v3)

```json
{
  "schema_version": 3,
  "shared": {
    "analyzer_name": "",
    "last_transcripts_folder": "",
    "last_audio_folder": "",
    "last_output_folder": "",
    "theme": "system"
  },
  "features": {
    "structural": true,
    "lexling_tokens": true,
    "lexling_pos": true,
    "lexling_morph": true,
    "lexling_dep": true,
    "freq_subtlex": true,
    "freq_ewfg_k2": true,
    "freq_ewfg_3to5": true,
    "freq_ewfg_6to8": true,
    "freq_ewfg_9to12": true,
    "freq_ewfg_overall": true,
    "pred_surprisal_gpt2": true,
    "pred_surprisal_llm": true,
    "pred_cloze": true,
    "pred_topmatch": true,
    "sem_indicators": true,
    "sem_composite_psych": true,
    "acoustic_boundaries": true
  },
  "predictability": {
    "llm_model": "gpt-4o-mini"
  },
  "semantic": {
    "backend": "auto",
    "categories_liwc": ["affect", "emo_pos", "..."],
    "categories_empath": ["social", "cognition", "..."]
  },
  "output": {
    "word_level_csv": true,
    "document_summary": true,
    "doc_counts": true,
    "doc_aggregates": true,
    "doc_lex_diversity": true,
    "doc_readability": true,
    "doc_category_proportions": true
  }
}
```

### Migration v2 → v3

Triggered when `load_preferences()` reads a file with `schema_version == 2`.

1. Copy `shared.analyzer_name` and folder fields verbatim.
2. **Features:** map old `word_level.feature_toggles` and `document_level.feature_toggles` into the new flat `features` dict and `output` dict:
   - `word_level.spacy_parse` → split into `structural`, `lexling_tokens`, `lexling_pos`, `lexling_morph`, `lexling_dep` (all set to the same boolean).
   - `word_level.zipf_freq` → `freq_subtlex`
   - `word_level.edu_freq` → all five `freq_ewfg_*` bands set to the same boolean
   - `word_level.gpt2_surprisal` → `pred_surprisal_gpt2`
   - `word_level.llm_cloze` → `pred_surprisal_llm` AND `pred_cloze` AND `pred_topmatch` (set to same boolean)
   - `word_level.sem_domain` → `sem_indicators`
   - `word_level.alignment` → `acoustic_boundaries`
   - `document_level.counts` → `output.doc_counts`
   - `document_level.mean_*` → `output.doc_aggregates`
   - `document_level.ttr`, `mtld` → `output.doc_lex_diversity`
   - `document_level.flesch_*`, `fk_*` → `output.doc_readability`
   - `document_level.category_proportions` → `output.doc_category_proportions`
   - `document_level.liwc_*` → `features.sem_composite_psych` (note: CSV column names also change from `liwc_authentic/analytic/tone/clout` to `psych_authentic/analytic/tone/clout` — see "Behavior changes" below)
   - Both output formats default to true: `output.word_level_csv = true`, `output.document_summary = true`.
3. **Semantic categories:** copy `word_level.semantic_categories` to whichever list matches the active backend at migration time. The other list defaults to that backend's SOP set.
4. **LLM model:** copy `word_level.llm_model` to `predictability.llm_model`.
5. **Dictionary backend:** copy `shared.dictionary_backend` to `semantic.backend`.
6. Rewrite the file with `schema_version: 3`.

The migration runs once and produces a clean v3 file. The v2 schema constants remain in `utils/preferences.py` only as needed by the migration function.

---

## Behavior changes (intentional)

- **Two run buttons → one run button.** With independent output checkboxes, a single Run produces word-level CSVs, document-level summary, or both. The word-level and document-level pipelines remain fully independent — neither requires the other; document-level computes whatever intermediates it needs internally.
- **Modal category dialog → inline picker.** No more X-out-saves-vs-Cancel-discards ambiguity; auto-save replaces both behaviors.
- **Dictionary backend moves out of Inputs** into Semantic domain.
- **Continuous pre-flight, inline warnings.** Environment problems (missing model, missing dict, missing API key) are surfaced as amber/red indicators next to the offending toggle as soon as they're relevant, not only at Run time.
- **Cost estimate** for OpenAI runs — new affordance for budgeting, with explicit token counts via `tiktoken` and a clearly-dated rate table in `config/openai_pricing.json`.
- **CSV column rename** — `liwc_authentic, liwc_analytic, liwc_tone, liwc_clout` become `psych_authentic, psych_analytic, psych_tone, psych_clout`. These are Pennebaker-formula approximations applied to a LIWC-style dictionary; the previous naming implied they were official LIWC scores, which they are not. Downstream consumers of the document-level CSV must update column references. The Feature Extraction Guide will be updated to document this.
- **EWFG band granularity** — `freq_ewfg` toggle splits into five per-band toggles (`freq_ewfg_k2`, `freq_ewfg_3to5`, etc.). v2 migration sets all five to the previous single value.

## Behavior preserved (no change)

- All extractor logic, output CSV schemas, file naming, and run-log artifacts (`run_<ts>.log`, `run_<ts>_manifest.json`, `run_<ts>_no-match.csv`).
- CLI behavior in `main.py` (flags, levels, defaults).
- Conda env, model caches, dictionary file locations.
- Pairing logic (top-level scan, case-insensitive base-filename matching).
- Threading model — runs happen in a background thread, GUI stays responsive.

---

## Open questions to resolve during implementation

1. **LIWC SOP default list** — the 15-item starter above is a placeholder. The actual LIWC-22 category names need to be confirmed against `dictionaries/liwc-2022.dic` and reviewed for EEG relevance with the parent project's stakeholders.
2. **Past-runs list pagination** — if a lab accumulates hundreds of runs, scanning every `run_*.log` in the output folder on every Logs-section visit becomes slow. Likely fine for the foreseeable future; revisit if it becomes a problem.
3. **PySide6 packaging** — confirm `PySide6` installs cleanly in the existing `sensi` conda environment alongside Kaldi/MFA. Adds ~80 MB to the install but should not conflict.

## Future work (raise in parent project)

- **Developmental-frequency alternative to EWFG.** EWFG is license-bound and not bundled. Open alternatives that overlap the *frequency-stratified-by-developmental-stage* concept don't exist cleanly: Kuperman et al. (2012) Age of Acquisition norms give an age-of-acquisition *rating* (when speakers learned the word) rather than frequency at each age, so it's a different construct. CHILDES gives child-directed speech frequency but not banded by reader grade. Worth raising in `../Data Curation/` whether a SUBTLEX-CHILDES-style banded resource exists or should be derived for the project. Until then, EWFG remains the only grade-banded source and the GUI emits NaN when its parquet is missing.

---

## Acceptance criteria

A reviewer should be able to verify the redesign succeeded by:

1. Launching `python gui.py` and seeing a PySide6 sidebar-and-content window, not a tabbed tkinter dialog.
2. Resizing the window to 1000×700 and confirming every feature, the Run button, and the status strip remain reachable (scroll if needed) — no clipped content.
3. Navigating to Features and seeing six subsections (Structural, Lexical-linguistic, Frequency, Predictability, Semantic domain, Acoustic) plus Document-level summaries, each with feature-first naming.
4. Selecting an OpenAI model in Predictability and seeing the in-session API-key paste row appear; switching to a local model and seeing it disappear. Confirming the key is never written to `preferences.json`.
5. Opening Semantic domain → seeing all backend categories inline with SOP-default flags; clicking [SOP default] resets to the backend's curated set; switching backend reloads the list.
6. With the EWFG parquet missing, opening Features and seeing the Grade-banded frequency band checkboxes disabled with an inline amber warning — *before* navigating to Run.
7. Pasting an OpenAI key, then deleting one transcript file, and seeing the pairing preview update live and the Run button warning indicator change.
8. Running an extraction and watching the status strip update while navigating between sections.
9. Loading a v2 `preferences.json` from before this change and confirming it auto-migrates to v3 with no lost toggles; verifying the `liwc_*` doc-level columns become `psych_*` in the v3-era output.
10. Toggling the OS theme between light and dark while the app is open and confirming the GUI re-themes within 1 second (signal listener on `colorScheme()` changes).
