# User Manual (self-contained HTML) — design

Date: 2026-10-27 · Owner: Amilcar Malave · Status: owner said "take the rails" after the
chat description below; this document records it for the build.

## Intent

The PI wants to review every feature SensiExtractor produces; the owner's manager wants a
manual. One document serves both: a **User Manual** that explains what the tool is, how to
install and use it (GUI, `run.py`, Sherlock), and — in depth — every feature: what it means,
what it looks like, where it lands in the outputs, how to load it in MATLAB, how it is
computed, which function computes it, why it matters for EEG, and the papers behind it. It
must be readable **without downloading the project**: a folder of files the owner can zip
and send, or open from the repo, by double-clicking `index.html`.

## Goals

1. A static, self-contained folder `docs/user-manual/` (committed): opens from `file://` by
   double-click, no server, no network needed (outbound links to papers/GitHub excepted).
2. **Three-panel layout** (owner's choice): left outline of the whole manual that unfolds to
   the chapter/feature you're in; content in the middle; right "On this page" list that
   highlights the part you're reading. Search box in the left panel.
3. **Look** (owner: "resemble the spirit of the UI, easier on the eye, more professional";
   took the recommendation): documentation style — white page, serif body text, sans-serif
   headings, numbered sections, figure captions, SENSI cyan `#00B2D0` for links and the
   current place — with the app's look (white cards, cyan controls, the app's greys) for
   hands-on elements: interactive figures, code blocks, output-file boxes. **All colours and
   fonts live in one theme file** (CSS custom properties), so restyling is a one-file swap.
4. Chapters: Introduction · Requirements · Installation (Windows / macOS / Linux) · Using the
   app (GUI walkthrough with real screenshots of every page; then `run.py` with an annotated
   `features.yaml` / `run.yaml`) · Running on Sherlock · Features · Outputs reference (every
   file and folder a run writes, what each column/variable is) · References.
5. **Feature pages**, grouped by level and by the GUI's section cards (Word, Document,
   Sub-word, Sentence, Acoustic), one page per `docs/features/<key>.md`. Standard parts, in
   order: **What it is** · **See it** · **In your output** · **MATLAB** · **How it's
   computed** · **In the code** · **Library** · **EEG rationale** · **Papers**. Extra doc
   sections (Models, Usage, How to set the analysis rate, Example…) keep their own headings.
6. **Single source with the app:** narrative text comes from `docs/features/*.md` (the files
   the in-app ⓘ pop-ups render), so the manual and the app can't disagree. Manual-only
   additions (example choice, figure spec, output columns, MATLAB snippet) live in a
   separate manifest, never by forking the doc text.
7. **No invented values** (owner's rule): every number, table and plot in the manual is the
   real output of SensiExtractor (`run.py`) on the example inputs, produced by the build.
   Figures say so. **Example inputs** (owner: "use some of the examples we have"): the
   LJSpeech clips from the validation set (public-domain read speech, natural voice — safe in
   a public repo and in a shared folder): `LJ002-0321`, `LJ003-0007`, `LJ005-0063`,
   `LJ005-0066`, `LJ019-0027` (audio + transcript) and `LJ006-0284` (audio only, acoustic
   level). The ISC study stimuli are **not** used (the repo is public). Where a text feature
   needs a contrast the clips don't contain (e.g. a predictable vs. a surprising final word),
   one or two invented sentences are run text-only (no audio) — their values are still the
   pipeline's real output.
8. Citations: every paper link from the feature docs works and points to the paper; a
   References page lists all papers with links back to every page that cites them. Pages
   cross-link related features, the output files, and the code on GitHub.

## Non-goals

- Analysis code (TRF fitting etc.) and Python snippets — owner: "don't worry about that for
  now"; MATLAB load / inspect / plot only.
- Hosting (GitHub Pages etc.) and PDF output.
- Rewriting the feature docs' substance. Fixes found while building go into the docs
  themselves (so the pop-ups get them too).

## Anti-goals

- Any number, score or plot not produced by SensiExtractor.
- The manual saying something the app's pop-up contradicts.
- Needing a server, internet, or the project to read it.
- A wall of text: every feature page has at least one visual (figure, table excerpt, or
  annotated file view).

## How it gets made (builder's calls, owner to confirm when reading)

1. **Build script** `scripts/build_manual.py` (maintainer tool, like the other scripts):
   markdown-it-py (already in the env) for Markdown, Jinja2 for page templates, Pygments for
   code highlighting, matplotlib for figures (SVG). No new dependencies. Output is
   deterministic (same inputs → same files) so diffs are reviewable.
2. **Sources** in `dev/manual/` (maintainer material, not shipped):
   - `chapters/*.md` — Introduction, Requirements, Installation, Using the app, Sherlock,
     Outputs reference (written for the manual; more detailed than README/RUNBOOK, which stay
     the short versions).
   - `features.yaml` — the outline (level → section → feature keys, mirroring the GUI) and
     per-feature manual additions: which example, figure type and variants, output file and
     columns/variables to excerpt, MATLAB snippet.
   - `examples/` — copies of the LJSpeech clips and transcripts above, the invented
     text-only sentences, and the scratch `features.yaml`/`run.yaml` that turns every feature
     on (predictability: GPT-2, which needs no token). The audio clips are also copied into
     the manual folder so pages can play them.
   - `templates/`, `static/` (theme CSS, layout CSS, JS).
3. **Example outputs** are produced by running `run.py` on `dev/manual/examples/` and are
   committed under `dev/manual/example_outputs/` (CSVs and `.mat` files are small), so the
   manual can be rebuilt without models; `--refresh-examples` re-runs the pipeline (needs
   the models). Variants (e.g. envelope method, compression, pitch units, mel preset) are
   separate real runs with that one setting changed, stored side by side; the page switches
   between pre-rendered figures — the browser never computes a value.
4. **GUI screenshots** are taken by the build (`--refresh-screenshots`) from the real app,
   offscreen, with a demo config pointing at the examples — so they show the current UI.
5. **Interactivity** (plain JS, no framework): outline fold/unfold and current-page
   highlight; scroll-spy for "On this page"; variant switches on figures; MATLAB/output tabs;
   copy buttons on code; audio players for example sentences; client-side search over a
   prebuilt index (titles, headings, column and variable names, feature keys). Works on
   narrow screens (side panels collapse) and prints cleanly (print stylesheet hides panels).
6. **Math**: the feature docs keep their current plain/monospace math (the app renders the
   same files with Qt, which has no TeX). Manual-only "How it's computed" additions may use
   TeX, rendered by KaTeX bundled in the folder (offline).
7. **MATLAB snippets** are checked against the real files at build time (variable names,
   shapes, column names read with scipy/pandas from the example outputs); they are not run in
   MATLAB here (no MATLAB on this machine) — kept to `readtable` / `load` / indexing /
   `plot` / `imagesc`, and the owner spot-checks one `.csv` and one `.mat` in MATLAB.
8. **Code links** point to the function in the public repo
   (`github.com/edneuro/SENSI-Speech-Features/blob/main/<path>#L<line>`), line numbers
   resolved at build time from the current source.
9. **Checks the build enforces** (fails otherwise): every `docs/features/*.md` appears in the
   outline exactly once; every internal link and anchor resolves; every figure/table comes
   from an existing example output file; every column/variable a page names exists in that
   output. `--check-links` additionally requests each external paper URL and reports
   failures (not part of every build — network).
10. Size target: well under 30 MB (SVG figures, a few seconds of audio, PNG screenshots).

## Builder-check answers (builder's calls)

11. **Doc headings → page parts** by a fixed mapping in the build script, not by editing the
    docs (the pop-ups stay as they are): `Meaning` → What it is; `Math`, `Math / linguistics`,
    `What it is / how it's computed` → How it's computed; `Implementation in SensiExtractor`
    → In the code; `Library`, `EEG rationale`, `Papers` → same; `Datasets` → its own part after
    Papers; any other heading → an extra part in document order. The YAML front matter
    (`name`, `levels`, `gui_toggles`, `output_columns`, `mat_variables`, `libraries`,
    `status`) is authoritative for the page header, output columns and libraries; the
    manifest adds only what front matter lacks. "See it", "In your output", "MATLAB" are
    generated.
12. **Every doc gets exactly one place.** Reference docs that aren't features
    (`llm_credentials`) are placed in the chapter they serve (Using the app → tokens), still
    counted by the "exactly once" check. Features that can't produce real output here get
    their page with a labelled panel instead of a figure: "Not computed in this manual —
    why, and what it would look like" naming the columns, never values: EWFG grade bands
    (no data exists), LIWC indicators (licensed dictionary; the page shows the Empath backend's
    real output, which uses the same column scheme), Llama models (gated, needs a token and
    GPU; predictability pages show GPT-2's real values and state that Llama columns have the
    same shape). The "at least one visual" rule is satisfied there by the annotated column
    layout. `g2p_oov` gets a real example: one example sentence contains a made-up word that is
    not in the pronunciation dictionary.
13. **Dependencies:** markdown-it-py, Jinja2, Pygments and matplotlib are in `conda-lock.yml`
    for all three platforms (pulled in by other packages); the build is a maintainer step, so
    nothing new is declared.
14. **Screenshots:** attempted in the first version (offscreen Qt already works on this
    machine: the reorganization's end-to-end check drove the real window offscreen); if a
    page can't be captured, the chapter ships with the text and the screenshot is added later.
15. **Phasing:** first version = all chapters + every feature page + layout, outline, scroll
    spy, search, code copy, tabs, audio players, print and narrow-screen CSS. Variant switches
    only where listed in 16. KaTeX only if a page needs it.
16. **Variants** (each a real run changing one setting): envelope method (`rms` / `hilbert`),
    pitch units (semitones / Hz), mel preset (`speech` / `music`), predictability context size
    (model default / a short context). Nothing else in the first version.

## Verification

- Build runs clean with all checks; open the folder from disk in a browser (no server) and
  walk every chapter and a sample of feature pages per level, desktop and narrow width.
- Each feature page shows a real figure/table, its output location, and a MATLAB snippet
  whose names match the files.
- External paper links: `--check-links` report reviewed; anything paywalled is still the
  correct DOI.
- Owner reads it with the PI's eyes before sharing.

## Left to the builder

Template markup, exact figure designs per feature, wording of chapters, example sentences
(chosen to make each feature's behaviour visible, e.g. a predictable vs. a surprising final
word for surprisal), search ranking, and the order of features within a section (follow the
GUI).
