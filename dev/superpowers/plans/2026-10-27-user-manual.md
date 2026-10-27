# User Manual Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** a self-contained, double-click HTML User Manual in `docs/user-manual/`, built from
`docs/features/*.md` plus real SensiExtractor output on public-domain examples.

**Architecture:** one maintainer script `scripts/build_manual.py` (package `scripts/manual/`
for its parts) reads the feature docs, a manifest and committed example outputs, and writes
static pages with Jinja2. Example outputs come from real `run.py` runs (`--refresh-examples`);
GUI screenshots from the real app offscreen (`--refresh-screenshots`).

**Tech Stack:** Python (markdown-it-py, Jinja2, Pygments, matplotlib, pandas, scipy — all in
`conda-lock.yml`), plain HTML/CSS/JS (no framework, no CDN).

**Spec:** `dev/superpowers/specs/2026-10-27-user-manual-design.md`

## Global Constraints

- Opens from `file://` with no server and no network: no `fetch`/XHR of local files, no CDN;
  all data a page needs is inlined or in `<script src>` files in the folder.
- Every number, table and figure comes from a file under `dev/manual/example_outputs/`
  written by `run.py`; the build fails if a named file/column/variable is missing.
- Inputs: LJSpeech `LJ002-0321`, `LJ003-0007`, `LJ005-0063`, `LJ005-0066`, `LJ019-0027`
  (audio + transcript), `LJ006-0284` (audio only), plus invented text-only sentences. No ISC
  stimuli anywhere in the output.
- Feature narrative comes only from `docs/features/*.md` (front matter authoritative for
  name, levels, toggles, columns, `.mat` variables, libraries, status); never copied into
  sources.
- Every `docs/features/*.md` placed exactly once (feature page or chapter).
- All colours/fonts as CSS custom properties in `static/theme.css` only; `layout.css` uses
  the variables. SENSI cyan `#00B2D0`; app greys from `sensi/gui/theme.py`.
- Code links: `https://github.com/edneuro/SENSI-Speech-Features/blob/main/<path>#L<n>`.
- Size of `docs/user-manual/` < 30 MB. Deterministic output (sorted iteration, fixed
  matplotlib `svg.hashsalt`, no timestamps in pages except the build date in the footer).
- No Claude attribution in commits.

## Review Focus

1. Opening `index.html` by double-click (file://) on Windows: every link, script, style,
   image and audio file uses relative forward-slash URLs that work from any page depth.
2. Narrow window (< 900 px): panels collapse behind toggles; content readable; no
   horizontal scroll.
3. Search with no matches / with a column name (`gpt2_Surprisal`) / with a feature key: shows
   "no results" or jumps to the right page section.
4. A feature whose example output lacks a named column (e.g. a disabled feature): build
   fails with the feature key and the missing name, unless the manifest marks it
   `not_computed` (then the page shows the labelled panel).
5. Printing a feature page: panels hidden, figures and code fully visible.

Tests for 1, 3 (index content), 4 go in Task 6's test file; 2 and 5 are checked by hand in
Task 7 (browser).

---

### Task 1: Examples and real example outputs

**Files:**
- Create: `dev/manual/examples/{transcripts,audio}/` — copies of the six LJ clips and five
  transcripts, plus two invented text-only transcripts with no audio: `predictable.txt`
  ("She spread butter on her toast.") and `surprising.txt` ("She spread butter on her
  violin."). G2P needs audio, so `g2p_oov` uses an LJ word missing from the MFA dictionary
  (checked against `english_us_arpa.dict`) if one exists; otherwise its page is
  `not_computed` with that reason.
- Create: `scripts/manual/examples.py` (configs are generated in code, not stored)
- Output (committed): `dev/manual/example_outputs/<variant>/…` (what `run.py` writes, minus
  `logs/` and `Montreal Forced Alignment/`)

**Interfaces:**
- Produces: `VARIANTS: dict[str, dict]` — `"base"` = every boolean in `features`/`output`
  on (as `scripts/calibrate_estimates.py` does), `predictability.models = ["gpt2"]`,
  EWFG toggles off; `"env_hilbert"` (acoustic only, `acoustic_ts.env_method="hilbert"`),
  `"pitch_hz"` (word level, `prosody.pitch_units="hz"`), `"mel_music"` (acoustic only,
  `acoustic_ts.mel_preset="music"`), `"ctx_short"` (word level,
  predictability context 32 tokens — use the pref key the GUI spinbox writes).
- Produces: `refresh_examples(out_root: Path) -> None` — for each variant: write
  `features.yaml`/`run.yaml` to a temp config dir via `sensi.utils.preferences`
  (`SENSI_CONFIG_DIR`), run `python run.py --config-dir <tmp>` in a subprocess (PATH with
  `env`, `env/Scripts`, `env/Library/bin` as POSIX-safe entries — never `C:`-style in a
  `:`-separated PATH), check the log line `[mfa] N stimuli:` shows alignment ran for the
  audio variants, copy outputs into `out_root/<variant>/`.

- [ ] Step 1: copy the inputs; write the invented sentences.
- [ ] Step 2: implement `scripts/manual/examples.py`; `python scripts/build_manual.py --refresh-examples --examples-only`.
- [ ] Step 3: verify: `base` has word/ syllable/ phoneme/ sentence/ document/ acoustic/ files
  for the LJ clips, word/ files for the text-only sentences, and no `Edu_Zipf_*` columns;
  each variant differs from `base` only in its setting's outputs.
- [ ] Step 4: commit (`feat(manual): example inputs and real example outputs`).

### Task 2: Manifest

**Files:** Create `dev/manual/features.yaml`.

**Interfaces:**
- Produces schema:
  ```yaml
  levels:
    - id: word            # word | document | subword | sentence | acoustic
      title: Word level
      sections:           # mirror the GUI cards on that page, in GUI order
        - title: Structural
          features:
            - key: structural_ids        # docs/features/<key>.md
              example: LJ002-0321        # stem in example_outputs/base
              output: word/{stem}_features.csv   # or .mat; several allowed (list)
              show: [doc_id, sentence_id, token_id, token]   # columns / .mat vars
              figure: table              # see Task 4 figure types
              variants: []               # keys of VARIANTS
              not_computed: null         # or a one-line reason
  chapter_docs: {llm_credentials: usage}  # docs placed in chapters, not feature pages
  ```
- [ ] Step 1: write the outline from the GUI pages (`sensi/gui/section_*.py` card order) and
  every doc's front matter; every one of the 49 docs placed once.
- [ ] Step 2: commit.

### Task 3: Site skeleton — templates, theme, layout, JS

**Files:** Create `scripts/build_manual.py` (CLI), `scripts/manual/{__init__,docs,site}.py`,
`dev/manual/templates/{base,chapter,feature,references}.html.j2`,
`dev/manual/static/{theme.css,layout.css,manual.js}`.

**Interfaces:**
- `docs.load_feature_doc(path) -> FeatureDoc` (front matter dict + ordered
  `[(part_id, title, html)]` using the heading map in spec item 11; reference links
  resolved; `[Author Year]` anchors kept and made unique per page).
- `site.build(out_dir: Path, *, manifest, docs, outputs_root, screenshots_dir) -> BuildReport`
  writes `index.html`, `chapters/*.html`, `features/<level>/<key>.html`, `references.html`,
  `assets/…`, `assets/search-index.js` (`window.MANUAL_INDEX = [...]`).
- Page shell: left outline (whole manual, current branch expanded, current page marked),
  content, right "On this page" (scroll-spy); breadcrumb; prev/next links; footer with build
  date + commit hash.
- [ ] Step 1: templates + CSS (style B + app cards, per spec Goal 3) + JS (outline toggle,
  scroll-spy, search over `MANUAL_INDEX`, tabs, copy buttons, variant switch, narrow-screen
  panel toggles, print CSS).
- [ ] Step 2: build with placeholder parts; open `docs/user-manual/index.html` from disk in
  the browser pane; check layout at 1400 px and 380 px.
- [ ] Step 3: commit.

### Task 4: Generated parts — See it, In your output, MATLAB, In the code

**Files:** Create `scripts/manual/{figures,excerpts,matlab,codelinks}.py`.

**Interfaces:**
- `figures.render(kind, spec, outputs_root, out_svg) -> None`; kinds: `table` (no figure,
  excerpt only), `word_timeline` (words on Onset/End with a value bar per word),
  `contour` (pitch/volume per word or syllable over time), `timeseries` (acoustic 1-D vs
  `t`), `spectrogram` (log-mel / MFCC image), `matrix` (embedding layers × dims, or
  similarity), `doc_bars` (document summary values across the LJ clips), `alignment`
  (phones/syllables strip under the waveform), `categories` (Sem_* indicators per word).
  All values read from the output files; caption states file + variant + "computed by
  SensiExtractor".
- `excerpts.html_table(path, columns, rows=12) -> str` (highlight the feature's columns);
  `.mat` → variable list with shapes/dtypes read via scipy.
- `matlab.snippet(output_path_pattern, names, kind) -> str` — `readtable` for CSV,
  `load` for `.mat`; one plot line matching `kind`; names checked against the real file.
- `codelinks.resolve(spec "module:function") -> (path, line)` via `ast`; error if missing.
- [ ] Step 1: implement; render every manifest feature; fix manifest until the build has no
  errors.
- [ ] Step 2: eyeball a sample per level in the browser.
- [ ] Step 3: commit.

### Task 5: Chapters and screenshots

**Files:** Create `dev/manual/chapters/{introduction,requirements,installation,usage,
sherlock,outputs}.md`, `scripts/manual/screenshots.py`, `docs/user-manual/assets/screens/`.

**Interfaces:** `screenshots.capture(out_dir) -> list[Path]` — `QT_QPA_PLATFORM=offscreen`,
scratch `SENSI_CONFIG_DIR` pointing at the examples, one PNG per sidebar page
(`MainWindow`, select each page, `widget.grab()`), fixed window size 1280×820.

- [ ] Step 1: write chapters from README, `sherlock/RUNBOOK.md`, `setup.*`,
  `config/*_defaults.yaml`, `sensi/gui/*` (facts only from those sources); `outputs.md`
  lists every folder/file a run writes, linking each column to its feature page.
- [ ] Step 2: capture screenshots; reference them in `usage.md`.
- [ ] Step 3: commit.

### Task 6: Checks and tests

**Files:** `scripts/manual/checks.py`; Test: `tests/scripts/test_build_manual.py`.

- `checks.run(out_dir, manifest, docs) -> list[str]` (errors): each doc placed once; every
  `href`/`src` relative and resolving on disk (forward slashes); every `#anchor` exists;
  every manifest `show` name exists in its output; folder size < 30 MB; no ISC stem
  (`1_FB_P`, `1_M`, `2_FB_P`, `2_M`, `3_M`) in any output file.
- `--check-links`: HEAD/GET each external URL (timeout 10 s), print a report.
- Tests (test-after): build into `tmp_path` from the committed outputs →
  `checks.run(...) == []`; `MANUAL_INDEX` contains `gpt2_Surprisal` and `acoustic_envelope`;
  a manifest entry naming a missing column → build raises naming the key and column; a
  `not_computed` entry renders its reason and no figure.
- [ ] Step 1: implement, run `pytest tests/scripts/test_build_manual.py -q`; full suite.
- [ ] Step 2: commit.

### Task 7: Verify and review

- [ ] Build clean; `--check-links` report reviewed (fix wrong DOIs in the feature docs).
- [ ] Browser walk from `file://`: every chapter, two features per level, narrow width,
  print preview, search cases from Review Focus.
- [ ] Fresh whole-branch review (most capable model); fix; update `dev/master_plan.md` §3,
  README (link to the manual), CLAUDE.md (local) structure entry.
