# Repo reorganization — implementation plan

**Goal:** the layout in the spec, with every output byte-identical and no
connection left pointing at an old path.
**Spec:** `docs/superpowers/specs/2026-10-27-repo-reorganization-design.md`
(moves to `dev/superpowers/specs/` in Task 6).
**Execution:** done in this session, one commit per task (pure `git mv`
commits separate from edit commits), suite after every task; one fresh
whole-branch review at the end on the most capable model.

## Global constraints

- Outputs identical to the baseline run of `main` @ 56e5672 on the
  validation corpus (all levels on, scratch config; manifests' timestamps,
  durations, versions and absolute paths excluded from the comparison).
- MFA still runs at most once per document; persisted alignments reused.
- `MKL_THREADING_LAYER` still set first thing in `gui.py`, `run.py`,
  `tests/conftest.py`, and `sensi/pipeline.py`.
- Settings files (`config/features.yaml`, `run.yaml`) load unedited.
- No Claude attribution in commits.
- Never edit `env/`; the user's live `config/` is never written.

## Fit with how Claude Code works on this repo

- `CLAUDE.md` must stay at the repo root (Claude Code loads it from there):
  untracked via `.gitignore`, contents updated to the new layout.
- `.claude/` and `.superpowers/` stay at the root (tool-owned, ignored).
- The superpowers skills default to `docs/superpowers/{specs,plans}`;
  ours move to `dev/superpowers/`, so `CLAUDE.md` gets an explicit line:
  "specs and plans go in `dev/superpowers/specs|plans/`" (skills honour a
  user preference over their default).
- `master_plan.md` moves to `dev/`; `CLAUDE.md`'s "Start here" points there.
- The memory folder is keyed by the repo path, which does not change;
  memory files naming moved files are updated.

## Tasks

### 1 · Package moves (pure `git mv`)
`extractors/ gui/ utils/` → `sensi/…`; `main.py` → `sensi/pipeline.py`;
root `__init__.py` → `sensi/__init__.py`. Delete leftover untracked
`__pycache__/` in the emptied folders. Commit (tree will not import yet).

### 2 · Imports and module-path strings
Rewrite, by script with a dry-run listing first:
`from|import extractors|gui|utils` → `sensi.…`; `from main import` /
`import main` → `sensi.pipeline`; quoted module paths used by
`mock.patch`, `monkeypatch.setattr`, `importlib`, and the planner's
`_lazy("extractors.…")`; `from __init__ import __version__` →
`from sensi import __version__`. Then `git grep` for every old form
(quoted and unquoted) must return nothing outside `dev/`/history.
Suite green. Commit.

### 3 · One paths module + `data/`
`sensi/paths.py`: `PROJECT_ROOT, DATA_DIR, DICT_DIR, ASSETS_DIR,
DOCS_FEATURES_DIR, CONFIG_DIR_DEFAULT, HF_CACHE_DIR, MFA_DIR, GENSIM_DIR`.
Replace every `Path(__file__)…parent.parent / X` anchor (aligner G2P,
educator_freq, lexical_norms, semantic_domain, word2vec default,
phoneme_surprisal dict, feature_docs, info_popup, theme, preflight,
preferences config dir, local_caches, gui.py icon, scripts). `git mv
dictionaries data/dictionaries`. `local_caches`: point at `data/…`; if a
`data/X` is missing but a legacy top-level `X/` exists, use the legacy one.
Launchers (`gui.py`, `run.py`, `scripts/download_models.py`) first move
legacy top-level model folders into `data/` (rename, print what moved;
on failure keep using the legacy folder and say so). Move this machine's
`hf_cache/ mfa_data/ gensim_data/` into `data/`. Update the cache test.
Suite green. Commit.

### 4 · Retire the `main.py` CLI
Delete `build_parser()`, `main()`, the `__main__` block and
`_resolve_sem_backend` if only `main()` uses it. Tests: delete
`tests/test_main_cli.py`; tests that used `build_parser()` for default
args build them with `plan_from_prefs` + `to_namespace` (the GUI/run.py
path). Check no runner reads an attribute only the parser set (diff the
attribute sets parser-vs-planner first). Suite green. Commit.

### 5 · Helper scripts
`git mv` `sensi/utils/download_models.py`, `download_lexical_norms.py`,
`validate_cloze.py` → `scripts/` (bootstrap `sys.path` like the other
scripts). Update `setup.sh`, `setup.bat` (and the `runGUI.bat` they write),
`scripts/import_smoke.py`, `.github/workflows/env-verify.yml`,
`scripts/make_mac_app.sh`, `scripts/calibrate_estimates.py`.
Suite + import smoke green. Commit.

### 6 · Docs, dev/, CLAUDE.md, Sherlock
`git mv` `docs/{master_plan,changelog,feature_dependencies,feature-docs-audit,AoA&MRC}.md`
and `docs/superpowers` → `dev/`; `validation/verify_phoneme_batch.py` →
`dev/validation/`; local `validation/ audit/ unrelated/` → `dev/…`
(untracked, moved on disk). `in_app_documentation_needs_fix.md` → `dev/`,
untracked (owner's notes). Fix relative links in moved docs and in
`docs/features/*.md`, `README.md` (layout tree; remove the Command line
section; model/dictionary paths; `scripts/` helpers), code comments naming
`docs/feature_dependencies.md`. Sherlock: `RUNBOOK.md` rsync excludes
(`data/hf_cache data/mfa_data data/gensim_data`, NOT `data/dictionaries`),
Word2Vec path, any `validation/` path; `run_container.sbatch`;
`Apptainer.def` comments. `.gitignore` + `pytest.ini` `norecursedirs`.
`git rm --cached CLAUDE.md`, update its contents; update memory files.
Feature-doc tests + suite green. Commit.

### 7 · Verify
Full suite; `scripts/import_smoke.py`; leftover sweep (`git grep` for
`extractors/`, `utils/`, `gui/`, `main.py`, `dictionaries/`, `hf_cache`,
`mfa_data`, `docs/master_plan`, `docs/superpowers`, `validation/` outside
`dev/`); fresh clone into scratch, check top level and run the suite there
(no models: model tests skip); real `run.py` on the scratch validation
config → compare every output file with the baseline; launch `gui.py`
and run the same batch through the Run tab's dispatch path; confirm MFA
reuse in the log.

### 8 · Review
Fresh reviewer (most capable model) gets the spec, this plan and the full
diff, asked specifically for missed connections. Fix, re-verify, merge on
the owner's word.
