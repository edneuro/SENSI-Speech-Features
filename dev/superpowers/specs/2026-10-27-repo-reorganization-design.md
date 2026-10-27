# Repo reorganization — design

Date: 2026-10-27 · Owner: Amilcar Malave · Status: draft for owner approval

## Intent

The repo's top level looks like a mess: code packages, data, model caches,
planning notes, audits and scratch work all sit side by side. A new user or
collaborator can't tell what they need. Group everything into a few folders
with obvious names, keep the two ways the owner actually starts the app
(`gui.py`, `run.py`), and retire the command-line flags of `main.py`, which
the owner never uses (GUI or `features.yaml` only).

## Goals

1. Top level holds only: the two launchers, setup scripts, the README, the
   three environment files, and a handful of folders (layout below).
2. All application code lives in one package, `sensi/`.
3. Downloaded models and lookup tables live together under `data/`.
4. `docs/` holds only documents a user reads; project history and
   development material move to `dev/`.
5. The `main.py` command line is retired; its pipeline code moves into the
   package. `run.py` + `config/` is the one headless path.
6. `CLAUDE.md` leaves the public repo but stays on the owner's disk.

## Non-goals / anti-goals

- No change to any output: every CSV/.mat a run writes must be identical to
  current `main` (manifests' timestamps/version aside).
- No change to settings files: existing `config/features.yaml` / `run.yaml`
  keep working unedited.
- No re-download of models: existing local model folders are reused.
- Not a refactor of the code inside the files (beyond import lines and the
  path constants that must change).

## Layout

```
SENSI-Speech-Features/
├── gui.py  run.py                 # the two ways to start the app
├── setup.sh  setup.bat  README.md
├── environment.yml  requirements.txt  conda-lock.yml
├── sensi/                         # all application code
│   ├── __init__.py                # __version__ (was the root __init__.py)
│   ├── paths.py                   # ONE place that knows where folders are
│   ├── pipeline.py                # was main.py minus its argparse CLI
│   ├── extractors/  gui/  utils/
├── scripts/                       # maintainer + one-time helper scripts
├── config/                        # *_defaults.yaml templates (+ local yaml, ignored)
├── data/
│   ├── dictionaries/              # tracked lookup tables
│   └── hf_cache/ mfa_data/ gensim_data/   # downloaded models (git-ignored)
├── assets/                        # icons only
├── docs/                          # user docs: features/, reproducibility.md
├── dev/                           # master_plan, changelog, superpowers/, audits, validation
├── sherlock/
├── tests/
└── .github/                       # install check on Windows/Linux/macOS
```

## Mapping (old → new)

| Old | New |
|---|---|
| `extractors/`, `gui/`, `utils/` | `sensi/extractors/`, `sensi/gui/`, `sensi/utils/` |
| `main.py` (runners, drop maps, `run()`) | `sensi/pipeline.py`; `build_parser()` and `main()` deleted |
| root `__init__.py` (`__version__`) | `sensi/__init__.py` |
| `dictionaries/` | `data/dictionaries/` |
| `hf_cache/`, `mfa_data/`, `gensim_data/` | `data/…` (local folders, not in git) |
| `utils/download_models.py`, `download_lexical_norms.py`, `validate_cloze.py` | `scripts/` (they are run by hand, not imported) |
| `docs/master_plan.md`, `changelog.md`, `feature_dependencies.md`, `feature-docs-audit.md`, `in_app_documentation_needs_fix.md`, `AoA&MRC.md`, `superpowers/` | `dev/` (same names) |
| `validation/` (tracked script + local data) | `dev/validation/` |
| `audit/`, `unrelated/` (local only) | `dev/audit/`, `dev/unrelated/` (still local only) |
| `CLAUDE.md` | stays on disk, removed from git, added to `.gitignore` |
| `pytest.ini` | stays at top (pytest looks for it there); `norecursedirs` updated |

## Decided with the owner

- Folder names and grouping above (owner approved the proposal; `sensi/`
  not `code/` because Python ships a built-in `code` module).
- Retire the `main.py` CLI entirely (owner: "I will use either the gui or
  the features.yaml").
- `CLAUDE.md` off GitHub, kept locally.
- Local model folders move into `data/`.

## Calls made by the builder (owner to confirm)

1. **One paths module.** ~20 places compute folders with
   `Path(__file__).parent.parent / "dictionaries"` etc. Each would break
   after the move; all of them read from `sensi/paths.py` instead
   (`PROJECT_ROOT`, `DATA_DIR`, `DICT_DIR`, `ASSETS_DIR`, `DOCS_DIR`,
   `CONFIG_DIR` default), so the next move is a one-line change.
2. **Old model folders are moved automatically, once.** If `hf_cache/`,
   `mfa_data/` or `gensim_data/` exist at the top level and the `data/`
   copy doesn't, `use_project_caches()` renames them into `data/` at start-up
   and prints what it moved. Without this, every existing checkout (yours,
   Neha's) would lose its aligner models after pulling — alignment would
   fail, and the language models would download again. A same-disk rename is
   instant; if it fails (file locked), it falls back to using the old folder
   and says so.
3. **Tests keep their names**, only imports and `mock.patch("…")` target
   strings change (`"extractors.x.f"` → `"sensi.extractors.x.f"`). Tests of
   the retired CLI (`tests/test_main_cli.py`, `build_parser` uses) are
   deleted or rewritten to build the args through the planner, which is what
   the GUI and `run.py` use.
4. **History is kept**: every move is a `git mv`, in its own commit,
   separate from the content edits, so `git log --follow` keeps working.

## Changes made during the build (recorded after the fact)

5. **MFA files follow `MFA_ROOT_DIR`.** The aligner's G2P model and the
   phoneme-surprisal / syllabify pronunciation dictionary were hard-wired to
   `<repo>/mfa_data/…`; they now come from `sensi.paths.mfa_pretrained()`,
   which follows `MFA_ROOT_DIR`. Identical on a normal install (both point at
   `data/mfa_data`). It does change Sherlock outputs: the container sets
   `MFA_ROOT_DIR=/opt/mfa_data` and never had `<repo>/mfa_data`, so phoneme
   surprisal and `Pron_Source` now work there instead of failing. (The
   container still downloads no G2P model, so `g2p_oov` remains a no-op on
   Sherlock, as before.)
6. **Tests keep the old parser** instead of building arguments through the
   planner (call 3 above): `tests/batch_args.py` is the retired CLI parser,
   test scaffolding only, because the batch-runner tests were written against
   its every-feature-on namespace and the planner's opt-in defaults would
   change what they exercise. Checked: the pipeline reads no attribute that
   only that parser sets.
7. **Old `dictionaries/` leftovers** (local-only LIWC `.dic`, EWFG parquet,
   self-built tables in an old checkout) are moved into `data/dictionaries/`
   file by file by the same launcher step (never overwriting), and the old
   folder is git-ignored so a licensed file there can't be committed.
   `tests/conftest.py` runs the move first too, so a test run never moves
   folders mid-session.

## Everything that references the old layout (must be updated)

Imports and patch strings (≈760 lines; `gui/run_dispatch.py`, `run.py`,
tests); the path constants above; `.gitignore`; `pytest.ini`;
`.github/workflows/env-verify.yml` and `scripts/import_smoke.py`;
`scripts/make_mac_app.sh`; `setup.sh` / `setup.bat` (and the `runGUI.bat`
they generate); `sherlock/Apptainer.def`, `run_container.sbatch`,
`RUNBOOK.md` (rsync excludes, Word2Vec pre-download path);
`gui/feature_docs.py` (docs root); `README.md` (layout tree, remove the
Command line section, model paths); `CLAUDE.md` and memory files (local).

## Verification

- Full test suite passes.
- `scripts/import_smoke.py` passes (what CI runs).
- Real runs on the validation corpus — `python gui.py` (all levels) and
  `python run.py` — and every output file is byte-identical to the same run
  on current `main` (excluding manifest timestamps/version/paths).
- A fresh-clone check: clone the branch into a scratch folder, confirm the
  top level matches the layout and that nothing references a removed path.
- MFA runs at most once per document (cache reuse unchanged).
