# Reproducible, cross-platform install — design

**Date:** 2026-07-10
**Status:** approved (brainstorm), pending spec review
**Topic:** Make SensiExtractor deterministically installable across Windows / Linux / Apple-Silicon macOS, now and years from now, without sacrificing "one universal source" ergonomics.

## Problem

`requirements.txt` uses `>=` floors and `environment.yml` pins only `python=3.10` + `kaldi=*=cpu_*`. Two installs months apart resolve to different versions. There is no lock, so:

- Re-installs are not deterministic (the core gap).
- Cross-platform parity is asserted, never verified.
- The `sherlock/Apptainer.def` container rebuilds from the unpinned floors, so even the "frozen" `.sif` is not actually reproducible.
- The Intel-Mac / PyTorch-2 breakage the user hit is a symptom of unmanaged, unpinned, upstream-shifting deps.

## Goals (from brainstorm)

Priority order, per the user:

1. **Easy installation** — one command, same on every OS.
2. **Live re-install stability** — same working versions next month / next year.
3. **Cross-platform parity** — Windows 11+, Linux, Apple-Silicon macOS produce the same env.
4. **Frozen archival snapshot** — a best-effort "this worked on this date" artifact for ~10-year reproducibility, stored with the published dataset.

Docker is a documented *nice-to-have*, not the primary path. Zero-install-for-outsiders (Binder/Colab/hosted) is explicitly **out of scope**.

## Decisions locked in the brainstorm

- **Two-tier model.** One hand-edited *source* (universal); generated *locks* (exact, per-platform-internally).
- **Tool: `conda-lock`.** Bedrock (conda-forge + conda-lock), lowest churn, keeps the conda(MFA/Kaldi)+pip mental model, and lets the Apptainer container install *from the lock* — fixing its reproducibility for free. (`pixi` and manual pins considered and rejected — see brainstorm.)
- **Target platforms: `linux-64`, `osx-arm64`, `win-64`.** Intel Mac (`osx-64`) is **dropped** — Apple made macOS 26 the last Intel release (macOS 27+ is Apple-Silicon-only) and PyTorch dropped x86_64-mac wheels after 2.2.2. So "any Mac that runs current/future macOS" == `osx-arm64`. This is documented as upstream-imposed, not a project preference. (GitHub Actions confirms the same trajectory: x86_64 macOS runners retire Fall 2027.)

## Verified technical facts (checked 2026-07-10, not assumed)

- conda-lock reads a **non-standard top-level `platforms:` key** in `environment.yml`; `-p` on the CLI overrides it.
- conda-lock's **default output is a single `conda-lock.yml`** in CWD containing **all requested platforms** — one universal lockfile, exact-per-platform inside. Install with `conda-lock install -p ./env conda-lock.yml` (auto-selects the current platform).
- conda-lock locks pip deps only when they are listed **inline** in `environment.yml`'s `dependencies.pip:` section (vendored Poetry solver). It does **not** reliably support `- -r requirements.txt` inside that section, and **`requirements.txt` is not a supported `-f` input**. → drives the "relock splice" decision below.
- GitHub Actions `macos-latest` runs on **Apple Silicon (arm64)**; `ubuntu-latest` = linux-64, `windows-latest` = win-64. All three lock targets are CI-verifiable on hosted runners.

## Architecture

### Tier 1 — Source (human-edited, universal)

- **`environment.yml`** — conda source of truth. Changes:
  - Add top-level `platforms: [linux-64, osx-arm64, win-64]`.
  - Promote the compiled scientific stack the Apptainer def already conda-installs (`pandas numpy scipy pyarrow gensim librosa`) into the conda `dependencies:` so conda-forge supplies prebuilt binaries and pip never compiles from source.
  - Keeps `kaldi=*=cpu_*`, `montreal-forced-aligner`, `python=3.10`, `pip`.
- **`requirements.txt`** — **remains the single source of truth for pip deps** (invariant preserved, per memory). Floors + rich per-line comments stay. Its only live consumers today are the install-flow files (rewritten here) + a docstring in `extractors/bert_embed.py` (no runtime coupling).

### Tier 1.5 — Relock splice (the wiring fix)

Because conda-lock can't read `requirements.txt` directly, a small script bridges the two source files:

- **`scripts/relock` (Python, ~20–30 lines)**:
  1. Read `environment.yml` and `requirements.txt`.
  2. Emit a temporary merged spec that inlines the requirements.txt entries under `dependencies.pip:` (comments/blank lines stripped; environment markers preserved verbatim).
  3. Run `conda-lock lock -f <merged> --lockfile conda-lock.yml` for the three platforms.
  4. Delete the temp file; leave only `conda-lock.yml` changed.
- One command to regenerate the lock; no tribal knowledge. Run whenever `environment.yml` or `requirements.txt` changes.
- **Known risk to validate during implementation:** pip **environment markers** — specifically `bitsandbytes ; platform_system != "Darwin"`. Confirm conda-lock's solver honors the marker per-platform (excluded from `osx-arm64`, included in win/linux). If it doesn't, handle explicitly (e.g. per-platform conda `sys_platform` selector or a documented post-install step). Do not assume; test the resolved lock.

### Tier 2 — Lock (generated, committed)

- **`conda-lock.yml`** at repo root. Single file, all three platforms, exact versions + hashes. Committed like `package-lock.json`; never hand-edited.

### Install flow (the files that drift together — audit all)

- **`setup.sh` / `setup.bat`**: replace `conda env create … && pip install -r requirements.txt` with:
  1. bootstrap conda-lock into base (`conda install -n base -c conda-forge conda-lock`),
  2. `conda-lock install -p ./env conda-lock.yml`,
  3. `python utils/download_models.py` (vendored MFA models),
  4. `python -m spacy download en_core_web_sm` **pinned** to an exact version (the one model not vendored in `mfa_data/`/`hf_cache/`).
- **`README.md`**: install section rewritten to the lock-based flow; add a short "how to update deps" note (edit source → run `scripts/relock` → commit `conda-lock.yml`).
- **`environment.yml`** header comment updated to describe the new role.

### Apptainer container fix (`sherlock/Apptainer.def`)

- `%files`: add `conda-lock.yml` (keep `requirements.txt` only if still referenced elsewhere).
- `%post`: replace `conda create … && pip install -r requirements.txt` with `conda-lock install -p /opt/sensienv conda-lock.yml`. The `.sif` becomes bit-reproducible from the same lock as everyone else. spaCy/MFA-model steps unchanged.

### Cross-platform verification (CI) — *included per user*

- **`.github/workflows/env-smoke.yml`**: matrix over `ubuntu-latest`, `windows-latest`, `macos-latest`. Each job: install `./env` from `conda-lock.yml`, then run a fast pytest smoke subset (not the full 490 — a representative slice that exercises import + one extractor). This is what *proves* parity instead of asserting it. Phased: lands after the locks work locally.

### Archival snapshot (the "10-year" layer) — *documented, best-effort*

- **`docs/reproducibility.md`**: defines the release ritual. At each dataset release, record together: (a) git commit hash, (b) `conda-lock.yml`, (c) a built image — the existing `sensiextractor.sif`, and optionally a `docker save`-d image `.tar` + SHA256 — stored beside the published dataset in `../Data Curation/`. Stated honestly: the lock + commit is the primary record; the image tarball is the backup; a decade-old runtime may still need coaxing.
- **`Dockerfile`** (installing from the same `conda-lock.yml`): **optional deliverable**, deferred unless the user wants it. Docker stays a documented archival option, not a required path.

## Scope & phasing

1. **Core** (delivers goals 1–3): Tier-1 edits + `scripts/relock` + generate `conda-lock.yml` + rewrite `setup.sh`/`setup.bat`/`README` + Apptainer fix.
2. **Verify**: GitHub Actions matrix smoke test.
3. **Archive**: `docs/reproducibility.md`; optional `Dockerfile` only on request.

## Testing gate

TDD does not apply (no `extractors/` math — this is env/tooling; per memory, test-after/verify-by-running). The completion gate is empirical:

- After generating `conda-lock.yml`, build `./env` **from the lock** on this Windows machine and confirm the pytest baseline (**490 passed / 2 skipped**) still holds. A lock that resolves but doesn't run is a failure.
- Inspect the resolved lock to confirm the `bitsandbytes` Darwin marker resolved correctly per platform.
- CI job green on all three OSes (phase 2).

## Out of scope

- Binder / Colab / hosted zero-install.
- Intel-Mac (`osx-64`) support.
- Migrating off conda to pixi/uv.
- Re-pinning individual libraries for their own sake (the lock captures exact versions; `requirements.txt` floors express intent and stay as-is).
