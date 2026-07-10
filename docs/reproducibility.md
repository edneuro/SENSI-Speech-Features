# Reproducibility runbook

How SensiExtractor's Python environment is pinned, how to change it safely, and
what to archive alongside a dataset release so the pipeline can be re-run years
later. Design rationale (why two tiers, why conda-lock, why these platforms) is
in [`docs/superpowers/specs/2026-07-10-reproducible-cross-platform-install-design.md`](superpowers/specs/2026-07-10-reproducible-cross-platform-install-design.md);
this doc is the day-to-day and archival *runbook*.

## The two-tier model

Two hand-edited **source** files declare intent, split by installer:

- `environment.yml` — conda deps: Python version, Kaldi (CPU build) + Montreal
  Forced Aligner, and the compiled scientific stack (numpy, pandas, scipy,
  pyarrow, gensim, librosa) as conda-forge prebuilt binaries so nothing
  compiles from source and NumPy has one provider.
- `requirements.txt` — everything else, pip-only.

Neither file is installed directly. `python scripts/relock.py` splices them
into a single generated **lock**, `conda-lock.yml` at the repo root — exact
versions and hashes for all three supported platforms in one file. Every
install path (setup.sh/.bat, the Apptainer container, CI) runs
`conda-lock install -p <prefix> conda-lock.yml` against that same committed
file, so a dev machine, a Sherlock container, and a CI runner all get
byte-identical dependency graphs. **`conda-lock.yml` is generated and never
hand-edited** — treat it like a build artifact, not source.

## How to change a dependency

1. Edit `environment.yml` (conda-level dep) or `requirements.txt`
   (pip-level dep). Don't add a package to both — conda-lock has no overlap
   between the two files by design.
2. Run `python scripts/relock.py` (needs `conda-lock` on PATH: `conda install
   -n base -c conda-forge conda-lock`). It reads both source files, splices
   `requirements.txt`'s pip lines into a temporary merged env spec (conda-lock
   cannot read a bare `-r requirements.txt` reference), solves all three
   platforms, and overwrites `conda-lock.yml`.
3. Rebuild your local `./env` from the new lock and run the full test suite
   before committing — see `docs/superpowers/plans/2026-07-10-reproducible-cross-platform-install.md`
   for the exact verification each platform needs.
4. Commit `conda-lock.yml` alongside the source-file change in the same
   commit. A lock that doesn't match its sources is worse than no lock.

CI (`.github/workflows/env-verify.yml` — the install+import merge gate across
all three OSes; see "Verification status" below for what it does and doesn't
prove) will catch a lock that fails to install, but it won't catch a lock that
installs but is stale relative to the source files — that's on the author.

## Supported platforms

`linux-64`, `osx-arm64` (Apple Silicon), `win-64`. These three, and only
these three, appear in `environment.yml`'s `platforms:` key and in
`conda-lock.yml`.

**Intel macOS (`osx-64`) is not supported, and this is not our choice.**
Apple ended Intel-Mac macOS support at macOS 26 (Tahoe is the last release
with Intel builds), and PyTorch stopped publishing Intel-Mac wheels at
2.2.2 — a version that predates the NumPy 2 baseline the rest of this
project's stack (gensim, pandas 2.x) requires. There is no pin that
satisfies both "runs on Intel Mac" and "modern NumPy 2 stack," so Intel Mac
is dropped upstream, not by a SensiExtractor decision. If a future PyTorch
release resurrects Intel-Mac wheels against NumPy 2, revisit; until then,
never add `osx-64` back to `environment.yml`.

## Why torch is capped `<2.8`

`requirements.txt` pins `torch>=2.0,<2.8` (conda-lock currently resolves
`2.7.1` on all three platforms). This is a **tooling limitation, not a
SensiExtractor preference**: conda-lock's pip resolver cannot solve torch
2.8+'s dependency chain, because torch 2.8+ declares newer NVIDIA CUDA
wheels (`nvidia-*-cu12` 12.8.x, and CUDA-13-era `nvidia-*` 13.x packages)
that the resolver can't find candidates for — the lock solve fails outright,
not just for macOS but for every platform in the same solve. This was
verified directly, not guessed:

| Bound | Result |
|---|---|
| unpinned (torch latest) | FAIL — resolver wants `nvidia-cublas` 13.1.1.3 (CUDA-13 era) |
| `<2.9` (2.8.x) | FAIL — resolver wants `nvidia-cublas-cu12` 12.8.4.1 |
| `<2.8` (2.7.x) | **OK** — locks `torch==2.7.1` |
| `<2.6` (2.5.x) | OK — locks `torch==2.5.1` (more headroom lost for no benefit) |

`<2.8` is the newest bound that locks cleanly on all three platforms, and it
keeps GPU support intact on Linux (the bundled `nvidia-*-cu12` 12.6-series
wheels resolve fine, preserving the Sherlock `--nv` GPU path). **Bump this
bound only after confirming conda-lock's resolver can handle torch's current
NVIDIA dependency declarations** — re-run the same kind of scratch solve
(temporarily loosen the bound, run `python scripts/relock.py` in a throwaway
copy, see whether it locks) before changing the pin in the real
`requirements.txt`. Don't just bump the version number and hope.

## Tribal knowledge worth preserving

A few dependency facts that aren't obvious from the source files and have
already caused confusion once (they were previously only captured as inline
comments in `sherlock/Apptainer.def`, which got trimmed during the
conda-lock migration):

- **`praat-parselmouth` and `soundfile` are PyPI-only spellings.** Their
  conda-forge package names differ from the PyPI names, so both stay in
  `requirements.txt` (pip) rather than moving to `environment.yml` (conda) —
  don't "simplify" them into the conda block; that spelling doesn't resolve
  on conda-forge.
- **`bitsandbytes` (4-bit Llama quantization, used by the LLM-cloze
  validator) is Linux/Windows-only.** `requirements.txt` declares it with
  a PEP 508 environment marker — `bitsandbytes>=0.42 ; platform_system !=
  "Darwin"` — so it's excluded from the `osx-arm64` solve entirely rather
  than installed-and-broken on macOS. If you ever see `bitsandbytes` show up
  in a macOS lock section, the marker has been lost; put it back verbatim.

## Windows: activate the env, don't call `python.exe` directly

On Windows, invoke everything through an **activated** environment
(`conda activate ./env`) or `conda run --prefix ./env ...` — both `setup.bat`
and `setup.sh` already do this for the MFA/spaCy download steps. Calling
`./env/python.exe` directly (bypassing activation) can fail to load native
DLLs such as `libsndfile.dll` (soundfile's backend), because activation is
what puts the env's DLL search paths on `PATH`. This bit Task 3's verification
directly: the full pytest run passed only once the environment was invoked
via `conda run`, not via the bare interpreter path.

## Verification status (what's actually proven, and where)

Be precise about this — it's the difference between "should work" and
"known to work":

- **win-64**: proven. Full test suite (**608 passing**) run natively on
  Windows against an env installed from the committed `conda-lock.yml`.
- **linux-64**: proven. Same full suite, same pass count, run inside WSL
  against an independently-installed env from the same lock — two different
  machines/environments converging on identical results is the strongest
  local evidence this effort produced.
- **osx-arm64 (Apple Silicon)**: proven only at the **install + import**
  level, via CI (`.github/workflows/env-verify.yml`). No one on this project
  has run the full pipeline — MFA forced alignment, GPT-2 surprisal, the
  acoustic/prosody extractors — on real macOS hardware, because CI runners
  don't carry the vendored `mfa_data/`/`hf_cache/` models and no Mac was
  available. **"MFA produces byte-identical alignment output on macOS" is
  not a verified claim by anyone on this project** — the lock guarantees the
  same *package versions* install, not that every extractor's numeric output
  has been checked bit-for-bit on that OS. If you get access to a real Mac,
  running the full suite there (and reporting the result) would close this
  gap; until then, treat macOS as "installs and imports correctly," not
  "fully verified."

## Release ritual: what to archive at each dataset release

The goal is that someone — possibly you, a decade from now, possibly someone
else entirely — can reproduce the exact feature-extraction environment used to
produce a published dataset. At each dataset release, record and store
**three things together**, beside the published dataset in
`../Data Curation/` (not just in this repo, which may itself change or move):

1. **The git commit hash** of SensiExtractor at the time the dataset's
   features were extracted (`git rev-parse HEAD`).
2. **`conda-lock.yml`** from that same commit — the exact, hashed dependency
   graph. This is the primary, lightweight record: given the commit + this
   file, `conda-lock install -p <anywhere> conda-lock.yml` reconstructs the
   same environment on any of the three supported platforms, indefinitely,
   as long as conda-forge/PyPI keep serving the pinned package versions.
3. **A built image**, as a backup in case package indices go stale or
   disappear: the existing `sherlock/sensiextractor.sif` Apptainer image (if
   one was built for that release), and optionally a portable `docker save`d
   tarball of the same environment plus its SHA256 checksum, so bit-rot in
   the tarball itself is detectable.

**Honest boundary**: the commit hash + `conda-lock.yml` is the primary
record — it's small, text-diffable, and sufficient as long as the package
ecosystem it points at still exists. The built image is the backup for when
it doesn't. Neither is a guarantee of effortless reproduction on a decade-old
or far-future system: container runtimes, base OSes, and even CPU
architectures move on, and a resurrection attempt years out should expect to
do some coaxing (finding a compatible Apptainer/Docker runtime, possibly
re-solving pinned versions that vanished from package indices). This ritual
minimizes that coaxing; it doesn't eliminate it.
