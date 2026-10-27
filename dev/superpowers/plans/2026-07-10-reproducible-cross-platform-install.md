# Reproducible Cross-Platform Install — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make SensiExtractor deterministically installable on Windows 11+, Linux, and Apple-Silicon macOS via a single committed `conda-lock.yml`, with real-machine proof of cross-platform correctness.

**Architecture:** Two-tier. One hand-edited *source* (`environment.yml` = conda deps, `requirements.txt` = pip deps) is spliced by `scripts/relock.py` into a single `conda-lock.yml` locking `linux-64`/`osx-arm64`/`win-64` with exact versions + hashes. Install everywhere (setup scripts, Apptainer, CI) is `conda-lock install -p ./env conda-lock.yml`. A GitHub Actions matrix proves the locked env installs and imports on all three OSes.

**Tech Stack:** conda / conda-forge, conda-lock, PyYAML, pytest, GitHub Actions, Apptainer.

Design spec: `docs/superpowers/specs/2026-07-10-reproducible-cross-platform-install-design.md` (read it for rationale; this plan implements it).

## Global Constraints

- **Target platforms (exact):** `linux-64`, `osx-arm64`, `win-64`. **Never** add `osx-64` (Intel Mac) — dropped upstream (Apple + PyTorch).
- **Single lockfile:** one `conda-lock.yml` at repo root, all three platforms, committed, never hand-edited.
- **Source of truth split:** `environment.yml` = conda deps only; `requirements.txt` = pip deps only. No package appears in both.
- **conda-lock cannot read `-r requirements.txt`** — pip deps must be spliced inline; that is `scripts/relock.py`'s job.
- **Preserve existing pins:** `transformers>=4.30,<6`, `torch>=2.0`, `PySide6>=6.9,<6.10`, `gensim>=4.3.3` (needs numpy 2), `pyarrow>=14.0`, `bitsandbytes>=0.42 ; platform_system != "Darwin"` (must resolve OUT of `osx-arm64`).
- **numpy 2** is the target (gensim floor requires it); do not let any platform fall back to numpy 1.x.
- **MKL crash guard:** `MKL_THREADING_LAYER=sequential` must be set wherever the code runs (already in `main.py`/`gui.py`/conftest/Apptainer; also set it in CI + import smoke).
- **Verification is by real machines, not inspection:** win-64 local (native), linux-64 local (WSL), osx-arm64 CI-only. CI green on all three is the merge gate.
- **TDD does not apply** (env/tooling, no `extractors/` math). Each task verifies by running commands and checking output.

---

### Task 1: Rework the two source specs

Split deps cleanly so conda-lock has no overlap and pip never compiles from source.

**Files:**
- Modify: `environment.yml`
- Modify: `requirements.txt`

**Interfaces:**
- Produces: an `environment.yml` with a top-level `platforms:` key and the compiled scientific stack as conda deps; a `requirements.txt` with the pip-only deps (five packages removed). `scripts/relock.py` (Task 2) consumes both.

- [ ] **Step 1: Rewrite `environment.yml`** to this exact content:

```yaml
name: sensi
channels:
  - conda-forge
  - defaults
# conda-lock reads this non-standard key to know which platforms to lock.
# NEVER add osx-64 — Intel Mac is dropped upstream (see design spec).
platforms:
  - linux-64
  - osx-arm64
  - win-64
dependencies:
  - python=3.10
  # Kaldi (MFA's engine) pinned to its CPU build: forced alignment is CPU-only,
  # and the CUDA build drags in ~3 GB of NVIDIA/MAGMA libs it never uses.
  - kaldi=*=cpu_*
  - montreal-forced-aligner
  # Compiled scientific stack from conda-forge PREBUILT binaries so the pip layer
  # never compiles from source and NumPy has a single (conda) provider. Mirrors
  # the proven sherlock/Apptainer.def split. These are deliberately NOT in
  # requirements.txt (that file is pip-only).
  - numpy
  - pandas
  - scipy
  - pyarrow
  - gensim
  - librosa
  - pip
  # The full pip dependency set lives in requirements.txt and is spliced into the
  # lock by scripts/relock.py (conda-lock cannot read `-r requirements.txt`).
```

- [ ] **Step 2: Edit `requirements.txt`** — remove the five packages now provided by conda. Delete the `pandas>=2.0`, `pyarrow>=14.0`, `gensim>=4.3.3`, `scipy>=1.11`, and `librosa>=0.10` lines (keep their surrounding comments' intent by replacing the block with a pointer). Where the `# Data` / pandas lines were, and the gensim/scipy/librosa lines were, leave a single note. Add near the top of the file:

```
# ---
# The compiled scientific stack (numpy, pandas, scipy, pyarrow, gensim, librosa)
# is installed via conda-forge (see environment.yml), NOT pip — conda-forge ships
# prebuilt binaries so nothing compiles from source and NumPy has one provider.
# Do not re-add them here.
# ---
```

Keep everything else (spacy, wordfreq, num2words, tgt, transformers, torch, soundfile, praat-parselmouth, empath, liwc, vaderSentiment, accelerate, bitsandbytes, openai, textstat, PySide6, tiktoken, pytest*, psutil, pyphen, pyyaml). **Keep `soundfile`** (PyPI-only spelling; tests import it directly). **Keep the `bitsandbytes>=0.42  ; platform_system != "Darwin"` line verbatim** — the marker is load-bearing.

- [ ] **Step 3: Verify both files parse and there is no overlap**

Run:
```bash
conda run -p ./env python - <<'PY'
import yaml
env = yaml.safe_load(open("environment.yml"))
conda_deps = {d.split("=")[0].strip() for d in env["dependencies"] if isinstance(d, str)}
reqs = [l.split("#")[0].split(";")[0].strip() for l in open("requirements.txt") if l.strip() and not l.startswith("#")]
reqs = {r.split(">")[0].split("=")[0].split("<")[0].strip() for r in reqs if r}
overlap = conda_deps & reqs
print("platforms:", env.get("platforms"))
print("conda deps:", sorted(conda_deps))
print("overlap (must be empty):", overlap)
assert env.get("platforms") == ["linux-64", "osx-arm64", "win-64"], "platforms wrong"
assert not overlap, f"package in BOTH conda and pip: {overlap}"
print("OK")
PY
```
Expected: `platforms: ['linux-64', 'osx-arm64', 'win-64']`, `overlap (must be empty): set()`, `OK`.

- [ ] **Step 4: Commit**

```bash
git add environment.yml requirements.txt
git commit -m "build(repro): split conda/pip source specs, add lock platforms"
```

---

### Task 2: relock script + generate the lockfile

The heart of the change. This produces the committed artifact and is where cross-platform resolution issues (if any) surface.

**Files:**
- Create: `scripts/relock.py`
- Create: `conda-lock.yml` (generated, committed)

**Interfaces:**
- Consumes: `environment.yml`, `requirements.txt` (Task 1).
- Produces: `conda-lock.yml` (all three platforms). Consumed by setup scripts (Task 5), Apptainer (Task 6), CI (Task 7).

- [ ] **Step 1: Ensure conda-lock is available**

Run:
```bash
conda install -n base -c conda-forge conda-lock -y
conda-lock --version
```
Expected: a version prints (any recent 2.x+).

- [ ] **Step 2: Create `scripts/relock.py`** with this exact content:

```python
#!/usr/bin/env python
"""Regenerate conda-lock.yml from the two source specs.

Source of truth is split by installer:
  - environment.yml   -> conda deps (python, Kaldi/MFA, compiled sci stack)
  - requirements.txt  -> pip deps

conda-lock only locks pip deps listed *inline* in environment.yml's
dependencies.pip section (it cannot read `-r requirements.txt`). So we splice
requirements.txt into a temporary merged env file, lock that, then delete it.
The committed artifact is conda-lock.yml (all platforms, exact + hashed).

Usage:   python scripts/relock.py
Requires: conda-lock on PATH  (conda install -n base -c conda-forge conda-lock)
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml  # PyYAML is a project dependency

ROOT = Path(__file__).resolve().parent.parent
ENV_YML = ROOT / "environment.yml"
REQS = ROOT / "requirements.txt"
LOCKFILE = ROOT / "conda-lock.yml"


def pip_requirements() -> list[str]:
    """Non-comment, non-blank lines from requirements.txt (markers preserved)."""
    lines = []
    for raw in REQS.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        lines.append(line)
    return lines


def build_merged_env() -> str:
    """environment.yml with requirements.txt spliced into dependencies.pip."""
    env = yaml.safe_load(ENV_YML.read_text(encoding="utf-8"))
    deps = env.setdefault("dependencies", [])
    if "pip" not in [d for d in deps if isinstance(d, str)]:
        deps.append("pip")
    deps.append({"pip": pip_requirements()})
    return yaml.safe_dump(env, sort_keys=False)


def main() -> int:
    if shutil.which("conda-lock") is None:
        sys.exit(
            "conda-lock not found. Install:\n"
            "  conda install -n base -c conda-forge conda-lock"
        )
    merged = build_merged_env()
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", suffix=".yml", delete=False, dir=ROOT, encoding="utf-8"
        ) as fh:
            fh.write(merged)
            tmp = Path(fh.name)
        print(f"Locking {ENV_YML.name} + {REQS.name} -> {LOCKFILE.name} ...")
        subprocess.run(
            ["conda-lock", "lock", "-f", str(tmp), "--lockfile", str(LOCKFILE)],
            check=True,
            cwd=ROOT,
        )
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)
    print(f"Wrote {LOCKFILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Generate the lock** (this may take several minutes — it solves for three platforms)

Run:
```bash
conda run -p ./env python scripts/relock.py
```
Expected: ends with `Wrote .../conda-lock.yml` and exit 0. If it fails to resolve, read the error — the likely culprits are (a) a pip dep with no wheel for a platform, or (b) the `bitsandbytes` marker. See Step 5 fallback.

- [ ] **Step 4: Sanity-check the lockfile contents**

Run:
```bash
conda run -p ./env python - <<'PY'
import yaml
lock = yaml.safe_load(open("conda-lock.yml"))
plats = set()
for pkg in lock["package"]:
    plats.add(pkg["platform"])
print("platforms in lock:", sorted(plats))
assert plats == {"linux-64", "osx-arm64", "win-64"}, plats
# torch must be the SAME version on all platforms (parity), builds may differ
tv = {p["platform"]: p["version"] for p in lock["package"] if p["name"] == "torch"}
print("torch per platform:", tv)
assert len(set(tv.values())) == 1, f"torch version diverges: {tv}"
print("OK")
PY
```
Expected: all three platforms present; identical torch version across them; `OK`.

- [ ] **Step 5: Verify the `bitsandbytes` Darwin marker resolved correctly** (the flagged risk)

Run:
```bash
conda run -p ./env python - <<'PY'
import yaml
lock = yaml.safe_load(open("conda-lock.yml"))
bnb = {p["platform"] for p in lock["package"] if p["name"] == "bitsandbytes"}
print("bitsandbytes present on:", sorted(bnb))
assert "osx-arm64" not in bnb, "bitsandbytes leaked onto macOS — marker NOT honored"
assert {"linux-64", "win-64"} <= bnb, "bitsandbytes missing on linux/win"
print("OK — marker honored")
PY
```
Expected: `bitsandbytes present on: ['linux-64', 'win-64']`, `OK`.

**Fallback if the assertion fails** (marker not honored by the solver): remove the `bitsandbytes` line from `requirements.txt`'s pip splice and instead add it to `environment.yml` as a platform-selected conda-forge dep, e.g. under `dependencies:` add `- bitsandbytes  # [not osx]` is NOT valid conda-lock syntax — instead keep it pip but move the marker handling into `scripts/relock.py`'s `pip_requirements()` to drop the line when it can't be expressed. Document whichever path you take in the relock docstring. Re-run Steps 3–5.

- [ ] **Step 6: Commit**

```bash
git add scripts/relock.py conda-lock.yml
git commit -m "build(repro): add relock script and generate conda-lock.yml"
```

---

### Task 3: Prove it on win-64 (local, native)

Install a throwaway env **from the committed lock** and run the full suite. Does not touch the working `./env`.

**Files:** none (verification only).

- [ ] **Step 1: Install the locked env to a temp prefix**

Run (temp prefix keeps your working `./env` untouched):
```bash
conda-lock install -p ./env-verify conda-lock.yml
conda run -p ./env-verify python -m spacy download en_core_web_sm
```
Expected: env solves and installs; spaCy model downloads. Exit 0.

- [ ] **Step 2: Run the full test suite from the locked env**

Run:
```bash
MKL_THREADING_LAYER=sequential conda run -p ./env-verify python -m pytest -q
```
Expected: **490 passed, 2 skipped** (the known baseline). If counts differ, investigate before proceeding — a lock that resolves but fails tests is a failure.

- [ ] **Step 3: Clean up the temp env**

Run:
```bash
conda env remove -p ./env-verify -y
```

- [ ] **Step 4: Record the result** — no commit (no files changed). Note the pass count in the task hand-off.

---

### Task 4: Prove it on linux-64 (local, WSL)

Second of three targets proven locally with full models. Runs inside WSL Ubuntu (the same env used to build the `.sif`).

**Files:** none (verification only).

- [ ] **Step 1: In WSL, ensure conda + conda-lock exist**

In a WSL Ubuntu shell:
```bash
cd /mnt/c/Users/amilcar/Desktop/code/Antigravity/SensiExtractor
which conda || echo "install miniforge in WSL first"
conda install -n base -c conda-forge conda-lock -y
```
Expected: conda present; conda-lock installs. (If conda is missing, install Miniforge in WSL, then retry.)

- [ ] **Step 2: Install the locked env (linux-64 auto-selected) and download the spaCy model**

```bash
conda-lock install -p ./env-verify-linux conda-lock.yml
conda run -p ./env-verify-linux python -m spacy download en_core_web_sm
```
Expected: solves for linux-64 and installs. Exit 0.

- [ ] **Step 3: Run the suite**

```bash
MKL_THREADING_LAYER=sequential conda run -p ./env-verify-linux python -m pytest -q
```
Expected: full suite passes (same 490/2 baseline, allowing for any genuinely OS-specific skips — note any difference).

- [ ] **Step 4: Clean up**

```bash
conda env remove -p ./env-verify-linux -y
```

- [ ] **Step 5: Record the result** in the hand-off (no commit).

---

### Task 5: Update the install flow (setup scripts + README)

Point installs at the lock. Same deterministic command on every OS.

**Files:**
- Modify: `setup.sh`
- Modify: `setup.bat`
- Modify: `README.md`

**Interfaces:**
- Consumes: `conda-lock.yml` (Task 2).

- [ ] **Step 1: Rewrite `setup.sh`** so steps 1–2 install from the lock:

```bash
#!/bin/bash
# SensiExtractor - First-time setup (macOS / Linux)
# Run from inside the SensiExtractor folder:  bash setup.sh
set -e

echo "============================================================"
echo " SensiExtractor - First-time setup"
echo "============================================================"
echo ""
echo "Everything will be downloaded into this folder:"
echo "  ./env/        Python environment (~2 GB)"
echo "  ./mfa_data/   MFA acoustic models (~1 GB)"
echo "  ./hf_cache/   GPT-2 weights (~500 MB, on first run)"
echo ""
read -p "Press Enter to continue (Ctrl+C to cancel)..."

echo ""
echo "[1/4] Installing conda-lock (deterministic installer) ..."
conda install -n base -c conda-forge conda-lock -y

echo ""
echo "[2/4] Creating ./env from conda-lock.yml (exact, pinned versions) ..."
# conda-lock.yml pins every package for this platform. CPU Kaldi is pinned in
# environment.yml; the lock captures it exactly.
conda-lock install -p ./env conda-lock.yml

echo ""
echo "[3/4] Downloading MFA models into ./mfa_data ..."
conda run --prefix ./env python utils/download_models.py

echo ""
echo "[4/4] Downloading spaCy model into ./env ..."
conda run --prefix ./env python -m spacy download en_core_web_sm

echo ""
echo "============================================================"
echo " Setup complete!"
echo ""
echo " To run the app:"
echo "   conda activate ./env"
echo "   python gui.py"
echo "============================================================"
```

- [ ] **Step 2: Rewrite `setup.bat`** with the equivalent Windows flow. Read the current file first, then mirror the four steps: (1) `conda install -n base -c conda-forge conda-lock -y`, (2) `conda-lock install -p .\env conda-lock.yml`, (3) `conda run --prefix .\env python utils\download_models.py`, (4) `conda run --prefix .\env python -m spacy download en_core_web_sm`. Preserve the existing banner/echo style and `pause` behavior of the current `setup.bat`.

- [ ] **Step 3: Update `README.md`** install section. Replace any `conda env create` / `pip install -r requirements.txt` instructions with:
  - Install: run `setup.sh` (mac/Linux) or `setup.bat` (Windows), which now installs from `conda-lock.yml`.
  - Add a short **"Updating dependencies"** subsection: "Edit `environment.yml` (conda) or `requirements.txt` (pip), then run `python scripts/relock.py` and commit the regenerated `conda-lock.yml`. Never edit `conda-lock.yml` by hand."
  - Note supported platforms: Windows 11+, Linux, Apple-Silicon macOS. Intel Macs are not supported (upstream: Apple + PyTorch).

- [ ] **Step 4: Verify the scripts reference the lock and match the commands proven in Tasks 3–4**

Run:
```bash
grep -n "conda-lock install" setup.sh setup.bat
grep -n "relock" README.md
```
Expected: both setup scripts contain `conda-lock install`; README mentions `relock`.

- [ ] **Step 5: Commit**

```bash
git add setup.sh setup.bat README.md
git commit -m "build(repro): install from conda-lock.yml in setup scripts + README"
```

---

### Task 6: Make the Apptainer container reproducible

The `.sif` currently rebuilds from unpinned floors. Install it from the same lock so it's bit-reproducible.

**Files:**
- Modify: `sherlock/Apptainer.def`

- [ ] **Step 1: Add the lockfile to `%files`** — change the `%files` block to copy the lock instead of (or in addition to) requirements.txt:

```
%files
    # conda-lock.yml pins the entire env exactly (all platforms). The container
    # installs from it so the .sif matches every other install target.
    conda-lock.yml /opt/conda-lock.yml
```

- [ ] **Step 2: Replace the `%post` env-build** — swap the `conda create ... && pip install -r requirements.txt` block (the numbered steps 1, 1b, 2) for a single lock install:

```
    # 1. Build the env from the committed lock (identical to setup.sh / CI).
    #    Kaldi CPU pin, MFA, compiled sci stack, and the full pip layer are all
    #    captured in conda-lock.yml — no floors, no source compilation, no drift.
    conda install -n base -c conda-forge conda-lock -y
    conda-lock install -p "$ENV_PREFIX" /opt/conda-lock.yml
```

Keep the spaCy model step (3), the MFA model downloads (4), and `conda clean -afy` unchanged.

- [ ] **Step 3: Update the header comment and `%labels`** — the def's opening comment says the env is built from requirements.txt; update it to say it's built from `conda-lock.yml`. Update the `org.sensi.contains` label from `pip(requirements.txt)` to `pip(conda-lock.yml)`.

- [ ] **Step 4: Verify the def no longer creates an unpinned env**

Run:
```bash
grep -n "conda-lock install" sherlock/Apptainer.def
grep -n "pip install -r" sherlock/Apptainer.def || echo "no unpinned pip install (good)"
```
Expected: `conda-lock install` present; no `pip install -r requirements.txt` line remains.

- [ ] **Step 5: Note the build-time verification** — the `.sif` is proven at the next real rebuild (WSL + `apptainer build`, ~30 min, out of band). The install commands are identical to Task 4's linux-64 verification, which *is* proven here. Record this in the hand-off.

- [ ] **Step 6: Commit**

```bash
git add sherlock/Apptainer.def
git commit -m "build(repro): build the Apptainer .sif from conda-lock.yml"
```

---

### Task 7: CI — prove all three OSes on real machines

The only proof `osx-arm64` runs. Author pushes the branch; sign-off rests on green checks.

**Files:**
- Create: `.github/workflows/env-verify.yml`
- Create: `scripts/import_smoke.py`

**Interfaces:**
- Consumes: `conda-lock.yml` (Task 2).

- [ ] **Step 1: Create `scripts/import_smoke.py`** with this exact content:

```python
#!/usr/bin/env python
"""Import every extractor/util module + PySide6 + the CLI entry point.

Used by CI to prove the locked env is installable AND importable on each OS,
without needing the ~1.5 GB vendored models (a fresh checkout lacks them).
Exits nonzero if any import fails. Imports must be side-effect-free (no model
loading at import time — the codebase loads models lazily).
"""
import importlib
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # headless PySide6
os.environ.setdefault("MKL_THREADING_LAYER", "sequential")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

targets = ["main", "PySide6.QtWidgets"]
for pkg in ("extractors", "utils"):
    for path in sorted((ROOT / pkg).glob("*.py")):
        if path.stem == "__init__":
            continue
        targets.append(f"{pkg}.{path.stem}")

failed = []
for mod in targets:
    try:
        importlib.import_module(mod)
        print(f"ok    {mod}")
    except Exception as exc:  # noqa: BLE001 - report every failure
        print(f"FAIL  {mod}: {type(exc).__name__}: {exc}")
        failed.append(mod)

if failed:
    raise SystemExit(f"{len(failed)} import(s) failed: {failed}")
print("all imports ok")
```

- [ ] **Step 2: Create `.github/workflows/env-verify.yml`** with this exact content:

```yaml
name: env-verify

on:
  push:
    branches: [main, reproducible-install]
  pull_request:

jobs:
  verify:
    name: ${{ matrix.os }}
    runs-on: ${{ matrix.os }}
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
    defaults:
      run:
        shell: bash -el {0}
    env:
      MKL_THREADING_LAYER: sequential
      QT_QPA_PLATFORM: offscreen
    steps:
      - uses: actions/checkout@v4

      - uses: conda-incubator/setup-miniconda@v3
        with:
          miniforge-version: latest
          activate-environment: ""

      - name: Install conda-lock
        run: conda install -n base -c conda-forge conda-lock -y

      - name: Install env from the committed lock
        run: conda-lock install -p ./env conda-lock.yml

      - name: Download spaCy model
        run: conda run -p ./env python -m spacy download en_core_web_sm

      - name: Import smoke (env installs AND imports on this OS)
        run: conda run -p ./env python scripts/import_smoke.py

      - name: bitsandbytes absent on macOS (Darwin marker honored)
        if: matrix.os == 'macos-latest'
        run: |
          conda run -p ./env python -c "import importlib.util, sys; \
          sys.exit(0 if importlib.util.find_spec('bitsandbytes') is None else 1)"
```

- [ ] **Step 3: Verify the import-smoke locally first** (catch problems before burning CI minutes)

Run:
```bash
conda run -p ./env python scripts/import_smoke.py
```
Expected: `all imports ok`. If a module loads a model at import (network call / missing file), guard that import or exclude the module here and note it.

- [ ] **Step 4: Commit and push to trigger CI**

```bash
git add .github/workflows/env-verify.yml scripts/import_smoke.py
git commit -m "ci(repro): verify locked env installs+imports on win/linux/macos"
git push -u origin reproducible-install
```

- [ ] **Step 5: Confirm CI is green on all three OSes**

Open the repo's **Actions** tab on GitHub and confirm the `env-verify` run shows green for `ubuntu-latest`, `windows-latest`, and `macos-latest`.
- If no run appears: Actions may be disabled at the `edneuro` org/repo level — Settings → Actions → General → allow Actions (may need an org admin). Note this in hand-off.
- If macOS fails on `bitsandbytes`: the Task 2 fallback wasn't applied correctly — return to Task 2 Step 5.
- **This green matrix is the merge gate.** Do not consider the feature done until it is green.

---

### Task 8: Archival doc + orientation updates

Document the "10-year" release ritual and update project orientation to mention the new files.

**Files:**
- Create: `docs/reproducibility.md`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Create `docs/reproducibility.md`** covering:
  - The two-tier model (source → lock) in 3–4 sentences; link the design spec.
  - How to update deps (edit source → `python scripts/relock.py` → commit `conda-lock.yml`).
  - Supported platforms + the honest Intel-Mac note (upstream-dropped).
  - **Release ritual for archival:** at each dataset release, record together (a) the git commit hash, (b) `conda-lock.yml`, (c) a built image — the existing `sherlock/sensiextractor.sif`, and optionally a `docker save`d image `.tar` + its SHA256 — stored beside the published dataset in `../Data Curation/`.
  - The honest boundary: lock + commit is the primary record; the image tarball is the backup; a decade-old runtime may still need coaxing; full MFA/GPT-2 pipeline is verified only where the vendored models exist (local Windows + WSL-Linux), not on macOS CI.

- [ ] **Step 2: Update `CLAUDE.md`** project-structure block to add: `scripts/relock.py`, `conda-lock.yml` (root), `.github/workflows/env-verify.yml`, `docs/reproducibility.md`. Update the "Environment" / "Running" sections so the install instruction is "run setup.sh/.bat (installs from conda-lock.yml)" and add a one-line pointer to `docs/reproducibility.md` for dependency changes.

- [ ] **Step 3: Verify the doc references resolve**

Run:
```bash
grep -n "conda-lock.yml\|relock" docs/reproducibility.md CLAUDE.md
ls conda-lock.yml scripts/relock.py .github/workflows/env-verify.yml
```
Expected: references present; all three files exist.

- [ ] **Step 4: Commit**

```bash
git add docs/reproducibility.md CLAUDE.md
git commit -m "docs(repro): reproducibility runbook + orientation updates"
```

---

## Self-Review

**Spec coverage:**
- Two-tier model → Tasks 1–2. ✅
- conda-lock, single `conda-lock.yml`, three platforms → Task 2. ✅
- relock splice (conda-lock can't read `-r`) → Task 2. ✅
- Compiled sci stack to conda, no overlap → Task 1. ✅
- bitsandbytes Darwin marker verification + fallback → Task 2 Steps 5. ✅
- Install-flow audit (setup.sh/.bat, README) → Task 5. ✅
- Apptainer reproducibility fix → Task 6. ✅
- CI as core, all three OSes, merge gate, bitsandbytes-absent-on-mac → Task 7. ✅
- Local proof: win-64 native (Task 3) + linux-64 WSL (Task 4). ✅
- Archival ritual + honest gap → Task 8. ✅
- spaCy model: kept as post-install `spacy download` (deterministic given locked-exact spaCy) rather than URL-pinned, to avoid conda-lock URL-dep risk — refinement noted for user sign-off. ✅
- Dockerfile: deferred (optional, out of scope unless requested). ✅ (design)

**Placeholder scan:** No TBD/TODO; every code file shown in full; every command has expected output. Task 5 Step 2 (setup.bat) says "read current file then mirror" rather than pasting — acceptable because it must preserve unseen banner/pause lines; the four exact commands are given.

**Type/name consistency:** `scripts/relock.py`, `conda-lock.yml`, `./env`, `env-verify.yml`, `import_smoke.py`, `./env-verify` used consistently across tasks. relock produces `conda-lock.yml` at root; every consumer references that path. ✅
