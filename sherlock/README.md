# Running SensiExtractor on Sherlock (experimental)

This folder is a **self-contained, additive orchestration layer** for running the
existing `main.py` batch CLI on Stanford's Sherlock HPC cluster — primarily to get
a GPU for the local-Llama LLM-cloze feature, and faster MFA on many CPU cores.

**It changes nothing in the main app.** No edits to `gui/`, `extractors/`, or
`main.py`. The scripts here only *call* the unchanged CLI. The model backend
(`extractors/llm_cloze.py::_HuggingFaceBackend`) already auto-selects by available
VRAM, so the same code runs on a local GPU PC, on a Sherlock GPU node, or on CPU —
this layer is just about *where* the batch runs, not *how* models are chosen.

If this experiment doesn't pan out, delete this folder; the app is unaffected.

---

## The plan (recap)

- **GUI stays local** on your Windows machine — it's only a config builder.
- **`main.py` runs on Sherlock** inside a SLURM job — MFA, spaCy, prosody, and the
  Llama LLM-cloze all run there.
- **LLM-cloze uses the existing HuggingFace backend on a GPU node** — no new
  extractor code; weights are cached on `$OAK`/`$GROUP_HOME`.
- Compute nodes are **offline**, so all model downloads happen once on a login node.

---

## Who does what

**You (one-time, on Sherlock — I can't do these):**
1. Confirm your account has a GPU partition. `sh_part` lists what you can submit to
   (`normal`, `gpu`, `dev`, or an `owners` partition if you're in an owner group).
2. Pick a storage root **not** in `$HOME` (tiny quota). Use `$GROUP_HOME` or `$OAK`
   for the conda env + model caches; `$SCRATCH` for working input/output.
3. Accept the Meta Llama-3 license on HuggingFace and create an `HF_TOKEN`
   (needed only for the one-time download).
4. Get the repo onto Sherlock (`git clone` or `rsync` from your PC).

**These scripts (run in order):**
| Script | Where | What it does |
|---|---|---|
| `setup_env.sh`      | **sh_dev node** | Builds the conda env: MFA (conda) + `requirements.txt` (pip) + spaCy model + MFA models. Must run on a compute node — see "Conda on Sherlock" below. |
| `download_models.sh`| login/sh_dev | Prefetches HF weights (GPT-2, Llama-3-8B) into `$HF_HOME`. |
| `run_extract.sbatch`| `sbatch`   | The GPU batch job: activates the env, sets cache env vars, runs `python main.py …`. |

---

## Quickstart

```bash
# 0. edit the CONFIG block at the top of each script to point at your paths
# 1. on a login node:
bash sherlock/setup_env.sh
HF_TOKEN=hf_xxx bash sherlock/download_models.sh
# 2. submit the GPU smoke test:
sbatch sherlock/run_extract.sbatch
squeue --me            # watch it
# outputs land in $OUTPUT_DIR; logs in slurm-<jobid>.out
```

---

## Conda on Sherlock (the main friction — read this)

MFA/Kaldi is **conda-forge-only** (not pip-installable), so the full pipeline *requires*
conda. But Sherlock **officially discourages conda** ("We recommend NOT using Anaconda on
Sherlock") because it dumps huge file counts into `$HOME` (blows quota), edits `~/.bashrc`
(slows logins), and conflicts with the module system. You have to thread that needle:

1. Use **miniforge** (conda-forge's minimal installer), **not** full Anaconda.
2. Install it **outside `$HOME`** — under `$GROUP_HOME` or `$OAK`.
3. Do the install **on an `sh_dev` compute node**, not a login node (the installer's file
   count "will likely fail on a login node" per Sherlock docs).

Bootstrap (once, inside an `sh_dev` session):
```bash
sh_dev -c 8 -m 32GB -t 02:00:00                       # grab a compute node
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p $GROUP_HOME/miniforge3
source $GROUP_HOME/miniforge3/etc/profile.d/conda.sh
bash sherlock/setup_env.sh                             # then build the env
```

**Hardening path (recommended once it works):** Sherlock's blessed way to carry a hard
conda dependency is an **Apptainer container** with the env baked in — no `$HOME`/module
conflicts, fully reproducible. Worth converting to once the smoke test proves out.

## Other gotchas (verify against current Sherlock docs)

- **Compute-node internet — verify.** I earlier assumed compute nodes are offline; but
  Sherlock telling you to `conda install` on `sh_dev` implies they *do* have outbound
  internet. If so, the `HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE` flags in the sbatch are
  just belt-and-suspenders (harmless when weights are cached). Confirm before relying
  on either assumption.
- **`bitsandbytes` ↔ CUDA.** The 4-bit quantization path needs a `bitsandbytes` build
  matching Sherlock's CUDA/torch. If it fails, the backend falls back to fp16 (fine for
  8B on a large-VRAM card). Match the CUDA module to your torch build.
- **MFA is easier here than on Windows** — conda-forge `montreal-forced-aligner` is
  best-supported on Linux. Your AMD/Windows MKL crash also does not occur on Sherlock.
- **Module names are placeholders.** I could not verify exact Lmod module names on
  Sherlock. Check with `ml spider cuda` and adjust the sbatch.

## ⚠️ One app-code caveat to resolve before a real run

`extractors/llm_cloze.py` lists model ids like `meta-llama/Llama-3-8B`, but the actual
HuggingFace repo id is `meta-llama/Meta-Llama-3-8B` (and 70B is `Meta-Llama-3-70B`).
`download_models.sh` uses the **real** repo ids. When we wire up the CLI invocation
(step #3, the `Namespace→CLI` emitter), we'll pass `--llm-model meta-llama/Meta-Llama-3-8B`
explicitly so it matches the downloaded weights. Flagging now — **not** fixing app code
in this experimental folder.
