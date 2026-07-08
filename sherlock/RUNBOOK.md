# Sherlock run — quick reference

## What this is and why
Sherlock is a remote, **headless** HPC cluster — there's no desktop to run the GUI
on, and a local conda env is built for one OS/CPU so it won't run on Sherlock's
nodes as-is. So we package the whole environment — Python + MFA/Kaldi + models —
into a single portable Linux **container image**, `sensi.sif`, built with
**Apptainer** (the container runtime HPC clusters allow). That one file runs
unchanged on Sherlock's compute nodes, including a GPU for the Llama LLM‑cloze.

The flow: **build the container** and **configure the run** on your own computer →
**ship** the container + code + a `config.yaml` to Sherlock → **run** the pipeline
there, inside the container, via `run_extract.py`.

## Two folders on Sherlock (fixed names — used everywhere below)
| Purpose | Folder | Holds |
|---|---|---|
| **Durable** (code + container) | `$GROUP_HOME/SensiExtractor` | the repo, incl. `sherlock/sensi.sif` and `sherlock/config.yaml` |
| **Working** (data + weights) | `$SCRATCH/sensi` | `transcripts/`, `audio/`, `output/`, `hf_cache/` |

`$GROUP_HOME` is shared with the whole group — if a labmate also uses a folder
called `SensiExtractor`, put a personal one in front (e.g. `$GROUP_HOME/amilcar/…`)
and adjust the paths. Step 3 creates both folders.

## Which terminal runs what
Each step's heading names the terminal. Don't mix them.
- **[local]** — your computer, in the project's conda env (`conda activate ./env`). Runs the GUI + `export_config.py`. *(Linux/macOS: your terminal. Windows: Anaconda Prompt.)*
- **[linux]** — a Linux shell with Apptainer, for building the container and rsync/ssh. *(Linux: your terminal. Windows: WSL. macOS: a Linux VM — Apptainer is Linux-only.)*
- **[Sherlock-login]** — inside `ssh amilcarm@login.sherlock.stanford.edu`. Setup / transfer / submit only.
- **[Sherlock-compute]** — a node from `sh_dev` (or the batch job). Everything that actually runs the pipeline.

Commands use forward-slash paths and `./env`, which work on every OS. The only 3
things I edit are marked **✎**: the GPU partition, the HF token, and the `run:`
block of `config.yaml`.

---

## 1 · Build the container — [linux], once
One-time tools *(Debian/Ubuntu shown; use your distro's package manager)*:
```bash
sudo apt update && sudo apt install -y apptainer uidmap squashfuse fuse2fs
```
Build it from the repo root (~5 GB → `sherlock/sensi.sif`). On Windows/WSL the repo
is under `/mnt/c/...`; on Linux/macOS use its path:
```bash
cd /mnt/c/Users/amilcar/Desktop/code/Antigravity/SensiExtractor
export APPTAINER_TMPDIR=/var/tmp/apptmp APPTAINER_CACHEDIR=/var/tmp/apptmp/cache
apptainer build sherlock/sensi.sif sherlock/Apptainer.def
```

## 2 · Configure the run — [local]
From the repo root: set features/model in the GUI, then close it:
```bash
conda activate ./env
python gui.py
```
Export those choices to `sherlock/config.yaml`:
```bash
python sherlock/export_config.py
```
Then open `sherlock/config.yaml` and fill the **`run:`** block ✎ (`$SCRATCH`/`~` are
expanded on Sherlock, so these point at the working folder from the table):
```yaml
run:
  transcripts_folder: $SCRATCH/sensi/transcripts
  audio_folder:       $SCRATCH/sensi/audio
  output_folder:      $SCRATCH/sensi/output
  analyzer:           sherlock-run
  hf_token_file:      ~/hf_token.txt
```

## 3 · Create the two Sherlock folders — [Sherlock-login], once
Must exist *before* the upload writes into them. Connect:
```bash
ssh amilcarm@login.sherlock.stanford.edu
```
Make them:
```bash
mkdir -p $GROUP_HOME/SensiExtractor $SCRATCH/sensi/{transcripts,audio,output,hf_cache}
```

## 4 · Upload code, container, data — [linux]
`$GROUP_HOME` does NOT expand over rsync, so the literal path is used.
Code first (small; carries `sherlock/config.yaml`; `*.sif` excluded so the container isn't sent here):
```bash
cd /mnt/c/Users/amilcar/Desktop/code/Antigravity/SensiExtractor
rsync -avP --exclude env --exclude hf_cache --exclude mfa_data --exclude .git \
  --exclude '*.sif' --exclude '__pycache__' \
  ./ amilcarm@login.sherlock.stanford.edu:/home/groups/brucemc/SensiExtractor/
```
Container into the repo's `sherlock/` folder (~5 GB):
```bash
rsync -avP sherlock/sensi.sif \
  amilcarm@login.sherlock.stanford.edu:/home/groups/brucemc/SensiExtractor/sherlock/
```
Sample data (paths must match the `run:` block from step 2):
```bash
rsync -avP ./my_transcripts/ \
  amilcarm@login.sherlock.stanford.edu:/scratch/users/amilcarm/sensi/transcripts/
```

## 5 · Download model weights — [Sherlock-login], once
Needs internet + a Llama-3-approved token. Create the token file ✎:
```bash
umask 077; printf 'hf_xxx\n' > ~/hf_token.txt; chmod 600 ~/hf_token.txt
```
Download into the cache the job reads:
```bash
export HF_HOME=$SCRATCH/sensi/hf_cache; mkdir -p $HF_HOME
apptainer exec --bind $SCRATCH:$SCRATCH --env HF_HOME=$HF_HOME \
  --env HF_TOKEN="$(cat ~/hf_token.txt)" --env HF_HUB_DISABLE_XET=1 \
  $GROUP_HOME/SensiExtractor/sherlock/sensi.sif \
  bash -c 'hf download gpt2 && hf download meta-llama/Meta-Llama-3-8B'
```

## 6 · (Optional) dry-run — [Sherlock-compute]
Validate the config + plumbing before spending a GPU allocation. Get a node:
```bash
sh_dev -c 4 -m 16GB -t 01:00:00
```
Run (reads `sherlock/config.yaml`; keep it text-only here for speed — GPU Llama is slow on a CPU node):
```bash
ulimit -n "$(ulimit -Hn)"
export HF_HOME=$SCRATCH/sensi/hf_cache
apptainer exec \
  --bind $SCRATCH:$SCRATCH \
  --bind $GROUP_HOME/SensiExtractor:$GROUP_HOME/SensiExtractor \
  --env HF_HOME=$HF_HOME --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
  $GROUP_HOME/SensiExtractor/sherlock/sensi.sif \
  python $GROUP_HOME/SensiExtractor/sherlock/run_extract.py
```

## 7 · Submit the GPU job — [Sherlock-login]
Set the GPU partition once ✎ (the sbatch already points at the two folders otherwise):
```bash
nano $GROUP_HOME/SensiExtractor/sherlock/run_container.sbatch
```
Submit:
```bash
sbatch $GROUP_HOME/SensiExtractor/sherlock/run_container.sbatch
```
Check the queue (note the JobID):
```bash
squeue --me
```
Watch the log:
```bash
tail -f slurm-<JobID>.out
```

## 8 · Collect results — [linux]
```bash
rsync -avP \
  amilcarm@login.sherlock.stanford.edu:/scratch/users/amilcarm/sensi/output/ \
  ./sherlock_out/
```

## Gotchas
- Anything that runs the pipeline must be on a **compute node** (`sh_dev`), never the login node (256 open-file cap → `Too many open files`). The sbatch raises the limit itself.
- HuggingFace: use `hf download` (not the removed `huggingface-cli`) with `HF_HUB_DISABLE_XET=1`. A `401` means the Llama-3 license isn't accepted/approved yet (`hf auth whoami` checks the token).
- A bad `config.yaml` makes `run_extract.py` print the errors and exit before running — fix the flagged keys/types, or re-run `export_config.py`.
