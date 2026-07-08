# Sherlock run — quick reference

## What this is and why
Sherlock is a remote, **headless** HPC cluster — there's no desktop to run the GUI
on, and a local conda env is built for one OS/CPU so it won't run on Sherlock's
nodes as-is. So we package the whole environment — Python + MFA/Kaldi + models —
into a single portable Linux **container image**, `sensiextractor.sif`, built with
**Apptainer** (the container runtime HPC clusters allow). That one file runs
unchanged on Sherlock's compute nodes, including a GPU for the Llama LLM‑cloze.

The flow: **build the container** and **configure the run** on your own computer →
**ship** the container + code + a `config.yaml` to Sherlock → **run** the pipeline
there, inside the container, via `run_extract.py`.

## Fill in your own identifiers
- `<sunetid>` — your Stanford SUNet ID (used for `ssh` and in `$SCRATCH`).
- `<GROUP_HOME>` / `<SCRATCH>` — the *literal* values of `$GROUP_HOME` and `$SCRATCH`
  on Sherlock. rsync can't expand `$GROUP_HOME` on the remote side, so get the real
  values once (step 3: `echo $GROUP_HOME $SCRATCH`) and paste them into the rsync
  commands. Sherlock-side commands keep `$GROUP_HOME`/`$SCRATCH` — they resolve for you.

## Two folders on Sherlock (fixed names — used everywhere below)
| Purpose | Folder | Holds |
|---|---|---|
| **Durable** (code + container) | `$GROUP_HOME/SensiExtractor` | the repo, incl. `sherlock/sensiextractor.sif` and `sherlock/config.yaml` |
| **Working** (data + weights) | `$SCRATCH/sensi` | `transcripts/`, `audio/`, `output/`, `hf_cache/` |

`$GROUP_HOME` is shared with your whole group — if a labmate also uses a folder
called `SensiExtractor`, add a personal one in front (e.g. `$GROUP_HOME/<sunetid>/…`)
and adjust the paths. Step 3 creates both folders.

## Which terminal runs what
Each step's heading names the terminal. Don't mix them.
- **[local]** — your computer, in the project's conda env (`conda activate ./env`). Runs the GUI + `export_config.py`. *(Linux/macOS: your terminal. Windows: Anaconda Prompt.)*
- **[linux]** — a Linux shell with Apptainer, for building the container and rsync/ssh. *(Linux: your terminal. Windows: WSL. macOS: a Linux VM — Apptainer is Linux-only.)*
- **[Sherlock-login]** — inside `ssh <sunetid>@login.sherlock.stanford.edu`. Setup / transfer / submit only.
- **[Sherlock-compute]** — a node from `sh_dev` (or the batch job). Everything that actually runs the pipeline.

Commands use forward-slash paths and `./env`, which work on every OS. The only 3
things to edit are marked **✎**: the GPU partition, the HF token, and the `run:`
block of `config.yaml`.

---

## 1 · Build the container — [linux], once
One-time tools *(Debian/Ubuntu shown; use your distro's package manager)*:
```bash
sudo apt update && sudo apt install -y apptainer uidmap squashfuse fuse2fs
```
Build it from the repo root. On Windows/WSL the repo is under `/mnt/c/...`; on
Linux/macOS use its path. Output ≈ a few GB (mostly the CUDA torch for GPU Llama):
```bash
cd path/to/SensiExtractor
export APPTAINER_TMPDIR=/var/tmp/apptmp APPTAINER_CACHEDIR=/var/tmp/apptmp/cache
apptainer build sherlock/sensiextractor.sif sherlock/Apptainer.def
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
  hf_token_file:      ~/hf_token.txt          # local Llama; you create this file in step 5
  # openai_api_key_file: ~/openai_key.txt     # only for gpt-* models on an internet-connected machine
```
`hf_token_file` / `openai_api_key_file` are just **paths** — you create those files
on Sherlock (step 5), so no secret ever goes into this config. Leave the paths as-is.

## 3 · Connect, note your paths, make the folders — [Sherlock-login], once
```bash
ssh <sunetid>@login.sherlock.stanford.edu
```
Print the literal paths (copy them for the rsync in step 4):
```bash
echo $GROUP_HOME $SCRATCH
```
Create both folders (these expand automatically here):
```bash
mkdir -p $GROUP_HOME/SensiExtractor $SCRATCH/sensi/{transcripts,audio,output,hf_cache}
```

## 4 · Upload code, container, data — [linux]
Use the **literal** `<GROUP_HOME>`/`<SCRATCH>` from step 3 (rsync won't expand them).
Code first (small; carries `sherlock/config.yaml`; `*.sif` excluded so the container isn't sent here):
```bash
cd path/to/SensiExtractor
rsync -avP --exclude env --exclude hf_cache --exclude mfa_data --exclude .git \
  --exclude '*.sif' --exclude '__pycache__' \
  ./ <sunetid>@login.sherlock.stanford.edu:<GROUP_HOME>/SensiExtractor/
```
Container into the repo's `sherlock/` folder (a few GB):
```bash
rsync -avP sherlock/sensiextractor.sif \
  <sunetid>@login.sherlock.stanford.edu:<GROUP_HOME>/SensiExtractor/sherlock/
```
Sample data (paths must match the `run:` block from step 2):
```bash
rsync -avP ./my_transcripts/ \
  <sunetid>@login.sherlock.stanford.edu:<SCRATCH>/sensi/transcripts/
```

## 5 · Get your HF token, save it, download the weights — [Sherlock-login], once
The local Llama model is gated, so you need a HuggingFace token — **made once on the
HF website, saved once on Sherlock:**

1. On huggingface.co (in a browser): sign in → open `meta-llama/Meta-Llama-3-8B` →
   accept the license (wait for "access granted") → Settings → Access Tokens →
   create a **Read** token, and copy it.
2. On Sherlock, save that token into the file your `config.yaml` points at
   (`hf_token_file: ~/hf_token.txt`, i.e. your home dir), readable only by you —
   replace `hf_xxx…` with the token you just copied ✎:
```bash
umask 077; printf 'hf_xxxYOURTOKEN\n' > ~/hf_token.txt; chmod 600 ~/hf_token.txt
```
3. Download the weights into the cache the job reads. This is the only step that
   uses the token (for the gated download) — the batch job runs offline and needs
   no token:
```bash
export HF_HOME=$SCRATCH/sensi/hf_cache; mkdir -p $HF_HOME
apptainer exec --bind $SCRATCH:$SCRATCH --env HF_HOME=$HF_HOME \
  --env HF_TOKEN="$(cat ~/hf_token.txt)" --env HF_HUB_DISABLE_XET=1 \
  $GROUP_HOME/SensiExtractor/sherlock/sensiextractor.sif \
  bash -c 'hf download gpt2 && hf download meta-llama/Meta-Llama-3-8B'
```

> **Using an OpenAI (`gpt-*`) model instead of Llama?** Save your OpenAI API key the
> same way — into `~/openai_key.txt` (the config's `openai_api_key_file`), `chmod
> 600`. But OpenAI needs internet, so it won't run on Sherlock's offline compute
> nodes; use it only when running `run_extract.py` on an internet-connected machine.
> With a `gpt-*` model you can skip the Llama download above.

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
  $GROUP_HOME/SensiExtractor/sherlock/sensiextractor.sif \
  python $GROUP_HOME/SensiExtractor/sherlock/run_extract.py
```

## 7 · Build & submit the GPU job — [Sherlock-login]

**a. See what you can request.** Which partitions you can submit to, and which VRAM
sizes actually exist on the cluster:
```bash
sh_part
```
```bash
sinfo -o "%P %f" | grep -o "GPU_MEM:[0-9]*GB" | sort -u
```

**b. Submit — pass the partition and VRAM as flags** (no file editing; these `sbatch`
flags override the script's defaults). Pick a partition from `sh_part` and a VRAM
size from the `sinfo` list that fits your model:
- **Llama-3-8B** (default, 4-bit) → `16GB` is plenty (needs ~6–8 GB).
- **Llama-3-70B** (4-bit) → `48GB` or `80GB` (also set `settings.llm_model` + download
  those weights in step 5).
- Bigger VRAM = **longer queue**, so don't over-request.
```bash
sbatch --partition=gpu -C GPU_MEM:16GB \
  $GROUP_HOME/SensiExtractor/sherlock/run_container.sbatch
```
(Drop `--partition=gpu` if `gpu` is already your default; swap it for your partition
otherwise. Change `16GB` to any value from the `sinfo` list.)

**c. Watch it.** Note the JobID, then follow the log:
```bash
squeue --me
```
```bash
tail -f slurm-<JobID>.out
```
The top of the log runs `nvidia-smi` — that confirms which GPU you landed on and its
VRAM, so you can verify you got the size you asked for.

## 8 · Collect results — [linux]
Literal `<SCRATCH>` again:
```bash
rsync -avP \
  <sunetid>@login.sherlock.stanford.edu:<SCRATCH>/sensi/output/ \
  ./sherlock_out/
```

## Gotchas
- Anything that runs the pipeline must be on a **compute node** (`sh_dev`), never the login node (256 open-file cap → `Too many open files`). The sbatch raises the limit itself.
- HuggingFace: use `hf download` (not the removed `huggingface-cli`) with `HF_HUB_DISABLE_XET=1`. A `401` means the Llama-3 license isn't accepted/approved yet (`hf auth whoami` checks the token).
- A bad `config.yaml` makes `run_extract.py` print the errors and exit before running — fix the flagged keys/types, or re-run `export_config.py`.
