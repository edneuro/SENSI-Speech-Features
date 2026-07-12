# Sherlock run — quick reference

## What this is and why
Sherlock is a remote, **headless** HPC cluster — there's no desktop to run the GUI
on, and a local conda env is built for one OS/CPU so it won't run on Sherlock's
nodes as-is. So we package the whole environment — Python + MFA/Kaldi + models —
into a single portable Linux **container image**, `sensiextractor.sif`, built with
**Apptainer** (the container runtime HPC clusters allow). That one file runs
unchanged on Sherlock's compute nodes, including a GPU for the Llama LLM‑cloze.

The flow: **build the container** on your own computer → **ship** the container +
code (which carries your GUI's live `config/features.yaml`) to Sherlock → **fill in**
a machine-specific `config/run.yaml` there → **run** the pipeline inside the
container, via the top-level `run.py`.

## Fill in your own identifiers
- `<sunetid>` — your Stanford SUNet ID (used for `ssh` and in `$SCRATCH`).
- `<GROUP_HOME>` / `<SCRATCH>` — the *literal* values of `$GROUP_HOME` and `$SCRATCH`
  on Sherlock. rsync can't expand `$GROUP_HOME` on the remote side, so get the real
  values once (step 3: `echo $GROUP_HOME $SCRATCH`) and paste them into the rsync
  commands. Sherlock-side commands keep `$GROUP_HOME`/`$SCRATCH` — they resolve for you.

## Two folders on Sherlock (fixed names — used everywhere below)
| Purpose | Folder | Holds |
|---|---|---|
| **Durable** (code + container) | `$GROUP_HOME/SensiExtractor` | the repo, incl. `sherlock/sensiextractor.sif`, `config/features.yaml` (rsync'd from your machine), and `config/run.yaml` (created on Sherlock) |
| **Working** (data + weights) | `$SCRATCH/sensi` | `transcripts/`, `audio/`, `output/`, `hf_cache/` |

`$GROUP_HOME` is shared with your whole group — if a labmate also uses a folder
called `SensiExtractor`, add a personal one in front (e.g. `$GROUP_HOME/<sunetid>/…`)
and adjust the paths. Step 3 creates both folders.

## Which terminal runs what
Each step's heading names the terminal. Don't mix them.
- **[local]** — your computer, in the project's conda env (`conda activate ./env`). Runs the GUI, which saves your feature choices straight to `config/features.yaml`. *(Linux/macOS: your terminal. Windows: Anaconda Prompt.)*
- **[linux]** — a Linux shell with Apptainer, for building the container and rsync/ssh. *(Linux: your terminal. Windows: WSL. macOS: a Linux VM — Apptainer is Linux-only.)*
- **[Sherlock-login]** — inside `ssh <sunetid>@login.sherlock.stanford.edu`. Setup / transfer / submit only.
- **[Sherlock-compute]** — a node from `sh_dev` (or the batch job). Everything that actually runs the pipeline.

Commands use forward-slash paths and `./env`, which work on every OS. The only 3
things to edit are marked **✎**: the GPU partition, the HF token, and the `shared:`
block of `config/run.yaml`.

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

## 2 · Configure the extraction — [local]
From the repo root: set your feature toggles / model choices in the GUI, then close
it. The GUI saves them straight to `config/features.yaml` — that file **is** the
shareable config now, no export step needed:
```bash
conda activate ./env
python gui.py
```
`config/features.yaml` is gitignored (machine-local like the old `preferences.json`
was), but it rsyncs to Sherlock in step 4 like any other file. It carries WHAT to
extract (feature toggles, predictability models, semantic categories, …) — nothing
machine-specific lives in it.

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
Create `config/run.yaml` in the repo folder — this is **machine-specific** (folders,
analyzer name, token-file path) and is never rsync'd from your machine, so you write
it once, directly here. The repo's own `config/run_defaults.yaml` (a committed
template, generated from the code defaults — it isn't on Sherlock yet, since the
code lands in step 4) is the shape to copy; fill in your paths ✎:
```bash
mkdir -p $GROUP_HOME/SensiExtractor/config
cat > $GROUP_HOME/SensiExtractor/config/run.yaml <<'YAML'
shared:
  analyzer_name: sherlock-run
  last_transcripts_folder: $SCRATCH/sensi/transcripts
  last_audio_folder: $SCRATCH/sensi/audio
  last_output_folder: $SCRATCH/sensi/output
  theme: system
  hf_token_file: hf_token.txt
YAML
```
`$SCRATCH` is expanded by `run.py` at run time, so that literal string is fine as
written. `hf_token_file` is just a **path** — you create that file in step 5, so no
secret ever goes into this config. The shipped default, `hf_token.txt`, is a
*relative* path, which `run.py` resolves against `config/` itself (not your home
dir) — so it means `$GROUP_HOME/SensiExtractor/config/hf_token.txt` here. Use an
absolute path or `~/...` instead if you'd rather keep the token outside the repo
folder. (Once the code arrives in step 4, `cp config/run_defaults.yaml config/run.yaml`
on Sherlock is an equivalent way to start this file for any *future* re-run.)

## 4 · Upload code, container, data — [linux]
Use the **literal** `<GROUP_HOME>`/`<SCRATCH>` from step 3 (rsync won't expand them).
Code first (small; carries `config/features.yaml` — your GUI's live feature choices;
`*.sif` excluded so the container isn't sent here; `config/run.yaml` and
`config/preferences.json*` excluded so your local machine's folders/legacy prefs
never clobber the `run.yaml` you wrote on Sherlock in step 3):
```bash
cd path/to/SensiExtractor
rsync -avP --exclude env --exclude hf_cache --exclude mfa_data --exclude .git \
  --exclude '*.sif' --exclude '__pycache__' \
  --exclude 'config/run.yaml' --exclude 'config/preferences.json*' \
  ./ <sunetid>@login.sherlock.stanford.edu:<GROUP_HOME>/SensiExtractor/
```
Container into the repo's `sherlock/` folder (a few GB):
```bash
rsync -avP sherlock/sensiextractor.sif \
  <sunetid>@login.sherlock.stanford.edu:<GROUP_HOME>/SensiExtractor/sherlock/
```
**Your input files — the data you actually want to analyze.** These go into
`$SCRATCH/sensi/transcripts` and (only if using alignment / prosody / acoustic
features) `$SCRATCH/sensi/audio`. **Each `.wav` must share its transcript's
basename** — e.g. `story1.txt` ↔ `story1.wav`; a transcript with no matching `.wav`
is processed text-only. You change these run-to-run, so the easy way is the web:

- **Easiest — Sherlock's web file browser (no rsync):** open
  <https://ondemand.sherlock.stanford.edu> → **Files** → browse to
  `$SCRATCH/sensi/transcripts` (and `/audio`) → **Upload** (drag-and-drop). Also how
  you'll grab results later (step 8).
- **Bulk / scripted — rsync** (better for hundreds of files) [linux]:
  ```bash
  rsync -avP path/to/your/transcripts/ <sunetid>@login.sherlock.stanford.edu:<SCRATCH>/sensi/transcripts/
  rsync -avP path/to/your/audio/       <sunetid>@login.sherlock.stanford.edu:<SCRATCH>/sensi/audio/
  ```

(The code + container above still go by rsync — they're large/many-file and
one-time — but your *data* can just drag-and-drop through the browser.)

## 5 · Get your HF token, save it, download the weights — [Sherlock-login], once
The local Llama model is gated, so you need a HuggingFace token — **made once on the
HF website, saved once on Sherlock:**

1. On huggingface.co (in a browser): sign in → open `meta-llama/Meta-Llama-3-8B` →
   accept the license (wait for "access granted") → Settings → Access Tokens →
   create a **Read** token, and copy it.
2. On Sherlock, save that token into the file your `config/run.yaml` points at —
   the shipped default `hf_token_file: hf_token.txt` (see step 3) is a relative
   path, so it resolves against the repo's own `config/` dir, i.e.
   `$GROUP_HOME/SensiExtractor/config/hf_token.txt` — readable only by you.
   Replace `hf_xxx…` with the token you just copied ✎:
```bash
umask 077
printf 'hf_xxxYOURTOKEN\n' > $GROUP_HOME/SensiExtractor/config/hf_token.txt
chmod 600 $GROUP_HOME/SensiExtractor/config/hf_token.txt
```
   (This file is gitignored, same as on your local machine, so it's never rsync'd
   by the code-upload step or committed by accident.)
3. Download the weights into the cache the job reads. This is the only step that
   uses the token (for the gated download) — the batch job runs offline and needs
   no token:
```bash
export HF_HOME=$SCRATCH/sensi/hf_cache; mkdir -p $HF_HOME
apptainer exec --bind $SCRATCH:$SCRATCH --env HF_HOME=$HF_HOME \
  --env HF_TOKEN="$(cat $GROUP_HOME/SensiExtractor/config/hf_token.txt)" \
  --env HF_HUB_DISABLE_XET=1 \
  $GROUP_HOME/SensiExtractor/sherlock/sensiextractor.sif \
  bash -c 'hf download gpt2 && hf download meta-llama/Meta-Llama-3-8B'
```

## 6 · (Optional) dry-run — [Sherlock-compute]
Validate the config + plumbing before spending a GPU allocation. Get a node:
```bash
sh_dev -c 4 -m 16GB -t 01:00:00
```
Run (reads `config/features.yaml` + `config/run.yaml`, default `--config-dir`;
keep the model selection text-only here for speed — GPU Llama is slow on a CPU node):
```bash
ulimit -n "$(ulimit -Hn)"
export HF_HOME=$SCRATCH/sensi/hf_cache
apptainer exec \
  --bind $SCRATCH:$SCRATCH \
  --bind $GROUP_HOME/SensiExtractor:$GROUP_HOME/SensiExtractor \
  --env HF_HOME=$HF_HOME --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
  $GROUP_HOME/SensiExtractor/sherlock/sensiextractor.sif \
  python $GROUP_HOME/SensiExtractor/run.py
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

> **⚠️ Paths.** The sbatch assumes the folder layout from **step 3**
> (`$GROUP_HOME/SensiExtractor` for code+container, `$SCRATCH/sensi/hf_cache` for
> weights). If you followed step 3, it just works. **If you used different folders,**
> tell the sbatch where they are — add them to the same `--export` flag:
> ```bash
> sbatch --partition=gpu -C GPU_MEM:16GB \
>   --export=ALL,REPO=/your/path/SensiExtractor,HF_HOME=/your/scratch/hf_cache \
>   /your/path/SensiExtractor/sherlock/run_container.sbatch
> ```
> If a path is wrong the job exits immediately with `ERROR: … not found at <path>`
> and the override hint — so it fails clearly, never silently.

**c. Watch it.** Note the JobID, then follow the log:
```bash
squeue --me
```
```bash
tail -f slurm-<JobID>.out
```
The top of the log runs `nvidia-smi` — that confirms which GPU you landed on and its
VRAM, so you can verify you got the size you asked for.

## 8 · Collect results
The output CSVs are in `$SCRATCH/sensi/output`. Grab them either way:

- **Easiest — web:** <https://ondemand.sherlock.stanford.edu> → **Files** → browse to
  `$SCRATCH/sensi/output` → select → **Download**.
- **rsync** [linux] (literal `<SCRATCH>`):
  ```bash
  rsync -avP <sunetid>@login.sherlock.stanford.edu:<SCRATCH>/sensi/output/ ./sherlock_out/
  ```

## Gotchas
- Anything that runs the pipeline must be on a **compute node** (`sh_dev`), never the login node (256 open-file cap → `Too many open files`). The sbatch raises the limit itself.
- HuggingFace: use `hf download` (not the removed `huggingface-cli`) with `HF_HUB_DISABLE_XET=1`. A `401` means the Llama-3 license isn't accepted/approved yet (`hf auth whoami` checks the token).
- A bad `features.yaml`/`run.yaml` makes `run.py` print the errors and exit before running — fix the flagged keys/types (schema validation runs before anything else).
