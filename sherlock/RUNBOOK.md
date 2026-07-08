# Sherlock run — quick reference

## What this is and why
Sherlock (the HPC cluster) can't run our Windows GUI or the local `./env` conda
environment. So we package the whole environment — Python + MFA/Kaldi + models —
into a single Linux **container image**, `sensi.sif`, using **Apptainer** (the
container runtime HPC clusters allow). That one file runs unchanged on Sherlock's
compute nodes, including a GPU for the Llama LLM‑cloze.

The flow: **build the container** and **configure the run** on my laptop → **ship**
the container + code + a `config.yaml` to Sherlock → **run** the pipeline there,
inside the container, via `run_extract.py`.

## Which terminal runs what
Each step's heading names the terminal. Don't mix them.
- **[WSL]** — the Ubuntu terminal on my laptop. Apptainer and rsync/ssh live here.
- **[Anaconda Prompt]** — Windows, after `conda activate .\env`. The GUI + Python tools.
- **[Sherlock-login]** — inside `ssh amilcarm@login.sherlock.stanford.edu`. Setup / transfer / submit only.
- **[Sherlock-compute]** — a node from `sh_dev` (or the batch job). Everything that actually runs the pipeline.

The only 3 things I edit are marked **✎**: the GPU partition, the HF token, and the
`run:` block of `config.yaml`. Every path below is already mine.

---

## 1 · Build the container — [WSL], once
One-time tools:
```bash
sudo apt update && sudo apt install -y apptainer uidmap squashfuse fuse2fs
```
Build it (~5 GB → `sherlock/sensi.sif`):
```bash
cd /mnt/c/Users/amilcar/Desktop/code/Antigravity/SensiExtractor
export APPTAINER_TMPDIR=/var/tmp/apptmp APPTAINER_CACHEDIR=/var/tmp/apptmp/cache
apptainer build sherlock/sensi.sif sherlock/Apptainer.def
```

## 2 · Configure the run — [Anaconda Prompt]
Set features/model in the GUI, then close it:
```bat
cd C:\Users\amilcar\Desktop\code\Antigravity\SensiExtractor
conda activate .\env
python gui.py
```
Export those choices to `sherlock\config.yaml`:
```bat
python sherlock\export_config.py
```
Then open `sherlock\config.yaml` and fill the **`run:`** block ✎ (the GUI can't know
Sherlock's paths; `$SCRATCH`/`~` are expanded on Sherlock):
```yaml
run:
  transcripts_folder: $SCRATCH/sensi/sample/transcripts
  audio_folder:       $SCRATCH/sensi/sample/audio
  output_folder:      $SCRATCH/sensi/sample/out
  analyzer:           sherlock-run
  hf_token_file:      ~/hf_token.txt
```

## 3 · Create the Sherlock folders — [Sherlock-login], once
Must exist *before* the upload writes into them. Connect:
```bash
ssh amilcarm@login.sherlock.stanford.edu
```
Make the folders:
```bash
mkdir -p $GROUP_HOME/amilcar/sensi $SCRATCH/sensi/sample/{transcripts,audio,out}
```

## 4 · Upload container + code + data — [WSL]
`$GROUP_HOME` does NOT expand over rsync, so the literal path is used.
Container (~5 GB):
```bash
cd /mnt/c/Users/amilcar/Desktop/code/Antigravity/SensiExtractor
rsync -avP sherlock/sensi.sif \
  amilcarm@login.sherlock.stanford.edu:/home/groups/brucemc/amilcar/sensi/
```
Code (small; carries `sherlock/config.yaml`; the `*.sif` exclude avoids re-sending the container):
```bash
rsync -avP --exclude env --exclude hf_cache --exclude mfa_data --exclude .git \
  --exclude '*.sif' --exclude '__pycache__' \
  ./ amilcarm@login.sherlock.stanford.edu:/home/groups/brucemc/amilcar/sensi/SensiExtractor/
```
Sample data (paths must match the `run:` block from step 2):
```bash
rsync -avP ./my_transcripts/ \
  amilcarm@login.sherlock.stanford.edu:/scratch/users/amilcarm/sensi/sample/transcripts/
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
  $GROUP_HOME/amilcar/sensi/sensi.sif \
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
  --bind $GROUP_HOME/amilcar/sensi/SensiExtractor:$GROUP_HOME/amilcar/sensi/SensiExtractor \
  --env HF_HOME=$HF_HOME --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
  $GROUP_HOME/amilcar/sensi/sensi.sif \
  python $GROUP_HOME/amilcar/sensi/SensiExtractor/sherlock/run_extract.py
```

## 7 · Submit the GPU job — [Sherlock-login]
Set the GPU partition once ✎ (the sbatch already points at my paths otherwise):
```bash
nano $GROUP_HOME/amilcar/sensi/SensiExtractor/sherlock/run_container.sbatch
```
Submit:
```bash
sbatch $GROUP_HOME/amilcar/sensi/SensiExtractor/sherlock/run_container.sbatch
```
Check the queue (note the JobID):
```bash
squeue --me
```
Watch the log:
```bash
tail -f slurm-<JobID>.out
```

## 8 · Collect results — [WSL]
```bash
rsync -avP \
  amilcarm@login.sherlock.stanford.edu:/scratch/users/amilcarm/sensi/sample/out/ \
  ./sherlock_out/
```

## Gotchas
- Anything that runs the pipeline must be on a **compute node** (`sh_dev`), never the login node (256 open-file cap → `Too many open files`). The sbatch raises the limit itself.
- HuggingFace: use `hf download` (not the removed `huggingface-cli`) with `HF_HUB_DISABLE_XET=1`. A `401` means the Llama-3 license isn't accepted/approved yet (`hf auth whoami` checks the token).
- A bad `config.yaml` makes `run_extract.py` print the errors and exit before running — fix the flagged keys/types, or re-run `export_config.py`.
