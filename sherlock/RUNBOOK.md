# Sherlock run — end-to-end runbook (container path)

Build the environment once as a single `sensi.sif` file on your PC, ship it to
Sherlock, and run the pipeline there.

## The 3 machines you switch between

Every command block below starts with a **▶ Run on:** banner saying which one.
Most of your earlier trouble was running a command on the wrong one.

| Banner | What it is | How you get there |
|---|---|---|
| **💻 PC (WSL)** | Your Windows machine's Ubuntu shell | a WSL terminal |
| **🔑 Login node** | Sherlock front-end; setup, transfers, downloads, job submit | `ssh amilcarm@login.sherlock.stanford.edu` |
| **⚙️ Compute node** | Where the pipeline actually runs (has CPUs/GPU, high limits) | `sh_dev …` (interactive) or a `sbatch` job |

**Rule of thumb:** the login node is *only* for setup/transfer/submit. Anything
that actually runs the pipeline (imports MFA, extracts features) must be on a
**compute node** — the login node has a 256-open-file cap that kills heavy imports.

## What you must fill in

Your paths are already filled in below (`amilcarm`, group `brucemc`,
`$GROUP_HOME/amilcar/sensi/…`, `/scratch/users/amilcarm/…`). Only two things are
yours to edit, both marked **✎**:
- your **GPU partition** name (from `sh_part`)
- your **HuggingFace token** (stored once in step 4)

---

# Part A — Build the container (💻 PC, once)

Apptainer is Linux-only → build in WSL Ubuntu. This bakes the whole fragile env
(conda MFA/Kaldi + pip + spaCy + MFA models) into one ~5 GB `sensi.sif`.

### A1. Install build tools

**▶ Run on: 💻 PC (WSL)**
```bash
sudo apt update
sudo apt install -y apptainer uidmap squashfuse fuse2fs
```
`uidmap` = rootless build; `squashfuse`+`fuse2fs` = ability to run the .sif locally.

### A2. Point Apptainer's scratch at a big disk

**▶ Run on: 💻 PC (WSL)** — WSL's `/tmp` is a ~7.6 GB RAM disk; the build overflows it.
```bash
mkdir -p /var/tmp/apptmp
export APPTAINER_TMPDIR=/var/tmp/apptmp
export APPTAINER_CACHEDIR=/var/tmp/apptmp/cache
```

### A3. Build

**▶ Run on: 💻 PC (WSL)** — from the repo root.
```bash
cd /mnt/c/Users/amilcar/Desktop/code/Antigravity/SensiExtractor
apptainer build sherlock/sensi.sif sherlock/Apptainer.def
```
Slow (does a full conda setup). Produces `sherlock/sensi.sif`. If it demands
root/fakeroot: prefix with `sudo -E env APPTAINER_TMPDIR=$APPTAINER_TMPDIR APPTAINER_CACHEDIR=$APPTAINER_CACHEDIR`.

---

# Part B — Set up Sherlock

## 1. First login: find your paths & make folders (🔑, once)

**▶ Run on: 🔑 Login node**
```bash
ssh amilcarm@login.sherlock.stanford.edu

sh_part                                    # ← note your GPU partition name (step 6 ✎)
apptainer --version || ml spider apptainer # confirm apptainer (ml load it if needed)

mkdir -p $GROUP_HOME/amilcar/sensi $SCRATCH/sensi/sample/{transcripts,audio,out}
```
Layout this runbook uses:
- **Durable** (`.sif` + code): `$GROUP_HOME/amilcar/sensi/` → holds `sensi.sif` and `SensiExtractor/`
- **Working** (weights, data, output): `$SCRATCH/sensi/`

## 2. Upload the container + code (💻 PC → Sherlock)

**▶ Run on: 💻 PC (WSL)** — from the repo root.

> ⚠️ Use the **literal** path `/home/groups/brucemc/…`, not `$GROUP_HOME` —
> `$GROUP_HOME` does NOT expand through rsync's remote shell.

```bash
cd /mnt/c/Users/amilcar/Desktop/code/Antigravity/SensiExtractor

# 2a. the container (~5 GB):
rsync -avP sherlock/sensi.sif \
  amilcarm@login.sherlock.stanford.edu:/home/groups/brucemc/amilcar/sensi/

# 2b. the code (small; --exclude '*.sif' avoids re-sending the 5 GB container):
rsync -avP \
  --exclude env --exclude hf_cache --exclude mfa_data --exclude .git \
  --exclude '*.sif' --exclude 'sherlock/*_sandbox' --exclude '__pycache__' \
  ./ amilcarm@login.sherlock.stanford.edu:/home/groups/brucemc/amilcar/sensi/SensiExtractor/
```

## 3. (Optional) upload sample data

**▶ Run on: 💻 PC (WSL)** — put a few `.txt` (and matching-basename `.wav` for
alignment) into the sample folders. Skip if your stimuli already live on Sherlock.
```bash
rsync -avP ./my_transcripts/ \
  amilcarm@login.sherlock.stanford.edu:/scratch/users/amilcarm/sensi/sample/transcripts/
# (and similarly ./my_audio/ → .../sample/audio/ if you have wavs)
```

## 4. Download the model weights (🔑, once)

**▶ Run on: 🔑 Login node** (compute nodes are offline — download here).

First, one-time on HuggingFace's website: accept the license at
huggingface.co/meta-llama/Meta-Llama-3-8B (wait for "access granted"), then make a
**Read** token (Settings → Access Tokens).

Store the token privately, then download:
```bash
umask 077
printf 'hf_xxx\n' > $HOME/.hf_token && chmod 600 $HOME/.hf_token     # ✎ your token (once)

export HF_HOME=$SCRATCH/sensi/hf_cache && mkdir -p $HF_HOME
apptainer exec \
  --bind $SCRATCH:$SCRATCH \
  --env HF_HOME=$HF_HOME \
  --env HF_TOKEN="$(cat $HOME/.hf_token)" \
  --env HF_HUB_DISABLE_XET=1 \
  $GROUP_HOME/amilcar/sensi/sensi.sif bash -c '
    hf download gpt2 &&
    hf download meta-llama/Meta-Llama-3-8B
  '
```
Notes: use `hf download` (not the removed `huggingface-cli`); `HF_HUB_DISABLE_XET=1`
is **required** (Xet backend 404s on Sherlock); a 401 means the license/token isn't
approved yet. This is download-only — the job itself runs offline and needs no token.

---

# Part C — Run the pipeline (⚙️ compute node)

## 5. Dry-run test first (⚙️, no GPU, no queue)

**▶ Run on: ⚙️ Compute node** — this both verifies the container and does a real
text-only extraction. Do this before spending a GPU allocation.

```bash
sh_dev -c 4 -m 16GB -t 01:00:00          # gets you an interactive compute node
ulimit -n "$(ulimit -Hn)"                # raise open-file limit (login-node habit; safe here)

apptainer exec \
  --bind $SCRATCH:$SCRATCH \
  --bind $GROUP_HOME/amilcar/sensi/SensiExtractor:$GROUP_HOME/amilcar/sensi/SensiExtractor \
  $GROUP_HOME/amilcar/sensi/sensi.sif \
  python $GROUP_HOME/amilcar/sensi/SensiExtractor/main.py \
    --transcripts-folder $SCRATCH/sensi/sample/transcripts \
    --audio-folder       $SCRATCH/sensi/sample/audio \
    --output-folder      $SCRATCH/sensi/sample/out \
    --analyzer sherlock-drytest --level word --no-align --no-llm-cloze
```
Expected: spaCy → freq → GPT-2 surprisal messages, then `Done. N rows written`.
Batch mode requires **all three** folder flags even in text-only mode; `--no-align`
just lets the audio folder be empty.

Check it worked:
```bash
head $SCRATCH/sensi/sample/out/word/*.csv    # Surprisal column should be populated
```

## 6. Submit the full GPU job (🔑 submit → ⚙️ runs)

**▶ Run on: 🔑 Login node.** The full run adds MFA alignment, prosody, and the
Llama LLM-cloze (GPU). It uses `sherlock/run_container.sbatch`.

Edit the partition once:
```bash
cd $GROUP_HOME/amilcar/sensi/SensiExtractor
nano sherlock/run_container.sbatch     # set  #SBATCH --partition=  ✎ to your GPU partition
```
Submit (paths passed as overrides so the script points at your `/amilcar/` layout):
```bash
sbatch --export=ALL,\
SIF=$GROUP_HOME/amilcar/sensi/sensi.sif,\
REPO=$GROUP_HOME/amilcar/sensi/SensiExtractor \
  sherlock/run_container.sbatch
```

## 7. Watch, then collect

**▶ Run on: 🔑 Login node.**
```bash
squeue --me                    # PD = pending, R = running; note the JobID
tail -f slurm-<JobID>.out      # live log; Ctrl-C to stop watching
# scancel <JobID>              # to kill it
ls -lh $SCRATCH/sensi/sample/out/word/
```

**▶ Run on: 💻 PC (WSL)** — pull the results back (literal path; `$SCRATCH` won't
expand through rsync):
```bash
rsync -avP amilcarm@login.sherlock.stanford.edu:/scratch/users/amilcarm/sensi/sample/out/ ./sherlock_out/
```

---

## Troubleshooting

| Symptom | Where | Fix |
|---|---|---|
| `Too many open files` (Errno 24) | login node | You ran a pipeline/MFA command on the login node (256 fd cap). Move to a compute node (`sh_dev`), then `ulimit -n $(ulimit -Hn)`. |
| `Batch mode requires --transcripts-folder, --audio-folder, and --output-folder` | anywhere | Pass all three folder flags, even for text-only. |
| `apptainer: command not found` | login node | `ml spider apptainer` then `ml load` the module it names. |
| rsync `No such file or directory` with `$GROUP_HOME` in the path | PC | Use the literal `/home/groups/brucemc/…` path, not `$GROUP_HOME`. |
| `hf: huggingface-cli deprecated` / Xet `404` | download | Use `hf download` and `--env HF_HUB_DISABLE_XET=1`. |
| Llama download `401` / "requires approval" | download | Accept the license on huggingface.co and wait for access; check the token is valid (`hf auth whoami`). |
| Job stuck pending (`PD`) | login node | Wrong/busy partition. `sh_part`; `squeue --me --start` for an ETA. |
| `nvidia-smi failed` in the log | job | Not a GPU node — fix `--partition`/`--gpus` in the sbatch. |
| `LLM_Cloze` columns empty/NaN | job | Weights not found offline — job's `HF_HOME` must match step 4's, and `--llm-model` must match the downloaded repo id. |
| `CUDA out of memory` | job | Pin a bigger card: `#SBATCH -C GPU_MEM:24GB`, or it falls back to CPU. |
| MFA/alignment errors | job | Each `.wav` needs a matching-basename `.txt`. |

Stuck on something else? Paste the `slurm-<JobID>.out` tail (or the compute-node
error) and I'll debug it.
