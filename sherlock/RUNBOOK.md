# Sherlock run — my personal runbook (container + config workflow)

Build the environment once as a single `sensi.sif` on my PC, configure the run in
the GUI and export it to a `config.yaml`, ship both to Sherlock, and run the
pipeline there with `run_extract.py` — which reuses the app's planner + batch
runners, so behavior matches the GUI exactly.

## The 3 machines I switch between

Every command block says **▶ Run on:** which one. Most earlier trouble was running
a command on the wrong machine.

| Banner | What it is | How I get there |
|---|---|---|
| **💻 PC (WSL)** | My Windows machine's Ubuntu shell | a WSL terminal |
| **🔑 Login node** | Sherlock front-end; setup, transfers, downloads, submit | `ssh amilcarm@login.sherlock.stanford.edu` |
| **⚙️ Compute node** | Where the pipeline runs (CPUs/GPU, high limits) | `sh_dev …` (interactive) or an `sbatch` job |

**Rule of thumb:** the login node is *only* for setup/transfer/download/submit.
Anything that runs the pipeline (imports MFA, extracts features, loads Llama) must
be on a **compute node** — the login node's 256-open-file cap kills heavy imports.

## The only things I edit

- my **GPU partition** name (from `sh_part`) — in `run_container.sbatch` (step 7) — **✎**
- my **HuggingFace token** — stored once in `~/hf_token.txt` (step 6) — **✎**
- the **`run:` block of `config.yaml`** — folders/analyzer/token path (step A4) — **✎**

Everything else (`amilcarm`, group `brucemc`, `$GROUP_HOME/amilcar/sensi/…`,
`/scratch/users/amilcarm/…`) is already correct below.

---

# Part A — On my PC (build + configure)

## A1. Install build tools (💻 PC (WSL), once)
```bash
sudo apt update
sudo apt install -y apptainer uidmap squashfuse fuse2fs
```
`uidmap` = rootless build; `squashfuse`+`fuse2fs` = ability to run the `.sif` locally.

## A2. Point Apptainer's scratch at a big disk (💻 PC (WSL))
WSL's `/tmp` is a ~7.6 GB RAM disk; the build overflows it.
```bash
mkdir -p /var/tmp/apptmp
export APPTAINER_TMPDIR=/var/tmp/apptmp
export APPTAINER_CACHEDIR=/var/tmp/apptmp/cache
```

## A3. Build the container (💻 PC (WSL), from the repo root)
```bash
cd /mnt/c/Users/amilcar/Desktop/code/Antigravity/SensiExtractor
apptainer build sherlock/sensi.sif sherlock/Apptainer.def
```
Slow (full conda setup). Produces `sherlock/sensi.sif` (~5 GB). If it demands
root/fakeroot: prefix with
`sudo -E env APPTAINER_TMPDIR=$APPTAINER_TMPDIR APPTAINER_CACHEDIR=$APPTAINER_CACHEDIR`.

## A4. Configure the run and export it (💻 PC)
Set everything up in the GUI as usual (features, model, categories), then export
those choices to a portable config:
```bash
cd /mnt/c/Users/amilcar/Desktop/code/Antigravity/SensiExtractor
python gui.py                    # configure, then close
python sherlock/export_config.py # writes sherlock/config.yaml from the GUI's saved prefs
```
Then open `sherlock/config.yaml` and fill the **`run:`** block (the GUI can't supply
Sherlock paths). Use `$SCRATCH`/`~` — `run_extract.py` expands them on Sherlock:
```yaml
run:
  transcripts_folder: $SCRATCH/sensi/sample/transcripts
  audio_folder:       $SCRATCH/sensi/sample/audio
  output_folder:      $SCRATCH/sensi/sample/out
  analyzer:           sherlock-run
  hf_token_file:      ~/hf_token.txt          # local Llama (created in step 6)
  # openai_api_key_file: ~/openai_key.txt     # only if using a gpt-* model
```
`features:` / `settings:` / `semantic_categories:` came from the GUI — leave them.
`config.yaml` holds **no secrets** (only the token *path*), so it's safe to ship.

---

# Part B — Set up Sherlock

## 1. First login: paths, folders, checks (🔑 Login node, once)
```bash
ssh amilcarm@login.sherlock.stanford.edu

sh_part                                     # ← note my GPU partition (step 7 ✎)
apptainer --version || ml spider apptainer  # confirm apptainer (ml load if needed)

mkdir -p $GROUP_HOME/amilcar/sensi $SCRATCH/sensi/sample/{transcripts,audio,out}
```
Layout: **durable** (`.sif` + code) under `$GROUP_HOME/amilcar/sensi/`; **working**
(weights, data, output) under `$SCRATCH/sensi/`.

## 2. Upload the container + code (💻 PC (WSL) → Sherlock)
> ⚠️ Use the **literal** path `/home/groups/brucemc/…`, not `$GROUP_HOME` — it does
> NOT expand through rsync's remote shell.
```bash
cd /mnt/c/Users/amilcar/Desktop/code/Antigravity/SensiExtractor

# 2a. the container (~5 GB):
rsync -avP sherlock/sensi.sif \
  amilcarm@login.sherlock.stanford.edu:/home/groups/brucemc/amilcar/sensi/

# 2b. the code — this carries sherlock/config.yaml with it. --exclude '*.sif'
#     avoids re-sending the 5 GB container that lives inside sherlock/.
rsync -avP \
  --exclude env --exclude hf_cache --exclude mfa_data --exclude .git \
  --exclude '*.sif' --exclude 'sherlock/*_sandbox' --exclude '__pycache__' \
  ./ amilcarm@login.sherlock.stanford.edu:/home/groups/brucemc/amilcar/sensi/SensiExtractor/
```
(Re-run 2b alone whenever I change `config.yaml` or the code — it's small.)

## 3. (Optional) upload sample data (💻 PC (WSL) → Sherlock)
Put `.txt` (and matching-basename `.wav` for alignment) into the sample folders,
matching the paths in `config.yaml`'s `run:` block:
```bash
rsync -avP ./my_transcripts/ \
  amilcarm@login.sherlock.stanford.edu:/scratch/users/amilcarm/sensi/sample/transcripts/
# and ./my_audio/ → .../sample/audio/ if I have wavs
```

## 4. Download the model weights (🔑 Login node, once)
Compute nodes are offline — download here. First, one-time on huggingface.co: accept
the license at meta-llama/Meta-Llama-3-8B (wait for "access granted"), make a **Read**
token. Store it in a plain file (matches `hf_token_file` in the config):
```bash
umask 077
printf 'hf_xxx\n' > ~/hf_token.txt && chmod 600 ~/hf_token.txt     # ✎ my token (once)

export HF_HOME=$SCRATCH/sensi/hf_cache && mkdir -p $HF_HOME
apptainer exec \
  --bind $SCRATCH:$SCRATCH \
  --env HF_HOME=$HF_HOME \
  --env HF_TOKEN="$(cat ~/hf_token.txt)" \
  --env HF_HUB_DISABLE_XET=1 \
  $GROUP_HOME/amilcar/sensi/sensi.sif bash -c '
    hf download gpt2 &&
    hf download meta-llama/Meta-Llama-3-8B
  '
```
Notes: `hf download` (not the removed `huggingface-cli`); `HF_HUB_DISABLE_XET=1` is
**required** (Xet 404s on Sherlock); a 401 means the license/token isn't approved.
Download-only — the job runs offline and reads this cache.

---

# Part C — Run the pipeline

Both run steps use one shorthand — set it first in each shell:
```bash
REPO=$GROUP_HOME/amilcar/sensi/SensiExtractor    # code lives here; config at $REPO/sherlock/config.yaml
```

## 5. Dry-run on a compute node (⚙️ Compute node, no queue)
Validates the config + plumbing before spending a GPU allocation. `run_extract.py`
with no argument reads `sherlock/config.yaml` next to it.
```bash
sh_dev -c 4 -m 16GB -t 01:00:00          # interactive CPU node
ulimit -n "$(ulimit -Hn)"                # lift the open-file cap
export HF_HOME=$SCRATCH/sensi/hf_cache

apptainer exec \
  --bind $SCRATCH:$SCRATCH --bind $REPO:$REPO \
  --env HF_HOME=$HF_HOME --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
  $GROUP_HOME/amilcar/sensi/sensi.sif \
  python $REPO/sherlock/run_extract.py
```
`run_extract.py` validates the config (exits with a clear message on bad keys/types),
then runs whichever levels the config enables. **Tip:** for a fast check, dry-run with
a text-only config (LLM-cloze + alignment off); the GPU Llama path is slow on this CPU
node. To test the GPU path interactively instead, add `-g 1` to `sh_dev` and `--nv` to
`apptainer exec`. Check output: `ls $SCRATCH/sensi/sample/out/`.

## 6. Submit the full GPU job (🔑 Login node → ⚙️ runs)
The job runs `run_extract.py` inside the container on a GPU node via
`run_container.sbatch` (it defaults `CONFIG=$REPO/sherlock/config.yaml`).
```bash
nano $REPO/sherlock/run_container.sbatch   # set  #SBATCH --partition=  ✎ to my GPU partition (once)

sbatch --export=ALL,SIF=$GROUP_HOME/amilcar/sensi/sensi.sif,REPO=$REPO \
  $REPO/sherlock/run_container.sbatch
```

## 7. Watch, then collect
**🔑 Login node:**
```bash
squeue --me                    # PD = pending, R = running; note the JobID
tail -f slurm-<JobID>.out      # live log; Ctrl-C to stop watching
# scancel <JobID>              # to kill it
```
**💻 PC (WSL)** — pull results back (literal path; `$SCRATCH` won't expand through rsync):
```bash
rsync -avP amilcarm@login.sherlock.stanford.edu:/scratch/users/amilcarm/sensi/sample/out/ ./sherlock_out/
```

---

## Troubleshooting

| Symptom | Where | Fix |
|---|---|---|
| `Too many open files` (Errno 24) | login node | Ran a pipeline command on the login node (256 fd cap). Use a compute node (`sh_dev`), then `ulimit -n $(ulimit -Hn)`. |
| `Invalid config:` + a list, then exit | run | `run_extract.py` rejected `config.yaml` — fix the flagged keys/types, or re-run `export_config.py` to regenerate a valid skeleton. |
| `run.analyzer is required` / `missing required folder(s)` | run | Fill the `run:` block in `config.yaml` (analyzer + transcripts/output folders). |
| `apptainer: command not found` | login node | `ml spider apptainer`, then `ml load` the module it names. |
| rsync `No such file or directory` with `$GROUP_HOME` in the path | PC | Use the literal `/home/groups/brucemc/…` path, not `$GROUP_HOME`. |
| `hf: huggingface-cli deprecated` / Xet `404` | download | Use `hf download` and `--env HF_HUB_DISABLE_XET=1`. |
| Llama download `401` / "requires approval" | download | Accept the license on huggingface.co, wait for access; verify the token: `hf auth whoami`. |
| Job stuck pending (`PD`) | login node | Wrong/busy partition. `sh_part`; `squeue --me --start` for an ETA. |
| `nvidia-smi failed` in the log | job | Not a GPU node — fix `--partition`/`--gpus` in `run_container.sbatch`. |
| `LLM_Cloze` columns empty/NaN | job | Weights not found offline — job's `HF_HOME` must match step 4's, and `config.yaml`'s `settings.llm_model` must match the downloaded repo id. |
| `CUDA out of memory` | job | Pin a bigger card: `#SBATCH -C GPU_MEM:24GB` in the sbatch, or it falls back to CPU. |
| MFA/alignment errors | job | Each `.wav` needs a matching-basename `.txt` in the audio/transcripts folders. |

Stuck on something else? Paste the `slurm-<JobID>.out` tail (or the compute-node error).
