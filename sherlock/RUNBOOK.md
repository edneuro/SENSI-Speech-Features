# Sherlock run — quick reference

Only 3 things to edit: GPU partition (in `run_container.sbatch`), the HF token
(`~/hf_token.txt`), and the `run:` block of `sherlock/config.yaml`. Paths below
are already mine (`amilcarm` / group `brucemc` / `$GROUP_HOME/amilcar/sensi`).

Machines: **PC** = WSL on my laptop · **login** = `ssh amilcarm@login.sherlock.stanford.edu`
· **compute** = `sh_dev …` or the batch job. Pipeline commands must run on a
compute node, never login.

## Build container — PC, once
```bash
cd /mnt/c/Users/amilcar/Desktop/code/Antigravity/SensiExtractor
# one-time deps: sudo apt install -y apptainer uidmap squashfuse fuse2fs
export APPTAINER_TMPDIR=/var/tmp/apptmp APPTAINER_CACHEDIR=/var/tmp/apptmp/cache
apptainer build sherlock/sensi.sif sherlock/Apptainer.def          # ~5 GB
```

## Configure — PC
```bash
python gui.py                       # set features/model, then close
python sherlock/export_config.py    # writes sherlock/config.yaml
# edit sherlock/config.yaml run:  folders ($SCRATCH/…), analyzer, hf_token_file: ~/hf_token.txt
```

## Upload — PC → Sherlock  (literal path — $GROUP_HOME won't expand over rsync)
```bash
rsync -avP sherlock/sensi.sif \
  amilcarm@login.sherlock.stanford.edu:/home/groups/brucemc/amilcar/sensi/
rsync -avP --exclude env --exclude hf_cache --exclude mfa_data --exclude .git \
  --exclude '*.sif' --exclude '__pycache__' \
  ./ amilcarm@login.sherlock.stanford.edu:/home/groups/brucemc/amilcar/sensi/SensiExtractor/
```

## Weights — login, once (needs internet + Llama-3-approved token)
```bash
printf 'hf_xxx\n' > ~/hf_token.txt && chmod 600 ~/hf_token.txt      # ✎ token
export HF_HOME=$SCRATCH/sensi/hf_cache && mkdir -p $HF_HOME
apptainer exec --bind $SCRATCH:$SCRATCH --env HF_HOME=$HF_HOME \
  --env HF_TOKEN="$(cat ~/hf_token.txt)" --env HF_HUB_DISABLE_XET=1 \
  $GROUP_HOME/amilcar/sensi/sensi.sif \
  bash -c 'hf download gpt2 && hf download meta-llama/Meta-Llama-3-8B'
```

## Run — login (edit partition once, then submit)
```bash
REPO=$GROUP_HOME/amilcar/sensi/SensiExtractor
nano $REPO/sherlock/run_container.sbatch     # ✎ #SBATCH --partition=
sbatch --export=ALL,SIF=$GROUP_HOME/amilcar/sensi/sensi.sif,REPO=$REPO \
  $REPO/sherlock/run_container.sbatch
squeue --me                                  # then: tail -f slurm-<JobID>.out
```

## Collect — PC
```bash
rsync -avP amilcarm@login.sherlock.stanford.edu:/scratch/users/amilcarm/sensi/sample/out/ ./sherlock_out/
```

## Gotchas
- **Compute node only** for anything that runs the pipeline; after `sh_dev`, `ulimit -n $(ulimit -Hn)` (login caps at 256 → `Too many open files`).
- HF: use `hf download` + `HF_HUB_DISABLE_XET=1`; `401` = accept the Llama license first (`hf auth whoami` checks the token).
- Dry-run before the GPU job (compute node, reads `sherlock/config.yaml`):
  ```bash
  sh_dev -c 4 -m 16GB -t 01:00:00 ; ulimit -n "$(ulimit -Hn)"
  apptainer exec --bind $SCRATCH:$SCRATCH --bind $REPO:$REPO \
    --env HF_HOME=$SCRATCH/sensi/hf_cache --env HF_HUB_OFFLINE=1 \
    $GROUP_HOME/amilcar/sensi/sensi.sif python $REPO/sherlock/run_extract.py
  ```
- Bad `config.yaml` → `run_extract.py` prints the errors and exits (fix keys/types, or re-run `export_config.py`).
