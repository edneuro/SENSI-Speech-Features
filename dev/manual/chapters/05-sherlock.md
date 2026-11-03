---
id: sherlock
title: Running on Sherlock
lede: Large batches and the bigger language models run on Stanford's Sherlock cluster, inside a container, driven by the same features.yaml you set up in the app.
keywords: sherlock hpc cluster apptainer container slurm sbatch gpu
---

## The idea

The app's environment is built for one operating system, and Sherlock's compute nodes are
headless Linux machines, so SENSI SpeechFeatures runs there inside a portable **Apptainer
container** (`sensiextractor.sif`) that carries the whole environment. The flow:

<ol class="steps">
<li><b>On your computer</b>, set up the features in the app as usual — that writes <code>config/features.yaml</code>.</li>
<li><b>Build the container</b> once (Linux, or WSL on Windows) and upload it with the code and your stimuli.</li>
<li><b>On Sherlock</b>, write a <code>run.yaml</code> with the cluster's folders, download the model weights once, and submit the job.</li>
<li>The job runs <code>run.py</code> in the container — the same headless runner as on your computer — and the outputs land in your scratch folder.</li>
</ol>

Everything below is the project's Sherlock runbook (`sherlock/RUNBOOK.md`), included here
as-is so the two never disagree.

{{include:sherlock/RUNBOOK.md}}
