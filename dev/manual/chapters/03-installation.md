---
id: installation
title: Installation
lede: Three steps — install Miniconda, run the setup script once, launch the app.
keywords: install setup miniconda conda env setup.bat setup.sh launcher update
---

## Step 1 — Install Miniconda

Download the installer for your system from
[anaconda.com/docs/getting-started/installation](https://www.anaconda.com/docs/getting-started/installation).

<div data-tabs class="os-tabs">
<div class="tabs" role="tablist"><button type="button" class="on">Windows</button><button type="button">macOS (Apple Silicon)</button><button type="button">Linux</button></div>
<div class="tabpanel on">

Run the `.exe` with the default options. Use the **Anaconda Prompt** (in your Start menu) for
every command below — it already has `conda` set up.

</div>
<div class="tabpanel">

Run the Apple-Silicon `.pkg` (or `.sh`) installer, then set conda up for your shell and
**open a new Terminal window**:

```bash
conda init zsh          # zsh is the macOS default
```

</div>
<div class="tabpanel">

Run the `.sh` installer, then set conda up for your shell and open a new terminal:

```bash
conda init bash         # or zsh / fish
```

</div>
</div>

<div class="callout warn">
<b>"Run 'conda init' before 'conda activate'"?</b> The init step hasn't been loaded. Run
<code>conda init &lt;your-shell&gt;</code>, open a new terminal, and try again.
</div>

## Step 2 — Get the project and run the setup script

Download the project from
[github.com/edneuro/SENSI-Speech-Features](https://github.com/edneuro/SENSI-Speech-Features)
(green **Code** button → *Download ZIP*, or `git clone`). Then, from inside the project
folder:

<div data-tabs class="os-tabs">
<div class="tabs" role="tablist"><button type="button" class="on">Windows</button><button type="button">macOS / Linux</button></div>
<div class="tabpanel on">

In the Anaconda Prompt:

```bat
cd path\to\SENSI-Speech-Features
setup.bat
```

</div>
<div class="tabpanel">

```bash
cd path/to/SENSI-Speech-Features
bash setup.sh
```

</div>
</div>

The script does five things, and prints each as it goes:

<ol class="steps">
<li><b>Installs conda-lock</b> into your base conda — the tool that installs exact, pinned versions.</li>
<li><b>Creates <code>./env</code></b> inside the project folder from <code>conda-lock.yml</code>, a lock file that pins every package for your platform. Everyone gets the same versions.</li>
<li><b>Downloads the forced-aligner models</b> into <code>data/mfa_data/</code>.</li>
<li><b>Downloads the spaCy English model</b> into the environment.</li>
<li><b>Creates a double-click launcher</b> for your machine (with the app icon).</li>
</ol>

It takes a while (the environment is about 6 GB). You only do it once.

## Step 3 — Launch

Double-click the **SENSI-SpeechFeatures** launcher the setup created in the project folder:

| System | Launcher | Notes |
|---|---|---|
| Windows | **SENSI-SpeechFeatures** shortcut (runs `runGUI.bat`) | a console window opens next to the app with live progress — that's intentional |
| macOS | **SENSI-SpeechFeatures.app** | no Terminal window; the log goes to `~/Library/Logs/SENSI-SpeechFeatures.log`. For live logs, double-click `runGUI.command` instead |
| Linux | **SENSI-SpeechFeatures** desktop entry (runs `runGUI.sh`) | |

Or from a terminal in the project folder:

```bash
conda activate ./env
python gui.py
```

## Updating to a newer version

From the project folder, `git pull` (or download the new ZIP over the old folder), then
start the app as usual. If the dependencies changed, run the setup script again — it
rebuilds `env/` from the new lock file.
