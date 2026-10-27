---
id: usage
title: Using the app
lede: The app is eight pages in a sidebar — choose your files, tick the features you want, run. The same settings also drive a headless run with run.py.
keywords: gui usage inputs run logs features.yaml run.yaml headless token reset settings
---

## The workflow at a glance

<ol class="steps">
<li><b>Inputs</b> — say who is running it, and pick the transcripts, audio and output folders.</li>
<li><b>Feature pages</b> (Word, Document, Sub-word, Sentence, Acoustic) — switch on the sections you want and tick their columns.</li>
<li><b>Run</b> — switch on the output levels to write, check the size / memory / time estimates and readiness, press <b>Run extraction</b>.</li>
<li><b>Logs</b> — watch progress live; reopen past runs.</li>
</ol>

Your choices are saved as you make them, in two plain-text files in `config/`:
`features.yaml` (*what* to extract — the same on every computer, safe to share) and
`run.yaml` (*where*: folders, your name — specific to this computer). Close the app and
everything is still there next time.

## Inputs

{{screen:inputs|The Inputs page with the manual's example files}}

- **Analyzer** — your name; it is recorded in every run's log and manifest, so you can tell
  later who produced a set of outputs.
- **Transcripts folder** — one plain-text file per stimulus (`.txt`; `.md` and `.tsv` work too; numbers are spelled out
  for you).
- **Audio folder** — the matching recordings (`.wav`, `.flac` or `.mp3`). A transcript and a
  recording are paired when they have the **same file name** (`story1.txt` ↔ `story1.wav`).
  Optional: without audio you still get every text feature; timing, prosody and the
  sub-word levels need it.
- **Output folder** — where results go (created if missing).
- **Pairing preview** — shows which files paired and which didn't, before you run. In the
  example above, two text-only sentences have no recording and one recording has no
  transcript (it is used only by the audio-only Acoustic level).

## Feature pages

{{screen:word|The Word-level features page}}

Each page groups its features in **cards** — the same grouping as the
[Features]({{root}}features/index.html) chapters of this manual. On every card:

- The **switch** next to the title includes or leaves out the whole section. Switched off,
  the card greys out but keeps your ticks for when you switch it back on.
- The **ticks** choose individual columns; **All · None** set them all at once.
- The **ⓘ** icon opens the feature's explanation — the same text as its page in this manual.
- Some cards have settings (pitch range, pitch units, model choice, analysis rate…); each is
  explained on the feature's page.

The other feature pages work the same way:

<div data-tabs>
<div class="tabs" role="tablist"><button type="button" class="on">Document</button><button type="button">Sub-word</button><button type="button">Sentence</button><button type="button">Acoustic</button></div>
<div class="tabpanel on">{{screen:document|Document-level features}}</div>
<div class="tabpanel">{{screen:subword|Sub-word features}}</div>
<div class="tabpanel">{{screen:sentence|Sentence features}}</div>
<div class="tabpanel">{{screen:acoustic|Acoustic features}}</div>
</div>

**Reset settings** (top right) puts every feature choice back to the shipped defaults — a
minimal run with only the structural word columns. It rewrites `features.yaml` only; your
folders and name in `run.yaml` are kept.

## Run

{{screen:run|The Run page: output levels, estimates, readiness}}

- **Output levels** — one switch per level. Each lists the files it will write for every
  stimulus; a file is crossed out when nothing you ticked would go into it. A level writes
  only when its switch is on *and* at least one ticked feature produces that file.
- **Estimates** — before you start, the page shows the **disk space** the outputs will take
  (per level and per file), the **peak memory** compared with this computer's (with a
  warning if it may not fit), and an **expected time** range. The time starts as a wide first
  estimate and narrows as the app learns how fast *this* computer is; stimuli that already
  have a saved alignment in the output folder aren't charged for alignment again.
- **Readiness** — the run button is enabled only when everything needed is in place; the
  line above it says what's missing (a folder, a token for a gated model, …).

Alignment runs once per stimulus and is saved in the output folder
(`Montreal Forced Alignment/`); the next run on unchanged files reuses it.

## Logs

{{screen:logs|The Logs page}}

**Current run** shows the progress messages live (with a search box and **Copy**); **Past
runs** lists earlier runs in this output folder. Each run also leaves files in
`<output>/logs/`: the log, a JSON manifest (inputs, settings, versions, per-stimulus results
and any problems), and a table of the units of every column.

## Headless runs: `run.py`

`python run.py` does exactly what the **Run** button does, using the saved
`config/features.yaml` and `config/run.yaml`. Use it for long batches, on a server, or on
Sherlock.

```bash
conda activate ./env
python run.py                       # uses config/
python run.py --config-dir path/to/other/config
```

The easiest way to make the two files is to set everything up in the app once. Without the
app, copy the templates and edit them:

```bash
cp config/features_defaults.yaml config/features.yaml
cp config/run_defaults.yaml config/run.yaml
```

### `run.yaml` — where

{{file:config/run_defaults.yaml|12-19}}

Set `analyzer_name` and the three folders. Folder paths can be absolute (recommended) or
relative — a relative path is taken from the folder that holds `run.yaml`.

### `features.yaml` — what

The file has one block per part of the app. The output levels:

{{file:config/features_defaults.yaml|184-189}}

Feature switches are `true` / `false` under `features:` (each feature page in this manual
lists the keys it uses). Settings that aren't switches have their own blocks, for example:

{{file:config/features_defaults.yaml|210-215}}

{{file:config/features_defaults.yaml|223-231}}

Predictability is chosen by model and metric:

```yaml
predictability:
  models: [gpt2]           # gpt2, or the gated llama3_* models (need a token)
  metrics:
    surprisal: true
    cloze: true
    topmatch: false
```

`run.py` checks both files before doing any work and stops with a plain message for an
unknown key, a wrong type, an out-of-range value (such as a pitch floor above the ceiling) or
a missing folder.

## HuggingFace token

{{doc:llm_credentials}}
