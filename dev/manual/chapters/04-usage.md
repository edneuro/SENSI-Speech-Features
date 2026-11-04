---
id: usage
title: Using the app
lede: The app's main sections, the two ways to run it (the app or the run.py script), the HuggingFace token, then each page in turn.
keywords: gui usage inputs run logs features.yaml run.yaml headless token reset settings
---

## The main sections

The app is one window with eight pages in a sidebar on the left. You go through them top to
bottom:

| Page | What you do there |
|---|---|
| **Inputs** | give your name (recorded with every run) and pick the transcripts, audio and output folders |
| **Word-level features** | choose the per-word columns: ids, parts of speech, frequency, predictability, embeddings, word prosody… |
| **Document-level features** | choose the one-row-per-stimulus summary: counts, averages, readability, document prosody… |
| **Sub-word features** | choose the syllable and phoneme tables |
| **Sentence features** | choose the per-sentence table |
| **Acoustic** | choose the audio-only time series (envelope, spectrogram, MFCC) |
| **Run** | switch on the output levels to write, check the size / memory / time estimates and press **Run extraction** |
| **Logs** | watch progress live; reopen past runs |

The bar along the bottom of the window shows whether a run is in progress, how far it has
got (file *n* of *N*) and a **Cancel** button.

## Two ways to run it

<div class="grid-cards">
<div class="card"><div class="card-title">With the app (GUI)</div>
<p>Start the app, set things up on its pages, press <b>Run extraction</b>. Best for choosing
features and for small and medium batches. Described in the rest of this chapter.</p></div>
<div class="card"><div class="card-title">With a script (headless)</div>
<p><code>python run.py</code> runs exactly what the Run button would, with no window. Best for
long batches, servers and <a href="{{root}}chapters/sherlock.html">Sherlock</a>. See
<a href="#headless-runs-run-py">Headless runs</a>.</p></div>
</div>

Both read the same two plain-text settings files in `config/`, which the app writes as you
click:

- **`features.yaml`** — *what* to extract (every switch, tick and setting on the feature
  pages). The same on every computer, so you can share it with a colleague or copy it to
  Sherlock.
- **`run.yaml`** — *where*: your name, the three folders, the token file. Specific to this
  computer.

So you can set everything up once in the app and then run it headless, or skip the app and
edit the two files directly. Close the app and everything is still there next time.

## HuggingFace token

You need a token **only** to use the Llama language models for predictability
(surprisal, cloze, top-match). GPT-2, the default, and every other feature need none. The
token proves to HuggingFace (where the model files are downloaded from) that Meta has
approved your request to use Llama; you get it once and the app remembers it.

{{doc:llm_credentials}}

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
  example above, the invented text-only examples have no recording and one recording has no
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

{{yaml:config/run_defaults.yaml|shared}}

Set `analyzer_name` and the three folders. Folder paths can be absolute (recommended) or
relative — a relative path is taken from the folder that holds `run.yaml`.

### `features.yaml` — what

The file has one block per part of the app. The output levels:

{{yaml:config/features_defaults.yaml|output|5}}

Feature switches are `true` / `false` under `features:` (each feature page in this manual
lists the keys it uses). Settings that aren't switches have their own blocks, for example:

{{yaml:config/features_defaults.yaml|prosody}}

{{yaml:config/features_defaults.yaml|acoustic_ts}}

Predictability is chosen by model and metric:

```yaml
predictability:
  models: [gpt2]           # gpt2 (also gpt2_medium, gpt2_large, gpt2_xl), or the gated Llama models (llama32_3b, llama3_8b, … — need a token)
  metrics:
    surprisal: true
    cloze: true
    topmatch: false
```

`run.py` checks both files before doing any work and stops with a plain message for an
unknown key, a wrong type, an out-of-range value (such as a pitch floor above the ceiling) or
a missing folder.
