---
id: outputs
title: Outputs reference
after_features: true
lede: Every folder and file a run writes, and which feature pages explain their columns.
keywords: outputs files folders csv mat logs manifest
---

## The output folder

```text
<output folder>/
├── word/        <stim>_features.csv            one row per word (and punctuation token)
│                <stim>_word2vec.mat, <stim>_bert_word.mat
│                <stim>_word_events.mat         one entry per spoken word, for stem plots (opt-in)
├── syllable/    <stim>_syllables.csv           one row per syllable
│                <stim>_syllable_events.mat     one entry per syllable, for stem plots (opt-in)
├── phoneme/     <stim>_phonemes.csv            one row per phoneme
│                <stim>_phoneme_surprisal.mat    time series on the analysis grid
├── sentence/    <stim>_sentences.csv           one row per sentence
│                <stim>_bert_sentence.mat
├── document/    documents_summary_<timestamp>.csv   one row per stimulus
├── acoustic/    <stim>_acoustic.mat            audio-only time series
├── Montreal Forced Alignment/                   saved alignments, reused by later runs
└── logs/        word_run_…, doc_run_…, acoustic_run_… (.log, _manifest.json, …)
```

`<stim>` is the file name of the transcript (or recording) without its extension. A folder
appears only when its output level is switched on and at least one feature you ticked
writes into it.

## Which features are in which file

{{outputs_table}}

## Reading the files

- **CSV files** open in Excel, MATLAB (`readtable`), R or pandas. An empty cell means "no
  value": punctuation rows have no frequency or timing, a word missing from a norm table has
  no norm, a transcript without audio has no timing or prosody.
- **`.mat` files** load in MATLAB with `load` (and in Python with `scipy.io.loadmat`). Time
  series share one time grid: `t` (seconds, one row per sample) and `fs` (samples per
  second), so they stack directly as columns of a predictor matrix.
- **Times** (`Onset`, `End`, `t`) are in seconds from the start of the recording.
- **Units** of every column are in `logs/<pass>_session_units_<timestamp>.csv`.

## The run logs

Each run writes into `logs/`:

`<pass>` is `word` (the word, sub-word and sentence files are written in the word pass), `doc`
(the document summary) or `acoustic`.

| File | What it holds |
|---|---|
| `<pass>_run_<timestamp>.log` | every progress message (what the Logs page shows) |
| `<pass>_run_<timestamp>_manifest.json` | inputs, settings, software versions, analyzer name, per-stimulus results and any problems |
| `<pass>_session_units_<timestamp>.csv` | each output column with its unit |
| `<pass>_run_<timestamp>_no-match.csv` | transcripts or recordings that found no partner |

Keep the manifest with your outputs: it records exactly how they were made.
