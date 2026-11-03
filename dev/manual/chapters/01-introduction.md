---
id: introduction
title: Introduction
lede: SENSI SpeechFeatures turns a spoken stimulus — an audio recording and its transcript — into time-aligned tables of linguistic and acoustic features, ready to relate to EEG.
keywords: overview what is sensi speechfeatures sensiextractor levels
---

## What SENSI SpeechFeatures is for

Naturalistic EEG studies play people real speech — stories, lectures, conversations — and
ask how the brain follows it. To analyse that, every stimulus first has to be described in
numbers: *when* each word was spoken, how frequent or predictable it is, what part of speech
it is, how high and loud the voice was, how the sound energy moved over time. SENSI SpeechFeatures
computes those descriptions from a recording and its transcript.

You give it a folder of transcripts (`.txt`) and, usually, the matching recordings (`.wav`,
paired by file name). It writes, for each stimulus, CSV tables and MATLAB `.mat` files, plus
one summary table for the whole set. Every time-based value is on the recording's own clock,
so it lines up with your EEG events.

## How it works

<div class="card">
<div class="flow">
  <div class="flow-col"><div class="flow-box in">Transcript<br><small>story1.txt</small></div><div class="flow-box in">Recording<br><small>story1.wav</small></div></div>
  <div class="flow-arrow">→</div>
  <div class="flow-col">
    <div class="flow-box">spaCy<br><small>words, sentences, parts of speech, dependencies</small></div>
    <div class="flow-box">Montreal Forced Aligner<br><small>when each word, syllable and sound was spoken</small></div>
    <div class="flow-box">Praat (Parselmouth)<br><small>pitch, loudness, voice quality</small></div>
    <div class="flow-box">Language models &amp; lexicons<br><small>GPT-2 / Llama, Word2Vec, BERT, word norms</small></div>
    <div class="flow-box">Audio signal processing<br><small>envelope, spectrogram (librosa)</small></div>
  </div>
  <div class="flow-arrow">→</div>
  <div class="flow-col">
    <div class="flow-box out">Word level<br><small>word/</small></div>
    <div class="flow-box out">Sub-word level<br><small>syllable/ · phoneme/</small></div>
    <div class="flow-box out">Sentence level<br><small>sentence/</small></div>
    <div class="flow-box out">Document level<br><small>document/</small></div>
    <div class="flow-box out">Acoustic level<br><small>acoustic/</small></div>
  </div>
</div>
</div>

1. The transcript is cleaned (numbers spelled out, spacing normalized) and parsed by spaCy
   into tokens, sentences and grammatical annotations.
2. If there is a recording, the Montreal Forced Aligner finds when each word and each speech
   sound (phone) was spoken; syllables are then built from the phones. This runs **once per stimulus** and is saved, so
   later runs reuse it.
3. Each feature you switched on is computed from those pieces and written into the table for
   its level. Features you leave off are not computed and add no columns (a few identifying
   columns, such as the token and its ids, are always written).

## The five output levels

| Level | One row per | Typical use | Chapter |
|---|---|---|---|
| Word | word (and punctuation token) | word-locked ERPs, TRF predictors at word onsets | [Word level]({{root}}features/word/index.html) |
| Sub-word | syllable, phoneme | syllable / phoneme-rate tracking, cohort models | [Sub-word level]({{root}}features/subword/index.html) |
| Sentence | sentence | sentence-level prosody, sentence embeddings | [Sentence level]({{root}}features/sentence/index.html) |
| Document | stimulus | comparing or balancing stimuli | [Document level]({{root}}features/document/index.html) |
| Acoustic | time step (e.g. 128 Hz) | continuous TRF predictors from the audio alone | [Acoustic level]({{root}}features/acoustic/index.html) |

## How to use this manual

- New to the tool: read [Requirements]({{root}}chapters/requirements.html),
  [Installation]({{root}}chapters/installation.html) and
  [Using the app]({{root}}chapters/usage.html), in that order.
- Looking up a feature or a column name: type it in the search box (top left, or press
  <kbd>/</kbd>), or open [Features]({{root}}features/index.html).
- Running large batches on Stanford's cluster: [Running on Sherlock]({{root}}chapters/sherlock.html).
- Looking for a file in your output folder: [Outputs reference]({{root}}chapters/outputs.html).

<div class="callout">
<b>Where the examples come from.</b> Every figure, table and number in this manual is real
SENSI SpeechFeatures output, computed when the manual was built. The recordings are from the
<a href="https://keithito.com/LJ-Speech-Dataset/">LJ Speech dataset</a> (public domain, a single
speaker reading non-fiction). Where a feature needs a particular kind of text (a sentence whose
words range from common to rare, paragraphs in contrasting styles…), short invented texts are run
without audio; each page shows the text it used. The
numbers are still the program's real output. No study stimuli are included.
</div>

The text of each feature page is the same text the app shows when you click a feature's
ⓘ icon, so the manual and the app always say the same thing.
