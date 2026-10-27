---
id: requirements
title: Requirements
lede: What your computer needs before you install SensiExtractor.
keywords: requirements platform windows mac linux disk memory gpu miniconda
---

## Operating system

| Supported | Not supported |
|---|---|
| Windows 11 or later | Intel (x86) Macs |
| Linux (64-bit) | |
| macOS on Apple Silicon (M1/M2/M3/M4) | |

Intel Macs are not supported because PyTorch stopped publishing Intel-macOS builds after
version 2.2.2, which cannot run with the NumPy 2 this project uses.

## Software

- **Miniconda** (a small installer for the conda package manager). Everything else is
  installed by the setup script into the project folder. A plain Python `virtualenv` will
  **not** work: the forced aligner needs native Kaldi libraries that come from conda.
- An internet connection the first time, to download the environment and the models.

## Disk space

About **10 GB** free:

| What | Size | When |
|---|---|---|
| The environment (`env/`: PyTorch, spaCy, Kaldi/MFA, the GUI…) | ~6 GB | setup |
| Forced-aligner models (`data/mfa_data/`) | ~0.5 GB | setup |
| GPT-2 / BERT weights (`data/hf_cache/`) | ~0.9 GB | first use |
| Word2Vec Google News vectors (`data/gensim_data/`) | ~1.7 GB | first use of Word2Vec |

All models stay inside the project's `data/` folder, never in your home directory. The
outputs of a run are usually small (a few MB per stimulus; BERT embedding files are the
largest). The app's **Run** page shows the expected size of every output before you start.

## Memory and GPU

- Text features, alignment, prosody and the acoustic level run comfortably on an ordinary
  laptop.
- Language models add memory: BERT and GPT-2 a few GB; the gated Llama-3 models a lot more,
  and they are practical only with a CUDA GPU.
- You don't have to work this out yourself: the **Run** page shows the peak memory the
  selected features need next to this computer's memory (and GPU memory, when a model would
  run on the GPU), and warns you when it may not fit.

## Optional: a HuggingFace token

Only the Llama-3 predictability models need one (they are free but gated: you accept Meta's
licence on HuggingFace). GPT-2, BERT and everything else work without a token. See
[the token section of Using the app]({{root}}chapters/usage.html#huggingface-token).
