# Embeddings Batch — Design Spec

**Date:** 2026-05-22  
**Status:** Design finalized; plan at `docs/superpowers/plans/2026-05-22-embeddings.md`

---

## 1. Outputs

All outputs are `.mat` files (scipy.io.savemat) — not CSVs. EEG / MATLAB workflows need dense matrices; 300- or 768-dim vectors make CSVs unwieldy.

| File | Shape | Key in .mat |
|------|-------|-------------|
| `word/<stim>_word2vec.mat` | (n_words, 300) float32 | `embeddings` |
| `word/<stim>_bert_word.mat` | (n_words, 12, 768) float32 | `embeddings` |
| `sentence/<stim>_bert_sentence.mat` | (n_sentences, 12, 768) float32 | `embeddings` |

n_words = rows in `<stim>_features.csv`. n_sentences = unique `sentence_id` values.  
Row i in `_word2vec.mat` and `_bert_word.mat` corresponds to row i in the features CSV.

## 2. Models

### Word2Vec
- **Model:** Google News 300-dimensional pre-trained vectors (Mikolov 2013)
- **Format:** gensim KeyedVectors binary (`.bin`)
- **Expected path:** `hf_cache/word2vec/GoogleNews-vectors-negative300.bin` (~3.6 GB)
- **OOV words:** zero vector (row of zeros)
- **Lowercase fallback:** try `token.lower()` before zero-filling

### BERT
- **Model:** `bert-base-uncased` from HuggingFace Transformers
- **Layers:** all 12 transformer layers (hidden_states[1:13], not the embedding layer)
- **Word embedding:** mean-pool subword tokens → one (12, 768) vector per word
- **Sentence embedding:** mean-pool word embeddings for non-PUNCT words in sentence → one (12, 768) vector per sentence
- **Token limit fix:** left-context sentence-complete sliding window
  - Each sentence gets as much preceding context as fits in 510 effective tokens (2 reserved for [CLS]/[SEP])
  - Context is added by prepending complete preceding sentences (never cuts mid-sentence)
  - Tokenization via `BertTokenizerFast` with `is_split_into_words=True`

## 3. Prefs: v9 → v10 (additive, or v8 → v10 if sentence-level ships first)

New `output` keys: `word2vec_mat`, `bert_word_mat`, `bert_sentence_mat` (all default True)  
New `features` keys: `master_word2vec`, `master_bert_word`, `master_bert_sentence` (master toggles)

## 4. Planner

New stages: `word2vec_embed`, `bert_embed` (shared forward pass for word+sentence)

New clusters: `word2vec`, `bert word embeddings`, `bert sentence embeddings`

`_OUTPUT_CSV_MAP` gains three new entries mapping mat prefs keys → identifiers  
`_WORD_LEVEL_CSVS` gains all three embedding identifiers

## 5. GUI

New sidebar page "Embeddings" (grouped under "Semantic Similarity" label).  
Three toggles: Word2Vec, BERT word-level, BERT sentence-level.

## 6. Requirements

- `gensim>=4.3` — add to `requirements.txt`
- `scipy`, `transformers`, `torch` — already present
- Word2Vec model download is user responsibility; `download_models.py` documents the URL
