"""BERT contextual embedding extractor (bert-base-uncased, all 12 layers).

Word-level:     (n_words, 12, 768) float32
                mean of subword token embeddings per word, per layer
Sentence-level: (n_sentences, 12, 768) float32
                mean of non-PUNCT word embeddings per sentence, per layer

Sliding window: left-context sentence-complete.
  - Each sentence gets as much preceding context as fits in 510 effective
    tokens (2 reserved for [CLS]/[SEP]).
  - Context is added by prepending complete preceding sentences (never
    cuts mid-sentence).

Requires: transformers, torch (already in requirements.txt).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_NAME = "bert-base-uncased"
_MAX_EFFECTIVE_TOKENS = 510   # 512 - 2 for [CLS] and [SEP]


def _load_bert(model_name: str):
    from transformers import BertTokenizerFast, BertModel  # type: ignore
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name, output_hidden_states=True)
    model.eval()
    return tokenizer, model


def _sentence_groups(word_df: pd.DataFrame) -> list[dict]:
    """Return list of {doc_id, sent_id, tokens: list[str], indices: list[int], upos: list[str]}"""
    groups = []
    for (doc_id, sent_id), grp in word_df.groupby(
            ["doc_id", "sentence_id"], sort=False):
        groups.append({
            "doc_id":  doc_id,
            "sent_id": sent_id,
            "tokens":  grp["token"].tolist(),
            "indices": grp.index.tolist(),
            "upos":    grp["upos"].tolist() if "upos" in grp.columns else ["NOUN"] * len(grp),
        })
    return groups


def _count_tokens(tokenizer, words: list[str]) -> int:
    enc = tokenizer(words, is_split_into_words=True, add_special_tokens=False)
    return len(enc["input_ids"])


def embed_bert_words_and_sentences(
    word_df: pd.DataFrame,
    *,
    model_name: str = _DEFAULT_MODEL_NAME,
    tokenizer=None,   # injectable for tests
    model=None,       # injectable for tests
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (word_embeddings, word_valid, sentence_embeddings, sentence_valid).

    word_embeddings:     (n_words, n_layers, hidden_size) float32
    word_valid:          (n_words,) bool — True iff a non-PUNCT word was embedded
    sentence_embeddings: (n_sentences, n_layers, hidden_size) float32
    sentence_valid:      (n_sentences,) bool — True iff the sentence had >=1 embedded content word

    n_layers = 12 for bert-base-uncased (2 for the test fixture).

    Mask names are level-qualified (word_valid, sent_valid) because this
    extractor returns two levels; the single-level Word2Vec extractor uses
    `is_valid`.
    """
    if tokenizer is None or model is None:
        tokenizer, model = _load_bert(model_name)

    n_layers = model.config.num_hidden_layers
    hidden   = model.config.hidden_size

    groups = _sentence_groups(word_df)
    n_words     = len(word_df)
    n_sentences = len(groups)

    word_emb = np.zeros((n_words, n_layers, hidden), dtype=np.float32)
    sent_emb = np.zeros((n_sentences, n_layers, hidden), dtype=np.float32)
    word_valid = np.zeros(n_words, dtype=bool)
    sent_valid = np.zeros(n_sentences, dtype=bool)

    idx_to_pos = {idx: i for i, idx in enumerate(word_df.index.tolist())}

    for s_pos, s in enumerate(groups):
        # PUNCT tokens are excluded from BERT input so their presence
        # does not alter attention-derived representations of content words.
        content_indices = [
            (orig_idx, tok)
            for orig_idx, tok, upos in zip(s["indices"], s["tokens"], s["upos"])
            if upos != "PUNCT"
        ]
        punct_indices = [
            orig_idx
            for orig_idx, upos in zip(s["indices"], s["upos"])
            if upos == "PUNCT"
        ]
        current_tokens = [tok for _, tok in content_indices]
        n_current      = _count_tokens(tokenizer, current_tokens) if current_tokens else 0

        # A single sentence longer than the window can't be given any left
        # context, and the tokenizer truncates its own tail — those tail
        # words end up zero-vectored with word_valid=False. Warn so the
        # silent data loss is visible (name the sentence for locating it).
        if n_current > _MAX_EFFECTIVE_TOKENS:
            sent_preview = str(s["tokens"][0] if s["tokens"] else "")[:40]
            logger.warning(
                "BERT: sentence (doc_id=%s, sentence_id=%s, starts %r) has %d "
                "subwords, exceeding the %d-token window — it is truncated and "
                "its tail words get no embedding (word_valid=False).",
                s["doc_id"], s["sent_id"], sent_preview, n_current,
                _MAX_EFFECTIVE_TOKENS)

        context_token_lists: list[list[str]] = []
        remaining = _MAX_EFFECTIVE_TOKENS - n_current
        for prev_s in reversed(groups[:s_pos]):
            prev_content = [
                tok for tok, upos in zip(prev_s["tokens"], prev_s["upos"])
                if upos != "PUNCT"
            ]
            if not prev_content:
                continue
            n_prev = _count_tokens(tokenizer, prev_content)
            if remaining - n_prev < 0:
                break
            context_token_lists.insert(0, prev_content)
            remaining -= n_prev

        context_words = [tok for tlist in context_token_lists for tok in tlist]
        n_context_words = len(context_words)
        window_words = context_words + current_tokens

        if not window_words:
            continue

        enc = tokenizer(
            window_words,
            is_split_into_words=True,
            return_tensors="pt",
            max_length=512,
            truncation=True,
        )
        word_ids = enc.word_ids()

        with torch.no_grad():
            out = model(**enc)

        # hidden_states[0] is embedding layer; [1:] are the n_layers transformer layers
        stacked = torch.stack(out.hidden_states[1:], dim=0)   # (n_layers, 1, seq_len, hidden)
        stacked = stacked.squeeze(1).permute(1, 0, 2).numpy()  # (seq_len, n_layers, hidden)

        content_positions: list[int] = []
        for w_local, (orig_idx, _tok) in enumerate(content_indices):
            w_global = n_context_words + w_local
            positions = [p for p, wid in enumerate(word_ids) if wid == w_global]
            if not positions:
                continue
            word_vec = stacked[positions].mean(axis=0)
            word_emb[idx_to_pos[orig_idx]] = word_vec
            word_valid[idx_to_pos[orig_idx]] = True
            content_positions.extend(positions)

        if content_positions:
            sent_emb[s_pos] = stacked[content_positions].mean(axis=0)
            sent_valid[s_pos] = True

    return word_emb, word_valid, sent_emb, sent_valid
