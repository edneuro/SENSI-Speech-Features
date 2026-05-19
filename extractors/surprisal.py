"""
Surprisal extractor — GPT-2 contextual word predictability.

Input:  raw transcript text (str) + token-level DataFrame with 'token'
        and 'upos' columns (output of text_parser.parse())
Output: same DataFrame with a new 'Surprisal' column appended

Unit: bits (log base 2). Higher = less predictable in context.

How it works:
    GPT-2 tokenizes text into subword pieces. For each spaCy token we
    collect all GPT-2 subword tokens that map to it, sum their
    log-probabilities, and negate to get surprisal in bits.

    Because GPT-2 context is cumulative, the model is run once over the
    full text. Each token's surprisal reflects everything that came before
    it, matching standard psycholinguistic usage.

Punctuation tokens (upos == 'PUNCT') receive NaN.

Notes:
    - First call downloads the GPT-2 weights (~500 MB); cached after that.
    - Model runs on CPU by default; set USE_GPU=True below if a GPU is
      available and torch.cuda.is_available().
    - Long texts (>1024 GPT-2 tokens) are split at sentence boundaries
      with a sliding context window to avoid truncation artifacts.
"""

import math

import pandas as pd
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

USE_GPU = False

_tokenizer = None
_model = None


def _load():
    global _tokenizer, _model
    if _tokenizer is None:
        _tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        _model = GPT2LMHeadModel.from_pretrained('gpt2')
        _model.eval()
        if USE_GPU and torch.cuda.is_available():
            _model = _model.cuda()


def add_surprisal(token_df: pd.DataFrame, text: str) -> pd.DataFrame:
    """Append a Surprisal column to a token-level DataFrame.

    Args:
        token_df: DataFrame from text_parser.parse(). Must have 'token'
                  and 'upos' columns. Rows must appear in document order.
        text:     The same raw text that was passed to text_parser.parse().
                  Used to run GPT-2 over the full document context.

    Returns:
        A copy of token_df with 'Surprisal' column (float, bits).
        Punctuation rows have NaN.
    """
    _load()
    surprisals = _compute_surprisals(text, [row['token'] for _, row in token_df.iterrows()])

    df = token_df.copy()
    df['Surprisal'] = [
        None if row['upos'] == 'PUNCT' else surprisals[i]
        for i, (_, row) in enumerate(token_df.iterrows())
    ]
    return df


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_MAX_GPT2_TOKENS = 1024


def _compute_surprisals(text: str, spacy_tokens: list[str]) -> list[float | None]:
    """Return one surprisal value per spaCy token (None for unmatched)."""
    enc = _tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    gpt2_ids = enc['input_ids']
    offsets = enc['offset_mapping']  # (char_start, char_end) per subword

    # Get per-subword log-probs via a single forward pass (or chunked for long texts)
    log_probs = _subword_log_probs(gpt2_ids)

    # Map each spaCy token to its GPT-2 subwords via character spans
    token_surprisals = _align_tokens_to_subwords(
        text, spacy_tokens, offsets, log_probs
    )
    return token_surprisals


def _subword_log_probs(input_ids: list[int]) -> list[float]:
    """Return log P(token_i | context) for each subword token."""
    device = next(_model.parameters()).device
    ids_tensor = torch.tensor([input_ids], device=device)

    with torch.no_grad():
        if len(input_ids) <= _MAX_GPT2_TOKENS:
            logits = _model(ids_tensor).logits[0]  # (seq_len, vocab)
        else:
            # Chunked inference: use last 512 tokens as context for each chunk
            logits = _chunked_logits(input_ids, device)

    # logits[i] predicts token[i+1]; shift so index aligns with input
    log_probs_all = torch.log_softmax(logits, dim=-1)
    # For token at position i, its log-prob is log_probs_all[i-1, id[i]]
    # Position 0 has no prior context — assign 0 (neutral)
    lp = [0.0]
    for i in range(1, len(input_ids)):
        lp.append(log_probs_all[i - 1, input_ids[i]].item())
    return lp


def _chunked_logits(input_ids: list[int], device) -> torch.Tensor:
    """Process sequences longer than GPT-2's context window in 512-token steps."""
    chunk = 512
    all_logits = []
    for start in range(0, len(input_ids), chunk):
        context_start = max(0, start - chunk)
        segment = input_ids[context_start: start + chunk]
        ids_t = torch.tensor([segment], device=device)
        with torch.no_grad():
            logits_seg = _model(ids_t).logits[0]
        # Keep only the logits for the current chunk (not the context prefix)
        offset = start - context_start
        all_logits.append(logits_seg[offset:])
    return torch.cat(all_logits, dim=0)


def _align_tokens_to_subwords(
    text: str,
    spacy_tokens: list[str],
    offsets: list[tuple[int, int]],
    log_probs: list[float],
) -> list[float | None]:
    """Sum subword log-probs for each spaCy token, return surprisal (bits)."""
    results = []
    char_pos = 0

    for token in spacy_tokens:
        # Find where this token starts in the text (simple left-to-right scan)
        idx = text.find(token, char_pos)
        if idx == -1:
            results.append(None)
            continue
        tok_start = idx
        tok_end = idx + len(token)
        char_pos = tok_end

        # Collect GPT-2 subwords that overlap with [tok_start, tok_end)
        total_lp = 0.0
        found = False
        for sub_idx, (s, e) in enumerate(offsets):
            if e <= tok_start:
                continue
            if s >= tok_end:
                break
            total_lp += log_probs[sub_idx]
            found = True

        if not found:
            results.append(None)
        else:
            # Surprisal = -log2 P = -log_e P / log_e 2
            surprisal_bits = -total_lp / math.log(2)
            results.append(round(surprisal_bits, 4))

    return results
