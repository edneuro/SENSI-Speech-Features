"""
Surprisal extractor — GPT-2 contextual word predictability.

Input:  raw transcript text (str) + token-level DataFrame with 'token'
        and 'upos' columns (output of text_parser.parse())
Output: same DataFrame with a new 'Surprisal' column appended

Unit: bits (log base 2). Higher = less predictable in context.

How it works:
    GPT-2 tokenizes text into subword pieces. For each spaCy token we
    collect the GPT-2 subword tokens that map to it, sum their
    log-probabilities, and negate to get surprisal in bits. Each subword
    is charged to exactly one spaCy token (the first it overlaps), so a
    subword straddling a token boundary is never double-counted.

    Because GPT-2 context is cumulative, the model is run once over the
    full text. Each token's surprisal reflects everything that came before
    it, matching standard psycholinguistic usage.

Punctuation tokens (upos == 'PUNCT') receive NaN. The document's FIRST
word also receives NaN: its first subword has no preceding context, so
GPT-2 assigns it no probability — emitting 0 bits there would fake a
"perfectly predictable" first word.

Results are memoized per (text, token sequence): when the word-level and
document-level batches process the same stimulus in one run, GPT-2 runs
once and the second pass is served from cache (without even loading the
model).

Notes:
    - First call downloads the GPT-2 weights (~500 MB); cached after that.
    - Model runs on CPU by default; set USE_GPU=True below if a GPU is
      available and torch.cuda.is_available().
    - Long texts (>1024 GPT-2 tokens) are split with a sliding context
      window to avoid truncation artifacts.
"""

import hashlib
import math
from collections import OrderedDict

import pandas as pd
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

USE_GPU = False

_tokenizer = None
_model = None

# (text, tokens) → per-token surprisal list. Sized to comfortably hold one
# GUI dispatch (word pass + doc pass over the same stimuli); FIFO-evicted.
_SURPRISAL_CACHE: OrderedDict[str, list] = OrderedDict()
_CACHE_MAX_ENTRIES = 256


def _load():
    global _tokenizer, _model
    if _tokenizer is None:
        _tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        _model = GPT2LMHeadModel.from_pretrained('gpt2')
        _model.eval()
        if USE_GPU and torch.cuda.is_available():
            _model = _model.cuda()


def _cache_key(text: str, tokens: list[str]) -> str:
    h = hashlib.sha256(text.encode('utf-8'))
    h.update(b'\x00')
    h.update('\x00'.join(tokens).encode('utf-8'))
    return h.hexdigest()


def add_surprisal(token_df: pd.DataFrame, text: str) -> pd.DataFrame:
    """Append a Surprisal column to a token-level DataFrame.

    Args:
        token_df: DataFrame from text_parser.parse(). Must have 'token'
                  and 'upos' columns. Rows must appear in document order.
        text:     The same raw text that was passed to text_parser.parse().
                  Used to run GPT-2 over the full document context.

    Returns:
        A copy of token_df with 'Surprisal' column (float, bits).
        Punctuation rows and the document's first word have NaN.
    """
    tokens = [str(t) for t in token_df['token'].tolist()]
    key = _cache_key(text, tokens)
    surprisals = _SURPRISAL_CACHE.get(key)
    if surprisals is None:
        _load()
        surprisals = _compute_surprisals(text, tokens)
        _SURPRISAL_CACHE[key] = surprisals
        while len(_SURPRISAL_CACHE) > _CACHE_MAX_ENTRIES:
            _SURPRISAL_CACHE.popitem(last=False)

    df = token_df.copy()
    df['Surprisal'] = [
        None if upos == 'PUNCT' else surprisals[i]
        for i, upos in enumerate(token_df['upos'].tolist())
    ]
    return df


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_MAX_GPT2_TOKENS = 1024


def map_tokens_to_subwords(
    text: str,
    spacy_tokens: list[str],
    offsets: list[tuple[int, int]],
) -> list[list[int]]:
    """Assign each subword to exactly one spaCy token.

    Tokens are located in `text` left to right; a subword belongs to the
    FIRST token whose character span it overlaps, and is then consumed —
    a subword straddling a token boundary (e.g. GPT-2 "don" over spaCy
    "do" + "n't") is never charged twice. Returns one (possibly empty)
    list of subword indices per token. Shared with the LLM-cloze
    HuggingFace backend, which does the same bookkeeping.
    """
    spans: list[list[int]] = []
    char_pos = 0
    sub_idx = 0
    n_subs = len(offsets)

    for token in spacy_tokens:
        idx = text.find(token, char_pos)
        if idx == -1:
            spans.append([])
            continue
        tok_start, tok_end = idx, idx + len(token)
        char_pos = tok_end

        # Skip subwords that end before this token starts.
        while sub_idx < n_subs and offsets[sub_idx][1] <= tok_start:
            sub_idx += 1
        own: list[int] = []
        while sub_idx < n_subs and offsets[sub_idx][0] < tok_end:
            own.append(sub_idx)
            sub_idx += 1
        spans.append(own)

    return spans


def _token_surprisals(
    spans: list[list[int]],
    log_probs: list,
) -> list:
    """Per-token surprisal (bits) from subword spans and log-probs.

    A token is None when it has no subwords, or when any of its subwords
    has no log-prob (log_probs[i] is None — the context-less position 0).
    """
    results = []
    for own in spans:
        if not own or any(log_probs[i] is None for i in own):
            results.append(None)
            continue
        total_lp = sum(log_probs[i] for i in own)
        results.append(round(-total_lp / math.log(2), 4))
    return results


def _compute_surprisals(text: str, spacy_tokens: list[str]) -> list:
    """Return one surprisal value per spaCy token (None for unmatched)."""
    enc = _tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    gpt2_ids = enc['input_ids']
    offsets = enc['offset_mapping']  # (char_start, char_end) per subword

    # Per-subword log-probs via a single forward pass (chunked for long texts)
    log_probs = _subword_log_probs(gpt2_ids)

    spans = map_tokens_to_subwords(text, spacy_tokens, offsets)
    return _token_surprisals(spans, log_probs)


def _subword_log_probs(input_ids: list[int]) -> list:
    """Return log P(token_i | context) for each subword token.

    Position 0 has no prior context — GPT-2 cannot score it, so its entry
    is None (propagated to NaN for whichever word owns it).
    """
    if not input_ids:
        return []
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
    lp: list = [None]
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
