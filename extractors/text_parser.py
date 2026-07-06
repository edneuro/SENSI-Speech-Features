"""
Text parsing extractor — spaCy linguistic annotations.

Input:  raw transcript text (str)
Output: pandas DataFrame, one row per token

Columns produced:
    doc_id, paragraph_id, sentence_id, sentence,
    start (char offset in sentence), end (char offset in sentence),
    token_id, token_in_sentence_id, token, lemma,
    upos, xpos, feats, dep_rel, head_token_id

token_id is the global whole-document token index (1-based); token_in_sentence_id
is the position within the current sentence (resets to 1 each sentence).

head_token_id references the syntactic head by its global token_id, so it is
self-contained against the token_id column and never ambiguous across sentence
boundaries. A root token's head_token_id equals its own token_id.

This extractor is purely text-based and has no dependency on audio.
It can be run independently of alignment.
"""

import re

import pandas as pd
import spacy

_nlp = None

# One or more blank lines (allowing intra-line whitespace) separate paragraphs.
_PARAGRAPH_SEP_RE = re.compile(r'\n[ \t]*\n+')


def _get_nlp():
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load('en_core_web_sm')
        except OSError:
            raise RuntimeError(
                "spaCy model not found. Run: python -m spacy download en_core_web_sm"
            )
    return _nlp


def _paragraph_spans(text: str) -> list[tuple[int, int]]:
    """Return (start_char, end_char) for each paragraph in document order.

    Paragraphs are runs of text separated by one or more blank lines. A
    document with no blank lines is a single paragraph spanning the whole
    text. Returned spans are half-open: [start, end).
    """
    spans: list[tuple[int, int]] = []
    cursor = 0
    for sep in _PARAGRAPH_SEP_RE.finditer(text):
        spans.append((cursor, sep.start()))
        cursor = sep.end()
    spans.append((cursor, len(text)))
    return spans


def _paragraph_id_for_char(char_pos: int, spans: list[tuple[int, int]]) -> int:
    """Return 1-indexed paragraph_id for the paragraph containing char_pos.

    If char_pos lands inside a blank-line gap (rare — sentences usually start
    at the first non-blank position of their paragraph), it falls through to
    the next paragraph that contains it.
    """
    for idx, (start, end) in enumerate(spans, start=1):
        if start <= char_pos < end:
            return idx
    # char_pos is past the last paragraph end (e.g. trailing whitespace);
    # attribute it to the final paragraph.
    return len(spans) if spans else 1


def parse(text: str, doc_id: int = 1) -> pd.DataFrame:
    """Run spaCy pipeline on transcript text and return a token-level DataFrame.

    Args:
        text:   Raw transcript text. Punctuation is preserved in the output
                but flagged by upos='PUNCT'. Numbers should be in their
                original written form (digits); the aligner extractor handles
                spelling them out separately. Blank lines in the input are
                treated as paragraph breaks (see paragraph_id).
        doc_id: 1-indexed identifier for this document. Defaults to 1 for
                single-document runs. Batch callers pass the document's
                position in the batch (1, 2, 3, ...) so each document's rows
                are distinguishable when concatenated.

    Returns:
        DataFrame with one row per token, including punctuation.
    """
    nlp = _get_nlp()
    doc = nlp(text)
    paragraphs = _paragraph_spans(text)

    rows = []
    token_id = 1

    for sent_idx, sent in enumerate(doc.sents, start=1):
        sent_start_char = sent.start_char
        paragraph_id = _paragraph_id_for_char(sent_start_char, paragraphs)

        for tok_idx, token in enumerate(sent, start=1):
            # Character offsets relative to the sentence string (1-indexed)
            char_start = token.idx - sent_start_char + 1
            char_end = char_start + len(token.text) - 1

            feats = str(token.morph) if token.morph else 'NA'

            # head_token_id is the token_id (global, whole-document index) of
            # the syntactic head, so it is self-contained against the token_id
            # column and unambiguous across sentence boundaries. For the root
            # token its own head is itself. token_id counts tokens in document
            # order starting at 1, so the head's token_id is token.head.i + 1.
            head_token_id = token.head.i + 1

            rows.append({
                'doc_id': doc_id,
                'paragraph_id': paragraph_id,
                'sentence_id': sent_idx,
                'sentence': sent.text,
                'start': char_start,
                'end': char_end,
                'token_id': token_id,
                'token_in_sentence_id': tok_idx,
                'token': token.text,
                'lemma': token.lemma_,
                'upos': token.pos_,
                'xpos': token.tag_,
                'feats': feats,
                'dep_rel': token.dep_,
                'head_token_id': head_token_id,
            })
            token_id += 1

    return pd.DataFrame(rows)
