"""
Text parsing extractor — spaCy linguistic annotations.

Input:  raw transcript text (str)
Output: pandas DataFrame, one row per token

Columns produced:
    doc_id, paragraph_id, sentence_id, sentence,
    start (char offset in sentence), end (char offset in sentence),
    token_id, token_in_sentence_id, token, lemma,
    upos, xpos, feats, dep_rel, head_token_id

Whitespace tokens (spaCy emits a literal newline as a token, upos='SPACE', for
hard-wrapped transcripts) are excluded from the output entirely — they are not
words. token_id is the global whole-document index (1-based) over emitted
(non-whitespace) tokens only; token_in_sentence_id is the position within the
current sentence, also counting emitted tokens only (resets to 1 each
sentence).

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

    # Pass 1: assign token_id over EMITTED tokens only, and record the
    # spaCy-doc-index -> token_id mapping.
    #
    # Whitespace tokens (a literal newline in the transcript) are not words:
    # they used to occupy a row with NaN for every feature and to consume both
    # a token_id and a token_in_sentence_id, shifting every following word by
    # one. They are dropped here.
    #
    # This mapping is what keeps head_token_id correct. spaCy's token.head.i is
    # a doc-level index that still counts the skipped whitespace, so
    # `token.head.i + 1` is only equal to the head's token_id while nothing is
    # skipped. Resolving through id_by_doc_index is mandatory, not defensive.
    id_by_doc_index: dict[int, int] = {}
    next_id = 1
    for token in doc:
        if token.is_space:
            continue
        id_by_doc_index[token.i] = next_id
        next_id += 1

    rows = []

    for sent_idx, sent in enumerate(doc.sents, start=1):
        sent_start_char = sent.start_char
        paragraph_id = _paragraph_id_for_char(sent_start_char, paragraphs)

        tok_idx = 0
        for token in sent:
            if token.is_space:
                continue
            tok_idx += 1

            # Character offsets relative to the sentence string (1-indexed)
            char_start = token.idx - sent_start_char + 1
            char_end = char_start + len(token.text) - 1

            # spaCy's MorphAnalysis already stringifies to '' when a token has
            # no morphology; writing a literal 'NA' sentinel instead collides
            # with pandas' default na_values on any standard pd.read_csv().
            feats = str(token.morph)

            # head_token_id is the head's token_id (global, whole-document,
            # 1-based over emitted tokens), so it is self-contained against the
            # token_id column and unambiguous across sentence boundaries. A
            # root token's head is itself. If a head is itself whitespace
            # (spaCy can attach to one in malformed input) the token is treated
            # as its own head rather than dangling.
            head_token_id = id_by_doc_index.get(
                token.head.i, id_by_doc_index[token.i])

            rows.append({
                'doc_id': doc_id,
                'paragraph_id': paragraph_id,
                'sentence_id': sent_idx,
                'sentence': sent.text,
                'start': char_start,
                'end': char_end,
                'token_id': id_by_doc_index[token.i],
                'token_in_sentence_id': tok_idx,
                'token': token.text,
                'lemma': token.lemma_,
                'upos': token.pos_,
                'xpos': token.tag_,
                'feats': feats,
                'dep_rel': token.dep_,
                'head_token_id': head_token_id,
            })

    return pd.DataFrame(rows)
