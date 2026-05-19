"""
Text parsing extractor — spaCy linguistic annotations.

Input:  raw transcript text (str)
Output: pandas DataFrame, one row per token

Columns produced:
    doc_id, paragraph_id, sentence_id, sentence,
    start (char offset in sentence), end (char offset in sentence),
    term_id, token_id, token, lemma,
    upos, xpos, feats, dep_rel, head_token_id, deps, misc

This extractor is purely text-based and has no dependency on audio.
It can be run independently of alignment.
"""

import pandas as pd
import spacy

_nlp = None


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


def parse(text: str) -> pd.DataFrame:
    """Run spaCy pipeline on transcript text and return a token-level DataFrame.

    Args:
        text: Raw transcript text. Punctuation is preserved in the output
              but flagged by upos='PUNCT'. Numbers should be in their
              original written form (digits); the aligner extractor handles
              spelling them out separately.

    Returns:
        DataFrame with one row per token, including punctuation.
    """
    nlp = _get_nlp()
    doc = nlp(text)

    rows = []
    term_id = 1

    for sent_idx, sent in enumerate(doc.sents, start=1):
        sent_start_char = sent.start_char

        for tok_idx, token in enumerate(sent, start=1):
            # Character offsets relative to the sentence string (1-indexed)
            char_start = token.idx - sent_start_char + 1
            char_end = char_start + len(token.text) - 1

            feats = str(token.morph) if token.morph else 'NA'
            misc = 'SpaceAfter=No' if not token.whitespace_ else 'NA'

            # head_token_id is the token_id (within-sentence index) of the
            # syntactic head. For the root token its own head is itself.
            head_token_id = token.head.i - sent.start + 1

            rows.append({
                'doc_id': 1,
                'paragraph_id': 1,
                'sentence_id': sent_idx,
                'sentence': sent.text,
                'start': char_start,
                'end': char_end,
                'term_id': term_id,
                'token_id': tok_idx,
                'token': token.text,
                'lemma': token.lemma_,
                'upos': token.pos_,
                'xpos': token.tag_,
                'feats': feats,
                'dep_rel': token.dep_,
                'head_token_id': head_token_id,
                'deps': 'NA',
                'misc': misc,
            })
            term_id += 1

    return pd.DataFrame(rows)
