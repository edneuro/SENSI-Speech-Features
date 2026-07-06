"""Tests for extractors.text_parser — focused on doc_id and paragraph_id.

The spaCy-driven columns (POS, lemma, dep) are covered by integration
tests via the CLI; this module only verifies the structural bookkeeping
added when paragraph-splitting and doc_id threading went in.
"""

import pytest

from extractors.text_parser import (
    _paragraph_id_for_char,
    _paragraph_spans,
    parse,
)


# ---------------------------------------------------------------------------
# Pure helpers — no spaCy required
# ---------------------------------------------------------------------------

class TestParagraphSpans:
    def test_single_paragraph_no_blank_lines(self):
        text = "One sentence. Two sentences. Three."
        assert _paragraph_spans(text) == [(0, len(text))]

    def test_two_paragraphs_one_blank_line(self):
        text = "First para.\n\nSecond para."
        spans = _paragraph_spans(text)
        assert len(spans) == 2
        assert text[spans[0][0]:spans[0][1]] == "First para."
        assert text[spans[1][0]:spans[1][1]] == "Second para."

    def test_multiple_blank_lines_collapse_to_one_break(self):
        text = "Para A.\n\n\n\nPara B."
        spans = _paragraph_spans(text)
        assert len(spans) == 2
        assert text[spans[0][0]:spans[0][1]] == "Para A."
        assert text[spans[1][0]:spans[1][1]] == "Para B."

    def test_blank_line_with_intra_line_whitespace(self):
        text = "Para A.\n  \nPara B."
        spans = _paragraph_spans(text)
        assert len(spans) == 2

    def test_empty_text(self):
        assert _paragraph_spans("") == [(0, 0)]

    def test_paragraph_id_for_char_first_paragraph(self):
        spans = [(0, 10), (12, 20)]
        assert _paragraph_id_for_char(0, spans) == 1
        assert _paragraph_id_for_char(9, spans) == 1

    def test_paragraph_id_for_char_second_paragraph(self):
        spans = [(0, 10), (12, 20)]
        assert _paragraph_id_for_char(12, spans) == 2
        assert _paragraph_id_for_char(19, spans) == 2

    def test_paragraph_id_for_char_in_gap_falls_to_next(self):
        # Char 10 and 11 are in the blank-line gap. The lookup attributes them
        # to the last paragraph (fallthrough behaviour).
        spans = [(0, 10), (12, 20)]
        assert _paragraph_id_for_char(10, spans) == 2


# ---------------------------------------------------------------------------
# End-to-end — requires the spaCy model
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def spacy_available():
    try:
        import spacy  # noqa: F401
        from extractors.text_parser import _get_nlp
        _get_nlp()
    except Exception as exc:
        pytest.skip(f'spaCy en_core_web_sm not available: {exc}')


def test_default_doc_id_is_one(spacy_available):
    df = parse("Hello world.")
    assert set(df['doc_id']) == {1}


def test_doc_id_parameter_propagates(spacy_available):
    df = parse("Hello world.", doc_id=7)
    assert set(df['doc_id']) == {7}


def test_single_paragraph_yields_paragraph_id_one(spacy_available):
    df = parse("First sentence. Second sentence in the same paragraph.")
    assert set(df['paragraph_id']) == {1}


def test_two_paragraphs_get_distinct_ids(spacy_available):
    text = "First paragraph sentence.\n\nSecond paragraph sentence."
    df = parse(text)
    # Rows from the first sentence must have paragraph_id == 1; rows from
    # the second sentence must have paragraph_id == 2.
    p1 = df[df['sentence_id'] == 1]['paragraph_id'].unique()
    p2 = df[df['sentence_id'] == 2]['paragraph_id'].unique()
    assert list(p1) == [1]
    assert list(p2) == [2]


# ---------------------------------------------------------------------------
# head_token_id — global token_id reference (self-contained against token_id)
# ---------------------------------------------------------------------------

def test_head_token_id_references_head_global_token_id(spacy_available):
    # Two sentences. Every token's head_token_id must equal the GLOBAL
    # token_id of its syntactic head, not a within-sentence index.
    df = parse("The dog barked. The cat slept.")
    slept = df[df['token'] == 'slept'].iloc[0]
    cat = df[df['token'] == 'cat'].iloc[0]
    # 'cat' depends on 'slept' -> head_token_id is slept's global token_id.
    assert cat['head_token_id'] == slept['token_id']


def test_root_head_points_to_its_own_global_token_id(spacy_available):
    # A ROOT points to itself. With global token_id encoding, the second
    # sentence's root must reference its own (large) token_id, NOT a small
    # within-sentence index like 3 — that difference is the whole bug.
    df = parse("The dog barked. The cat slept.")
    roots = df[df['dep_rel'] == 'ROOT']
    assert len(roots) >= 2
    assert (roots['head_token_id'] == roots['token_id']).all()
    second_root = roots.iloc[1]
    # First sentence is 4 tokens; a within-sentence root id would be <= 3.
    assert second_root['head_token_id'] >= 5


def test_token_in_sentence_id_resets_each_sentence(spacy_available):
    # The within-sentence index must restart at 1 for every sentence, while
    # token_id keeps counting globally.
    df = parse("The dog barked. The cat slept.")
    assert df[df['sentence_id'] == 1]['token_in_sentence_id'].iloc[0] == 1
    assert df[df['sentence_id'] == 2]['token_in_sentence_id'].iloc[0] == 1
    assert list(df['token_id']) == list(range(1, len(df) + 1))


def test_every_head_token_id_is_a_valid_token_id(spacy_available):
    # Invariant: head references are self-contained against token_id, so every
    # head_token_id must correspond to an actual row's token_id in the doc.
    df = parse("The dog barked loudly. Then the cat slept for many hours.")
    assert set(df['head_token_id']).issubset(set(df['token_id']))
