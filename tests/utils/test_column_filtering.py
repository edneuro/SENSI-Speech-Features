"""Tests for the v4 per-column CSV filtering helpers in main.py."""


def test_word_columns_to_drop_returns_empty_when_all_on():
    from main import _word_columns_to_drop
    all_on = {
        "feat_struct_doc_id": True, "feat_struct_paragraph_id": True,
        "feat_struct_sentence_id": True, "feat_struct_sentence_text": True,
        "feat_struct_char_start": True, "feat_struct_char_end": True,
        "feat_struct_term_id": True, "feat_struct_token_id": True,
        "feat_acoustic_onset": True, "feat_acoustic_end": True,
        "feat_acoustic_word_duration": True, "feat_acoustic_word_gap": True,
    }
    assert _word_columns_to_drop(all_on) == set()


def test_word_columns_to_drop_includes_disabled_structural():
    from main import _word_columns_to_drop
    drop = _word_columns_to_drop({
        "feat_struct_doc_id": False,
        "feat_struct_paragraph_id": False,
        "feat_struct_sentence_id": True,
    })
    assert "doc_id" in drop
    assert "paragraph_id" in drop
    assert "sentence_id" not in drop


def test_word_columns_to_drop_includes_disabled_acoustic():
    from main import _word_columns_to_drop
    drop = _word_columns_to_drop({
        "feat_acoustic_onset": False,
        "feat_acoustic_end": True,
        "feat_acoustic_word_duration": False,
        "feat_acoustic_word_gap": True,
    })
    assert "Onset" in drop
    assert "Word_Duration" in drop
    assert "End" not in drop
    assert "Word_Gap" not in drop


def test_word_columns_to_drop_treats_missing_keys_as_on():
    """Backward compat — when an old args object has no `features` dict,
    or features dict is missing some keys, every column is retained."""
    from main import _word_columns_to_drop
    assert _word_columns_to_drop({}) == set()


def test_doc_columns_to_drop_filters_counts():
    from main import _doc_columns_to_drop
    drop = _doc_columns_to_drop(
        output={"out_count_tokens": False, "out_count_sentences": False,
                "out_count_content_tokens": True, "out_count_paragraphs": True},
        features={"feat_psych_authentic": True, "feat_psych_analytic": True,
                  "feat_psych_tone": True, "feat_psych_clout": True},
    )
    assert "n_tokens" in drop
    assert "n_sentences" in drop
    assert "n_content_tokens" not in drop
    assert "n_paragraphs" not in drop


def test_doc_columns_to_drop_filters_psych():
    from main import _doc_columns_to_drop
    drop = _doc_columns_to_drop(
        output={"out_count_tokens": True, "out_count_content_tokens": True,
                "out_count_sentences": True, "out_count_paragraphs": True},
        features={"feat_psych_authentic": False, "feat_psych_analytic": False,
                  "feat_psych_tone": True, "feat_psych_clout": True},
    )
    assert "psych_authentic" in drop
    assert "psych_analytic" in drop
    assert "psych_tone" not in drop
    assert "psych_clout" not in drop


def test_doc_columns_to_drop_empty_when_all_on():
    from main import _doc_columns_to_drop
    all_on_output = {
        "out_count_tokens": True, "out_count_content_tokens": True,
        "out_count_sentences": True, "out_count_paragraphs": True,
    }
    all_on_features = {
        "feat_psych_authentic": True, "feat_psych_analytic": True,
        "feat_psych_tone": True, "feat_psych_clout": True,
    }
    assert _doc_columns_to_drop(all_on_output, all_on_features) == set()
