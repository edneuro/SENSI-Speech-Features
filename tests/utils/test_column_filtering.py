"""Tests for the v4 per-column CSV filtering helpers in main.py."""


def test_word_columns_to_drop_returns_empty_when_all_on():
    from sensi.pipeline import _word_columns_to_drop
    all_on = {
        "feat_struct_doc_id": True, "feat_struct_paragraph_id": True,
        "feat_struct_sentence_id": True, "feat_struct_sentence_text": True,
        "feat_struct_char_start": True, "feat_struct_char_end": True,
        "feat_struct_token_id": True, "feat_struct_token_in_sentence_id": True,
        "feat_acoustic_onset": True, "feat_acoustic_end": True,
        "feat_acoustic_word_duration": True, "feat_acoustic_word_gap": True,
    }
    assert _word_columns_to_drop(all_on) == set()


def test_word_columns_to_drop_includes_disabled_structural():
    from sensi.pipeline import _word_columns_to_drop
    drop = _word_columns_to_drop({
        "feat_struct_doc_id": False,
        "feat_struct_paragraph_id": False,
        "feat_struct_sentence_id": True,
    })
    assert "doc_id" in drop
    assert "paragraph_id" in drop
    assert "sentence_id" not in drop


def test_word_columns_to_drop_includes_disabled_acoustic():
    from sensi.pipeline import _word_columns_to_drop
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
    from sensi.pipeline import _word_columns_to_drop
    assert _word_columns_to_drop({}) == set()


def test_word_columns_to_drop_keeps_token_id_when_dep_parse_on():
    """head_token_id references the head by global token_id, so token_id is its
    anchor. Dropping token_id while the dependency parse is emitted would leave
    head_token_id dangling — the guard must retain token_id in that case."""
    from sensi.pipeline import _word_columns_to_drop
    drop = _word_columns_to_drop({
        "feat_struct_token_id": False,  # user tried to drop the anchor
        "lexling_dep": True,            # but dependency parse is emitted
    })
    assert "token_id" not in drop


def test_word_columns_to_drop_drops_token_id_when_dep_parse_off():
    """With no dependency columns emitted, token_id is a plain structural column
    and honours its own toggle."""
    from sensi.pipeline import _word_columns_to_drop
    drop = _word_columns_to_drop({
        "feat_struct_token_id": False,
        "lexling_dep": False,
    })
    assert "token_id" in drop


def test_word_columns_to_drop_drops_lexling_columns_when_off():
    """The lexling per-column toggles must actually remove their columns from
    features.csv (previously a no-op — the toggles were ignored)."""
    from sensi.pipeline import _word_columns_to_drop
    drop = _word_columns_to_drop({
        "lexling_tokens": False, "lexling_pos": False,
        "lexling_morph": False, "lexling_dep": False,
    })
    for col in ("token", "lemma", "upos", "xpos", "feats", "dep_rel", "head_token_id"):
        assert col in drop, col


def test_word_columns_to_drop_keeps_lexling_columns_when_on():
    from sensi.pipeline import _word_columns_to_drop
    drop = _word_columns_to_drop({
        "lexling_tokens": True, "lexling_pos": True,
        "lexling_morph": True, "lexling_dep": True,
    })
    for col in ("token", "lemma", "upos", "xpos", "feats", "dep_rel", "head_token_id"):
        assert col not in drop, col


def test_word_columns_to_drop_lexling_groups_are_independent():
    from sensi.pipeline import _word_columns_to_drop
    drop = _word_columns_to_drop({
        "lexling_tokens": True, "lexling_pos": False,
        "lexling_morph": True, "lexling_dep": True,
    })
    assert "upos" in drop and "xpos" in drop        # POS off → dropped
    assert "token" not in drop and "lemma" not in drop  # tokens on → kept
    assert "feats" not in drop                       # morph on → kept
    assert "dep_rel" not in drop                     # dep on → kept


def test_doc_columns_to_drop_filters_counts():
    from sensi.pipeline import _doc_columns_to_drop
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
    from sensi.pipeline import _doc_columns_to_drop
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
    from sensi.pipeline import _doc_columns_to_drop
    all_on_output = {
        "out_count_tokens": True, "out_count_content_tokens": True,
        "out_count_sentences": True, "out_count_paragraphs": True,
    }
    all_on_features = {
        "feat_psych_authentic": True, "feat_psych_analytic": True,
        "feat_psych_tone": True, "feat_psych_clout": True,
    }
    assert _doc_columns_to_drop(all_on_output, all_on_features) == set()
