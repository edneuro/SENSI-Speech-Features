def test_word_section_does_not_duplicate_output_toggle(qtbot):
    """Output toggle ('Word-level CSV') is owned by the Run page —
    Word-level page is purely feature configuration. Confirms no duplicate."""
    from gui.section_word_features import WordFeaturesSection
    w = WordFeaturesSection()
    qtbot.addWidget(w)
    assert w.toggle("output_word_level_csv") is None


def test_word_section_structural_has_eight_per_column_toggles(qtbot):
    from gui.section_word_features import WordFeaturesSection
    w = WordFeaturesSection()
    qtbot.addWidget(w)
    for k in ("feat_struct_doc_id", "feat_struct_paragraph_id",
              "feat_struct_sentence_id", "feat_struct_sentence_text",
              "feat_struct_char_start", "feat_struct_char_end",
              "feat_struct_term_id", "feat_struct_token_id"):
        assert w.toggle(k) is not None, k


def test_word_section_acoustic_has_four_per_column_toggles(qtbot):
    from gui.section_word_features import WordFeaturesSection
    w = WordFeaturesSection()
    qtbot.addWidget(w)
    for k in ("feat_acoustic_onset", "feat_acoustic_end",
              "feat_acoustic_word_duration", "feat_acoustic_word_gap"):
        assert w.toggle(k) is not None, k


def test_word_section_has_configure_categories_button(qtbot):
    from gui.section_word_features import WordFeaturesSection
    w = WordFeaturesSection()
    qtbot.addWidget(w)
    assert w.configure_cats_button is not None
    assert "word" in w.configure_cats_button.text().lower()


def test_word_section_lexling_freq_predict_sem_carried_over(qtbot):
    """The lex-ling/freq/predictability/semantic subsections still render."""
    from gui.section_word_features import WordFeaturesSection
    w = WordFeaturesSection()
    qtbot.addWidget(w)
    for k in ("lexling_tokens", "freq_subtlex", "pred_surprisal_gpt2",
              "sem_indicators", "freq_ewfg_k2"):
        assert w.toggle(k) is not None, k


def test_word_prosody_card_has_ten_toggles(qtbot, prefs_tmp):
    from gui.section_word_features import WordFeaturesSection
    section = WordFeaturesSection()
    qtbot.addWidget(section)
    for key in (
        'feat_prosody_pitch_mean', 'feat_prosody_pitch_min',
        'feat_prosody_pitch_max', 'feat_prosody_pitch_std',
        'feat_prosody_pitch_delta',
        'feat_prosody_volume_mean', 'feat_prosody_volume_min',
        'feat_prosody_volume_max', 'feat_prosody_volume_std',
        'feat_prosody_volume_delta',
    ):
        assert section.toggle(key) is not None, f"{key} toggle missing"


def test_word_prosody_pitch_units_combo_persists(qtbot, prefs_tmp):
    from gui.section_word_features import WordFeaturesSection
    from utils.preferences import load_preferences
    section = WordFeaturesSection()
    qtbot.addWidget(section)
    section.pitch_units_combo.setCurrentText('Hz')
    assert load_preferences()['prosody']['pitch_units'] == 'hz'
