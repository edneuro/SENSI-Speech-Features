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
              "feat_struct_token_id", "feat_struct_token_in_sentence_id"):
        assert w.toggle(k) is not None, k


def test_word_section_acoustic_has_four_per_column_toggles(qtbot):
    from gui.section_word_features import WordFeaturesSection
    w = WordFeaturesSection()
    qtbot.addWidget(w)
    for k in ("feat_acoustic_onset", "feat_acoustic_end",
              "feat_acoustic_word_duration", "feat_acoustic_word_gap"):
        assert w.toggle(k) is not None, k


def test_credentials_help_icon_opens_shared_doc(qtbot, monkeypatch):
    """The Predictability card exposes the standard blue ⓘ info-icon next to
    text, targeting the shared LLM-credentials doc; clicking it opens the
    feature-info dialog on 'llm_credentials'."""
    from gui.section_word_features import WordFeaturesSection
    from gui.info_popup import InfoIcon
    w = WordFeaturesSection()
    qtbot.addWidget(w)
    assert isinstance(w.credentials_help_icon, InfoIcon)
    assert w.credentials_help_icon._toggle_key == "llm_credentials"

    opened = {}
    import gui.info_popup as ip

    class _FakeDialog:
        def __init__(self, key, parent=None):
            opened["key"] = key
        def exec(self):
            opened["exec"] = True

    monkeypatch.setattr(ip, "FeatureInfoDialog", _FakeDialog)
    w.credentials_help_icon.click()
    assert opened == {"key": "llm_credentials", "exec": True}


def test_llm_credentials_doc_exists_and_is_loadable():
    """The shared credentials doc the button targets must resolve as a real
    feature-doc file (key == filename), else the button opens a 'not found'."""
    from gui.feature_docs import load_feature_doc
    doc = load_feature_doc("llm_credentials")
    assert doc is not None
    # Local HF backend only; the doc must cover the HF token setup.
    assert "HF_TOKEN" in doc.body


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


def test_ewfg_toggles_are_disabled_unavailable(qtbot, prefs_tmp):
    """EWFG has no digital source (print-only), so its toggles are hard-disabled
    and unchecked — even though the SUBTLEX toggle in the same card is fine."""
    from gui.section_word_features import WordFeaturesSection
    section = WordFeaturesSection()
    qtbot.addWidget(section)
    for key in ("freq_ewfg_k2", "freq_ewfg_3to5", "freq_ewfg_6to8",
                "freq_ewfg_9to12", "freq_ewfg_overall"):
        row = section.toggle(key)
        assert row is not None, key
        assert row._perma_disabled is True, key
        assert not row.check.isEnabled(), key
        assert not row.is_checked(), key
    # SUBTLEX (same card) is unaffected
    assert section.toggle("freq_subtlex")._perma_disabled is False


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


def _label_for(model_id):
    from extractors.llm_cloze import LLM_OPTIONS
    return next(lab for mid, lab in LLM_OPTIONS if mid == model_id)


def test_no_openai_option_in_model_combo(qtbot):
    # OpenAI is no longer a selectable backend, so the API-key row can never
    # be shown by picking a model. The combo must offer local models only.
    from gui.section_word_features import WordFeaturesSection
    from extractors.llm_cloze import LLM_OPTIONS
    w = WordFeaturesSection(); qtbot.addWidget(w)
    labels = [w.llm_model_combo.itemText(i)
              for i in range(w.llm_model_combo.count())]
    assert labels == [lab for _mid, lab in LLM_OPTIONS]
    assert not any(mid.startswith('gpt-') for mid, _ in LLM_OPTIONS)


def test_gated_model_no_token_shows_hf_box(qtbot, monkeypatch):
    import extractors.llm_cloze as m
    monkeypatch.setattr(m, 'has_hf_token', lambda: False)
    from gui.section_word_features import WordFeaturesSection
    w = WordFeaturesSection(); qtbot.addWidget(w)
    w.show()  # isVisible() requires a shown top-level ancestor (Qt semantics)
    w.llm_model_combo.setCurrentText(_label_for('meta-llama/Meta-Llama-3-8B'))
    assert w.hf_token_row.isVisible() is True
    assert w.hf_token_detected_label.isVisible() is False
    assert w.api_key_row.isVisible() is False


def test_gated_model_with_token_shows_detected_label(qtbot, monkeypatch):
    import extractors.llm_cloze as m
    monkeypatch.setattr(m, 'has_hf_token', lambda: True)
    from gui.section_word_features import WordFeaturesSection
    w = WordFeaturesSection(); qtbot.addWidget(w)
    w.show()  # isVisible() requires a shown top-level ancestor (Qt semantics)
    w.llm_model_combo.setCurrentText(_label_for('meta-llama/Meta-Llama-3-8B'))
    assert w.hf_token_row.isVisible() is False
    assert w.hf_token_detected_label.isVisible() is True


def test_session_hf_token_returns_text_only_when_visible(qtbot, monkeypatch):
    import extractors.llm_cloze as m
    monkeypatch.setattr(m, 'has_hf_token', lambda: False)
    from gui.section_word_features import WordFeaturesSection
    w = WordFeaturesSection(); qtbot.addWidget(w)
    w.show()  # isVisible() requires a shown top-level ancestor (Qt semantics)
    w.llm_model_combo.setCurrentText(_label_for('meta-llama/Meta-Llama-3-8B'))
    w.hf_token_line.setText('hf_secret')
    assert w.session_hf_token() == 'hf_secret'
    # When a token is already detected, the input row hides → session empty.
    monkeypatch.setattr(m, 'has_hf_token', lambda: True)
    w.llm_model_combo.setCurrentText(_label_for('meta-llama/Llama-3.2-3B'))
    assert w.session_hf_token() == ''   # hidden → empty
