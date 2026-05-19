def test_doc_section_does_not_duplicate_output_toggle(qtbot):
    """Output toggle ('Document-level summary') is owned by the Run page —
    Doc-level page is purely feature configuration. Confirms no duplicate."""
    from gui.section_doc_features import DocFeaturesSection
    w = DocFeaturesSection()
    qtbot.addWidget(w)
    assert w.toggle("output_document_summary") is None


def test_doc_section_counts_has_four_per_column_toggles(qtbot):
    from gui.section_doc_features import DocFeaturesSection
    w = DocFeaturesSection()
    qtbot.addWidget(w)
    for k in ("output_out_count_tokens", "output_out_count_content_tokens",
              "output_out_count_sentences", "output_out_count_paragraphs"):
        assert w.toggle(k) is not None, k


def test_doc_section_psych_has_four_per_column_toggles(qtbot):
    from gui.section_doc_features import DocFeaturesSection
    w = DocFeaturesSection()
    qtbot.addWidget(w)
    for k in ("feat_psych_authentic", "feat_psych_analytic",
              "feat_psych_tone", "feat_psych_clout"):
        assert w.toggle(k) is not None, k


def test_doc_section_has_configure_categories_button(qtbot):
    from gui.section_doc_features import DocFeaturesSection
    w = DocFeaturesSection()
    qtbot.addWidget(w)
    assert w.configure_cats_button is not None
    assert "document" in w.configure_cats_button.text().lower()


def test_doc_prosody_card_has_sixteen_toggles(qtbot, prefs_tmp):
    from gui.section_doc_features import DocFeaturesSection
    section = DocFeaturesSection()
    qtbot.addWidget(section)
    expected_keys = [
        'feat_doc_prosody_pitch_mean', 'feat_doc_prosody_pitch_min',
        'feat_doc_prosody_pitch_max', 'feat_doc_prosody_pitch_std',
        'feat_doc_prosody_volume_mean', 'feat_doc_prosody_volume_min',
        'feat_doc_prosody_volume_max', 'feat_doc_prosody_volume_std',
        'feat_doc_prosody_pitch_delta_mean', 'feat_doc_prosody_volume_delta_mean',
        'feat_doc_prosody_hnr', 'feat_doc_prosody_articulation_rate',
        'feat_doc_prosody_pause_count', 'feat_doc_prosody_pause_mean_duration',
        'feat_doc_prosody_pause_total_duration', 'feat_doc_prosody_pause_pct_silence',
    ]
    for key in expected_keys:
        assert section.toggle(key) is not None, f"{key} toggle missing"


def test_pause_threshold_spinbox_persists(qtbot, prefs_tmp):
    from gui.section_doc_features import DocFeaturesSection
    from utils.preferences import load_preferences
    section = DocFeaturesSection()
    qtbot.addWidget(section)
    section.pause_threshold_spin.setValue(0.5)
    assert load_preferences()['prosody']['pause_threshold_s'] == 0.5
