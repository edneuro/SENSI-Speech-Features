def test_doc_section_does_not_duplicate_output_toggle(qtbot):
    """Output toggle ('Document-level summary') is owned by the Run page —
    Doc-level page is purely feature configuration. Confirms no duplicate."""
    from sensi.gui.section_doc_features import DocFeaturesSection
    w = DocFeaturesSection()
    qtbot.addWidget(w)
    assert w.toggle("output_document_summary") is None


def test_doc_section_counts_has_four_per_column_toggles(qtbot):
    from sensi.gui.section_doc_features import DocFeaturesSection
    w = DocFeaturesSection()
    qtbot.addWidget(w)
    for k in ("output_out_count_tokens", "output_out_count_content_tokens",
              "output_out_count_sentences", "output_out_count_paragraphs"):
        assert w.toggle(k) is not None, k


def test_doc_section_psych_has_four_per_column_toggles(qtbot):
    from sensi.gui.section_doc_features import DocFeaturesSection
    w = DocFeaturesSection()
    qtbot.addWidget(w)
    for k in ("feat_psych_authentic", "feat_psych_analytic",
              "feat_psych_tone", "feat_psych_clout"):
        assert w.toggle(k) is not None, k


def test_doc_section_shows_categories_count_and_hint(qtbot):
    """v7: doc-level page is read-only for categories. It shows a count label
    and a hint pointing to the word-level page; the old configure button is gone."""
    from sensi.gui.section_doc_features import DocFeaturesSection
    w = DocFeaturesSection()
    qtbot.addWidget(w)
    # Count label must exist and mention categories
    assert w.cats_count_label is not None
    assert "categor" in w.cats_count_label.text().lower()
    # No configure button on this page in v7
    assert not hasattr(w, "configure_cats_button")


def test_doc_prosody_card_has_sixteen_toggles(qtbot, prefs_tmp):
    from sensi.gui.section_doc_features import DocFeaturesSection
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
    from sensi.gui.section_doc_features import DocFeaturesSection
    from sensi.utils.preferences import load_preferences
    section = DocFeaturesSection()
    qtbot.addWidget(section)
    section.pause_threshold_spin.setValue(0.5)
    assert load_preferences()['prosody']['pause_threshold_s'] == 0.5


def test_agg_nan_warnings_track_shared_config(qtbot, prefs_tmp):
    """Doc means compute without the word-level section, so the only NaN
    warning left is an empty predictability roster; mean Zipf frequency
    never warns (word_freq is pulled in)."""
    from sensi.utils.preferences import save_preferences
    from sensi.gui.section_doc_features import DocFeaturesSection
    save_preferences({
        "output": {"level_word": False, "level_document": True,
                   "doc_agg_zipf_freq": True, "doc_agg_predictability": True},
        "predictability": {"models": [], "metrics": {"surprisal": False}},
    })
    w = DocFeaturesSection()
    qtbot.addWidget(w)
    w.refresh_from_prefs()
    assert w._toggles["output_doc_agg_zipf_freq"].warning is None
    assert w._toggles["output_doc_agg_predictability"].warning is not None

    save_preferences({
        "output": {"level_word": False, "level_document": True,
                   "doc_agg_zipf_freq": True, "doc_agg_predictability": True},
        "predictability": {"models": ["gpt2"], "metrics": {"surprisal": True}},
    })
    w.refresh_from_prefs()
    assert w._toggles["output_doc_agg_predictability"].warning is None
