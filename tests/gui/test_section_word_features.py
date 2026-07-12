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
    """The lex-ling/freq/predictability/semantic subsections still render.

    Predictability (v18) has no per-column feature toggles anymore — it's
    model_checkboxes + metric_checkboxes — so it's checked separately here
    rather than via toggle().
    """
    from gui.section_word_features import WordFeaturesSection
    w = WordFeaturesSection()
    qtbot.addWidget(w)
    for k in ("lexling_tokens", "freq_subtlex",
              "sem_indicators", "freq_ewfg_k2"):
        assert w.toggle(k) is not None, k
    assert "gpt2" in w.model_checkboxes
    assert "surprisal" in w.metric_checkboxes


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


# --- Predictability v18: model_checkboxes / metric_checkboxes ------------
# Replaces the old single llm_model_combo + api_key_row design (OpenAI
# removed; model selection is now a multi-select roster, see
# extractors/predictability_models.py).

def test_model_rows_persist_selection(qtbot, prefs_tmp):
    from gui.section_word_features import WordFeaturesSection
    from utils.preferences import load_preferences
    w = WordFeaturesSection(); qtbot.addWidget(w)
    w.model_checkboxes["gpt2"].setChecked(True)
    assert "gpt2" in load_preferences()["predictability"]["models"]


def test_metric_toggles_persist(qtbot, prefs_tmp):
    from gui.section_word_features import WordFeaturesSection
    from utils.preferences import load_preferences
    w = WordFeaturesSection(); qtbot.addWidget(w)
    w.metric_checkboxes["cloze"].setChecked(True)
    assert load_preferences()["predictability"]["metrics"]["cloze"] is True


def test_model_rows_cover_full_roster_with_gating_hint(qtbot):
    """One checkbox per extractors.predictability_models.all_specs() entry,
    with the spec's label + note (the 'free, no token' vs 'gated' hint) on
    the row's word-wrapping label (checkbox text can't wrap, so the text
    lives on a companion label — see the horizontal-scrollbar fix)."""
    from gui.section_word_features import WordFeaturesSection
    from extractors.predictability_models import all_specs
    w = WordFeaturesSection(); qtbot.addWidget(w)
    for spec in all_specs():
        assert spec.slug in w.model_checkboxes
        lbl = w.model_labels[spec.slug]
        assert spec.label in lbl.text()
        assert spec.note in lbl.text()
        assert lbl.wordWrap() is True


def test_no_openai_widgets_anymore(qtbot):
    """OpenAI is no longer a selectable backend — the old combo/API-key
    widgets from the pre-unification design must be gone entirely."""
    from gui.section_word_features import WordFeaturesSection
    w = WordFeaturesSection(); qtbot.addWidget(w)
    assert not hasattr(w, "llm_model_combo")
    assert not hasattr(w, "api_key_row")
    assert not hasattr(w, "api_key_line")


def test_hf_box_always_shown_when_no_token_detected(qtbot, prefs_tmp, monkeypatch):
    # Discoverability: with no token auto-detected, the input is always
    # present (so the user can find where to paste it) regardless of model
    # selection — including nothing selected or only the free GPT-2.
    import extractors.predictability as m
    monkeypatch.setattr(m, "has_hf_token", lambda: False)
    from gui.section_word_features import WordFeaturesSection
    w = WordFeaturesSection(); qtbot.addWidget(w); w.show()
    # Nothing selected → still visible.
    assert w.hf_token_row.isVisible() is True
    assert w.hf_token_detected_label.isVisible() is False
    # A gated model selected → still visible.
    w.model_checkboxes["llama3_8b"].setChecked(True)
    assert w.hf_token_row.isVisible() is True
    # Only the free GPT-2 selected → still visible (no longer hidden).
    w.model_checkboxes["llama3_8b"].setChecked(False)
    w.model_checkboxes["gpt2"].setChecked(True)
    assert w.hf_token_row.isVisible() is True


def test_gated_model_with_token_shows_detected_label(qtbot, monkeypatch):
    import extractors.predictability as m
    monkeypatch.setattr(m, 'has_hf_token', lambda: True)
    from gui.section_word_features import WordFeaturesSection
    w = WordFeaturesSection(); qtbot.addWidget(w)
    w.show()  # isVisible() requires a shown top-level ancestor (Qt semantics)
    w.model_checkboxes['llama3_8b'].setChecked(True)
    assert w.hf_token_row.isVisible() is False
    assert w.hf_token_detected_label.isVisible() is True


def test_session_hf_token_returns_text_only_when_visible(qtbot, monkeypatch):
    import extractors.predictability as m
    monkeypatch.setattr(m, 'has_hf_token', lambda: False)
    from gui.section_word_features import WordFeaturesSection
    w = WordFeaturesSection(); qtbot.addWidget(w)
    w.show()  # isVisible() requires a shown top-level ancestor (Qt semantics)
    w.model_checkboxes['llama3_8b'].setChecked(True)
    w.hf_token_line.setText('hf_secret')
    assert w.session_hf_token() == 'hf_secret'
    # When a token is already detected, the input row hides → session empty.
    monkeypatch.setattr(m, 'has_hf_token', lambda: True)
    w.model_checkboxes['llama3_8b'].setChecked(False)
    w.model_checkboxes['llama32_3b'].setChecked(True)
    assert w.session_hf_token() == ''   # hidden → empty


def test_session_api_key_is_inert(qtbot):
    """OpenAI backend removed — session_api_key() is a permanent no-op kept
    only so gui/main_window.py's existing wiring stays valid."""
    from gui.section_word_features import WordFeaturesSection
    w = WordFeaturesSection(); qtbot.addWidget(w)
    assert w.session_api_key() == ""


# --- Project B: per-model context spinner ------------------------------------

def test_context_spin_persists_cap_below_native(qtbot, prefs_tmp):
    from gui.section_word_features import WordFeaturesSection
    from utils.preferences import load_preferences
    w = WordFeaturesSection(); qtbot.addWidget(w)
    w.context_spins["llama3_8b"].setValue(2048)
    assert load_preferences()["predictability"]["context"]["llama3_8b"] == 2048


def test_context_spin_at_native_is_stored_as_absence(qtbot, prefs_tmp):
    from gui.section_word_features import WordFeaturesSection
    from utils.preferences import load_preferences
    from extractors.predictability_models import spec_for
    w = WordFeaturesSection(); qtbot.addWidget(w)
    # Lower it, then raise back to native → the key is removed (default).
    w.context_spins["llama3_8b"].setValue(2048)
    w.context_spins["llama3_8b"].setValue(spec_for("llama3_8b").native_ctx)
    assert "llama3_8b" not in load_preferences()["predictability"]["context"]


def test_context_spin_range_capped_at_native(qtbot, prefs_tmp):
    from gui.section_word_features import WordFeaturesSection
    from extractors.predictability_models import spec_for, FLOOR_CONTEXT
    w = WordFeaturesSection(); qtbot.addWidget(w)
    spin = w.context_spins["gpt2"]
    assert spin.maximum() == spec_for("gpt2").native_ctx
    assert spin.minimum() == FLOOR_CONTEXT


def test_context_spin_always_enabled(qtbot, prefs_tmp):
    # Discoverability (same lesson as the HF-token box): a greyed spinner
    # reads as "broken". The spinner is always editable; the value simply
    # takes effect whenever its model is selected.
    from gui.section_word_features import WordFeaturesSection
    w = WordFeaturesSection(); qtbot.addWidget(w)
    assert w.context_spins["gpt2"].isEnabled() is True    # even unchecked
    w.model_checkboxes["gpt2"].setChecked(True)
    assert w.context_spins["gpt2"].isEnabled() is True


def test_metric_rows_have_info_icons_routing_to_docs(qtbot, prefs_tmp):
    """Regression (user report): the metric checkboxes lost the ⓘ doc links
    the old per-column ToggleRows auto-rendered. Each metric row must carry
    an InfoIcon targeting its feature doc, and the doc must resolve."""
    from gui.section_word_features import WordFeaturesSection
    from gui.info_popup import InfoIcon
    from gui.feature_docs import load_feature_doc
    w = WordFeaturesSection(); qtbot.addWidget(w)
    expected = {"surprisal": "pred_surprisal", "cloze": "pred_cloze",
                "topmatch": "pred_topmatch"}
    for metric, doc_key in expected.items():
        icon = w.metric_info_icons[metric]
        assert isinstance(icon, InfoIcon)
        assert icon._toggle_key == doc_key
        assert load_feature_doc(doc_key) is not None, doc_key


def test_context_spin_uses_adaptive_steps(qtbot, prefs_tmp):
    """Regression (user report): a fixed 128 step against a 131072 range made
    arrow clicks look like the box 'does nothing'. Steps must be adaptive."""
    from PySide6.QtWidgets import QSpinBox
    from gui.section_word_features import WordFeaturesSection
    w = WordFeaturesSection(); qtbot.addWidget(w)
    for slug, spin in w.context_spins.items():
        assert spin.stepType() == QSpinBox.StepType.AdaptiveDecimalStepType, slug
    # One arrow click from the 131072 default moves the value visibly.
    spin = w.context_spins["llama32_3b"]
    before = spin.value()
    spin.stepDown()
    assert before - spin.value() >= 1000  # ~10%, not a 0.1% nudge
