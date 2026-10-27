"""Tests for sensi/gui/preflight.py.

Group A (planner-aware, kept): env/asset warnings that still apply when
  the relevant stage is actually in the plan.
Group B (deleted — toggle-dep warnings removed):
  - test_prosody_warns_when_alignment_off: INVERTED to
    test_no_prosody_alignment_warning_anymore (the planner resolves the dep).
  - test_prosody_does_not_warn_when_any_acoustic_toggle_on: DELETED
    (trivially passing, no longer documents anything meaningful).
  - test_warns_when_word_categories_selected_but_sem_indicators_off: DELETED
    (Task 7 replaces with picker UX disabling).
  - test_no_warning_when_word_categories_match_sem_indicators_on: DELETED
    (complement of the deleted warning).
  - test_warns_when_doc_categories_selected_but_proportions_off: DELETED
    (same as above).
  - test_no_warning_when_categories_empty: DELETED (same reason).
  - test_warns_when_subword_features_on_but_alignment_off: INVERTED to
    test_no_subword_alignment_warning_anymore.
  - test_no_subword_warning_when_acoustic_on: DELETED (complement of
    deleted warning).
"""


def test_preflight_warns_on_missing_ewfg(prefs_tmp):
    from sensi.utils.preferences import save_preferences
    from sensi.gui.preflight import compute_warnings
    # Need ewfg_freq in plan: word_level_csv + an EWFG band toggle on.
    save_preferences({
        "output": {"level_word": True},
        "features": {"freq_ewfg_k2": True},
    })
    warnings = compute_warnings()
    # EWFG parquet is not bundled by default
    keys = {w.toggle_key for w in warnings}
    assert any(k is not None and k.startswith("freq_ewfg_") for k in keys)


def test_preflight_ignores_unrecognized_model_slug(prefs_tmp):
    """A stale/unrecognized slug (e.g. left over from a hand-edited prefs
    file, or an old OpenAI id from before the predictability unification)
    must not crash preflight — is_gated() would KeyError on it, so it's
    simply skipped for the gating check rather than raising."""
    from sensi.utils.preferences import save_preferences
    from sensi.gui.preflight import compute_warnings
    save_preferences({
        "output": {"level_word": True},
        "predictability": {"models": ["gpt-4o-mini"],
                            "metrics": {"surprisal": True, "cloze": False, "topmatch": False}},
    })
    warnings = compute_warnings()  # must not raise
    assert not any("HuggingFace token" in w.message for w in warnings)


def test_preflight_warns_on_gated_model_without_token(prefs_tmp, monkeypatch):
    import sensi.extractors.predictability as m
    monkeypatch.setattr(m, 'has_hf_token', lambda: False)
    from sensi.utils.preferences import save_preferences
    from sensi.gui.preflight import compute_warnings
    save_preferences({
        "output": {"level_word": True},
        "predictability": {"models": ["llama3_8b"],
                            "metrics": {"surprisal": True, "cloze": False, "topmatch": False}},
    })
    warnings = compute_warnings(session_hf_token="")
    assert any("HuggingFace token" in w.message for w in warnings)


def test_preflight_no_gated_warning_when_token_present(prefs_tmp, monkeypatch):
    import sensi.extractors.predictability as m
    monkeypatch.setattr(m, 'has_hf_token', lambda: True)
    from sensi.utils.preferences import save_preferences
    from sensi.gui.preflight import compute_warnings
    save_preferences({
        "output": {"level_word": True},
        "predictability": {"models": ["llama3_8b"],
                            "metrics": {"surprisal": True, "cloze": False, "topmatch": False}},
    })
    warnings = compute_warnings(session_hf_token="")
    assert not any("HuggingFace token" in w.message for w in warnings)


def test_preflight_no_gated_warning_when_session_token_pasted(prefs_tmp, monkeypatch):
    import sensi.extractors.predictability as m
    monkeypatch.setattr(m, 'has_hf_token', lambda: False)
    from sensi.utils.preferences import save_preferences
    from sensi.gui.preflight import compute_warnings
    save_preferences({
        "output": {"level_word": True},
        "predictability": {"models": ["llama3_8b"],
                            "metrics": {"surprisal": True, "cloze": False, "topmatch": False}},
    })
    warnings = compute_warnings(session_hf_token="hf_pasted")
    assert not any("HuggingFace token" in w.message for w in warnings)


def test_preflight_no_gated_warning_for_free_model(prefs_tmp, monkeypatch):
    """GPT-2 is the roster's only ungated model — selecting it must never
    trigger the HuggingFace-token warning, even with no token available."""
    import sensi.extractors.predictability as m
    monkeypatch.setattr(m, 'has_hf_token', lambda: False)
    from sensi.utils.preferences import save_preferences
    from sensi.gui.preflight import compute_warnings
    save_preferences({
        "output": {"level_word": True},
        "predictability": {"models": ["gpt2"],
                            "metrics": {"surprisal": True, "cloze": False, "topmatch": False}},
    })
    warnings = compute_warnings(session_hf_token="")
    assert not any("HuggingFace token" in w.message for w in warnings)


def test_preflight_warns_models_without_metrics(prefs_tmp):
    """A model is picked but no metric is turned on — nothing will be
    produced; preflight should say so."""
    from sensi.utils.preferences import save_preferences
    from sensi.gui.preflight import compute_warnings
    save_preferences({
        "output": {"level_word": True},
        "predictability": {"models": ["gpt2"],
                            "metrics": {"surprisal": False, "cloze": False, "topmatch": False}},
    })
    warnings = compute_warnings()
    assert any("nothing will be produced" in w.message.lower() for w in warnings)


def test_preflight_warns_metrics_without_models(prefs_tmp):
    """A metric is turned on but no model is picked — the mirror case."""
    from sensi.utils.preferences import save_preferences
    from sensi.gui.preflight import compute_warnings
    save_preferences({
        "output": {"level_word": True},
        "predictability": {"models": [],
                            "metrics": {"surprisal": True, "cloze": False, "topmatch": False}},
    })
    warnings = compute_warnings()
    assert any("nothing will be produced" in w.message.lower() for w in warnings)


def test_preflight_no_predictability_warnings_when_fully_configured(prefs_tmp, monkeypatch):
    import sensi.extractors.predictability as m
    monkeypatch.setattr(m, 'has_hf_token', lambda: True)
    from sensi.utils.preferences import save_preferences
    from sensi.gui.preflight import compute_warnings
    save_preferences({
        "output": {"level_word": True},
        "predictability": {"models": ["gpt2"],
                            "metrics": {"surprisal": True, "cloze": False, "topmatch": False}},
    })
    warnings = compute_warnings()
    assert not any(
        "nothing will be produced" in w.message.lower() or "HuggingFace token" in w.message
        for w in warnings)


def test_no_prosody_alignment_warning_anymore(prefs_tmp):
    """The old prosody-needs-alignment warning is gone — the planner resolves
    the dependency automatically. Verify no such warning is emitted even when
    the acoustic toggles are off and prosody is on."""
    from sensi.utils.preferences import save_preferences
    from sensi.gui.preflight import compute_warnings
    save_preferences({
        "output": {"level_word": True},
        "features": {
            "feat_acoustic_onset": False,
            "feat_acoustic_end": False,
            "feat_acoustic_word_duration": False,
            "feat_acoustic_word_gap": False,
            "feat_prosody_pitch_mean": True,
        },
    })
    warnings = compute_warnings()
    assert not any("enable" in w.message.lower() for w in warnings)
    assert not any("requires MFA alignment" in w.message for w in warnings)


def test_prosody_warns_when_parselmouth_missing(prefs_tmp, monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "parselmouth", None)
    from sensi.utils.preferences import save_preferences
    from sensi.gui.preflight import compute_warnings
    # Need a prosody stage in plan: word_level_csv + prosody toggle on.
    save_preferences({
        "output": {"level_word": True},
        "features": {"feat_prosody_pitch_mean": True},
    })
    warnings = compute_warnings()
    msgs = " ".join(w.message for w in warnings)
    assert "parselmouth" in msgs.lower()


def test_no_subword_alignment_warning_anymore(prefs_tmp):
    """The old sub-word-needs-alignment warning is gone — the planner resolves
    the dependency automatically. Verify no such warning is emitted."""
    from sensi.utils.preferences import save_preferences
    from sensi.gui.preflight import compute_warnings
    save_preferences({
        "output": {"level_subword": True},
        "features": {
            "feat_syllable_onset": True,
            "feat_acoustic_onset": False,
            "feat_acoustic_end": False,
            "feat_acoustic_word_duration": False,
            "feat_acoustic_word_gap": False,
        },
    })
    warnings = compute_warnings()
    assert not any("Sub-word features require MFA alignment" in w.message
                   for w in warnings)
    assert not any("enable" in w.message.lower() for w in warnings)


def test_warns_when_g2p_oov_on_but_model_missing(prefs_tmp, monkeypatch):
    """Final-review Fix 4: enabling alignment.g2p_oov without the vendored
    MFA G2P model silently degrades every OOV word to a single spn phone
    (sensi/extractors/aligner._run_mfa) with only a run-time stderr line. This is
    the preflight-time equivalent of the existing MFA-on-PATH check."""
    import sensi.extractors.aligner as al
    monkeypatch.setattr(al, "g2p_model_available", lambda: False)
    from sensi.utils.preferences import save_preferences
    from sensi.gui.preflight import compute_warnings
    save_preferences({
        "output": {"level_word": True},
        "features": {"feat_acoustic_onset": True},
        "alignment": {"g2p_oov": True},
    })
    warnings = compute_warnings()
    msgs = " ".join(w.message for w in warnings)
    assert "g2p" in msgs.lower() and "spn" in msgs.lower()


def test_no_g2p_warning_when_model_present(prefs_tmp, monkeypatch):
    import sensi.extractors.aligner as al
    monkeypatch.setattr(al, "g2p_model_available", lambda: True)
    from sensi.utils.preferences import save_preferences
    from sensi.gui.preflight import compute_warnings
    save_preferences({
        "output": {"level_word": True},
        "features": {"feat_acoustic_onset": True},
        "alignment": {"g2p_oov": True},
    })
    warnings = compute_warnings()
    assert not any("g2p" in w.message.lower() for w in warnings)


def test_no_g2p_warning_when_toggle_off(prefs_tmp, monkeypatch):
    import sensi.extractors.aligner as al
    monkeypatch.setattr(al, "g2p_model_available", lambda: False)
    from sensi.utils.preferences import save_preferences
    from sensi.gui.preflight import compute_warnings
    save_preferences({
        "output": {"level_word": True},
        "features": {"feat_acoustic_onset": True},
        "alignment": {"g2p_oov": False},
    })
    warnings = compute_warnings()
    assert not any("g2p" in w.message.lower() for w in warnings)


def test_warns_when_pyphen_missing_and_syllable_label_on(prefs_tmp, monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "pyphen", None)
    from sensi.utils.preferences import save_preferences
    from sensi.gui.preflight import compute_warnings
    # Need syllabify_pairs in plan: syllables_csv + syllable label toggle on.
    save_preferences({
        "output": {"level_subword": True},
        "features": {"feat_syllable_label": True},
    })
    warnings = compute_warnings()
    msgs = " ".join(w.message for w in warnings)
    assert "pyphen not installed" in msgs


def test_gated_warning_for_doc_predictability_means_only(prefs_tmp, monkeypatch):
    """Word level off, document predictability means on: the model still
    runs (doc path), so the gated-model warning must fire."""
    import sensi.extractors.predictability as m
    monkeypatch.setattr(m, 'has_hf_token', lambda: False)
    from sensi.utils.preferences import save_preferences
    from sensi.gui.preflight import compute_warnings
    save_preferences({
        "output": {"level_word": False, "level_document": True,
                   "doc_agg_predictability": True},
        "predictability": {"models": ["llama3_8b"],
                            "metrics": {"surprisal": True, "cloze": False, "topmatch": False}},
    })
    warnings = compute_warnings(session_hf_token="")
    assert any("HuggingFace token" in w.message for w in warnings)


def test_no_predictability_warnings_when_section_off(prefs_tmp, monkeypatch):
    """Word level on but the Predictability section off and no doc means:
    nothing loads a model, so no token or mismatch warning."""
    import sensi.extractors.predictability as m
    monkeypatch.setattr(m, 'has_hf_token', lambda: False)
    from sensi.utils.preferences import save_preferences
    from sensi.gui.preflight import compute_warnings
    save_preferences({
        "output": {"level_word": True},
        "features": {"master_predictability": False},
        "predictability": {"models": ["llama3_8b"],
                            "metrics": {"surprisal": False, "cloze": False, "topmatch": False}},
    })
    warnings = compute_warnings(session_hf_token="")
    assert not any(
        "nothing will be produced" in w.message.lower() or "HuggingFace token" in w.message
        for w in warnings)


def test_mismatch_warning_for_doc_predictability_means_only(prefs_tmp):
    """Doc means on (Word level off) with a model but no metric: the doc
    means would be empty, so the mismatch warning fires."""
    from sensi.utils.preferences import save_preferences
    from sensi.gui.preflight import compute_warnings
    save_preferences({
        "output": {"level_word": False, "level_document": True,
                   "doc_agg_predictability": True},
        "predictability": {"models": ["gpt2"],
                            "metrics": {"surprisal": False, "cloze": False, "topmatch": False}},
    })
    warnings = compute_warnings()
    assert any("nothing will be produced" in w.message.lower() for w in warnings)
