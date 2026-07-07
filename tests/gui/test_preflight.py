"""Tests for gui/preflight.py.

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
    from utils.preferences import save_preferences
    from gui.preflight import compute_warnings
    # Need ewfg_freq in plan: word_level_csv + an EWFG band toggle on.
    save_preferences({
        "output": {"word_level_csv": True},
        "features": {"freq_ewfg_k2": True},
    })
    warnings = compute_warnings()
    # EWFG parquet is not bundled by default
    keys = {w.toggle_key for w in warnings}
    assert any(k is not None and k.startswith("freq_ewfg_") for k in keys)


def test_preflight_warns_on_openai_without_key(prefs_tmp):
    from utils.preferences import save_preferences
    from gui.preflight import compute_warnings
    # Need llm_predict in plan: word_level_csv + a predictability toggle on.
    save_preferences({
        "output": {"word_level_csv": True},
        "features": {"pred_cloze": True},
        "predictability": {"llm_model": "gpt-4o-mini"},
    })
    warnings = compute_warnings(session_api_key="")
    keys = {w.toggle_key for w in warnings}
    assert "pred_surprisal_llm" in keys or "pred_cloze" in keys


def test_preflight_no_warning_on_openai_with_key(prefs_tmp):
    from utils.preferences import save_preferences
    from gui.preflight import compute_warnings
    save_preferences({
        "output": {"word_level_csv": True},
        "features": {"pred_cloze": True},
        "predictability": {"llm_model": "gpt-4o-mini"},
    })
    warnings = compute_warnings(session_api_key="sk-fake")
    keys = {w.toggle_key for w in warnings}
    assert not any(k is not None and k.startswith("pred_") for k in keys)


def test_preflight_warns_on_gated_local_without_token(prefs_tmp, monkeypatch):
    import extractors.llm_cloze as m
    monkeypatch.setattr(m, 'has_hf_token', lambda: False)
    from utils.preferences import save_preferences
    from gui.preflight import compute_warnings
    save_preferences({
        "output": {"word_level_csv": True},
        "predictability": {"llm_model": "meta-llama/Meta-Llama-3-8B"},
        "features": {"pred_cloze": True},
    })
    warnings = compute_warnings(session_api_key="", session_hf_token="")
    msgs = [w.message for w in warnings if w.toggle_key == "pred_cloze"]
    assert any("HuggingFace token" in msg for msg in msgs)


def test_preflight_no_gated_warning_when_token_present(prefs_tmp, monkeypatch):
    import extractors.llm_cloze as m
    monkeypatch.setattr(m, 'has_hf_token', lambda: True)
    from utils.preferences import save_preferences
    from gui.preflight import compute_warnings
    save_preferences({
        "output": {"word_level_csv": True},
        "predictability": {"llm_model": "meta-llama/Meta-Llama-3-8B"},
        "features": {"pred_cloze": True},
    })
    warnings = compute_warnings(session_api_key="", session_hf_token="")
    assert not any("HuggingFace token" in w.message for w in warnings)


def test_preflight_no_gated_warning_when_session_token_pasted(prefs_tmp, monkeypatch):
    import extractors.llm_cloze as m
    monkeypatch.setattr(m, 'has_hf_token', lambda: False)
    from utils.preferences import save_preferences
    from gui.preflight import compute_warnings
    save_preferences({
        "output": {"word_level_csv": True},
        "predictability": {"llm_model": "meta-llama/Meta-Llama-3-8B"},
        "features": {"pred_cloze": True},
    })
    warnings = compute_warnings(session_api_key="", session_hf_token="hf_pasted")
    assert not any("HuggingFace token" in w.message for w in warnings)


def test_no_prosody_alignment_warning_anymore(prefs_tmp):
    """The old prosody-needs-alignment warning is gone — the planner resolves
    the dependency automatically. Verify no such warning is emitted even when
    the acoustic toggles are off and prosody is on."""
    from utils.preferences import save_preferences
    from gui.preflight import compute_warnings
    save_preferences({
        "output": {"word_level_csv": True},
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
    from utils.preferences import save_preferences
    from gui.preflight import compute_warnings
    # Need a prosody stage in plan: word_level_csv + prosody toggle on.
    save_preferences({
        "output": {"word_level_csv": True},
        "features": {"feat_prosody_pitch_mean": True},
    })
    warnings = compute_warnings()
    msgs = " ".join(w.message for w in warnings)
    assert "parselmouth" in msgs.lower()


def test_no_subword_alignment_warning_anymore(prefs_tmp):
    """The old sub-word-needs-alignment warning is gone — the planner resolves
    the dependency automatically. Verify no such warning is emitted."""
    from utils.preferences import save_preferences
    from gui.preflight import compute_warnings
    save_preferences({
        "output": {"syllables_csv": True},
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


def test_warns_when_pyphen_missing_and_syllable_label_on(prefs_tmp, monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "pyphen", None)
    from utils.preferences import save_preferences
    from gui.preflight import compute_warnings
    # Need syllabify_pairs in plan: syllables_csv + syllable label toggle on.
    save_preferences({
        "output": {"syllables_csv": True},
        "features": {"feat_syllable_label": True},
    })
    warnings = compute_warnings()
    msgs = " ".join(w.message for w in warnings)
    assert "pyphen not installed" in msgs
