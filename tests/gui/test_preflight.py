def test_preflight_warns_on_missing_ewfg(monkeypatch, tmp_path):
    monkeypatch.setenv("SENSI_PREFS_PATH", str(tmp_path / "prefs.json"))
    from gui.preflight import compute_warnings
    warnings = compute_warnings()
    # EWFG parquet is not bundled by default
    keys = {w.toggle_key for w in warnings}
    assert any(k.startswith("freq_ewfg_") for k in keys)


def test_preflight_warns_on_openai_without_key(monkeypatch, tmp_path):
    monkeypatch.setenv("SENSI_PREFS_PATH", str(tmp_path / "prefs.json"))
    # Force OpenAI model selection
    from utils.preferences import save_preferences
    save_preferences({"predictability": {"llm_model": "gpt-4o-mini"}})
    from gui.preflight import compute_warnings
    warnings = compute_warnings(session_api_key="")
    keys = {w.toggle_key for w in warnings}
    assert "pred_surprisal_llm" in keys or "pred_cloze" in keys


def test_preflight_no_warning_on_openai_with_key(tmp_path, monkeypatch):
    monkeypatch.setenv("SENSI_PREFS_PATH", str(tmp_path / "prefs.json"))
    from utils.preferences import save_preferences
    save_preferences({"predictability": {"llm_model": "gpt-4o-mini"}})
    from gui.preflight import compute_warnings
    warnings = compute_warnings(session_api_key="sk-fake")
    keys = {w.toggle_key for w in warnings}
    assert not any(k.startswith("pred_") for k in keys)


def test_prosody_warns_when_alignment_off(prefs_tmp):
    from utils.preferences import save_preferences
    from gui.preflight import compute_warnings
    # Alignment "off" means ALL 4 acoustic toggles are off (run_dispatch
    # enables alignment when any of them is on).
    save_preferences({"features": {
        "feat_acoustic_onset": False,
        "feat_acoustic_end": False,
        "feat_acoustic_word_duration": False,
        "feat_acoustic_word_gap": False,
        "feat_prosody_pitch_mean": True,
    }})
    warnings = compute_warnings()
    keys = [w.toggle_key for w in warnings]
    assert "feat_prosody_pitch_mean" in keys


def test_prosody_does_not_warn_when_any_acoustic_toggle_on(prefs_tmp):
    """Regression: prosody only needs alignment to run, which happens if any
    of the 4 acoustic toggles is on — not specifically feat_acoustic_onset."""
    from utils.preferences import save_preferences
    from gui.preflight import compute_warnings
    save_preferences({"features": {
        "feat_acoustic_onset": False,
        "feat_acoustic_end": False,
        "feat_acoustic_word_duration": False,
        "feat_acoustic_word_gap": True,  # only one on
        "feat_prosody_pitch_mean": True,
    }})
    warnings = compute_warnings()
    msgs = [w.message for w in warnings if w.toggle_key == "feat_prosody_pitch_mean"]
    assert not any("requires MFA alignment" in m for m in msgs)


def test_prosody_warns_when_parselmouth_missing(prefs_tmp, monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "parselmouth", None)
    from utils.preferences import save_preferences
    from gui.preflight import compute_warnings
    save_preferences({"features": {
        "feat_prosody_pitch_mean": True,
        "feat_acoustic_onset": True,
    }})
    warnings = compute_warnings()
    msgs = " ".join(w.message for w in warnings)
    assert "parselmouth" in msgs.lower()


def test_warns_when_word_categories_selected_but_sem_indicators_off(prefs_tmp):
    """Regression: selecting categories at word level without enabling
    the parent Semantic indicators toggle silently produces no Sem_*
    columns. Preflight should surface this so the user notices."""
    from utils.preferences import save_preferences
    from gui.preflight import compute_warnings
    save_preferences({
        "features": {"sem_indicators": False},
        "semantic": {
            "backend": "empath",
            "word_categories_empath": ["work", "family", "money"],
        },
    })
    warnings = compute_warnings()
    matches = [w for w in warnings
               if w.toggle_key == "sem_indicators"
               and "Semantic indicators is off" in w.message]
    assert matches, "expected a warning attached to sem_indicators"


def test_no_warning_when_word_categories_match_sem_indicators_on(prefs_tmp):
    from utils.preferences import save_preferences
    from gui.preflight import compute_warnings
    save_preferences({
        "features": {"sem_indicators": True},
        "semantic": {
            "backend": "empath",
            "word_categories_empath": ["work"],
        },
    })
    warnings = compute_warnings()
    matches = [w for w in warnings
               if w.toggle_key == "sem_indicators"
               and "Semantic indicators is off" in w.message]
    assert not matches


def test_warns_when_doc_categories_selected_but_proportions_off(prefs_tmp):
    """Mirror of the word-level case: doc-level category selections do
    nothing if doc_category_proportions is off."""
    from utils.preferences import save_preferences
    from gui.preflight import compute_warnings
    save_preferences({
        "output": {"doc_category_proportions": False},
        "semantic": {
            "backend": "empath",
            "doc_categories_empath": ["violence"],
        },
    })
    warnings = compute_warnings()
    matches = [w for w in warnings
               if w.toggle_key == "output_doc_category_proportions"
               and "Category proportions is off" in w.message]
    assert matches


def test_no_warning_when_categories_empty(prefs_tmp):
    """Zero categories selected = no mismatch, no warning, even when
    the parent toggle is off."""
    from utils.preferences import save_preferences
    from gui.preflight import compute_warnings
    save_preferences({
        "features": {"sem_indicators": False},
        "output": {"doc_category_proportions": False},
        "semantic": {
            "backend": "empath",
            "word_categories_empath": [],
            "doc_categories_empath": [],
        },
    })
    warnings = compute_warnings()
    msgs = " ".join(w.message for w in warnings)
    assert "Semantic indicators is off" not in msgs
    assert "Category proportions is off" not in msgs


def test_warns_when_subword_features_on_but_alignment_off(prefs_tmp):
    from utils.preferences import save_preferences
    from gui.preflight import compute_warnings
    save_preferences({"features": {
        "feat_syllable_onset": True,
        # All acoustic off
        "feat_acoustic_onset": False, "feat_acoustic_end": False,
        "feat_acoustic_word_duration": False, "feat_acoustic_word_gap": False,
    }})
    warnings = compute_warnings()
    msgs = " ".join(w.message for w in warnings)
    assert "Sub-word features require MFA alignment" in msgs


def test_no_subword_warning_when_acoustic_on(prefs_tmp):
    from utils.preferences import save_preferences
    from gui.preflight import compute_warnings
    save_preferences({"features": {
        "feat_syllable_onset": True,
        "feat_acoustic_onset": True,  # one acoustic on is enough
    }})
    warnings = compute_warnings()
    msgs = " ".join(w.message for w in warnings)
    assert "Sub-word features require MFA alignment" not in msgs


def test_warns_when_pyphen_missing_and_syllable_label_on(prefs_tmp, monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "pyphen", None)
    from utils.preferences import save_preferences
    from gui.preflight import compute_warnings
    save_preferences({"features": {"feat_syllable_label": True}})
    warnings = compute_warnings()
    msgs = " ".join(w.message for w in warnings)
    assert "pyphen not installed" in msgs
