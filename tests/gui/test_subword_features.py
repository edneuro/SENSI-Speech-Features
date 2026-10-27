import pytest
pytest.importorskip("PySide6")

from sensi.gui.section_subword_features import SubwordFeaturesSection


def test_phoneme_surprisal_card_present(qtbot, tmp_path, monkeypatch):
    monkeypatch.setenv("SENSI_PREFS_PATH", str(tmp_path / "preferences.json"))
    w = SubwordFeaturesSection()
    qtbot.addWidget(w)
    assert w.toggle("feat_phoneme_surprisal") is not None
    assert w.toggle("feat_phoneme_cohort_entropy") is not None


def test_phoneme_surprisal_analysis_sr_persists(qtbot, tmp_path, monkeypatch):
    monkeypatch.setenv("SENSI_PREFS_PATH", str(tmp_path / "preferences.json"))
    from sensi.utils.preferences import load_preferences
    w = SubwordFeaturesSection()
    qtbot.addWidget(w)
    w._psr_sr_spin.setValue(64)
    assert load_preferences()["phoneme_surprisal"]["analysis_sr"] == 64


def test_phoneme_card_shows_four_label_toggles(qtbot, tmp_path, monkeypatch):
    monkeypatch.setenv("SENSI_PREFS_PATH", str(tmp_path / "preferences.json"))
    w = SubwordFeaturesSection()
    qtbot.addWidget(w)
    keys = set(w._toggles.keys())
    assert {"feat_phoneme_arpabet", "feat_phoneme_stress",
            "feat_phoneme_ipa", "feat_phoneme_pron_source",
            "feat_syllable_ipa"} <= keys
    assert "feat_phoneme_label" not in keys


def test_g2p_oov_toggle_writes_to_alignment_section(qtbot, tmp_path, monkeypatch):
    """g2p_oov must round-trip through prefs["alignment"], not prefs["features"] —
    a toggle wired to the wrong section would silently do nothing."""
    monkeypatch.setenv("SENSI_PREFS_PATH", str(tmp_path / "preferences.json"))
    from sensi.utils.preferences import load_preferences
    w = SubwordFeaturesSection()
    qtbot.addWidget(w)
    row = w.toggle("g2p_oov")
    assert row is not None
    row.set_checked(True)
    row.toggled.emit(True)
    prefs = load_preferences()
    assert prefs["alignment"]["g2p_oov"] is True
    assert "g2p_oov" not in prefs["features"]
