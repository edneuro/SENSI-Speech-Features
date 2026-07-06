import pytest
pytest.importorskip("PySide6")

from gui.section_subword_features import SubwordFeaturesSection


def test_phoneme_surprisal_card_present(qtbot, tmp_path, monkeypatch):
    monkeypatch.setenv("SENSI_PREFS_PATH", str(tmp_path / "preferences.json"))
    w = SubwordFeaturesSection()
    qtbot.addWidget(w)
    assert w.toggle("feat_phoneme_surprisal") is not None
    assert w.toggle("feat_phoneme_cohort_entropy") is not None


def test_phoneme_surprisal_analysis_sr_persists(qtbot, tmp_path, monkeypatch):
    monkeypatch.setenv("SENSI_PREFS_PATH", str(tmp_path / "preferences.json"))
    from utils.preferences import load_preferences
    w = SubwordFeaturesSection()
    qtbot.addWidget(w)
    w._psr_sr_spin.setValue(64)
    assert load_preferences()["phoneme_surprisal"]["analysis_sr"] == 64
