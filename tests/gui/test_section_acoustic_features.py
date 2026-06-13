import pytest
pytest.importorskip("PySide6")

from gui.section_acoustic_features import AcousticFeaturesSection


def test_section_constructs_with_four_toggles(qtbot, tmp_path, monkeypatch):
    monkeypatch.setenv("SENSI_PREFS_PATH", str(tmp_path / "preferences.json"))
    sec = AcousticFeaturesSection()
    qtbot.addWidget(sec)
    for key in ("feat_acoustic_ts_envelope", "feat_acoustic_ts_onset_env",
                "feat_acoustic_ts_logmel", "feat_acoustic_ts_mfcc"):
        assert key in sec._toggles


def test_rate_change_persists(qtbot, tmp_path, monkeypatch):
    monkeypatch.setenv("SENSI_PREFS_PATH", str(tmp_path / "preferences.json"))
    from utils.preferences import load_preferences
    sec = AcousticFeaturesSection()
    qtbot.addWidget(sec)
    sec._rate_spin.setValue(64)
    assert load_preferences()["acoustic_ts"]["analysis_rate"] == 64
