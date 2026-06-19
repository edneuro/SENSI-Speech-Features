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


def test_param_greying_follows_feature_toggles(qtbot, tmp_path, monkeypatch):
    monkeypatch.setenv("SENSI_PREFS_PATH", str(tmp_path / "preferences.json"))
    sec = AcousticFeaturesSection()
    qtbot.addWidget(sec)

    # Envelope + onset off -> env params disabled.
    sec._toggles["feat_acoustic_ts_envelope"].set_checked(False)
    sec._toggles["feat_acoustic_ts_onset_env"].set_checked(False)
    assert not sec._env_params_w.isEnabled()

    # Onset back on -> env params re-enabled (onset consumes them).
    sec._toggles["feat_acoustic_ts_onset_env"].set_checked(True)
    assert sec._env_params_w.isEnabled()

    # Log-mel on, MFCC off -> mel params enabled, n_mfcc disabled.
    sec._toggles["feat_acoustic_ts_logmel"].set_checked(True)
    sec._toggles["feat_acoustic_ts_mfcc"].set_checked(False)
    assert sec._mel_params_w.isEnabled()
    assert not sec._mfcc_params_w.isEnabled()

    # Both log-mel and MFCC off -> mel params disabled.
    sec._toggles["feat_acoustic_ts_logmel"].set_checked(False)
    assert not sec._mel_params_w.isEnabled()
