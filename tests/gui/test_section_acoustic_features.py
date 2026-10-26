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
    from utils.preferences import save_preferences
    # The card (and so every param) is greyed while the Acoustic level is
    # off — the fresh-install default. Include it to test per-feature greying.
    save_preferences({"output": {"level_acoustic": True}})
    sec = AcousticFeaturesSection()
    qtbot.addWidget(sec)

    # With every feature on, the constructor's _sync_param_enabled() leaves all
    # param groups enabled.
    for k in ("feat_acoustic_ts_envelope", "feat_acoustic_ts_onset_env",
              "feat_acoustic_ts_logmel", "feat_acoustic_ts_mfcc"):
        sec._toggles[k].set_checked(True)
    assert sec._env_params_w.isEnabled()
    assert sec._mel_params_w.isEnabled()
    assert sec._mfcc_params_w.isEnabled()

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


def test_refresh_from_prefs_mirrors_run_tab_level_switch(qtbot, tmp_path, monkeypatch):
    monkeypatch.setenv("SENSI_PREFS_PATH", str(tmp_path / "preferences.json"))
    from utils.preferences import save_preferences
    save_preferences({"output": {"level_acoustic": False}})
    sec = AcousticFeaturesSection()
    qtbot.addWidget(sec)
    assert not sec._card.master_toggle().is_checked()
    assert not sec._rate_spin.isEnabled()
    # The Run tab writes the same key; the page mirrors it on refresh.
    save_preferences({"output": {"level_acoustic": True}})
    sec.refresh_from_prefs()
    assert sec._card.master_toggle().is_checked()
    assert sec._rate_spin.isEnabled()
