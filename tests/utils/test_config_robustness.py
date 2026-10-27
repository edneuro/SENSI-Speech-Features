"""Config I/O robustness: save retry, visible save failures, warn-once, and
run.py's relative-path anchoring."""
import os
import sys

import pytest
import yaml

from sensi.utils import config_yaml, preferences


def test_replace_retries_a_transient_lock(tmp_path, monkeypatch):
    real_replace = os.replace
    calls = []

    def flaky(src, dst):
        calls.append(1)
        if len(calls) < 3:
            raise PermissionError("locked")
        real_replace(src, dst)

    monkeypatch.setattr(config_yaml.os, "replace", flaky)
    monkeypatch.setattr("time.sleep", lambda s: None)
    target = tmp_path / "features.yaml"
    config_yaml._write_text(target, "a: 1\n")
    assert target.read_text(encoding="utf-8") == "a: 1\n"
    assert len(calls) == 3


def test_failed_save_goes_to_the_registered_handler(monkeypatch):
    def boom(*a, **kw):
        raise PermissionError("still locked")

    monkeypatch.setattr(preferences, "write_features_yaml", boom)
    messages = []
    preferences.set_save_error_handler(messages.append)
    try:
        preferences.save_preferences({"prosody": {"pause_threshold_s": 0.3}})
    finally:
        preferences.set_save_error_handler(None)
    assert len(messages) == 1 and "Couldn't save your settings" in messages[0]
    with pytest.raises(PermissionError):
        preferences.save_preferences({"prosody": {"pause_threshold_s": 0.3}})


def test_unparseable_config_warns_once_per_session(capsys, monkeypatch):
    monkeypatch.setattr(preferences, "_WARNED_UNPARSEABLE", set())
    preferences.save_preferences({"prosody": {"pause_threshold_s": 0.3}})
    preferences.get_features_yaml_path().write_text("a: [unclosed\n", encoding="utf-8")
    for _ in range(5):
        preferences.load_preferences()
    assert capsys.readouterr().err.count("could not be parsed") == 1


def test_run_py_anchors_relative_folders_to_the_config_dir(tmp_path, monkeypatch):
    cfg = tmp_path / "cfg"
    cfg.mkdir()
    (cfg / "run.yaml").write_text(yaml.safe_dump({"shared": {
        "analyzer_name": "me",
        "last_transcripts_folder": "data/transcripts",
        "last_audio_folder": str(tmp_path / "abs_audio"),
        "last_output_folder": "../out",
    }}), encoding="utf-8")
    import run
    captured = {}
    monkeypatch.setattr(run, "dispatch", lambda prefs: captured.update(prefs["shared"]))
    monkeypatch.setattr(sys, "argv", ["run.py", "--config-dir", str(cfg)])
    monkeypatch.chdir(tmp_path / "..")
    run.main()
    assert captured["last_transcripts_folder"] == str(cfg / "data" / "transcripts")
    assert captured["last_output_folder"] == str(tmp_path / "out")
    assert captured["last_audio_folder"] == str(tmp_path / "abs_audio")   # absolute: unchanged
