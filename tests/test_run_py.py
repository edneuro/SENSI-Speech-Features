import os
from copy import deepcopy
import pytest

from utils.preferences import DEFAULT_PREFS
from utils.config_yaml import (FEATURES_YAML, RUN_YAML, split_prefs,
                               write_features_yaml, write_run_yaml)


def _write_pair(tmp_path, monkeypatch, features_overrides=None, **shared_overrides):
    """Write a features.yaml + run.yaml pair into tmp_path and point the app at it."""
    monkeypatch.setenv("SENSI_CONFIG_DIR", str(tmp_path))
    prefs = deepcopy(DEFAULT_PREFS)
    prefs["shared"].update(
        last_transcripts_folder=str(tmp_path / "t"),
        last_audio_folder=str(tmp_path / "a"),
        last_output_folder=str(tmp_path / "o"),
        analyzer_name="tester",
    )
    prefs["shared"].update(shared_overrides)
    for section, values in (features_overrides or {}).items():
        prefs[section].update(values)
    features, run_data = split_prefs(prefs)
    write_features_yaml(tmp_path / FEATURES_YAML, features)
    write_run_yaml(tmp_path / RUN_YAML, run_data)
    return prefs


def test_load_credentials_sets_hf_token_from_file(tmp_path, monkeypatch):
    import run
    monkeypatch.delenv("HF_TOKEN", raising=False)
    hf = tmp_path / "hf.txt"; hf.write_text("hf_secret\n", encoding="utf-8")
    run.load_credentials({"hf_token_file": str(hf)})
    assert os.environ["HF_TOKEN"] == "hf_secret"


def test_load_credentials_missing_file_is_noop(tmp_path, monkeypatch):
    import run
    monkeypatch.delenv("HF_TOKEN", raising=False)
    run.load_credentials({"hf_token_file": str(tmp_path / "nope.txt")})
    assert "HF_TOKEN" not in os.environ


def test_dispatch_runs_word_level_from_output_toggle(tmp_path, monkeypatch):
    import run
    calls = {}
    def fake_word(ns, **kw):
        calls["word"] = (ns.transcripts_folder, ns.analyzer)
    monkeypatch.setattr("main.run_word_level_batch", fake_word, raising=False)
    monkeypatch.setattr("main.run_doc_level_batch",
                        lambda *a, **k: calls.setdefault("doc", True), raising=False)
    prefs = _write_pair(tmp_path, monkeypatch, features_overrides={
        "output": {"word_level_csv": True, "document_summary": False}})
    run.dispatch(prefs)
    assert calls["word"] == (str(tmp_path / "t"), "tester")
    assert "doc" not in calls


def test_main_exits_when_required_folder_missing(tmp_path, monkeypatch, capsys):
    import run
    _write_pair(tmp_path, monkeypatch, last_transcripts_folder="")
    monkeypatch.setattr("sys.argv", ["run.py"])
    with pytest.raises(SystemExit) as ei:
        run.main()
    assert ei.value.code == 1
    assert "last_transcripts_folder" in capsys.readouterr().err


def test_main_exits_on_invalid_features_yaml(tmp_path, monkeypatch, capsys):
    import run
    _write_pair(tmp_path, monkeypatch)
    (tmp_path / FEATURES_YAML).write_text(
        "features:\n  not_a_real_toggle: true\n", encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["run.py"])
    with pytest.raises(SystemExit) as ei:
        run.main()
    assert ei.value.code == 1
    assert "not_a_real_toggle" in capsys.readouterr().err


def test_main_exits_on_non_mapping_features_yaml_root(tmp_path, monkeypatch, capsys):
    # Fix 4: a YAML list (or other non-mapping) root must not silently
    # validate clean and run with defaults — read_yaml's tolerant coercion
    # to {} is bypassed for run.py's own validation pass.
    import run
    _write_pair(tmp_path, monkeypatch)
    (tmp_path / FEATURES_YAML).write_text("- a\n- b\n", encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["run.py"])
    with pytest.raises(SystemExit) as ei:
        run.main()
    assert ei.value.code == 1
    err = capsys.readouterr().err
    assert "must be a mapping" in err


def test_main_warns_when_features_yaml_missing(tmp_path, monkeypatch, capsys):
    # Fix 5: a present run.yaml + absent features.yaml must not run the
    # minimal opt-in defaults silently — warn to stderr but still proceed.
    import run
    _write_pair(tmp_path, monkeypatch)
    (tmp_path / FEATURES_YAML).unlink()
    monkeypatch.setattr("main.run_word_level_batch",
                        lambda *a, **k: None, raising=False)
    monkeypatch.setattr("main.run_doc_level_batch",
                        lambda *a, **k: None, raising=False)
    monkeypatch.setattr("sys.argv", ["run.py"])
    run.main()
    err = capsys.readouterr().err
    assert "features.yaml not found" in err
