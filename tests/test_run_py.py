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


# ---------------------------------------------------------------------------
# Final-review Fix 2: run.py's raw validation must not sys.exit(1) on a
# pre-v20 features.yaml carrying a retired toggle — it must warn and still
# reach dispatch(), which is where load_preferences() migrates it away.
# ---------------------------------------------------------------------------

def test_main_migrates_v19_features_yaml_with_retired_phoneme_label(
        tmp_path, monkeypatch, capsys):
    import run
    _write_pair(tmp_path, monkeypatch)
    (tmp_path / FEATURES_YAML).write_text(
        "schema_version: 19\nfeatures:\n  feat_phoneme_label: true\n",
        encoding="utf-8")
    calls = {}
    monkeypatch.setattr(
        "main.run_word_level_batch",
        lambda *a, **k: calls.setdefault("word", True), raising=False)
    monkeypatch.setattr(
        "main.run_doc_level_batch",
        lambda *a, **k: calls.setdefault("doc", True), raising=False)
    monkeypatch.setattr("sys.argv", ["run.py"])
    run.main()  # must NOT raise SystemExit
    assert calls.get("word") is True
    assert "retired toggle 'feat_phoneme_label'" in capsys.readouterr().err


def test_main_migrates_features_yaml_with_missing_schema_version(
        tmp_path, monkeypatch, capsys):
    # A hand-stripped/never-set schema_version must be assumed v19 (not
    # "current"), else the v20 finalize block never runs and
    # feat_phoneme_label survives forever (changed stays False).
    import run
    _write_pair(tmp_path, monkeypatch)
    (tmp_path / FEATURES_YAML).write_text(
        "features:\n  feat_phoneme_label: true\n", encoding="utf-8")
    monkeypatch.setattr("main.run_word_level_batch",
                        lambda *a, **k: None, raising=False)
    monkeypatch.setattr("main.run_doc_level_batch",
                        lambda *a, **k: None, raising=False)
    monkeypatch.setattr("sys.argv", ["run.py"])
    run.main()  # must NOT raise SystemExit
    from utils.preferences import load_preferences
    prefs = load_preferences()
    assert "feat_phoneme_label" not in prefs["features"]
    assert prefs["features"]["feat_phoneme_arpabet"] is True


def test_main_migrates_features_yaml_with_quoted_schema_version(
        tmp_path, monkeypatch, capsys):
    # schema_version: '19' (a YAML string, e.g. from a hand-quoted edit) must
    # migrate the same as an int 19, not silently fall through to the v1
    # flat-json migration path (which would mangle the sectioned shape).
    import run
    _write_pair(tmp_path, monkeypatch)
    (tmp_path / FEATURES_YAML).write_text(
        "schema_version: '19'\nfeatures:\n  feat_phoneme_label: true\n",
        encoding="utf-8")
    monkeypatch.setattr("main.run_word_level_batch",
                        lambda *a, **k: None, raising=False)
    monkeypatch.setattr("main.run_doc_level_batch",
                        lambda *a, **k: None, raising=False)
    monkeypatch.setattr("sys.argv", ["run.py"])
    run.main()  # must NOT raise SystemExit
    from utils.preferences import load_preferences
    prefs = load_preferences()
    assert "feat_phoneme_label" not in prefs["features"]
    assert prefs["features"]["feat_phoneme_arpabet"] is True


def test_main_still_exits_on_a_genuine_typo_not_a_retired_key(
        tmp_path, monkeypatch, capsys):
    import run
    _write_pair(tmp_path, monkeypatch)
    (tmp_path / FEATURES_YAML).write_text(
        "features:\n  feat_phoneme_arpabetX: true\n", encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["run.py"])
    with pytest.raises(SystemExit) as ei:
        run.main()
    assert ei.value.code == 1
    assert "feat_phoneme_arpabetX" in capsys.readouterr().err


@pytest.mark.parametrize("name", [
    "sentence_timing_word_off", "phoneme_surprisal_mat_only",
    "word2vec_only", "bert_sentence_only"])
def test_dispatch_runs_word_batch_for_sidecar_only_levels(tmp_path, monkeypatch, name):
    """run.dispatch starts the word batch for every artifact it writes, not
    only the word / syllable / phoneme CSVs (scenarios shared with
    tests/gui/test_run_dispatch.py)."""
    import run
    from tests.gui.test_run_dispatch import SIDECAR_ONLY_SCENARIOS
    prefs_over, artifact, flag = SIDECAR_ONLY_SCENARIOS[name]
    calls = {}
    monkeypatch.setattr("main.run_word_level_batch",
                        lambda ns, **kw: calls.setdefault("word", ns))
    monkeypatch.setattr("main.run_doc_level_batch",
                        lambda *a, **k: calls.setdefault("doc", True))
    prefs = _write_pair(tmp_path, monkeypatch, features_overrides=prefs_over)
    run.dispatch(prefs)
    assert "word" in calls
    assert calls["word"].no_word_csv is True
    assert getattr(calls["word"], flag) is False
    assert "doc" not in calls
