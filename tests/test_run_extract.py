import os
from utils.preferences import DEFAULT_PREFS
from utils.config_yaml import prefs_to_config


def _base_config(tmp_path):
    cfg = prefs_to_config(DEFAULT_PREFS)
    cfg["run"].update(
        transcripts_folder=str(tmp_path / "t"),
        audio_folder=str(tmp_path / "a"),
        output_folder=str(tmp_path / "o"),
        analyzer="tester",
    )
    return cfg


def test_load_credentials_sets_env_from_files(tmp_path, monkeypatch):
    import run_extract
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    hf = tmp_path / "hf.txt"; hf.write_text("hf_secret\n", encoding="utf-8")
    run_extract.load_credentials({"hf_token_file": str(hf)})
    assert os.environ["HF_TOKEN"] == "hf_secret"


def test_load_credentials_missing_file_is_noop(tmp_path, monkeypatch):
    import run_extract
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    run_extract.load_credentials({"openai_api_key_file": str(tmp_path / "nope.txt")})
    assert "OPENAI_API_KEY" not in os.environ


def test_dispatch_runs_word_level_from_output_toggle(tmp_path, monkeypatch):
    import run_extract
    calls = {}
    def fake_word(ns, **kw):
        calls["word"] = (ns.transcripts_folder, ns.analyzer)
    monkeypatch.setattr("main.run_word_level_batch", fake_word, raising=False)
    monkeypatch.setattr("main.run_doc_level_batch", lambda *a, **k: calls.setdefault("doc", True), raising=False)
    cfg = _base_config(tmp_path)
    # Ensure a word-level output is on and doc is off.
    cfg["features"]["word_level_csv"] = True
    cfg["features"]["document_summary"] = False
    run_extract.dispatch(cfg)
    assert calls["word"] == (str(tmp_path / "t"), "tester")
    assert "doc" not in calls
