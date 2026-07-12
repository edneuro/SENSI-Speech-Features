import os
from pathlib import Path

from utils.credentials import (
    resolve_hf_token_path, read_hf_token, export_hf_token, save_hf_token,
)
from utils.preferences import DEFAULT_PREFS, load_preferences


def test_default_hf_token_file_is_relative_conventional_name():
    assert DEFAULT_PREFS["shared"]["hf_token_file"] == "hf_token.txt"


def test_resolve_relative_against_config_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("SENSI_CONFIG_DIR", str(tmp_path))
    assert resolve_hf_token_path({"hf_token_file": "hf_token.txt"}) == tmp_path / "hf_token.txt"


def test_resolve_absolute_and_vars_and_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("SENSI_CONFIG_DIR", str(tmp_path))
    absolute = tmp_path / "elsewhere" / "tok.txt"
    assert resolve_hf_token_path({"hf_token_file": str(absolute)}) == absolute
    monkeypatch.setenv("TOKDIR", str(tmp_path / "vardir"))
    assert resolve_hf_token_path({"hf_token_file": "$TOKDIR/t.txt"}) == tmp_path / "vardir" / "t.txt"
    assert resolve_hf_token_path({"hf_token_file": ""}) is None
    assert resolve_hf_token_path({}) is None


def test_read_hf_token_missing_unreadable_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("SENSI_CONFIG_DIR", str(tmp_path))
    shared = {"hf_token_file": "hf_token.txt"}
    assert read_hf_token(shared) == ""                       # missing file
    (tmp_path / "hf_token.txt").write_text("  \n", encoding="utf-8")
    assert read_hf_token(shared) == ""                       # whitespace only
    (tmp_path / "hf_token.txt").write_text("hf_secret\n", encoding="utf-8")
    assert read_hf_token(shared) == "hf_secret"


def test_export_hf_token_sets_env_and_never_clobbers(tmp_path, monkeypatch):
    monkeypatch.setenv("SENSI_CONFIG_DIR", str(tmp_path))
    monkeypatch.delenv("HF_TOKEN", raising=False)
    shared = {"hf_token_file": "hf_token.txt"}
    assert export_hf_token(shared) is False                  # no file → no export
    assert "HF_TOKEN" not in os.environ
    (tmp_path / "hf_token.txt").write_text("hf_file\n", encoding="utf-8")
    assert export_hf_token(shared) is True
    assert os.environ["HF_TOKEN"] == "hf_file"
    monkeypatch.setenv("HF_TOKEN", "hf_session")
    assert export_hf_token(shared) is False                  # existing env wins
    assert os.environ["HF_TOKEN"] == "hf_session"


def test_save_hf_token_writes_file_and_points_pref_at_it(tmp_path, monkeypatch):
    monkeypatch.setenv("SENSI_CONFIG_DIR", str(tmp_path))
    p = save_hf_token("  hf_new_token  ")
    assert p == tmp_path / "hf_token.txt"
    assert p.read_text(encoding="utf-8") == "hf_new_token\n"
    assert load_preferences()["shared"]["hf_token_file"] == "hf_token.txt"


def test_has_hf_token_sees_configured_file(tmp_path, monkeypatch):
    # has_hf_token() must treat a readable, non-empty configured token file
    # as "token present" (in addition to env / `hf auth login`).
    import extractors.predictability as m
    monkeypatch.setenv("SENSI_CONFIG_DIR", str(tmp_path))
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr("huggingface_hub.get_token", lambda: None)
    assert m.has_hf_token() is False
    (tmp_path / "hf_token.txt").write_text("hf_x\n", encoding="utf-8")
    assert m.has_hf_token() is True
