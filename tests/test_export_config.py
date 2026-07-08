import subprocess, sys, os
from pathlib import Path
from utils.config_yaml import read_config, validate_config


def test_export_config_writes_valid_yaml(tmp_path, monkeypatch):
    # Point the app's prefs at a temp path so we export whatever defaults load.
    monkeypatch.setenv("SENSI_PREFS_PATH", str(tmp_path / "prefs.json"))
    out = tmp_path / "config.yaml"
    repo = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [sys.executable, str(repo / "export_config.py"), "--out", str(out)],
        capture_output=True, text=True, cwd=str(repo),
        env={**os.environ, "SENSI_PREFS_PATH": str(tmp_path / "prefs.json")},
    )
    assert result.returncode == 0, result.stderr
    assert out.exists()
    cfg = read_config(str(out))
    assert set(cfg) == {"run", "features", "settings", "semantic_categories"}
    assert validate_config(cfg) == []
