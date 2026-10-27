"""Model caches resolve inside the project, not the home directory."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_VARS = ("HF_HOME", "MFA_ROOT_DIR", "GENSIM_DATA_DIR")

# Import an entry module first, then the libraries that read their cache
# location at import time, in a fresh interpreter with no cache vars set.
_PROBE = """
import json, os, sys
sys.path.insert(0, {root!r})
import {entry}
import gensim.downloader as gd
import huggingface_hub.constants as hc
print(json.dumps({{"gensim": gd.BASE_DIR, "hf": hc.HF_HUB_CACHE,
                  "mfa": os.environ["MFA_ROOT_DIR"]}}))
"""


def _probe(entry: str, extra_env: dict | None = None) -> dict:
    env = {k: v for k, v in os.environ.items() if k not in _VARS}
    env.update(extra_env or {})
    out = subprocess.run(
        [sys.executable, "-c", _PROBE.format(root=str(_ROOT), entry=entry)],
        env=env, cwd=str(_ROOT), capture_output=True, text=True, timeout=300,
    )
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("entry", ["sensi.pipeline", "run", "sensi.extractors"])
def test_caches_default_to_project(entry):
    got = _probe(entry)
    assert Path(got["gensim"]) == _ROOT / "data" / "gensim_data"
    assert Path(got["hf"]).is_relative_to(_ROOT / "data" / "hf_cache")
    assert Path(got["mfa"]) == _ROOT / "data" / "mfa_data"


def test_preset_location_wins(tmp_path):
    got = _probe("sensi.extractors", {"GENSIM_DATA_DIR": str(tmp_path / "g"),
                                "HF_HOME": str(tmp_path / "h"),
                                "MFA_ROOT_DIR": str(tmp_path / "m")})
    assert Path(got["gensim"]) == tmp_path / "g"
    assert Path(got["hf"]).is_relative_to(tmp_path / "h")
    assert Path(got["mfa"]) == tmp_path / "m"


def _layout(tmp_path, monkeypatch):
    from sensi.utils import local_caches as lc
    new, old = tmp_path / "data" / "hf_cache", tmp_path / "hf_cache"
    monkeypatch.setattr(lc, "LEGACY_MODEL_DIRS", {new: old})
    monkeypatch.setattr(lc, "LEGACY_DICT_DIR", tmp_path / "dictionaries")
    monkeypatch.setattr(lc, "DICT_DIR", tmp_path / "data" / "dictionaries")
    for var in lc.CACHE_DIRS:
        monkeypatch.delenv(var, raising=False)
    return lc, new, old


def test_legacy_folder_used_until_moved(tmp_path, monkeypatch):
    lc, new, old = _layout(tmp_path, monkeypatch)
    old.mkdir()
    assert lc.resolve_cache_dir(new) == old          # old layout, not moved yet
    new.mkdir(parents=True)
    assert lc.resolve_cache_dir(new) == new          # data/ copy wins once it exists


def test_move_legacy_model_dirs_moves_once(tmp_path, monkeypatch, capsys):
    lc, new, old = _layout(tmp_path, monkeypatch)
    (old / "hub").mkdir(parents=True)
    (old / "hub" / "w.bin").write_text("weights")
    lc.move_legacy_model_dirs()
    assert not old.exists() and (new / "hub" / "w.bin").read_text() == "weights"
    assert "Moved hf_cache/" in capsys.readouterr().err
    lc.move_legacy_model_dirs()             # nothing left to move: no-op
    assert (new / "hub" / "w.bin").exists()


def test_move_never_overwrites_existing_data_dir(tmp_path, monkeypatch, capsys):
    lc, new, old = _layout(tmp_path, monkeypatch)
    old.mkdir(); (old / "old.txt").write_text("old")
    new.mkdir(parents=True); (new / "new.txt").write_text("new")
    lc.move_legacy_model_dirs()
    assert (old / "old.txt").exists() and (new / "new.txt").exists()
    assert "Both hf_cache/ and data/hf_cache/ exist" in capsys.readouterr().err


def test_folder_this_session_points_at_is_not_moved(tmp_path, monkeypatch):
    lc, new, old = _layout(tmp_path, monkeypatch)
    old.mkdir()
    monkeypatch.setenv("HF_HOME", str(old))     # already pinned by this process
    lc.move_legacy_model_dirs()
    assert old.exists() and not new.exists()


def test_leftover_dictionary_files_move_without_overwriting(tmp_path, monkeypatch):
    lc, _, _ = _layout(tmp_path, monkeypatch)
    old_d, new_d = tmp_path / "dictionaries", tmp_path / "data" / "dictionaries"
    old_d.mkdir(); new_d.mkdir(parents=True)
    (old_d / "liwc-2022.dic").write_text("local only")
    (old_d / "mrc.parquet").write_text("old copy")
    (new_d / "mrc.parquet").write_text("tracked copy")
    lc.move_legacy_model_dirs()
    assert (new_d / "liwc-2022.dic").read_text() == "local only"
    assert (new_d / "mrc.parquet").read_text() == "tracked copy"     # not overwritten
    assert (old_d / "mrc.parquet").exists()                          # left for the user
    (old_d / "mrc.parquet").unlink()
    lc.move_legacy_model_dirs()
    assert not old_d.exists()                                        # empty -> removed
