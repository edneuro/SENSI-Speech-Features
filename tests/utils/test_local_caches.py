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


@pytest.mark.parametrize("entry", ["main", "run", "extractors"])
def test_caches_default_to_project(entry):
    got = _probe(entry)
    assert Path(got["gensim"]) == _ROOT / "gensim_data"
    assert Path(got["hf"]).is_relative_to(_ROOT / "hf_cache")
    assert Path(got["mfa"]) == _ROOT / "mfa_data"


def test_preset_location_wins(tmp_path):
    got = _probe("extractors", {"GENSIM_DATA_DIR": str(tmp_path / "g"),
                                "HF_HOME": str(tmp_path / "h"),
                                "MFA_ROOT_DIR": str(tmp_path / "m")})
    assert Path(got["gensim"]) == tmp_path / "g"
    assert Path(got["hf"]).is_relative_to(tmp_path / "h")
    assert Path(got["mfa"]) == tmp_path / "m"
