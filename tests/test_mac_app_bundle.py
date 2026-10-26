"""scripts/make_mac_app.sh builds a valid SensiExtractor.app bundle.

Runs the real script with bash (Git Bash on Windows) into a temp copy of the
repo layout; only launching the app needs a Mac.
"""
import plistlib
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "make_mac_app.sh"
BASH = shutil.which("bash")


def _bash_path(p: Path) -> str:
    return p.as_posix()


@pytest.fixture
def built_app(tmp_path):
    if BASH is None:
        pytest.skip("bash not available")
    (tmp_path / "assets").mkdir()
    shutil.copy(ROOT / "assets" / "icon.icns", tmp_path / "assets" / "icon.icns")
    result = subprocess.run(
        [BASH, _bash_path(SCRIPT), _bash_path(tmp_path), "/opt/conda-test"],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return tmp_path / "SensiExtractor.app"


def test_bundle_layout_and_plist(built_app):
    contents = built_app / "Contents"
    plist = plistlib.loads((contents / "Info.plist").read_bytes())
    assert plist["CFBundleName"] == "SensiExtractor"
    assert plist["CFBundleExecutable"] == "SensiExtractor"
    assert plist["CFBundleIconFile"] == "icon"
    assert plist["CFBundlePackageType"] == "APPL"
    assert (contents / "MacOS" / plist["CFBundleExecutable"]).is_file()
    assert (contents / "Resources" / "icon.icns").read_bytes()[:4] == b"icns"


def test_launcher_script_is_valid_bash_with_baked_paths(built_app, tmp_path):
    launcher = built_app / "Contents" / "MacOS" / "SensiExtractor"
    text = launcher.read_text(encoding="utf-8")
    assert f'cd "{_bash_path(tmp_path)}"' in text
    assert 'source "/opt/conda-test/etc/profile.d/conda.sh"' in text
    assert "exec python gui.py" in text
    syntax = subprocess.run([BASH, "-n", _bash_path(launcher)],
                            capture_output=True, text=True)
    assert syntax.returncode == 0, syntax.stderr


def test_rebuild_is_idempotent(built_app, tmp_path):
    (built_app / "Contents" / "stale.txt").write_text("old", encoding="utf-8")
    subprocess.run([BASH, _bash_path(SCRIPT), _bash_path(tmp_path), "/opt/conda-test"],
                   check=True, capture_output=True)
    assert not (built_app / "Contents" / "stale.txt").exists()


def test_setup_sh_calls_the_builder_on_macos():
    setup = (ROOT / "setup.sh").read_text(encoding="utf-8")
    assert 'bash "$REPO_DIR/scripts/make_mac_app.sh" "$REPO_DIR" "$CONDA_BASE"' in setup
