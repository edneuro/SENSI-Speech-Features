import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import scipy.io
import soundfile as sf

from sensi import pipeline

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _make_audio(folder: Path, name: str, sr=16000, dur=0.5):
    folder.mkdir(parents=True, exist_ok=True)
    t = np.arange(int(sr * dur)) / sr
    sf.write(str(folder / name), 0.4 * np.sin(2 * np.pi * 200 * t), sr)


def _args(audio_dir, out_dir):
    return argparse.Namespace(
        audio_folder=str(audio_dir), output_folder=str(out_dir),
        analyzer="tester", audio_extensions="wav,flac,mp3",
        acoustic_ts={"envelope": True, "onset_env": True, "logmel": True,
                     "mfcc": False, "env_method": "rms", "env_comp": 0.3,
                     "n_mels": 16, "n_mfcc": 13, "mel_preset": "speech",
                     "analysis_rate": 128},
        run_ts="2026-06-13_00-00-00",
    )


def test_acoustic_batch_writes_mat_without_transcript(tmp_path):
    audio = tmp_path / "audio"
    out = tmp_path / "out"
    _make_audio(audio, "stimA.wav")
    pipeline.run_acoustic_level_batch(_args(audio, out))
    mat_path = out / "acoustic" / "stimA_acoustic.mat"
    assert mat_path.is_file()
    data = scipy.io.loadmat(str(mat_path))
    assert "envelope" in data and "onset_envelope" in data and "logmel" in data
    assert "mfcc" not in data           # disabled
    assert float(data["fs"].squeeze()) == 128
    # 1-D time series are saved as (N, 1) column vectors so they stack with
    # logmel (N, n_mels) as a time-by-feature TRF predictor matrix in MATLAB.
    n = data["t"].shape[0]
    assert data["envelope"].shape == (n, 1)
    assert data["onset_envelope"].shape == (n, 1)
    assert data["logmel"].shape[0] == n


def test_acoustic_run_py_in_fresh_process_without_mkl_env(tmp_path, monkeypatch):
    """Regression: a real headless run of the acoustic level must NOT crash in a
    fresh process.

    The acoustic level's mel-filterbank matmul (librosa einsum -> numpy/MKL) hard-
    crashes on AMD CPUs (Windows fatal exception 0xc06d007f) under the default
    Intel MKL threading layer. tests/conftest.py sets MKL_THREADING_LAYER for the
    test process, but the launchers (run.py / gui.py) must set it themselves --
    otherwise real runs crash for end users. This test launches `run.py` in a
    *fresh* subprocess with MKL_THREADING_LAYER explicitly removed to prove the
    launcher applies the workaround on its own. (On Intel CPUs the default layer
    is fine, so this is also a plain end-to-end smoke test of run.py there.)
    """
    from sensi.utils.preferences import save_preferences

    audio = tmp_path / "audio"
    out = tmp_path / "out"
    cfg = tmp_path / "cfg"
    _make_audio(audio, "stimCLI.wav")
    (tmp_path / "transcripts").mkdir()
    monkeypatch.setenv("SENSI_CONFIG_DIR", str(cfg))
    save_preferences({
        "output": {"level_word": False, "level_acoustic": True},
        "features": {"master_acoustic_ts": True, "feat_acoustic_ts_envelope": True,
                     "feat_acoustic_ts_onset_env": True, "feat_acoustic_ts_logmel": True,
                     "feat_acoustic_ts_mfcc": False},
        "acoustic_ts": {"analysis_rate": 128},
        "shared": {"analyzer_name": "t",
                   "last_transcripts_folder": str(tmp_path / "transcripts"),
                   "last_audio_folder": str(audio),
                   "last_output_folder": str(out)},
    })

    env = dict(os.environ)
    env.pop("MKL_THREADING_LAYER", None)   # do NOT inherit the conftest workaround

    proc = subprocess.run(
        [sys.executable, "run.py", "--config-dir", str(cfg)],
        cwd=str(_PROJECT_ROOT), env=env,
        capture_output=True, text=True, timeout=300,
    )

    mat_path = out / "acoustic" / "stimCLI_acoustic.mat"
    assert proc.returncode == 0, (
        f"run.py exited {proc.returncode} (native crash if non-zero with no traceback).\n"
        f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    assert mat_path.is_file(), f"no .mat produced; stderr:\n{proc.stderr}"
    data = scipy.io.loadmat(str(mat_path))
    assert "envelope" in data and "logmel" in data
