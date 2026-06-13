import argparse
from pathlib import Path

import numpy as np
import scipy.io
import soundfile as sf

import main


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
    main.run_acoustic_level_batch(_args(audio, out))
    mat_path = out / "acoustic" / "stimA_acoustic.mat"
    assert mat_path.is_file()
    data = scipy.io.loadmat(str(mat_path))
    assert "envelope" in data and "onset_envelope" in data and "logmel" in data
    assert "mfcc" not in data           # disabled
    assert float(data["fs"].squeeze()) == 128
