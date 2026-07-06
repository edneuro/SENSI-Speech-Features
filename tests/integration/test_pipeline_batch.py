import json
import shutil
from pathlib import Path

import pytest

from main import run_word_level_batch, build_parser

FIXTURES = Path(__file__).parent.parent / 'fixtures'


def _args_for_batch(tmp_path):
    parser = build_parser()
    args = parser.parse_args([
        '--transcript', 'unused.txt',  # ignored in batch mode but required by parser
        '--transcripts-folder', str(FIXTURES / 'transcripts'),
        '--audio-folder', str(FIXTURES / 'audio'),
        '--output-folder', str(tmp_path / 'out'),
        '--analyzer', 'TestRunner',
        '--no-align',  # skip MFA in tests
        '--no-llm-cloze',  # skip LLM in tests
        '--no-edu-freq',  # skip — no lookup file in fixtures
    ])
    return args


def test_batch_processes_all_pairs(tmp_path):
    args = _args_for_batch(tmp_path)
    run_word_level_batch(args)
    out = tmp_path / 'out'
    assert (out / 'word' / 'story1_features.csv').exists()
    assert (out / 'word' / 'story2_features.csv').exists()
    # Audio-less transcripts (story3, test_short) are now processed for their
    # text features — they are no longer skipped (2026-06-24 policy change:
    # "always process text features"). Their audio columns are simply absent.
    assert (out / 'word' / 'story3_features.csv').exists()
    assert (out / 'word' / 'test_short_features.csv').exists()
    # No per-stim summary sidecar in word-level runs anymore
    assert not (out / 'story1_summary.csv').exists()
    assert not (out / 'story2_summary.csv').exists()


def test_batch_emits_run_artifacts(tmp_path):
    args = _args_for_batch(tmp_path)
    run_word_level_batch(args)
    # Log artifacts live in <output>/logs/ (moved from output root to keep
    # the user's output folder focused on feature CSVs).
    logs = tmp_path / 'out' / 'logs'
    assert any(logs.glob('word_run_*.log'))
    assert any(logs.glob('word_run_*_manifest.json'))
    # no-match.csv is conditional — only written when something is unmatched.
    # These fixtures now fully process (audio-less transcripts included), and
    # there is no audio without a transcript, so no no-match.csv is expected.
    assert not any(logs.glob('word_run_*_no-match.csv'))


def test_batch_processes_audio_less_transcripts(tmp_path):
    """Audio-less transcripts are processed as text-only items, not recorded
    as unmatched-text (2026-06-24 policy change). All four fixture transcripts
    (2 paired + story3 + test_short) become successful pairs; nothing is left
    in n_unmatched_text. unmatched_audio still tracks audio with no transcript
    (none in these fixtures)."""
    args = _args_for_batch(tmp_path)
    run_word_level_batch(args)
    manifest = json.loads(next((tmp_path / 'out' / 'logs').glob('word_run_*_manifest.json')).read_text())
    assert manifest['n_unmatched_text'] == 0
    assert manifest['n_pairs_processed'] == 4  # story1, story2, story3, test_short
    assert manifest['analyzer'] == 'TestRunner'


def test_batch_emits_syllables_and_phonemes_sidecars(tmp_path):
    """End-to-end: a real fixture stimulus should produce both sidecars
    alongside the features CSV when alignment is on."""
    import wave
    import numpy as np
    import pandas as pd

    pytest.importorskip("parselmouth")

    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    txt_dir = tmp_path / "txt"
    txt_dir.mkdir()
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    # Synthesize a 2-second 200 Hz tone (same pattern as prosody test)
    wav = audio_dir / "tone.wav"
    sr, dur, f0 = 16000, 2.0, 200.0
    samples = (0.5 * np.sin(2 * np.pi * f0 * np.arange(int(sr * dur)) / sr) * 32767).astype(np.int16)
    with wave.open(str(wav), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(samples.tobytes())
    (txt_dir / "tone.txt").write_text("aaa")

    parser = build_parser()
    args = parser.parse_args([
        '--transcripts-folder', str(txt_dir),
        '--audio-folder', str(audio_dir),
        '--output-folder', str(out_dir),
        '--analyzer', 'tester',
        '--no-llm-cloze',
        '--no-edu-freq',
        # no --no-align, no --no-syllables, no --no-phonemes
    ])
    run_word_level_batch(args)

    csv_path = out_dir / 'word' / 'tone_features.csv'
    if not csv_path.exists():
        pytest.skip(
            "Pipeline didn't produce CSV (MFA likely couldn't align synthetic input). "
            "Sidecar check skipped."
        )

    # Sidecars require phoneme-level alignment data. spaCy always runs so the
    # word CSV is written regardless; check that Onset is not all-NaN (which
    # happens when the synthetic token is OOV in the MFA English lexicon).
    _df_check = pd.read_csv(csv_path)
    if _df_check["Onset"].isna().all():
        pytest.skip(
            "MFA aligned the audio but produced no phone matches for the "
            "synthetic token (OOV). Sidecar check skipped."
        )

    syll = list((out_dir / 'syllable').glob('*_syllables.csv'))
    phon = list((out_dir / 'phoneme').glob('*_phonemes.csv'))
    assert syll, 'expected at least one syllables sidecar'
    assert phon, 'expected at least one phonemes sidecar'

    syll_df = pd.read_csv(syll[0])
    phon_df = pd.read_csv(phon[0])
    for col in ('doc_id', 'sentence_id', 'token_in_sentence_id', 'token', 'syllable_id', 'Onset', 'End'):
        assert col in syll_df.columns, f'syllable CSV missing {col}'
    for col in ('doc_id', 'sentence_id', 'token_in_sentence_id', 'token', 'syllable_id', 'phoneme_id',
                'Label', 'Onset', 'End'):
        assert col in phon_df.columns, f'phoneme CSV missing {col}'
    assert len(syll_df) > 0
    assert len(phon_df) > 0


def test_word_level_batch_emits_prosody_columns(tmp_path):
    """End-to-end: 200 Hz sine WAV + 1-line transcript → CSV has Pitch_Mean
    column. Skips gracefully if MFA can't align synthetic audio."""
    import wave
    import numpy as np
    import pandas as pd

    pytest.importorskip("parselmouth")

    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    txt_dir = tmp_path / "txt"
    txt_dir.mkdir()
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    # Synthesize a 2-second 200 Hz tone
    wav = audio_dir / "tone.wav"
    sr, dur, f0 = 16000, 2.0, 200.0
    samples = (0.5 * np.sin(2 * np.pi * f0 * np.arange(int(sr * dur)) / sr) * 32767).astype(np.int16)
    with wave.open(str(wav), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(samples.tobytes())
    (txt_dir / "tone.txt").write_text("aaa")

    parser = build_parser()
    args = parser.parse_args([
        '--transcripts-folder', str(txt_dir),
        '--audio-folder', str(audio_dir),
        '--output-folder', str(out_dir),
        '--analyzer', 'tester',
        '--no-llm-cloze',
        '--no-edu-freq',
    ])
    run_word_level_batch(args)

    csv_path = out_dir / 'word' / 'tone_features.csv'
    if not csv_path.exists():
        pytest.skip(
            "Pipeline didn't produce CSV (MFA likely couldn't align synthetic input). "
            "Schema check skipped."
        )
    df = pd.read_csv(csv_path)
    for col in ("Pitch_Mean", "Pitch_Min", "Pitch_Max", "Pitch_Std",
                "Volume_Mean", "Volume_Delta"):
        assert col in df.columns, f"Missing {col} in word CSV"


def test_sidecars_emit_when_struct_and_acoustic_per_column_toggles_off(
        tmp_path, monkeypatch):
    """Regression for the silent-no-sidecars bug:

    A user with all feat_struct_* and feat_acoustic_* toggles OFF (so those
    columns get dropped from the features CSV) but with feat_syllable_* /
    feat_phoneme_* ON and the sidecar outputs enabled should still get
    syllables/phonemes CSVs. Previously the column-drop step ran BEFORE
    sidecar emission and stripped doc_id/sentence_id/token_in_sentence_id/Onset/End from
    df, so syllabify_pairs silently produced empty DataFrames.

    Monkeypatches main.run so this test doesn't need MFA — we synthesize the
    word_df + phone_df that a real MFA pass would have produced.
    """
    import pandas as pd
    import main

    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    txt_dir = tmp_path / "txt"
    txt_dir.mkdir()
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (audio_dir / "tone.wav").write_bytes(b"RIFF\x00\x00\x00\x00WAVE")  # placeholder
    (txt_dir / "tone.txt").write_text("hello world")

    fake_word_df = pd.DataFrame({
        "doc_id": [0, 0], "sentence_id": [0, 0], "token_in_sentence_id": [0, 1],
        "token": ["hello", "world"],
        "Onset": [0.0, 0.5], "End": [0.4, 0.9],
    })
    fake_phone_df = pd.DataFrame({
        "phone": ["HH", "EH1", "L", "OW1", "W", "ER1", "L", "D"],
        "onset":  [0.00, 0.10, 0.20, 0.30, 0.50, 0.60, 0.75, 0.85],
        "offset": [0.10, 0.20, 0.30, 0.40, 0.60, 0.75, 0.85, 0.90],
    })

    def fake_run(args):
        args._phones_df = fake_phone_df
        args._contours = None
        return fake_word_df.copy()

    monkeypatch.setattr(main, "run", fake_run)

    # All struct + acoustic toggles OFF (their columns will be in _word_columns_to_drop)
    # All syllable + phoneme toggles ON (so the per-column drops inside the
    # sidecar emission block don't remove FK columns from the sidecars).
    feature_toggles = {
        # struct off
        "feat_struct_doc_id": False, "feat_struct_paragraph_id": False,
        "feat_struct_sentence_id": False, "feat_struct_sentence_text": False,
        "feat_struct_char_start": False, "feat_struct_char_end": False,
        "feat_struct_token_id": False, "feat_struct_token_in_sentence_id": False,
        # acoustic off
        "feat_acoustic_onset": False, "feat_acoustic_end": False,
        "feat_acoustic_word_duration": False, "feat_acoustic_word_gap": False,
        # syllable + phoneme on (default True; spelled out for clarity)
        "feat_syllable_onset": True, "feat_syllable_end": True,
        "feat_syllable_duration": True, "feat_syllable_label": True,
        "feat_syllable_arpabet": True, "feat_syllable_pitch_mean": True,
        "feat_syllable_volume_mean": True, "feat_syllable_pitch_delta": True,
        "feat_syllable_volume_delta": True,
        "feat_phoneme_label": True, "feat_phoneme_onset": True,
        "feat_phoneme_end": True, "feat_phoneme_duration": True,
    }

    parser = build_parser()
    args = parser.parse_args([
        '--transcripts-folder', str(txt_dir),
        '--audio-folder', str(audio_dir),
        '--output-folder', str(out_dir),
        '--analyzer', 'tester',
        '--no-align', '--no-llm-cloze', '--no-edu-freq',
    ])
    args.feature_toggles = feature_toggles
    args.no_syllables = False
    args.no_phonemes = False
    args.no_word_csv = True  # user-reported config: only sidecars wanted

    run_word_level_batch(args)

    syll_files = list((out_dir / 'syllable').glob('*_syllables.csv'))
    phon_files = list((out_dir / 'phoneme').glob('*_phonemes.csv'))
    assert syll_files, "sidecar bug regressed: no syllables CSV emitted"
    assert phon_files, "sidecar bug regressed: no phonemes CSV emitted"
    syll_df = pd.read_csv(syll_files[0])
    phon_df = pd.read_csv(phon_files[0])
    assert len(syll_df) > 0, "syllables sidecar is empty — drop ran before emission"
    assert len(phon_df) > 0, "phonemes sidecar is empty — drop ran before emission"
    # And the features CSV must NOT be written when no_word_csv=True
    assert not (out_dir / 'word').exists()
