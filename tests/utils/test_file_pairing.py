from pathlib import Path

import pytest

from utils.file_pairing import find_pairs, PairingResult


def _touch(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text('x', encoding='utf-8')


def test_basic_pairing(tmp_path):
    txt_dir = tmp_path / 'txt'
    aud_dir = tmp_path / 'aud'
    _touch(txt_dir / 'story1.txt')
    _touch(txt_dir / 'story2.txt')
    _touch(aud_dir / 'story1.wav')
    _touch(aud_dir / 'story2.wav')

    result = find_pairs(txt_dir, aud_dir)
    assert len(result.pairs) == 2
    assert result.unmatched_text == []
    assert result.unmatched_audio == []
    assert result.warnings == []


def test_unmatched_text(tmp_path):
    txt = tmp_path / 'txt'
    aud = tmp_path / 'aud'
    _touch(txt / 'story1.txt')
    _touch(txt / 'story2.txt')
    _touch(aud / 'story1.wav')

    result = find_pairs(txt, aud)
    assert len(result.pairs) == 1
    assert [p.name for p in result.unmatched_text] == ['story2.txt']


def test_case_insensitive_matching(tmp_path):
    txt = tmp_path / 'txt'
    aud = tmp_path / 'aud'
    _touch(txt / 'Story1.txt')
    _touch(aud / 'story1.WAV')
    result = find_pairs(txt, aud)
    assert len(result.pairs) == 1


def test_stimulus_id_preserves_transcript_casing(tmp_path):
    """Matching is case-insensitive, but the stimulus_id / document name must
    keep the transcript file's real casing (e.g. '1_FB_P', not '1_fb_p')."""
    txt = tmp_path / 'txt'
    aud = tmp_path / 'aud'
    _touch(txt / '1_FB_P.txt')
    _touch(aud / '1_fb_p.wav')
    result = find_pairs(txt, aud)
    assert len(result.pairs) == 1
    assert result.pairs[0].stimulus_id == '1_FB_P'


def test_extension_priority_for_audio(tmp_path):
    txt = tmp_path / 'txt'
    aud = tmp_path / 'aud'
    _touch(txt / 'story1.txt')
    _touch(aud / 'story1.wav')
    _touch(aud / 'story1.mp3')
    result = find_pairs(txt, aud)
    assert len(result.pairs) == 1
    assert Path(result.pairs[0].audio_path).name == 'story1.wav'  # wav > mp3
    assert any('ambiguous' in w.lower() for w in result.warnings)


def test_top_level_only_no_recursion(tmp_path):
    txt = tmp_path / 'txt'
    aud = tmp_path / 'aud'
    _touch(txt / 'subdir' / 'story1.txt')
    _touch(aud / 'story1.wav')
    result = find_pairs(txt, aud)
    assert len(result.pairs) == 0


def test_custom_extensions(tmp_path):
    txt = tmp_path / 'txt'
    aud = tmp_path / 'aud'
    _touch(txt / 'story1.tsv')
    _touch(aud / 'story1.flac')
    result = find_pairs(txt, aud, text_exts=['tsv'], audio_exts=['flac'])
    assert len(result.pairs) == 1


def test_empty_text_exts_means_no_text_files(tmp_path):
    """Empty list should be honored, not silently replaced with defaults."""
    txt = tmp_path / 'txt'
    aud = tmp_path / 'aud'
    _touch(txt / 'story1.txt')
    _touch(aud / 'story1.wav')
    result = find_pairs(txt, aud, text_exts=[])
    assert len(result.pairs) == 0


def test_unmatched_audio_is_recorded(tmp_path):
    txt = tmp_path / 'txt'
    aud = tmp_path / 'aud'
    _touch(aud / 'orphan.wav')
    result = find_pairs(txt, aud)
    assert len(result.pairs) == 0
    assert [p.name for p in result.unmatched_audio] == ['orphan.wav']
