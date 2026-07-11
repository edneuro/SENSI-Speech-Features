"""Per-level output folder structure — lazy mkdir contract.

Each test verifies that a level subfolder (word/, syllable/, phoneme/,
document/) is created only when output is actually written there, and that
session_units CSVs land in logs/ not the output root.

All tests fail until Task 2 and Task 3 update main.py.
"""
from pathlib import Path

import pandas as pd
import pytest

import main
from main import build_parser, run_word_level_batch, run_doc_level_batch

FIXTURES = Path(__file__).parent.parent / 'fixtures'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _word_args(tmp_path):
    return build_parser().parse_args([
        '--transcripts-folder', str(FIXTURES / 'transcripts'),
        '--audio-folder', str(FIXTURES / 'audio'),
        '--output-folder', str(tmp_path / 'out'),
        '--analyzer', 'TestRunner',
        '--no-align',
        '--no-edu-freq',
    ])


def _doc_args(tmp_path):
    return build_parser().parse_args([
        '--transcript', '',
        '--transcripts-folder', str(FIXTURES / 'transcripts'),
        '--audio-folder', str(FIXTURES / 'audio'),
        '--output-folder', str(tmp_path / 'out'),
        '--analyzer', 'TestRunner',
        '--level', 'document',
        '--no-align',
        '--no-doc-mean-llm-cloze',
        '--no-edu-freq',
        '--no-doc-mean-edu-zipf',
    ])


# ---------------------------------------------------------------------------
# Word level
# ---------------------------------------------------------------------------

def test_word_features_land_in_word_subfolder(tmp_path):
    run_word_level_batch(_word_args(tmp_path))
    out = tmp_path / 'out'
    assert (out / 'word' / 'story1_features.csv').exists()
    assert (out / 'word' / 'story2_features.csv').exists()
    assert not list(out.glob('*_features.csv')), 'features CSV found at root'


def test_word_csv_disabled_no_word_folder_not_created(tmp_path):
    """When word CSV is off and --no-align (so no sidecars), word/ must not exist."""
    args = _word_args(tmp_path)
    args.no_word_csv = True   # not a CLI flag; set directly
    run_word_level_batch(args)
    assert not (tmp_path / 'out' / 'word').exists()


def test_sidecars_land_in_subfolders(tmp_path, monkeypatch):
    """syllables → syllable/, phonemes → phoneme/."""
    fake_word_df = pd.DataFrame({
        'doc_id': [0, 0], 'sentence_id': [0, 0], 'token_in_sentence_id': [0, 1],
        'token': ['hello', 'world'],
        'Onset': [0.0, 0.5], 'End': [0.4, 0.9],
    })
    fake_phone_df = pd.DataFrame({
        'phone':  ['HH', 'EH1', 'L',  'OW1', 'W',  'ER1', 'L',  'D'],
        'onset':  [0.00, 0.10, 0.20, 0.30, 0.50, 0.60, 0.75, 0.85],
        'offset': [0.10, 0.20, 0.30, 0.40, 0.60, 0.75, 0.85, 0.90],
    })

    def _fake_run(args):
        args._phones_df = fake_phone_df
        args._contours = None
        return fake_word_df.copy()

    monkeypatch.setattr(main, 'run', _fake_run)

    txt_dir = tmp_path / 'txt'
    txt_dir.mkdir()
    audio_dir = tmp_path / 'audio'
    audio_dir.mkdir()
    out_dir = tmp_path / 'out'
    (txt_dir / 'tone.txt').write_text('hello world')
    (audio_dir / 'tone.wav').write_bytes(b'RIFF\x00\x00\x00\x00WAVE')

    args = build_parser().parse_args([
        '--transcripts-folder', str(txt_dir),
        '--audio-folder', str(audio_dir),
        '--output-folder', str(out_dir),
        '--analyzer', 'tester',
        '--no-align', '--no-edu-freq',
    ])
    args.no_syllables = False
    args.no_phonemes = False
    run_word_level_batch(args)

    assert list((out_dir / 'syllable').glob('*_syllables.csv')), 'syllables not in syllable/'
    assert list((out_dir / 'phoneme').glob('*_phonemes.csv')), 'phonemes not in phoneme/'
    assert not list(out_dir.glob('*_syllables.csv')), 'syllables found at root'
    assert not list(out_dir.glob('*_phonemes.csv')), 'phonemes found at root'


def test_sidecars_disabled_folders_not_created(tmp_path):
    """With --no-align (phones_df=None) and sidecars off, no sidecar folder created."""
    args = _word_args(tmp_path)
    args.no_syllables = True
    args.no_phonemes = True
    run_word_level_batch(args)
    out = tmp_path / 'out'
    assert not (out / 'syllable').exists()
    assert not (out / 'phoneme').exists()


def test_session_units_not_at_word_output_root(tmp_path):
    """word_session_units_*.csv must be in logs/, never at the output root."""
    run_word_level_batch(_word_args(tmp_path))
    out = tmp_path / 'out'
    assert not list(out.glob('word_session_units_*.csv')), \
        'word_session_units found at root — must be in logs/'
    logs_units = list((out / 'logs').glob('word_session_units_*.csv'))
    assert logs_units, \
        'word_session_units not found in logs/ — either path is wrong or session_units block silently errored'


# ---------------------------------------------------------------------------
# Document level
# ---------------------------------------------------------------------------

def test_doc_summary_lands_in_document_subfolder(tmp_path):
    run_doc_level_batch(_doc_args(tmp_path))
    out = tmp_path / 'out'
    assert list((out / 'document').glob('documents_summary_*.csv')), \
        'documents_summary not in document/'
    assert not list(out.glob('documents_summary_*.csv')), \
        'documents_summary found at root'


def test_session_units_not_at_doc_output_root(tmp_path):
    """doc_session_units_*.csv must be in logs/, never at the output root."""
    run_doc_level_batch(_doc_args(tmp_path))
    out = tmp_path / 'out'
    assert not list(out.glob('doc_session_units_*.csv')), \
        'doc_session_units found at root — must be in logs/'
    logs_units = list((out / 'logs').glob('doc_session_units_*.csv'))
    assert logs_units, \
        'doc_session_units not found in logs/ — either path is wrong or session_units block silently errored'


def test_sentence_csv_lands_in_sentence_subfolder(tmp_path, monkeypatch):
    """Sentence CSV lands in sentence/ subfolder when add_sentence_prosody succeeds."""
    import extractors.sentence_prosody as sp_module

    def _fake_sentence_prosody(word_df, contours=None, *, pitch_units="semitones"):
        doc_ids = word_df["doc_id"].unique() if "doc_id" in word_df.columns else ["doc1"]
        rows = [{"doc_id": d, "sentence_id": 0, "sentence": "Hello world.",
                  "Onset": float("nan"), "End": float("nan"),
                  "Pitch_Mean": float("nan"), "Pitch_Min": float("nan"),
                  "Pitch_Max": float("nan"), "Pitch_Std": float("nan"),
                  "Pitch_Delta": float("nan"),
                  "Volume_Mean": float("nan"), "Volume_Min": float("nan"),
                  "Volume_Max": float("nan"), "Volume_Std": float("nan"),
                  "Volume_Delta": float("nan"), "HNR": float("nan")}
                 for d in doc_ids]
        return pd.DataFrame(rows)

    monkeypatch.setattr(sp_module, "add_sentence_prosody", _fake_sentence_prosody)

    run_word_level_batch(_word_args(tmp_path))
    out = tmp_path / 'out'
    assert (out / 'sentence' / 'story1_sentences.csv').exists()
    assert (out / 'sentence' / 'story2_sentences.csv').exists()


def test_sentence_csv_suppressed_by_no_sentences_flag(tmp_path):
    """Setting no_sentences=True means no sentence/ subfolder is created."""
    args = _word_args(tmp_path)
    args.no_sentences = True
    run_word_level_batch(args)
    out = tmp_path / 'out'
    assert not (out / 'sentence').exists()


# ---------------------------------------------------------------------------
# Embedding .mat output paths
# ---------------------------------------------------------------------------

def _fake_w2v(word_df, **kwargs):
    import numpy as np
    n = len(word_df)
    return (
        np.zeros((n, 300), dtype='float32'),
        np.zeros(n, dtype=bool),
    )


def _fake_bert(word_df, **kwargs):
    import numpy as np
    n = len(word_df)
    n_sent = word_df['sentence_id'].nunique() if 'sentence_id' in word_df.columns else 1
    return (
        np.zeros((n, 12, 768), dtype='float32'),
        np.zeros(n, dtype=bool),
        np.zeros((n_sent, 12, 768), dtype='float32'),
        np.zeros(n_sent, dtype=bool),
    )


def test_word2vec_mat_lands_in_word_subfolder(tmp_path, monkeypatch):
    import extractors.word2vec_embed as w2v_module
    monkeypatch.setattr(w2v_module, 'embed_word2vec', _fake_w2v)

    args = _word_args(tmp_path)
    args.no_word2vec = False
    run_word_level_batch(args)
    out = tmp_path / 'out'
    assert list((out / 'word').glob('*_word2vec.mat')), 'word2vec.mat not found in word/'
    assert not list(out.glob('*_word2vec.mat')), 'word2vec.mat found at output root'


def test_word2vec_disabled_mat_not_created(tmp_path):
    args = _word_args(tmp_path)
    args.no_word2vec = True
    run_word_level_batch(args)
    out = tmp_path / 'out'
    assert not list((out / 'word').glob('*_word2vec.mat'))


def test_bert_word_mat_lands_in_word_subfolder(tmp_path, monkeypatch):
    import extractors.bert_embed as bert_module
    monkeypatch.setattr(bert_module, 'embed_bert_words_and_sentences', _fake_bert)

    args = _word_args(tmp_path)
    args.no_bert_word = False
    args.no_bert_sentence = True
    run_word_level_batch(args)
    out = tmp_path / 'out'
    assert list((out / 'word').glob('*_bert_word.mat')), 'bert_word.mat not found in word/'
    assert not list((out / 'sentence').glob('*_bert_sentence.mat'))


def test_bert_sentence_mat_lands_in_sentence_subfolder(tmp_path, monkeypatch):
    import extractors.bert_embed as bert_module
    monkeypatch.setattr(bert_module, 'embed_bert_words_and_sentences', _fake_bert)

    args = _word_args(tmp_path)
    args.no_bert_word = True
    args.no_bert_sentence = False
    run_word_level_batch(args)
    out = tmp_path / 'out'
    assert list((out / 'sentence').glob('*_bert_sentence.mat')), 'bert_sentence.mat not found in sentence/'
    assert not list((out / 'word').glob('*_bert_word.mat'))


def test_bert_disabled_mats_not_created(tmp_path):
    args = _word_args(tmp_path)
    args.no_bert_word = True
    args.no_bert_sentence = True
    run_word_level_batch(args)
    out = tmp_path / 'out'
    assert not list((out / 'word').glob('*_bert_word.mat'))
    assert not list((out / 'sentence').glob('*_bert_sentence.mat'))
