import csv
import json
from pathlib import Path

from main import run_doc_level_batch, build_parser

FIXTURES = Path(__file__).parent.parent / 'fixtures'


def _args(tmp_path):
    parser = build_parser()
    return parser.parse_args([
        '--transcript', '',
        '--transcripts-folder', str(FIXTURES / 'transcripts'),
        '--audio-folder', str(FIXTURES / 'audio'),
        '--output-folder', str(tmp_path / 'out'),
        '--analyzer', 'DocTest',
        '--level', 'document',
        '--no-align',
        '--no-llm-cloze',
        '--no-doc-mean-llm-cloze',  # disable doc-level LLM aggregate (model not available in CI)
        '--no-edu-freq',
        '--no-doc-mean-edu-zipf',   # disable doc-level edu aggregate (lookup not in fixtures)
    ])


def test_doc_level_emits_master_csv(tmp_path):
    args = _args(tmp_path)
    run_doc_level_batch(args)
    out = tmp_path / 'out'
    masters = list(out.glob('documents_summary_*.csv'))
    assert len(masters) == 1


def test_doc_level_run_log_uses_doc_prefix(tmp_path):
    args = _args(tmp_path)
    run_doc_level_batch(args)
    # Log artifacts moved under <output>/logs/.
    logs = tmp_path / 'out' / 'logs'
    assert list(logs.glob('doc_run_*.log'))
    assert list(logs.glob('doc_run_*_manifest.json'))
    assert list(logs.glob('doc_run_*_no-match.csv'))


def test_doc_level_master_csv_columns(tmp_path):
    args = _args(tmp_path)
    run_doc_level_batch(args)
    out = tmp_path / 'out'
    master = next(out.glob('documents_summary_*.csv'))
    with master.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) >= 1  # at least one matched pair processed
    # Always-present columns
    assert 'stimulus_id' in rows[0]
    assert 'n_tokens' in rows[0]


def test_doc_level_does_not_emit_word_level_csvs(tmp_path):
    args = _args(tmp_path)
    run_doc_level_batch(args)
    out = tmp_path / 'out'
    assert not list(out.glob('*_features.csv'))


def test_main_dispatches_level_word_by_default(tmp_path):
    """`--level` defaults to 'word' — verifies backward compat."""
    parser = build_parser()
    args = parser.parse_args([
        '--transcript', '',
        '--transcripts-folder', str(FIXTURES / 'transcripts'),
        '--audio-folder', str(FIXTURES / 'audio'),
        '--output-folder', str(tmp_path / 'out'),
        '--analyzer', 'Default',
        '--no-align', '--no-llm-cloze', '--no-edu-freq',
    ])
    assert args.level == 'word'
