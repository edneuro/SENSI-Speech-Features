import csv
import json
from pathlib import Path

from utils.run_logger import RunLogger


def test_run_logger_creates_three_artifacts(tmp_path):
    out = tmp_path / 'out'
    out.mkdir()
    logger = RunLogger(
        output_folder=out,
        analyzer='Alice',
        config={'llm_model': 'llama-3-8b'},
        input_transcripts_folder=tmp_path / 'txt',
        input_audio_folder=tmp_path / 'aud',
        text_extensions=['txt'],
        audio_extensions=['wav'],
    )
    logger.add_warning('mock warning')
    logger.record_pair_success(
        stimulus_id='story1',
        transcript_path=tmp_path / 'txt/story1.txt',
        audio_path=tmp_path / 'aud/story1.wav',
        runtime_seconds=1.5,
        output_csv_path=out / 'story1_features.csv',
        output_summary_path=out / 'story1_summary.csv',
    )
    logger.record_pair_error(
        stimulus_id='story2',
        transcript_path=tmp_path / 'txt/story2.txt',
        audio_path=tmp_path / 'aud/story2.wav',
        runtime_seconds=0.2,
        error_message='boom',
    )
    logger.record_unmatched_text(tmp_path / 'txt/story3.txt')
    logger.record_unmatched_audio(tmp_path / 'aud/story4.wav')
    logger.finalize()

    # Artifacts now live under <output>/logs/ to keep the output folder clean.
    logs = out / 'logs'
    log_files = list(logs.glob('run_*.log'))
    manifests = list(logs.glob('run_*_manifest.json'))
    no_match = list(logs.glob('run_*_no-match.csv'))
    assert len(log_files) == 1
    assert len(manifests) == 1
    assert len(no_match) == 1, "no-match.csv should be written when there are unmatched files"


def test_manifest_contents(tmp_path):
    out = tmp_path / 'out'
    out.mkdir()
    logger = RunLogger(
        output_folder=out,
        analyzer='Bob',
        config={'feature': 'value'},
        input_transcripts_folder=tmp_path / 'txt',
        input_audio_folder=tmp_path / 'aud',
        text_extensions=['txt'],
        audio_extensions=['wav'],
    )
    logger.record_pair_success(
        stimulus_id='s1',
        transcript_path=tmp_path / 'txt/s1.txt',
        audio_path=tmp_path / 'aud/s1.wav',
        runtime_seconds=2.0,
        output_csv_path=out / 's1_features.csv',
        output_summary_path=out / 's1_summary.csv',
    )
    logger.finalize()

    manifest_path = next((out / 'logs').glob('run_*_manifest.json'))
    data = json.loads(manifest_path.read_text())
    assert data['analyzer'] == 'Bob'
    assert data['config'] == {'feature': 'value'}
    assert data['n_pairs_processed'] == 1
    assert data['pairs'][0]['stimulus_id'] == 's1'
    assert data['pairs'][0]['status'] == 'success'
    assert 'system_info' in data
    assert 'sensi_extractor_version' in data
    assert data['total_runtime_seconds'] >= 0


def test_no_match_csv_columns(tmp_path):
    out = tmp_path / 'out'
    out.mkdir()
    logger = RunLogger(
        output_folder=out, analyzer='C', config={},
        input_transcripts_folder=tmp_path / 'txt',
        input_audio_folder=tmp_path / 'aud',
        text_extensions=['txt'], audio_extensions=['wav'],
    )
    logger.record_unmatched_text(tmp_path / 'txt/orphan.txt')
    logger.record_unmatched_audio(tmp_path / 'aud/orphan.wav')
    logger.finalize()

    no_match_path = next((out / 'logs').glob('run_*_no-match.csv'))
    with no_match_path.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert {r['filename'] for r in rows} == {'orphan.txt', 'orphan.wav'}
    assert all(set(r.keys()) == {
        'filename', 'found_in_folder', 'missing_counterpart', 'reason'
    } for r in rows)


def test_filename_prefix(tmp_path):
    out = tmp_path / 'out'; out.mkdir()
    logger = RunLogger(
        output_folder=out, analyzer='A', config={},
        input_transcripts_folder=tmp_path / 'txt',
        input_audio_folder=tmp_path / 'aud',
        text_extensions=['txt'], audio_extensions=['wav'],
        filename_prefix='word_run',
    )
    # Add one unmatched so the no-match.csv would be written (the new
    # behavior skips it when empty).
    logger.record_unmatched_text(tmp_path / 'txt/orphan.txt')
    logger.finalize()
    logs = out / 'logs'
    assert list(logs.glob('word_run_*.log'))
    assert list(logs.glob('word_run_*_manifest.json'))
    assert list(logs.glob('word_run_*_no-match.csv'))


def test_no_match_skipped_when_empty(tmp_path):
    """No unmatched files → no empty no-match.csv litters the logs folder."""
    out = tmp_path / 'out'; out.mkdir()
    logger = RunLogger(
        output_folder=out, analyzer='A', config={},
        input_transcripts_folder=tmp_path / 'txt',
        input_audio_folder=tmp_path / 'aud',
        text_extensions=['txt'], audio_extensions=['wav'],
    )
    logger.finalize()
    logs = out / 'logs'
    assert list(logs.glob('run_*.log'))             # log still written
    assert list(logs.glob('run_*_manifest.json'))   # manifest still written
    assert not list(logs.glob('run_*_no-match.csv'))  # no empty no-match.csv


def test_skip_no_match_kwarg_suppresses_emission(tmp_path):
    """skip_no_match=True suppresses no-match.csv even when there ARE unmatched.

    Used by the second batch in a combined GUI run (its unmatched list is
    identical to the first batch's, so the file would be a duplicate)."""
    out = tmp_path / 'out'; out.mkdir()
    logger = RunLogger(
        output_folder=out, analyzer='A', config={},
        input_transcripts_folder=tmp_path / 'txt',
        input_audio_folder=tmp_path / 'aud',
        text_extensions=['txt'], audio_extensions=['wav'],
        skip_no_match=True,
    )
    logger.record_unmatched_text(tmp_path / 'txt/orphan.txt')
    logger.finalize()
    assert not list((out / 'logs').glob('*_no-match.csv'))


def test_shared_run_ts_yields_grouped_filenames(tmp_path):
    """Two RunLoggers with the same run_ts → log files share that timestamp,
    so the Logs tab can group them as one batch."""
    out = tmp_path / 'out'; out.mkdir()
    ts = '2026-05-12_14-30-22'
    RunLogger(
        output_folder=out, analyzer='A', config={},
        input_transcripts_folder=tmp_path / 'txt',
        input_audio_folder=tmp_path / 'aud',
        text_extensions=['txt'], audio_extensions=['wav'],
        filename_prefix='word_run', run_ts=ts,
    ).finalize()
    RunLogger(
        output_folder=out, analyzer='A', config={},
        input_transcripts_folder=tmp_path / 'txt',
        input_audio_folder=tmp_path / 'aud',
        text_extensions=['txt'], audio_extensions=['wav'],
        filename_prefix='doc_run', run_ts=ts,
    ).finalize()
    logs = out / 'logs'
    assert (logs / f'word_run_{ts}.log').exists()
    assert (logs / f'doc_run_{ts}.log').exists()
