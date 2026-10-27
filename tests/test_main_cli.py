"""CLI smoke tests for prosody flags added in Task 5."""
import pytest


def test_build_parser_accepts_prosody_flags():
    from sensi.pipeline import build_parser
    p = build_parser()
    args = p.parse_args([
        '--audio', 'a.wav', '--transcript', 't.txt',
        '--no-prosody', '--no-doc-prosody',
        '--pitch-units', 'hz',
        '--pause-threshold', '0.5',
        '--pitch-floor', '50', '--pitch-ceiling', '600',
    ])
    assert args.no_prosody is True
    assert args.no_doc_prosody is True
    assert args.pitch_units == 'hz'
    assert args.pause_threshold == 0.5
    assert args.pitch_floor == 50.0
    assert args.pitch_ceiling == 600.0


def test_pitch_units_default_is_semitones():
    from sensi.pipeline import build_parser
    args = build_parser().parse_args(['--audio', 'a.wav', '--transcript', 't.txt'])
    assert args.pitch_units == 'semitones'


# --- single-run guard: missing transcript ----------------------------------

def test_single_run_without_transcript_errors_cleanly(monkeypatch, capsys):
    """`python main.py` with no --transcript used to die with
    FileNotFoundError: '' — an opaque stack trace. It must exit(1) with a
    message that names the missing flag."""
    from sensi import pipeline
    monkeypatch.setattr('sys.argv', ['main.py', '--no-align'])
    with __import__('pytest').raises(SystemExit) as ei:
        pipeline.main()
    # SystemExit carries the human-readable message; it must name the missing
    # --transcript rather than surfacing FileNotFoundError on the empty path.
    assert 'transcript' in str(ei.value).lower()


# --- batch: audio-folder optional for text-only runs ------------------------

def test_batch_allows_missing_audio_folder(monkeypatch, tmp_path):
    """A text-only batch (transcripts + output, no audio) must not be rejected
    by the batch-arg validation."""
    from sensi import pipeline
    called = {}
    monkeypatch.setattr(pipeline, 'run_word_level_batch',
                        lambda args, **k: called.setdefault('ran', args))
    monkeypatch.setattr('sys.argv', [
        'main.py',
        '--transcripts-folder', str(tmp_path / 'tx'),
        '--output-folder', str(tmp_path / 'out'),
        '--analyzer', 'tester',
    ])
    pipeline.main()
    assert 'ran' in called
    # audio_folder defaulted to empty/None → no audio paired.
    assert not called['ran'].audio_folder


def test_batch_still_requires_transcripts_and_output(monkeypatch, tmp_path):
    from sensi import pipeline
    import pytest
    monkeypatch.setattr('sys.argv', [
        'main.py', '--audio-folder', str(tmp_path / 'a'),
    ])
    with pytest.raises(SystemExit):
        pipeline.main()


# --- acoustic batch: per-file failure is a recorded pair error --------------

def test_acoustic_batch_records_pair_error_on_failure(monkeypatch, tmp_path):
    """A file that fails to process in the acoustic level must be recorded as
    a pair error in the manifest (like word/doc batches), not only as a
    free-text warning."""
    import json
    from sensi import pipeline
    audio_dir = tmp_path / 'audio'
    audio_dir.mkdir()
    (audio_dir / 'bad.wav').write_bytes(b'not really a wav')
    out_dir = tmp_path / 'out'

    monkeypatch.setattr('sensi.extractors.acoustic_ts.extract_acoustic',
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError('boom')))

    import argparse
    args = argparse.Namespace(
        audio_folder=str(audio_dir), output_folder=str(out_dir),
        audio_extensions='wav', analyzer='tester', run_ts=None,
        acoustic_ts={'analysis_rate': 128, 'envelope': True, 'onset_env': False,
                     'logmel': False, 'mfcc': False, 'env_method': 'rms',
                     'env_comp': 0.3, 'env_window': 1.0, 'n_mels': 16,
                     'n_mfcc': 13, 'mel_preset': 'speech'},
    )
    pipeline.run_acoustic_level_batch(args)

    manifests = list((out_dir / 'logs').glob('acoustic_run_*_manifest.json'))
    assert manifests, 'no manifest written'
    manifest = json.loads(manifests[0].read_text(encoding='utf-8'))
    pairs = manifest.get('pairs', manifest.get('results', []))
    assert any(p.get('status') == 'error' and p.get('stimulus_id') == 'bad'
               for p in pairs), f'no error pair recorded: {pairs}'


# --- predictability: unified extractor replaces --no-surprisal/--no-llm-cloze ---

def test_removed_predictability_flags_are_gone():
    """The old separate surprisal/LLM-cloze CLI flags are retired in favor of
    a single ns.predictability config (models x metrics) consumed by
    extractors.predictability.add_predictability."""
    from sensi.pipeline import build_parser
    p = build_parser()
    for flag in ('--no-surprisal', '--no-llm-cloze', '--llm-model'):
        with pytest.raises(SystemExit):
            p.parse_args(['--transcript', 't.txt', flag, 'x'])


def test_run_reads_predictability_from_namespace(monkeypatch, tmp_path):
    """A namespace carrying predictability config drives add_predictability
    with the models/metrics it specifies, regardless of preferences.json."""
    from sensi import pipeline
    from sensi.pipeline import build_parser

    called = {}

    def fake_add_predictability(df, text, **kwargs):
        called['k'] = kwargs
        return df

    monkeypatch.setattr('sensi.extractors.predictability.add_predictability',
                        fake_add_predictability)

    transcript = tmp_path / 't.txt'
    transcript.write_text('Hello world.', encoding='utf-8')

    args = build_parser().parse_args([
        '--transcript', str(transcript),
        '--no-align', '--no-freq', '--no-edu-freq', '--no-sem-domain',
    ])
    args.predictability = {
        'models': ['fake_slug'],
        'metrics': {'surprisal': True, 'cloze': False, 'topmatch': False},
    }

    pipeline.run(args)

    assert called['k']['models'] == ['fake_slug']
    assert called['k']['metrics'] == ['surprisal']
