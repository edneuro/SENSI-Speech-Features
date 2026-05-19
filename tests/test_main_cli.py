"""CLI smoke tests for prosody flags added in Task 5."""


def test_build_parser_accepts_prosody_flags():
    from main import build_parser
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
    from main import build_parser
    args = build_parser().parse_args(['--audio', 'a.wav', '--transcript', 't.txt'])
    assert args.pitch_units == 'semitones'
