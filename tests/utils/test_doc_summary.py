import pandas as pd
import pytest

from utils.doc_summary import build_doc_summary, DOC_FEATURE_KEYS


def _all_off() -> dict:
    return {k: False for k in DOC_FEATURE_KEYS}


def _all_on() -> dict:
    return {k: True for k in DOC_FEATURE_KEYS}


def _basic_df() -> pd.DataFrame:
    return pd.DataFrame({
        'token': ['the', 'cat', 'sat'],
        'lemma': ['the', 'cat', 'sat'],
        'upos':  ['DET', 'NOUN', 'VERB'],
        'sentence_id': [1, 1, 1],
        'paragraph_id': [1, 1, 1],
    })


def test_counts_when_toggled_on():
    df = _basic_df()
    toggles = _all_off()
    toggles['counts'] = True
    summary = build_doc_summary(
        df, stimulus_id='s', dictionary_used='Empath',
        text='The cat sat.', toggles=toggles,
    )
    assert summary['n_tokens'] == 3
    assert summary['n_content_tokens'] == 2
    assert summary['n_sentences'] == 1


def test_means_omitted_when_toggled_off():
    df = _basic_df()
    df['Zipf_Freq'] = [5.5, 5.0, 4.0]
    toggles = _all_off()
    toggles['counts'] = True
    summary = build_doc_summary(
        df, stimulus_id='s', dictionary_used='Empath',
        text='The cat sat.', toggles=toggles,
    )
    assert 'mean_zipf_freq' not in summary


def test_means_when_toggled_on():
    df = _basic_df()
    df['Zipf_Freq'] = [5.5, 5.0, 4.0]
    toggles = _all_off()
    toggles['counts'] = True
    toggles['mean_zipf_freq'] = True
    summary = build_doc_summary(
        df, stimulus_id='s', dictionary_used='Empath',
        text='The cat sat.', toggles=toggles,
    )
    assert summary['mean_zipf_freq'] == pytest.approx(14.5 / 3)


def test_ttr_mtld_toggles():
    df = _basic_df()
    toggles = _all_off()
    toggles['ttr'] = True
    toggles['mtld'] = True
    summary = build_doc_summary(
        df, stimulus_id='s', dictionary_used='Empath',
        text='The cat sat.', toggles=toggles,
    )
    assert 'ttr' in summary
    assert 'mtld' in summary


def test_readability_toggles():
    df = _basic_df()
    toggles = _all_off()
    toggles['flesch_reading_ease'] = True
    toggles['flesch_kincaid_grade'] = True
    summary = build_doc_summary(
        df, stimulus_id='s', dictionary_used='Empath',
        text='The cat sat on the mat. The dog ran fast.',
        toggles=toggles,
    )
    assert 'flesch_reading_ease' in summary
    assert 'flesch_kincaid_grade' in summary


def test_category_proportions_toggled():
    df = _basic_df()
    df['Sem_Empath_animal'] = [None, 1, 0]  # only content tokens scored
    df['Sem_Empath_motion'] = [None, 0, 1]
    toggles = _all_off()
    toggles['counts'] = True
    toggles['category_proportions'] = True
    summary = build_doc_summary(
        df, stimulus_id='s', dictionary_used='Empath',
        text='The cat sat.', toggles=toggles,
    )
    assert summary['count_animal'] == 1
    assert summary['prop_animal'] == pytest.approx(1 / 2)


def test_psych_summary_columns_absent_when_empath():
    df = _basic_df()
    toggles = _all_off()
    toggles['psych_authentic'] = True  # requested but Empath active
    summary = build_doc_summary(
        df, stimulus_id='s', dictionary_used='Empath',
        text='Hello world.', toggles=toggles,
    )
    assert 'psych_authentic' not in summary


# ---------------------------------------------------------------------------
# Pipeline independence tests: doc-level must not depend on word-level toggles
# ---------------------------------------------------------------------------

def test_independence_doc_mean_zipf_when_word_freq_disabled(tmp_path):
    """Doc-level mean_zipf_freq must be computed even when --no-freq disables
    word-level Zipf columns.  run_doc_level_batch should internally enable the
    freq extractor regardless of the word-level toggle."""
    from pathlib import Path
    from main import run_doc_level_batch, build_parser
    import csv

    FIXTURES = Path(__file__).parent.parent / 'fixtures'

    parser = build_parser()
    args = parser.parse_args([
        '--transcript', '',
        '--transcripts-folder', str(FIXTURES / 'transcripts'),
        '--audio-folder', str(FIXTURES / 'audio'),
        '--output-folder', str(tmp_path / 'out'),
        '--analyzer', 'IndependenceTest',
        '--level', 'document',
        '--no-align',
        '--no-llm-cloze',       # word-level LLM OFF (slow)
        '--no-doc-mean-llm-cloze',  # doc-level LLM OFF (avoids force-enabling LLM)
        '--no-edu-freq',
        '--no-doc-mean-edu-zipf',
        '--no-surprisal',       # word-level surprisal OFF (slow)
        '--no-doc-mean-surprisal',  # doc-level surprisal OFF
        '--no-sem-domain',      # word-level sem-domain OFF
        '--no-doc-category-proportions',
        '--no-doc-liwc-authentic', '--no-doc-liwc-analytic',
        '--no-doc-liwc-tone', '--no-doc-liwc-clout',
        '--no-freq',            # word-level Zipf OFF  ← this is the toggle under test
        # doc-level mean_zipf_freq is ON (no --no-doc-mean-zipf)
    ])

    run_doc_level_batch(args)

    masters = list((tmp_path / 'out' / 'document').glob('documents_summary_*.csv'))
    assert masters, (
        'No documents_summary CSV created — run_doc_level_batch produced no output. '
        'Check run log for errors.'
    )
    master = masters[0]
    with master.open() as f:
        rows = list(csv.DictReader(f))

    assert len(rows) >= 1, 'Expected at least one processed stimulus'
    row = rows[0]

    # The key assertion: mean_zipf_freq must be present and non-empty
    assert 'mean_zipf_freq' in row, (
        'mean_zipf_freq missing from doc summary — doc-level pipeline depended '
        'on word-level --no-freq toggle and silently skipped it'
    )
    assert row['mean_zipf_freq'] not in ('', 'None', 'nan'), (
        f'mean_zipf_freq is blank/NaN: {row["mean_zipf_freq"]!r}'
    )


def test_independence_doc_mean_surprisal_when_word_surprisal_disabled(tmp_path):
    """Doc-level mean_surprisal must be computed even when --no-surprisal
    disables word-level GPT-2 surprisal columns."""
    from pathlib import Path
    from main import run_doc_level_batch, build_parser
    import csv

    FIXTURES = Path(__file__).parent.parent / 'fixtures'

    parser = build_parser()
    args = parser.parse_args([
        '--transcript', '',
        '--transcripts-folder', str(FIXTURES / 'transcripts'),
        '--audio-folder', str(FIXTURES / 'audio'),
        '--output-folder', str(tmp_path / 'out'),
        '--analyzer', 'IndependenceTest2',
        '--level', 'document',
        '--no-align',
        '--no-llm-cloze',           # word-level LLM OFF
        '--no-doc-mean-llm-cloze',  # doc-level LLM OFF
        '--no-edu-freq',
        '--no-doc-mean-edu-zipf',
        '--no-freq',                # word-level Zipf OFF
        '--no-doc-mean-zipf',       # doc-level Zipf OFF
        '--no-sem-domain',
        '--no-doc-category-proportions',
        '--no-doc-liwc-authentic', '--no-doc-liwc-analytic',
        '--no-doc-liwc-tone', '--no-doc-liwc-clout',
        '--no-surprisal',           # word-level surprisal OFF  ← toggle under test
        # doc-level mean_surprisal is ON (no --no-doc-mean-surprisal)
    ])

    run_doc_level_batch(args)

    masters = list((tmp_path / 'out' / 'document').glob('documents_summary_*.csv'))
    assert masters, (
        'No documents_summary CSV created — run_doc_level_batch produced no output. '
        'Check run log for errors.'
    )
    master = masters[0]
    with master.open() as f:
        rows = list(csv.DictReader(f))

    assert len(rows) >= 1
    row = rows[0]

    assert 'mean_surprisal' in row, (
        'mean_surprisal missing from doc summary — doc-level pipeline depended '
        'on word-level --no-surprisal toggle and silently skipped it'
    )
    assert row['mean_surprisal'] not in ('', 'None', 'nan'), (
        f'mean_surprisal is blank/NaN: {row["mean_surprisal"]!r}'
    )


def test_build_doc_summary_includes_prosody_when_contours_supplied():
    import numpy as np, pandas as pd
    from utils.doc_summary import build_doc_summary
    from extractors.audio_features import Contours
    from extractors.prosody import add_word_prosody

    times = np.arange(100) / 100.0
    contours = Contours(
        pitch_hz=np.full(100, 200.0), intensity_db=np.full(100, 60.0),
        times=times, frame_rate_hz=100,
        speaker_median_pitch_hz=200.0, hnr_db=15.0,
    )
    df = pd.DataFrame([{
        'token': 'cat', 'Onset': 0.0, 'End': 1.0, 'upos': 'NOUN',
    }])
    df = add_word_prosody(df, contours, pitch_units='hz')

    summary = build_doc_summary(
        df, stimulus_id='s1', dictionary_used='LIWC', text='cat',
        toggles={
            'counts': False, 'mean_zipf_freq': False, 'mean_edu_zipf': False,
            'mean_surprisal': False, 'mean_llm_cloze': False, 'ttr': False,
            'mtld': False, 'flesch_reading_ease': False,
            'flesch_kincaid_grade': False, 'category_proportions': False,
            'psych_authentic': False, 'psych_analytic': False,
            'psych_tone': False, 'psych_clout': False,
            # Prosody master gate
            'doc_prosody': True,
        },
        contours=contours, pause_threshold_s=0.25, pitch_units='hz',
    )
    assert summary['doc_pitch_mean'] == 200.0
    assert summary['doc_hnr'] == 15.0
    assert summary['doc_pause_count'] == 0


def test_build_doc_summary_no_contours_no_prosody_keys():
    import pandas as pd
    from utils.doc_summary import build_doc_summary
    df = pd.DataFrame([{'token': 'x', 'upos': 'NOUN'}])
    summary = build_doc_summary(
        df, stimulus_id='s1', dictionary_used='Empath', text='x',
        toggles={'counts': True}, contours=None,
    )
    assert 'doc_pitch_mean' not in summary
    assert 'doc_hnr' not in summary
