import json
import shutil
from pathlib import Path

import pytest

from sensi.pipeline import (
    run_word_level_batch,
    _phoneme_columns_to_drop, _syllable_columns_to_drop,
)
from tests.batch_args import build_parser

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

    Monkeypatches pipeline.run so this test doesn't need MFA — we synthesize the
    word_df + phone_df that a real MFA pass would have produced.
    """
    import pandas as pd
    from sensi import pipeline

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

    monkeypatch.setattr(pipeline, "run", fake_run)

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
        "feat_phoneme_arpabet": True, "feat_phoneme_stress": True,
        "feat_phoneme_ipa": True, "feat_phoneme_pron_source": True,
        "feat_phoneme_onset": True,
        "feat_phoneme_end": True, "feat_phoneme_duration": True,
    }

    parser = build_parser()
    args = parser.parse_args([
        '--transcripts-folder', str(txt_dir),
        '--audio-folder', str(audio_dir),
        '--output-folder', str(out_dir),
        '--analyzer', 'tester',
        '--no-align', '--no-edu-freq',
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


def test_phoneme_surprisal_mat_write_survives_savemat(tmp_path, monkeypatch):
    """Regression for a review finding on Task 7 (phoneme-surprisal batch):

    scipy.io.savemat raises `TypeError: Could not convert None to array` if
    the nested 'config' dict passed to phoneme_surprisal_mat_vars contains a
    bare Python None. pipeline.run_word_level_batch's config dict was originally
    fixed to use 'cap': float('nan') as a placeholder instead of 'cap': None,
    but no committed test drove that exact code path — tests only called
    phoneme_surprisal_mat_vars directly with a config dict that never had a
    'cap' key at all. A later fix threads the REAL OOV-surprisal cap from the
    cohort trie (get_cohort_trie(...).cap) into that dict for honest .mat
    provenance. This test drives run_word_level_batch itself (the real
    construction site of that config dict) with phoneme-surprisal .mat output
    enabled, so a future edit that reintroduces a None (or a fake NaN) field
    fails here instead of only at runtime.

    Follows the fake_run/fake_word_df/fake_phone_df pattern from
    test_sidecars_emit_when_struct_and_acoustic_per_column_toggles_off above,
    to avoid needing a real MFA alignment pass.
    """
    import numpy as np
    import pandas as pd
    import scipy.io
    import soundfile as sf
    from sensi import pipeline
    from sensi.extractors import phoneme_surprisal as ps

    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    txt_dir = tmp_path / "txt"
    txt_dir.mkdir()
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    # A real (tiny, silent) wav so _audio_duration()'s soundfile.info() call
    # — part of the real config-dict construction in sensi/pipeline.py — succeeds.
    sr = 16000
    sf.write(str(audio_dir / "story.wav"), np.zeros(sr, dtype=np.float32), sr)
    (txt_dir / "story.txt").write_text("hello world")

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

    monkeypatch.setattr(pipeline, "run", fake_run)

    # Synthetic cohort trie so add_phoneme_surprisal doesn't need the real
    # ~200k-entry MFA lexicon.
    trie = ps.build_cohort_trie([
        (("HH", "EH", "L", "OW"), 3.0), (("W", "ER", "L", "D"), 2.0),
    ])
    monkeypatch.setattr(ps, "get_cohort_trie", lambda *a, **k: trie)

    parser = build_parser()
    args = parser.parse_args([
        '--transcripts-folder', str(txt_dir),
        '--audio-folder', str(audio_dir),
        '--output-folder', str(out_dir),
        '--analyzer', 'tester',
        '--no-align', '--no-edu-freq',
    ])
    # Scope the run to just the phoneme-surprisal .mat sidecar.
    args.no_syllables = True
    args.no_phonemes = True
    args.no_sentences = True
    args.no_word_csv = True
    args.no_phoneme_surprisal = False
    args.no_phoneme_surprisal_mat = False
    args.phoneme_surprisal = {
        'analysis_sr': 100, 'strip_stress': True, 'primary_only': True,
    }

    # run_word_level_batch wraps each pair's processing in a per-pair
    # try/except (it logs failures via logger.record_pair_error rather than
    # raising), so a savemat TypeError here would NOT propagate as a pytest
    # exception — it would silently show up as a recorded pair error instead.
    # Assert on the manifest so a reintroduced None fails loudly and legibly
    # (surfacing the actual savemat TypeError) rather than as a vaguer
    # missing/corrupt .mat file downstream.
    run_word_level_batch(args)

    manifest_path = next((out_dir / 'logs').glob('word_run_*_manifest.json'))
    manifest = json.loads(manifest_path.read_text())
    assert not manifest['errors'], (
        f"pair processing recorded error(s) (expected none): {manifest['errors']}"
    )
    assert manifest['n_pairs_processed'] == 1

    mat_path = out_dir / 'phoneme' / 'story_phoneme_surprisal.mat'
    assert mat_path.exists(), "phoneme-surprisal .mat sidecar was not written"

    # Reload to confirm savemat produced a valid, readable file (not just
    # that no exception propagated) and that the config struct round-tripped
    # with no None-shaped fields.
    loaded = scipy.io.loadmat(str(mat_path), simplify_cells=True)
    cfg = loaded["config"]
    assert cfg, "config struct missing/empty after round-trip"
    for key, val in cfg.items():
        assert val is not None, f"config.{key} round-tripped as None"
    # 'cap' specifically must be the REAL OOV-surprisal cap from the cohort
    # trie actually used for scoring (not the earlier None/NaN placeholder) —
    # sensi/pipeline.py fetches it via get_cohort_trie(...).cap for honest provenance.
    assert np.isfinite(float(cfg["cap"]))
    assert float(cfg["cap"]) == pytest.approx(trie.cap)


def test_batch_emits_namespaced_predictability_columns(tmp_path):
    """End-to-end: a text-only batch with gpt2 + surprisal & cloze writes the
    namespaced <slug>_<Metric> columns and NONE of the retired names. Runs the
    real GPT-2 forward pass (weights cached in data/hf_cache/)."""
    import pandas as pd
    args = _args_for_batch(tmp_path)
    # Predictability is roster-driven (no CLI flags): configure it on the
    # namespace directly, exactly as sensi/gui/planner/emit.py does.
    args.predictability = {
        "models": ["gpt2"],
        "metrics": {"surprisal": True, "cloze": True, "topmatch": False},
    }
    args.no_sem_domain = True   # keep the run lean
    run_word_level_batch(args)

    df = pd.read_csv(tmp_path / 'out' / 'word' / 'story1_features.csv')
    # Namespaced columns present for the selected model x on-metrics.
    assert 'gpt2_Surprisal' in df.columns
    assert 'gpt2_Cloze' in df.columns
    # Not the un-selected metric, and not any retired legacy name.
    assert 'gpt2_TopMatch' not in df.columns
    for retired in ('Surprisal', 'LLM_Surprisal', 'LLM_Cloze', 'LLM_TopMatch'):
        assert retired not in df.columns
    # Document-first token has no context → NaN; a later content word is scored.
    assert pd.isna(df['gpt2_Surprisal'].iloc[0])
    assert df['gpt2_Surprisal'].notna().any()


# ---------------------------------------------------------------------------
# Task 8: phoneme/syllable column-drop maps for the split columns + --g2p-oov
# ---------------------------------------------------------------------------

def test_phoneme_drop_map_covers_the_split_columns():
    all_off = {
        "feat_phoneme_arpabet": False, "feat_phoneme_stress": False,
        "feat_phoneme_ipa": False, "feat_phoneme_pron_source": False,
        "feat_phoneme_onset": False, "feat_phoneme_end": False,
        "feat_phoneme_duration": False,
        "feat_phoneme_surprisal": False, "feat_phoneme_cohort_entropy": False,
    }
    drop = _phoneme_columns_to_drop(all_off)
    assert {"ARPABET", "Stress", "IPA", "Pron_Source"} <= drop
    assert "Label" not in drop, "the Label column no longer exists"


def test_phoneme_drop_map_is_per_column():
    drop = _phoneme_columns_to_drop({"feat_phoneme_ipa": False})
    assert drop == {"IPA"}


def test_syllable_drop_map_covers_ipa():
    assert _syllable_columns_to_drop({"feat_syllable_ipa": False}) == {"IPA"}


def test_planner_passes_g2p_oov_defaulting_off():
    # alignment.g2p_oov (features.yaml) -> ns.g2p_oov, which the batch
    # runners hand to the aligner. Replaces the retired --g2p-oov CLI flag test.
    import copy
    from sensi.gui.planner import plan_from_prefs, to_namespace
    from sensi.utils.preferences import DEFAULT_PREFS
    prefs = copy.deepcopy(DEFAULT_PREFS)
    assert to_namespace(plan_from_prefs(prefs), prefs, level="word").g2p_oov is False
    prefs["alignment"]["g2p_oov"] = True
    assert to_namespace(plan_from_prefs(prefs), prefs, level="word").g2p_oov is True
