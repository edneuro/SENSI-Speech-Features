"""Disk-size estimates and input statistics (utils/run_estimates.py)."""
import builtins
import copy

import numpy as np
import pytest
import soundfile as sf

from sensi.gui.planner.plan import LEVEL_ARTIFACTS, plan_from_prefs
from sensi.utils import estimate_constants as K
from sensi.utils import run_estimates as RE
from sensi.utils.preferences import DEFAULT_PREFS
from sensi.utils.run_estimates import (InputStats, estimate_disk, format_bytes,
                                 level_bytes, scan_inputs)


def _all_on() -> dict:
    prefs = copy.deepcopy(DEFAULT_PREFS)
    for section in ("features", "output"):
        for k, v in list(prefs[section].items()):
            if isinstance(v, bool):
                prefs[section][k] = True
    prefs["predictability"]["models"] = ["gpt2"]
    prefs["predictability"]["metrics"] = {"surprisal": True, "cloze": True,
                                          "topmatch": True}
    return prefs


def _stats(**kw) -> InputStats:
    base = dict(n_stimuli=2, n_with_audio=2, words=10_000,
                words_with_audio=10_000, audio_seconds=600.0)
    base.update(kw)
    return InputStats(**base)


def test_bert_word_bytes_scale_with_words():
    prefs = _all_on()
    disk = estimate_disk(plan_from_prefs(prefs), prefs, _stats())
    # One embedding row per word-CSV row (spaCy token), not per whitespace word.
    rows = 10_000 * K.ROWS_PER_WORD["features"]
    assert disk["bert_word"] == round(rows * 12 * 768 * 4) + 1024 * 2


def test_huge_input_formats_in_gb():
    assert format_bytes(3_700_000_000) == "≈ 3.4 GB"
    assert format_bytes(430_000_000) == "≈ 410 MB"


def test_text_only_items_add_no_audio_files():
    prefs = _all_on()
    plan = plan_from_prefs(prefs)
    for art in ("acoustic", "phoneme_surprisal_mat", "syllables", "phonemes"):
        assert art in plan.output_csvs
    disk = estimate_disk(plan, prefs, _stats(n_with_audio=0, words_with_audio=0,
                                             audio_seconds=0.0))
    for art in ("acoustic", "phoneme_surprisal_mat", "syllables", "phonemes"):
        assert disk.get(art, 0) == 0, art
    assert disk["features"] > 0


def test_level_bytes_sums_its_artifacts():
    prefs = _all_on()
    disk = estimate_disk(plan_from_prefs(prefs), prefs, _stats())
    for level, arts in LEVEL_ARTIFACTS.items():
        assert level_bytes(disk, level) == sum(disk.get(a, 0) for a in arts)
    assert level_bytes(disk, "level_word") > disk["features"]


def test_absent_artifacts_not_in_dict():
    prefs = _all_on()
    prefs["output"]["level_acoustic"] = False
    plan = plan_from_prefs(prefs)
    disk = estimate_disk(plan, prefs, _stats())
    assert set(disk) == set(plan.output_csvs)
    assert "acoustic" not in disk


def test_dropped_columns_shrink_features_csv():
    prefs = _all_on()
    full = estimate_disk(plan_from_prefs(prefs), prefs, _stats())["features"]
    # Whole cluster off (plan.column_drops)...
    prefs["features"]["master_word_prosody"] = False
    no_prosody = estimate_disk(plan_from_prefs(prefs), prefs, _stats())["features"]
    assert no_prosody < full
    # ...and a per-column toggle inside an active cluster (main.py drop map).
    prefs["features"]["feat_acoustic_onset"] = False
    fewer = estimate_disk(plan_from_prefs(prefs), prefs, _stats())["features"]
    assert fewer < no_prosody


def test_scan_inputs_none_without_folder(tmp_path):
    assert scan_inputs("", "") is None
    assert scan_inputs(str(tmp_path / "missing"), "") is None


def _wav(path, seconds, sr=8000):
    sf.write(str(path), np.zeros(int(seconds * sr), dtype=np.float32), sr)


def _corpus(tmp_path):
    tdir, adir = tmp_path / "t", tmp_path / "a"
    tdir.mkdir()
    adir.mkdir()
    (tdir / "s1.txt").write_text("I saw 3 cats.\n", encoding="utf-8")
    (tdir / "s2.txt").write_text("text only here", encoding="utf-8")
    _wav(adir / "s1.wav", 2.0)
    _wav(adir / "lonely.wav", 1.0)   # unpaired: acoustic level only
    return tdir, adir


def test_scan_inputs_counts(tmp_path):
    tdir, adir = _corpus(tmp_path)
    st = scan_inputs(str(tdir), str(adir))
    assert (st.n_stimuli, st.n_with_audio) == (2, 1)
    # "I saw 3 cats." = 4 raw whitespace words; + 3 text-only words.
    assert (st.words, st.words_with_audio) == (7, 4)
    assert st.audio_seconds == pytest.approx(2.0)
    assert st.n_audio_files == 2
    assert st.all_audio_seconds == pytest.approx(3.0)


def test_scan_inputs_audio_only(tmp_path):
    _, adir = _corpus(tmp_path)
    st = scan_inputs("", str(adir))
    assert (st.n_stimuli, st.words) == (0, 0)
    assert st.all_audio_seconds == pytest.approx(3.0)


def test_scan_inputs_cached(tmp_path, monkeypatch):
    tdir, adir = _corpus(tmp_path)
    first = scan_inputs(str(tdir), str(adir))

    def boom(*a, **kw):
        raise AssertionError("file reopened")

    monkeypatch.setattr(builtins, "open", boom)
    monkeypatch.setattr(sf, "info", boom)
    assert scan_inputs(str(tdir), str(adir)) == first


def test_cache_invalidated_on_change(tmp_path):
    tdir, adir = _corpus(tmp_path)
    scan_inputs(str(tdir), str(adir))
    (tdir / "s2.txt").write_text("now it has five words", encoding="utf-8")
    assert scan_inputs(str(tdir), str(adir)).words == 9


def test_cache_is_module_level():
    assert isinstance(RE._FILE_CACHE, dict)


def test_scan_words_match_recorded_units(tmp_path):
    # The estimator counts words exactly as main.py records them
    # (utils.timing_history.count_words on the raw transcript), so learned
    # seconds-per-word apply to the same unit. Number spelling would make
    # "1,500" four words; the recorder sees one.
    from sensi.utils.timing_history import count_words
    tdir = tmp_path / "t"
    tdir.mkdir()
    text = "He won 1,500 points in 2024.\n"
    (tdir / "s.txt").write_text(text, encoding="utf-8")
    st = scan_inputs(str(tdir), "")
    assert st.words == count_words(text) == 6
