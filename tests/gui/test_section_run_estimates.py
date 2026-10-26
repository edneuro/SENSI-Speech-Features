"""Run tab shows disk / memory / time estimates (display only)."""
import json

import numpy as np
import pytest
import soundfile as sf

from utils.preferences import load_preferences, save_preferences

_TEXT = "The quick brown fox jumps over the lazy dog. It was a bright day."


def _inputs(tmp_path, levels=("level_word",), pred=None):
    t = tmp_path / "t"; t.mkdir()
    a = tmp_path / "a"; a.mkdir()
    out = tmp_path / "out"; out.mkdir()
    for name in ("s1", "s2"):
        (t / f"{name}.txt").write_text(_TEXT, encoding="utf-8")
        sf.write(str(a / f"{name}.wav"), np.zeros(16000, dtype="float32"), 16000)
    prefs = {
        "shared": {"analyzer_name": "me", "last_transcripts_folder": str(t),
                   "last_audio_folder": str(a), "last_output_folder": str(out)},
        "output": {k: (k in levels) for k in (
            "level_word", "level_document", "level_subword",
            "level_sentence", "level_acoustic")},
    }
    if pred:
        prefs["predictability"] = pred
    save_preferences(prefs)


def _fake_system(monkeypatch, ram, vram=0.0):
    import utils.system_info as si
    monkeypatch.setattr(si, "gather_system_info", lambda: {
        "system_ram_gb": ram,
        "gpu": {"name": "GPU" if vram else None, "vram_gb": vram}})


def _section(qtbot):
    from gui.section_run import RunSection
    w = RunSection(get_session_api_key=lambda: "", status_strip=None)
    qtbot.addWidget(w)
    return w


def test_dash_until_folders_set(qtbot):
    w = _section(qtbot)
    w.show()
    assert [w.est_values[k].text() for k in ("disk", "ram", "time")] == ["—"] * 3
    assert not w.estimate_warning.isVisible()
    assert not w.level_rows["level_word"].size_label.isVisible()


def test_level_and_file_sizes_shown(qtbot, tmp_path, monkeypatch):
    _fake_system(monkeypatch, 64.0)
    _inputs(tmp_path)
    w = _section(qtbot)
    w.show()
    row = w.level_rows["level_word"]
    assert row.size_label.isVisible() and row.size_label.text().startswith("≈ ")
    assert w.est_values["disk"].text().startswith("≈ ")
    assert "this computer: 64 GB" in w.est_values["ram"].text()
    assert w.est_values["time"].text().startswith("≈ ")
    shown = [s for s in row.file_size_labels.values() if s.isVisible()]
    assert shown and all(s.text().startswith("≈ ") for s in shown)
    # Levels left out show no size.
    assert not w.level_rows["level_document"].size_label.isVisible()


def test_ram_warning_only_when_it_does_not_fit(qtbot, tmp_path, monkeypatch):
    _inputs(tmp_path, pred={"models": ["llama3_8b"],
                            "metrics": {"surprisal": True, "cloze": False,
                                        "topmatch": False}})
    from gui.planner import plan_from_prefs
    assert plan_from_prefs(load_preferences()).predictability_word
    _fake_system(monkeypatch, 4.0)
    w = _section(qtbot)
    w.show()
    assert w.estimate_warning.isVisible()
    assert w.estimate_warning.text().startswith("⚠")
    _fake_system(monkeypatch, 512.0)
    w2 = _section(qtbot)
    w2.show()
    assert not w2.estimate_warning.isVisible()


def test_estimates_never_disable_run(qtbot, tmp_path, monkeypatch):
    _fake_system(monkeypatch, 4.0)
    _inputs(tmp_path, pred={"models": ["llama3_8b"],
                            "metrics": {"surprisal": True, "cloze": False,
                                        "topmatch": False}})
    w = _section(qtbot)
    assert w.estimate_warning.isVisibleTo(w)
    assert w.run_button.isEnabled()
    # And a crashing estimator leaves the tab usable, with dashes.
    import utils.run_estimates as re_
    monkeypatch.setattr(re_, "estimate_disk",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    w.refresh()
    assert w.est_values["disk"].text() == "—"
    assert w.run_button.isEnabled()


def test_first_estimate_note_hidden_when_learned(qtbot, tmp_path, monkeypatch):
    _fake_system(monkeypatch, 64.0)
    _inputs(tmp_path)
    w = _section(qtbot)
    assert w.estimate_note.isVisibleTo(w)
    from gui.planner import plan_from_prefs
    from utils import run_estimates as re_
    from utils.preferences import get_config_dir
    from utils.timing_history import MIN_SAMPLES_FOR_RATE
    plan = plan_from_prefs(load_preferences())
    steps = {re_.STAGE_TO_STEP[s] for s in plan.stages_to_run
             if s in re_.STAGE_TO_STEP}
    samples = [{"step": st, "model": None, "device": "cpu", "units": 100.0,
                "seconds": 10.0}
               for st in steps for _ in range(MIN_SAMPLES_FOR_RATE + 1)]
    (get_config_dir() / "timing_history.json").write_text(
        json.dumps({"version": 1, "samples": samples, "load_seconds": {}}),
        encoding="utf-8")
    w.refresh()
    assert not w.estimate_note.isVisibleTo(w)
