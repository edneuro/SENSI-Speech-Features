def test_run_section_has_run_button(qtbot):
    from gui.section_run import RunSection
    w = RunSection(get_session_api_key=lambda: "", status_strip=None)
    qtbot.addWidget(w)
    assert w.run_button is not None


def test_run_button_disabled_when_no_analyzer(qtbot, tmp_path, monkeypatch):
    monkeypatch.setenv("SENSI_PREFS_PATH", str(tmp_path / "prefs.json"))
    from utils.preferences import save_preferences
    save_preferences({"shared": {"analyzer_name": ""}})
    from gui.section_run import RunSection
    w = RunSection(get_session_api_key=lambda: "", status_strip=None)
    qtbot.addWidget(w)
    w.refresh()
    assert not w.run_button.isEnabled()


def test_run_tab_refreshes_on_show_after_inputs_filled(qtbot, tmp_path, monkeypatch):
    """Regression: filling the Inputs on another page and then viewing the Run
    tab must update readiness. The Inputs page does not push a refresh to the Run
    section, so the Run section re-reads prefs on showEvent. (Previously the stale
    state was masked because Path('').is_dir() let empty folders pass.)"""
    monkeypatch.setenv("SENSI_PREFS_PATH", str(tmp_path / "prefs.json"))
    from PySide6.QtGui import QShowEvent
    from utils.preferences import save_preferences
    from gui.section_run import RunSection
    w = RunSection(get_session_api_key=lambda: "", status_strip=None)
    qtbot.addWidget(w)
    assert not w.run_button.isEnabled()   # nothing set at construction

    # Fill the Inputs (as the Inputs page would) for an audio-only acoustic run,
    # WITHOUT calling w.refresh() directly.
    audio = tmp_path / "audio"; audio.mkdir()
    out = tmp_path / "out"; out.mkdir()
    save_preferences({"shared": {
        "analyzer_name": "me", "last_audio_folder": str(audio),
        "last_output_folder": str(out), "last_transcripts_folder": ""}})
    _enable_only_acoustic(save_preferences)   # audio-only: every other output off
    # Viewing the page must pick up the new prefs.
    w.showEvent(QShowEvent())
    assert w.run_button.isEnabled()


def test_run_tab_has_five_level_switches(qtbot, tmp_path, monkeypatch):
    """One switch per level, named after the feature page, in page order."""
    monkeypatch.setenv("SENSI_PREFS_PATH", str(tmp_path / "prefs.json"))
    from gui.section_run import RunSection, _LEVEL_ROWS
    w = RunSection(get_session_api_key=lambda: "", status_strip=None)
    qtbot.addWidget(w)
    names = [row.name_label.text() for row in w.level_rows.values()]
    assert names == ["Word level", "Document level", "Sub-word level",
                     "Sentence level", "Acoustic level"]
    assert list(w.level_rows) == ["level_word", "level_document",
                                  "level_subword", "level_sentence",
                                  "level_acoustic"]
    assert [n for _, n, _ in _LEVEL_ROWS] == names
    # Defaults: Word level on, the rest left out.
    assert w.level_rows["level_word"].switch.is_checked()
    for key in ("level_document", "level_subword", "level_sentence",
                "level_acoustic"):
        assert not w.level_rows[key].switch.is_checked()
        assert w.level_rows[key].tag_label.text() == "Left out"


def test_level_row_strikes_unproduced_files(qtbot, tmp_path, monkeypatch):
    """Sub-word on with only syllable features ticked: the phoneme files are
    struck through and the tag counts the one live file."""
    monkeypatch.setenv("SENSI_PREFS_PATH", str(tmp_path / "prefs.json"))
    from utils.preferences import load_preferences, save_preferences
    from gui.section_run import RunSection
    feats = {k: False for k in load_preferences()["features"]
             if k.startswith(("feat_syllable", "feat_phoneme"))}
    feats.update({"feat_syllable_onset": True,
                  "master_syllable_intervals": True})
    save_preferences({"output": {"level_subword": True}, "features": feats})
    w = RunSection(get_session_api_key=lambda: "", status_strip=None)
    qtbot.addWidget(w)
    row = w.level_rows["level_subword"]
    assert row.tag_label.text() == "1 file per stimulus"
    files = row.file_labels
    assert files["syllables"].text() == "syllable/<stim>_syllables.csv"
    assert not files["syllables"].font().strikeOut()
    assert files["phonemes"].font().strikeOut()
    assert files["phoneme_surprisal_mat"].font().strikeOut()


def test_level_count_tag_wording():
    from gui.section_run import _count_tag
    assert _count_tag("level_word", False, 3) == "Left out"
    assert _count_tag("level_word", True, 0) == "Nothing selected"
    assert _count_tag("level_word", True, 1) == "1 file per stimulus"
    assert _count_tag("level_word", True, 3) == "3 files per stimulus"
    assert _count_tag("level_document", True, 1) == "1 file for the whole run"
    assert _count_tag("level_acoustic", True, 1) == "1 file per audio file"


def test_level_switch_persists_and_emits(qtbot, tmp_path, monkeypatch):
    monkeypatch.setenv("SENSI_PREFS_PATH", str(tmp_path / "prefs.json"))
    from utils.preferences import load_preferences
    from gui.section_run import RunSection
    w = RunSection(get_session_api_key=lambda: "", status_strip=None)
    qtbot.addWidget(w)
    assert load_preferences()["output"]["level_sentence"] is False
    with qtbot.waitSignal(w.settings_changed, timeout=1000):
        w.level_rows["level_sentence"].switch.set_checked(True)
    assert load_preferences()["output"]["level_sentence"] is True
    assert w.level_rows["level_sentence"].tag_label.text() != "Left out"


def test_acoustic_level_syncs_card(qtbot, tmp_path, monkeypatch):
    """The Run-tab Acoustic level and the Acoustic page's card switch are one
    setting: a Run-tab toggle shows on the card after settings_changed, and a
    card toggle shows on the Run tab after refresh."""
    monkeypatch.setenv("SENSI_PREFS_PATH", str(tmp_path / "prefs.json"))
    from utils.preferences import load_preferences
    from gui.section_run import RunSection
    from gui.section_acoustic_features import AcousticFeaturesSection
    run = RunSection(get_session_api_key=lambda: "", status_strip=None)
    acoustic = AcousticFeaturesSection()
    qtbot.addWidget(run); qtbot.addWidget(acoustic)
    run.settings_changed.connect(acoustic.refresh_from_prefs)
    acoustic.settings_changed.connect(run.refresh)
    assert not acoustic._card.master_toggle().is_checked()

    run.level_rows["level_acoustic"].switch.set_checked(True)
    assert load_preferences()["output"]["level_acoustic"] is True
    assert acoustic._card.master_toggle().is_checked()

    acoustic._card.master_toggle().set_checked(False)
    assert load_preferences()["output"]["level_acoustic"] is False
    assert not run.level_rows["level_acoustic"].switch.is_checked()
    assert run.level_rows["level_acoustic"].tag_label.text() == "Left out"


def _enable_only_acoustic(save_preferences):
    """Select the Acoustic level as the ONLY output (audio-only run)."""
    save_preferences({
        "output": {
            "level_word": False, "level_document": False,
            "level_subword": False, "level_sentence": False,
            "level_acoustic": True,
        },
        "features": {"feat_acoustic_ts_envelope": True},
    })


def test_level_rows_reflect_plan_for_audio_only_run(qtbot, tmp_path, monkeypatch):
    """The level rows show the files the plan will write: an audio-only run
    lists the acoustic .mat and leaves the Word level out."""
    monkeypatch.setenv("SENSI_PREFS_PATH", str(tmp_path / "prefs.json"))
    from utils.preferences import save_preferences
    from gui.section_run import RunSection
    _enable_only_acoustic(save_preferences)   # audio-only run
    w = RunSection(get_session_api_key=lambda: "", status_strip=None)
    qtbot.addWidget(w)
    w.refresh()
    ac = w.level_rows["level_acoustic"]
    assert ac.tag_label.text() == "1 file per audio file"
    assert ac.file_labels["acoustic"].text() == "acoustic/<stim>_acoustic.mat"
    assert not ac.file_labels["acoustic"].font().strikeOut()
    assert w.level_rows["level_word"].tag_label.text() == "Left out"


def test_run_button_enabled_for_audio_only_acoustic_without_transcripts(
        qtbot, tmp_path, monkeypatch):
    """The acoustic level is audio-only — Run must enable with audio + output +
    analyzer even when NO transcripts folder is set."""
    monkeypatch.setenv("SENSI_PREFS_PATH", str(tmp_path / "prefs.json"))
    from utils.preferences import save_preferences
    audio = tmp_path / "audio"; audio.mkdir()
    out = tmp_path / "out"; out.mkdir()
    save_preferences({"shared": {
        "analyzer_name": "me", "last_audio_folder": str(audio),
        "last_output_folder": str(out), "last_transcripts_folder": "",  # none!
    }})
    _enable_only_acoustic(save_preferences)
    from gui.section_run import RunSection
    w = RunSection(get_session_api_key=lambda: "", status_strip=None)
    qtbot.addWidget(w)
    w.refresh()
    assert w.run_button.isEnabled()


def test_run_button_disabled_when_text_output_lacks_transcripts(
        qtbot, tmp_path, monkeypatch):
    """A transcript-derived output (word CSV) with no transcripts folder must
    keep Run disabled — the acoustic relaxation must not leak to text outputs."""
    monkeypatch.setenv("SENSI_PREFS_PATH", str(tmp_path / "prefs.json"))
    from utils.preferences import save_preferences
    audio = tmp_path / "audio"; audio.mkdir()
    out = tmp_path / "out"; out.mkdir()
    save_preferences({
        "shared": {"analyzer_name": "me", "last_audio_folder": str(audio),
                   "last_output_folder": str(out), "last_transcripts_folder": ""},
        "output": {"level_word": True},
        "features": {"master_structural": True, "feat_struct_token_id": True},
    })
    from gui.section_run import RunSection
    w = RunSection(get_session_api_key=lambda: "", status_strip=None)
    qtbot.addWidget(w)
    w.refresh()
    assert not w.run_button.isEnabled()


def test_cost_panel_hidden_when_local_model(qtbot, tmp_path, monkeypatch):
    monkeypatch.setenv("SENSI_PREFS_PATH", str(tmp_path / "prefs.json"))
    from utils.preferences import save_preferences
    save_preferences({"predictability": {"llm_model": "llama-3.2-3b"}})
    from gui.section_run import RunSection
    w = RunSection(get_session_api_key=lambda: "", status_strip=None)
    qtbot.addWidget(w)
    w.refresh()
    assert not w.cost_panel.isVisible()


# --- Bug B regression tests: worker-thread → main-thread signal routing -----
#
# dispatch_run invokes its progress/done callbacks from a background thread.
# Those callbacks must not touch widgets directly. RunSection wires them
# through internal Qt signals (_progress_event, _done_event) so the slots
# run on the main thread via auto-QueuedConnection. These tests verify the
# wiring: emit the signal (including from a worker thread) and confirm the
# slot fires and mutates the status_strip.


class _FakeSignal:
    def __init__(self):
        self.slots = []

    def connect(self, slot):
        self.slots.append(slot)

    def emit(self):
        for slot in self.slots:
            slot()


class _FakeStatusStrip:
    """Minimal stand-in for StatusStrip — captures the last call made."""

    def __init__(self):
        self.calls = []
        self.cancel_requested = _FakeSignal()

    def set_cancelling(self):
        self.calls.append(("cancelling",))

    def set_cancelled(self):
        self.calls.append(("cancelled",))

    def set_running(self, current, total, file_name):
        self.calls.append(("running", current, total, file_name))

    def set_done(self, total, elapsed_s):
        self.calls.append(("done", total, elapsed_s))

    def set_errored(self, total):
        self.calls.append(("errored", total))


def test_progress_event_is_signal(qtbot):
    from gui.section_run import RunSection
    strip = _FakeStatusStrip()
    w = RunSection(get_session_api_key=lambda: "", status_strip=strip)
    qtbot.addWidget(w)
    # Bound signals expose .emit and .connect; verifies the class-level
    # Signal descriptor is wired (an instance attribute would be a plain int).
    assert hasattr(w._progress_event, "emit")
    assert hasattr(w._progress_event, "connect")
    assert hasattr(w._done_event, "emit")
    assert hasattr(w._done_event, "connect")


def test_progress_event_drives_status_strip(qtbot):
    from gui.section_run import RunSection
    strip = _FakeStatusStrip()
    w = RunSection(get_session_api_key=lambda: "", status_strip=strip)
    qtbot.addWidget(w)
    w._progress_event.emit(3, 10, "stim_07")
    assert ("running", 3, 10, "stim_07") in strip.calls


def test_done_event_drives_status_strip_on_success(qtbot, tmp_path):
    from gui.section_run import RunSection
    from utils.preferences import save_preferences
    audio = tmp_path / "audio"; audio.mkdir()
    out = tmp_path / "out"; out.mkdir()
    save_preferences({"shared": {
        "analyzer_name": "me", "last_audio_folder": str(audio),
        "last_output_folder": str(out), "last_transcripts_folder": ""}})
    _enable_only_acoustic(save_preferences)          # a ready config
    strip = _FakeStatusStrip()
    w = RunSection(get_session_api_key=lambda: "", status_strip=strip)
    qtbot.addWidget(w)
    w.run_button.setEnabled(False)
    w._done_event.emit("done")
    assert any(c[0] == "done" for c in strip.calls)
    # Slot re-enables the Run button (it refreshes, so readiness decides).
    assert w.run_button.isEnabled()
    # Settings broken during the run: the refreshed readiness keeps it off.
    save_preferences({"shared": {"analyzer_name": ""}})
    w.run_button.setEnabled(False)
    w._done_event.emit("done")
    assert not w.run_button.isEnabled()


def test_done_event_drives_status_strip_on_error(qtbot):
    from gui.section_run import RunSection
    strip = _FakeStatusStrip()
    w = RunSection(get_session_api_key=lambda: "", status_strip=strip)
    qtbot.addWidget(w)
    w._done_event.emit("error: boom")
    assert any(c[0] == "errored" for c in strip.calls)


def test_done_event_emits_public_run_finished(qtbot):
    from gui.section_run import RunSection
    strip = _FakeStatusStrip()
    w = RunSection(get_session_api_key=lambda: "", status_strip=strip)
    qtbot.addWidget(w)
    with qtbot.waitSignal(w.run_finished, timeout=1000) as blocker:
        w._done_event.emit("done")
    assert blocker.args == ["done"]


def test_done_event_from_worker_thread_runs_slot_on_main_thread(qtbot):
    """Regression for Bug B: emitting from a worker thread must not mutate
    widgets off-thread. We verify the slot fires (via the public
    run_finished signal) after a worker-thread emit; if the auto-
    QueuedConnection upgrade is broken, qtbot.waitSignal will time out."""
    import threading
    from gui.section_run import RunSection
    strip = _FakeStatusStrip()
    w = RunSection(get_session_api_key=lambda: "", status_strip=strip)
    qtbot.addWidget(w)

    def worker():
        w._done_event.emit("done")

    with qtbot.waitSignal(w.run_finished, timeout=2000) as blocker:
        threading.Thread(target=worker, daemon=True).start()
    assert blocker.args == ["done"]
    assert any(c[0] == "done" for c in strip.calls)


def _ready_acoustic(tmp_path, save_preferences):
    audio = tmp_path / "audio"; audio.mkdir()
    out = tmp_path / "out"; out.mkdir()
    save_preferences({"shared": {
        "analyzer_name": "me", "last_audio_folder": str(audio),
        "last_output_folder": str(out), "last_transcripts_folder": ""}})
    _enable_only_acoustic(save_preferences)


def test_inverted_pitch_range_blocks_run(qtbot, tmp_path, monkeypatch):
    monkeypatch.setenv("SENSI_PREFS_PATH", str(tmp_path / "prefs.json"))
    from utils.preferences import save_preferences
    from gui.section_run import RunSection
    _ready_acoustic(tmp_path, save_preferences)
    w = RunSection(get_session_api_key=lambda: "", status_strip=None)
    qtbot.addWidget(w)
    assert w.run_button.isEnabled()
    save_preferences({"prosody": {"pitch_floor_hz": 400.0, "pitch_ceiling_hz": 100.0}})
    w.refresh()
    assert not w.run_button.isEnabled()
    assert "Pitch floor" in w.readiness_label.text()


def test_mfcc_above_mel_bands_blocks_run(qtbot, tmp_path, monkeypatch):
    monkeypatch.setenv("SENSI_PREFS_PATH", str(tmp_path / "prefs.json"))
    from utils.preferences import save_preferences
    from gui.section_run import RunSection
    _ready_acoustic(tmp_path, save_preferences)
    save_preferences({"acoustic_ts": {"n_mels": 16, "n_mfcc": 40}})
    w = RunSection(get_session_api_key=lambda: "", status_strip=None)
    qtbot.addWidget(w)
    assert not w.run_button.isEnabled()
    assert "MFCC count" in w.readiness_label.text()


def _word_only(tmp_path, save_preferences, transcripts):
    out = tmp_path / "out"; out.mkdir()
    save_preferences({
        "shared": {"analyzer_name": "me", "last_audio_folder": "",
                   "last_output_folder": str(out),
                   "last_transcripts_folder": str(transcripts)},
        "output": {"level_word": True, "level_document": False,
                   "level_subword": False, "level_sentence": False,
                   "level_acoustic": False},
    })


def test_text_only_run_needs_no_audio_folder(qtbot, tmp_path, monkeypatch):
    monkeypatch.setenv("SENSI_PREFS_PATH", str(tmp_path / "prefs.json"))
    from utils.preferences import save_preferences
    from gui.section_run import RunSection
    from gui.planner import plan_from_prefs
    from utils.preferences import load_preferences
    t = tmp_path / "t"; t.mkdir(); (t / "a.txt").write_text("hi", encoding="utf-8")
    _word_only(tmp_path, save_preferences, t)
    assert "mfa_align" not in plan_from_prefs(load_preferences()).stages_to_run
    w = RunSection(get_session_api_key=lambda: "", status_strip=None)
    qtbot.addWidget(w)
    assert w.run_button.isEnabled(), w.readiness_label.text()


def test_empty_transcripts_folder_blocks_run(qtbot, tmp_path, monkeypatch):
    monkeypatch.setenv("SENSI_PREFS_PATH", str(tmp_path / "prefs.json"))
    from utils.preferences import save_preferences
    from gui.section_run import RunSection
    t = tmp_path / "t"; t.mkdir()
    _word_only(tmp_path, save_preferences, t)
    w = RunSection(get_session_api_key=lambda: "", status_strip=None)
    qtbot.addWidget(w)
    assert not w.run_button.isEnabled()
    assert "no .txt" in w.readiness_label.text()


def test_cancel_button_reaches_the_runners(qtbot):
    from gui.section_run import RunSection
    strip = _FakeStatusStrip()
    w = RunSection(get_session_api_key=lambda: "", status_strip=strip)
    qtbot.addWidget(w)
    assert not w._cancel.is_set()
    strip.cancel_requested.emit()
    assert w._cancel.is_set()
    assert ("cancelling",) in strip.calls
    w._done_event.emit("cancelled")
    assert ("cancelled",) in strip.calls
