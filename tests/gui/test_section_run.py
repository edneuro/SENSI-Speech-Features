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


def _enable_only_acoustic(save_preferences):
    """Select the acoustic .mat as the ONLY output (audio-only run)."""
    save_preferences({
        "output": {
            "word_level_csv": False, "document_summary": False,
            "syllables_csv": False, "phonemes_csv": False,
            "sentences_csv": False, "word2vec_mat": False,
            "bert_word_mat": False, "bert_sentence_mat": False,
            "acoustic_ts_mat": True,
        },
        "features": {"master_acoustic_ts": True, "feat_acoustic_ts_envelope": True},
    })


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
        "output": {"word_level_csv": True},
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


class _FakeStatusStrip:
    """Minimal stand-in for StatusStrip — captures the last call made."""

    def __init__(self):
        self.calls = []

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


def test_done_event_drives_status_strip_on_success(qtbot):
    from gui.section_run import RunSection
    strip = _FakeStatusStrip()
    w = RunSection(get_session_api_key=lambda: "", status_strip=strip)
    qtbot.addWidget(w)
    w.run_button.setEnabled(False)
    w._done_event.emit("done")
    assert any(c[0] == "done" for c in strip.calls)
    # Slot also re-enables the Run button.
    assert w.run_button.isEnabled()


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
