"""Tests for gui.section_logs (Logs tab — live tail + past runs)."""
from pathlib import Path


def test_logs_section_has_text_viewer(qtbot):
    from gui.section_logs import LogsSection
    w = LogsSection()
    qtbot.addWidget(w)
    assert w.viewer is not None


def test_past_runs_lists_logs_from_logs_subfolder(qtbot, tmp_path):
    """Log artifacts now live under <output>/logs/, not directly in the
    output folder."""
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "word_run_2026-05-12_14-30-22.log").write_text("done", encoding="utf-8")
    (logs / "doc_run_2026-05-12_14-30-22.log").write_text("done", encoding="utf-8")
    (logs / "word_run_2026-05-11_10-15-44.log").write_text("done", encoding="utf-8")

    from gui.section_logs import LogsSection
    w = LogsSection(output_folder=tmp_path)
    qtbot.addWidget(w)

    items = [w.past_runs_list.item(i).text()
             for i in range(w.past_runs_list.count())]
    log_items = [s for s in items if s.endswith(".log")]
    assert any("2026-05-12_14-30-22" in s and "word_run" in s for s in log_items)
    assert any("2026-05-12_14-30-22" in s and "doc_run" in s for s in log_items)
    assert any("2026-05-11_10-15-44" in s for s in log_items)


def test_past_runs_inserts_separator_between_batches(qtbot, tmp_path):
    """Word + doc logs sharing the same timestamp are one 'batch' and
    list consecutively; different timestamps get a separator line between
    them."""
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "word_run_2026-05-12_14-30-22.log").write_text("", encoding="utf-8")
    (logs / "doc_run_2026-05-12_14-30-22.log").write_text("", encoding="utf-8")
    (logs / "word_run_2026-05-11_10-15-44.log").write_text("", encoding="utf-8")

    from gui.section_logs import LogsSection, _GROUP_SEPARATOR
    w = LogsSection(output_folder=tmp_path)
    qtbot.addWidget(w)

    items = [w.past_runs_list.item(i).text()
             for i in range(w.past_runs_list.count())]
    # 2 batches → exactly 1 separator
    assert items.count(_GROUP_SEPARATOR) == 1


def test_live_tail_picks_up_new_log_file(qtbot, tmp_path):
    """When start_live is called and a new log appears under <output>/logs/,
    a subsequent _poll_tail picks up its contents into the viewer."""
    from gui.section_logs import LogsSection
    w = LogsSection(output_folder=tmp_path)
    qtbot.addWidget(w)

    w.start_live()
    # Don't actually run the QTimer — drive _poll_tail directly to keep the
    # test deterministic without spinning the event loop.
    logs = tmp_path / "logs"
    logs.mkdir(exist_ok=True)
    log_path = logs / "word_run_2026-05-12_14-30-22.log"
    log_path.write_text("[14:30:22] INFO: Run started\n", encoding="utf-8")
    w._poll_tail()
    assert "Run started" in w.viewer.toPlainText()

    # Append more bytes — second poll picks up only the new bytes.
    with log_path.open("a", encoding="utf-8") as f:
        f.write("[14:30:23] INFO: Processing story1\n")
    w._poll_tail()
    assert "Processing story1" in w.viewer.toPlainText()

    w.stop_live()


def test_load_past_run_replaces_viewer_contents(qtbot, tmp_path):
    """Activating a past-run list item loads that file into the viewer."""
    from gui.section_logs import LogsSection
    logs = tmp_path / "logs"
    logs.mkdir()
    log_path = logs / "word_run_2026-05-12_14-30-22.log"
    log_path.write_text(
        "[14:30:22] INFO: Old run line\n[14:30:25] INFO: Old run done\n",
        encoding="utf-8",
    )

    w = LogsSection(output_folder=tmp_path)
    qtbot.addWidget(w)
    # Find the QListWidgetItem matching the log file (skip any separator).
    target_item = None
    for i in range(w.past_runs_list.count()):
        item = w.past_runs_list.item(i)
        if str(log_path) in item.text():
            target_item = item
            break
    assert target_item is not None
    w._load_past_run(target_item)
    text = w.viewer.toPlainText()
    assert "Old run line" in text
    assert "Old run done" in text
