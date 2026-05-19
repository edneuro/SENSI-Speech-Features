def test_empty_analyzer_shows_error(qtbot):
    from gui.section_inputs import InputsSection
    w = InputsSection()
    qtbot.addWidget(w)
    w.analyzer_line.setText("")
    assert w.analyzer_line.property("error") == "true"


def test_filled_analyzer_clears_error(qtbot):
    from gui.section_inputs import InputsSection
    w = InputsSection()
    qtbot.addWidget(w)
    w.analyzer_line.setText("Amilcar")
    assert w.analyzer_line.property("error") != "true"


def test_nonexistent_folder_shows_error(qtbot, tmp_path):
    from gui.section_inputs import InputsSection
    w = InputsSection()
    qtbot.addWidget(w)
    w.transcripts_row.set_path("/does/not/exist")
    assert w.transcripts_row.line.property("error") == "true"


def test_valid_folder_shows_file_count(qtbot, tmp_path):
    from gui.section_inputs import InputsSection
    (tmp_path / "a.txt").write_text("hello", encoding="utf-8")
    (tmp_path / "b.txt").write_text("world", encoding="utf-8")
    w = InputsSection()
    qtbot.addWidget(w)
    w.transcripts_row.set_path(str(tmp_path))
    assert "2 files" in w.transcripts_count_label.text()
