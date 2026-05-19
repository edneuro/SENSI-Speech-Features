def test_dialog_word_level_title(qtbot):
    from gui.category_dialog import CategoryDialog
    d = CategoryDialog(backend="empath", level="word")
    qtbot.addWidget(d)
    assert "word-level" in d.windowTitle().lower()


def test_dialog_doc_level_title(qtbot):
    from gui.category_dialog import CategoryDialog
    d = CategoryDialog(backend="empath", level="document")
    qtbot.addWidget(d)
    assert "document-level" in d.windowTitle().lower()


def test_dialog_embeds_category_picker(qtbot):
    from gui.category_dialog import CategoryDialog
    d = CategoryDialog(backend="empath", level="word")
    qtbot.addWidget(d)
    assert d.picker is not None
    assert d.picker._level == "word"


def test_dialog_done_button_closes(qtbot):
    from gui.category_dialog import CategoryDialog
    d = CategoryDialog(backend="empath", level="word")
    qtbot.addWidget(d)
    d.show()
    d.done_button.click()
    assert not d.isVisible()
