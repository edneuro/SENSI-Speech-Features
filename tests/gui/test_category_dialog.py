def test_dialog_word_level_title(qtbot):
    from sensi.gui.category_dialog import CategoryDialog
    d = CategoryDialog(backend="empath", level="word")
    qtbot.addWidget(d)
    assert "word-level" in d.windowTitle().lower()


def test_dialog_doc_level_title(qtbot):
    from sensi.gui.category_dialog import CategoryDialog
    d = CategoryDialog(backend="empath", level="document")
    qtbot.addWidget(d)
    assert "document-level" in d.windowTitle().lower()


def test_dialog_embeds_category_picker(qtbot):
    from sensi.gui.category_dialog import CategoryDialog
    from sensi.gui.category_picker import CategoryPicker
    d = CategoryDialog(backend="empath", level="word")
    qtbot.addWidget(d)
    assert d.picker is not None
    # v7: CategoryPicker has no _level attribute — unified key only.
    assert isinstance(d.picker, CategoryPicker)
    assert not hasattr(d.picker, "_level")


def test_dialog_done_button_closes(qtbot):
    from sensi.gui.category_dialog import CategoryDialog
    d = CategoryDialog(backend="empath", level="word")
    qtbot.addWidget(d)
    d.show()
    d.done_button.click()
    assert not d.isVisible()
