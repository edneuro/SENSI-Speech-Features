def test_picker_lists_all_backend_categories(qtbot):
    from gui.category_picker import CategoryPicker
    w = CategoryPicker(backend="empath")
    qtbot.addWidget(w)
    assert w.total_count() > 100   # Empath ships ~200 categories


def test_picker_shows_selected_count(qtbot):
    from gui.category_picker import CategoryPicker
    w = CategoryPicker(backend="empath")
    qtbot.addWidget(w)
    w.set_selection(["friends", "philosophy"])
    assert "2 of" in w.summary_label.text()


def test_picker_search_filters_categories(qtbot):
    from gui.category_picker import CategoryPicker
    w = CategoryPicker(backend="empath")
    qtbot.addWidget(w)
    w.search_field.setText("emoti")
    visible = [name for name, cb in w._checks.items() if cb.isVisible()]
    assert all("emoti" in n.lower() or w._checks[n].isChecked() for n in visible)


def test_picker_sop_default_button_resets_selection(qtbot, tmp_path, monkeypatch):
    monkeypatch.setenv("SENSI_PREFS_PATH", str(tmp_path / "prefs.json"))
    from gui.category_picker import CategoryPicker
    from utils.preferences import _SOP_CATEGORIES_EMPATH
    w = CategoryPicker(backend="empath")
    qtbot.addWidget(w)
    w.set_selection([])
    w.sop_default_button.click()
    assert sorted(w.current_selection()) == sorted(_SOP_CATEGORIES_EMPATH)


def test_picker_auto_saves_on_toggle(qtbot, tmp_path, monkeypatch):
    monkeypatch.setenv("SENSI_PREFS_PATH", str(tmp_path / "prefs.json"))
    from gui.category_picker import CategoryPicker
    from utils.preferences import load_preferences
    w = CategoryPicker(backend="empath")
    qtbot.addWidget(w)
    w.set_selection([])
    w._checks["friends"].setChecked(True)
    prefs = load_preferences()
    assert "friends" in prefs["semantic"]["word_categories_empath"]


def test_picker_word_level_uses_word_categories_key(qtbot, tmp_path, monkeypatch):
    monkeypatch.setenv("SENSI_PREFS_PATH", str(tmp_path / "p.json"))
    from gui.category_picker import CategoryPicker
    from utils.preferences import load_preferences
    w = CategoryPicker(backend="empath", level="word")
    qtbot.addWidget(w)
    w.set_selection(["money"])
    # Toggle one
    w._checks["money"].setChecked(True)
    prefs = load_preferences()
    assert "money" in prefs["semantic"]["word_categories_empath"]
    # Doc-level list should be untouched
    assert "money" not in prefs["semantic"]["doc_categories_empath"] or \
           prefs["semantic"]["doc_categories_empath"] != \
           prefs["semantic"]["word_categories_empath"]


def test_picker_doc_level_uses_doc_categories_key(qtbot, tmp_path, monkeypatch):
    monkeypatch.setenv("SENSI_PREFS_PATH", str(tmp_path / "p.json"))
    from gui.category_picker import CategoryPicker
    from utils.preferences import load_preferences
    w = CategoryPicker(backend="empath", level="document")
    qtbot.addWidget(w)
    w._checks["money"].setChecked(True)
    prefs = load_preferences()
    assert "money" in prefs["semantic"]["doc_categories_empath"]


def test_picker_default_level_is_word_for_backward_compat(qtbot):
    from gui.category_picker import CategoryPicker
    w = CategoryPicker(backend="empath")  # no level arg
    qtbot.addWidget(w)
    # Should default to "word" so older test calls keep working
    assert w._level == "word"


def test_picker_doc_level_writes_to_doc_not_document_key(qtbot, tmp_path, monkeypatch):
    """Regression: the picker used to map level='document' to a key prefix
    of 'document_' which doesn't match the schema's 'doc_' prefix. The doc-
    level dialog's selections never reached the counter or the pipeline.
    Verify by toggling a non-SOP category so the assertion isn't a false
    positive on the default selection."""
    monkeypatch.setenv("SENSI_PREFS_PATH", str(tmp_path / "p.json"))
    from gui.category_picker import CategoryPicker
    from utils.preferences import load_preferences, _SOP_CATEGORIES_EMPATH

    non_sop = "violence"  # in empath but not in SOP defaults
    assert non_sop not in _SOP_CATEGORIES_EMPATH

    w = CategoryPicker(backend="empath", level="document")
    qtbot.addWidget(w)
    w._checks[non_sop].setChecked(True)

    prefs = load_preferences()
    assert non_sop in prefs["semantic"]["doc_categories_empath"]
    # And the bogus key MUST not appear in saved prefs.
    assert "document_categories_empath" not in prefs["semantic"]


def test_load_preferences_migrates_document_key_into_doc_key(tmp_path, monkeypatch):
    """If a user upgrades from a build that wrote to the buggy
    `document_categories_*` key, load_preferences must move the value into
    the canonical `doc_categories_*` slot so the selection isn't lost."""
    import json
    monkeypatch.setenv("SENSI_PREFS_PATH", str(tmp_path / "p.json"))
    prefs_path = tmp_path / "p.json"
    prefs_path.write_text(json.dumps({
        "schema_version": 5,
        "semantic": {
            "backend": "empath",
            "document_categories_empath": ["violence", "love"],
        },
    }))

    from utils.preferences import load_preferences
    prefs = load_preferences()
    assert prefs["semantic"]["doc_categories_empath"] == ["violence", "love"]
    assert "document_categories_empath" not in prefs["semantic"]
