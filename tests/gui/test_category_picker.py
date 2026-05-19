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
    # v7: unified categories_<backend> key (no split word/doc keys)
    assert "friends" in prefs["semantic"]["categories_empath"]


def test_picker_word_level_uses_unified_categories_key(qtbot, tmp_path, monkeypatch):
    """v7: word and doc levels both save to categories_<backend> (unified key)."""
    monkeypatch.setenv("SENSI_PREFS_PATH", str(tmp_path / "p.json"))
    from gui.category_picker import CategoryPicker
    from utils.preferences import load_preferences
    w = CategoryPicker(backend="empath", level="word")
    qtbot.addWidget(w)
    w.set_selection(["money"])
    w._checks["money"].setChecked(True)
    prefs = load_preferences()
    assert "money" in prefs["semantic"]["categories_empath"]
    # Split keys must not exist in v7 prefs
    assert "word_categories_empath" not in prefs["semantic"]
    assert "doc_categories_empath" not in prefs["semantic"]


def test_picker_doc_level_uses_unified_categories_key(qtbot, tmp_path, monkeypatch):
    """v7: doc-level picker saves to the same unified categories_<backend> key."""
    monkeypatch.setenv("SENSI_PREFS_PATH", str(tmp_path / "p.json"))
    from gui.category_picker import CategoryPicker
    from utils.preferences import load_preferences
    w = CategoryPicker(backend="empath", level="document")
    qtbot.addWidget(w)
    w._checks["money"].setChecked(True)
    prefs = load_preferences()
    assert "money" in prefs["semantic"]["categories_empath"]


def test_picker_default_level_is_word_for_backward_compat(qtbot):
    from gui.category_picker import CategoryPicker
    w = CategoryPicker(backend="empath")  # no level arg
    qtbot.addWidget(w)
    # Should default to "word" so older test calls keep working
    assert w._level == "word"


def test_picker_doc_level_writes_to_unified_not_document_key(qtbot, tmp_path, monkeypatch):
    """Regression: the picker used to map level='document' to a key prefix
    of 'document_' which doesn't match the schema's unified key. v7 uses
    categories_<backend> for both word and doc levels.
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
    assert non_sop in prefs["semantic"]["categories_empath"]
    # Neither the old split key nor the old buggy key should appear.
    assert "doc_categories_empath" not in prefs["semantic"]
    assert "document_categories_empath" not in prefs["semantic"]


def test_load_preferences_migrates_document_key_into_unified_key(tmp_path, monkeypatch):
    """If a user upgrades from a build that wrote to the buggy
    `document_categories_*` key, load_preferences must repair it to
    `doc_categories_*` (v5→v6 repair step) and then unify it into
    `categories_*` (v6→v7 migration) so the selection isn't lost."""
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
    # After repair (document→doc) and v7 union migration, selections land
    # in the unified key.
    cats = prefs["semantic"]["categories_empath"]
    assert "violence" in cats
    assert "love" in cats
    assert "document_categories_empath" not in prefs["semantic"]
    assert "doc_categories_empath" not in prefs["semantic"]
