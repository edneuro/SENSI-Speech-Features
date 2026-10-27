def test_picker_lists_all_backend_categories(qtbot):
    from sensi.gui.category_picker import CategoryPicker
    w = CategoryPicker(backend="empath")
    qtbot.addWidget(w)
    assert w.total_count() > 100   # Empath ships ~200 categories


def test_picker_shows_selected_count(qtbot):
    from sensi.gui.category_picker import CategoryPicker
    w = CategoryPicker(backend="empath")
    qtbot.addWidget(w)
    w.set_selection(["friends", "philosophy"])
    assert "2 of" in w.summary_label.text()


def test_picker_search_filters_categories(qtbot):
    from sensi.gui.category_picker import CategoryPicker
    w = CategoryPicker(backend="empath")
    qtbot.addWidget(w)
    w.search_field.setText("emoti")
    visible = [name for name, cb in w._checks.items() if cb.isVisible()]
    assert all("emoti" in n.lower() or w._checks[n].isChecked() for n in visible)


def test_picker_sop_default_button_resets_selection(qtbot, tmp_path, monkeypatch):
    monkeypatch.setenv("SENSI_PREFS_PATH", str(tmp_path / "prefs.json"))
    from sensi.gui.category_picker import CategoryPicker
    from sensi.utils.preferences import _SOP_CATEGORIES_EMPATH
    w = CategoryPicker(backend="empath")
    qtbot.addWidget(w)
    w.set_selection([])
    w.sop_default_button.click()
    assert sorted(w.current_selection()) == sorted(_SOP_CATEGORIES_EMPATH)


def test_picker_auto_saves_on_toggle(qtbot, tmp_path, monkeypatch):
    monkeypatch.setenv("SENSI_PREFS_PATH", str(tmp_path / "prefs.json"))
    from sensi.gui.category_picker import CategoryPicker
    from sensi.utils.preferences import load_preferences
    w = CategoryPicker(backend="empath")
    qtbot.addWidget(w)
    w.set_selection([])
    w._checks["friends"].setChecked(True)
    prefs = load_preferences()
    # v7: unified categories_<backend> key (no split word/doc keys)
    assert "friends" in prefs["semantic"]["categories_empath"]


def test_picker_uses_unified_categories_key(qtbot, tmp_path, monkeypatch):
    """v7: the picker saves to categories_<backend> (unified key, no level split)."""
    monkeypatch.setenv("SENSI_PREFS_PATH", str(tmp_path / "p.json"))
    from sensi.gui.category_picker import CategoryPicker
    from sensi.utils.preferences import load_preferences
    w = CategoryPicker(backend="empath")
    qtbot.addWidget(w)
    w.set_selection(["money"])
    w._checks["money"].setChecked(True)
    prefs = load_preferences()
    assert "money" in prefs["semantic"]["categories_empath"]
    # Split keys must not exist in v7 prefs
    assert "word_categories_empath" not in prefs["semantic"]
    assert "doc_categories_empath" not in prefs["semantic"]


def test_picker_writes_to_unified_not_split_key(qtbot, tmp_path, monkeypatch):
    """Regression: the picker must not write any level-prefixed key.
    Verify by toggling a non-SOP category so the assertion isn't a false
    positive on the default selection."""
    monkeypatch.setenv("SENSI_PREFS_PATH", str(tmp_path / "p.json"))
    from sensi.gui.category_picker import CategoryPicker
    from sensi.utils.preferences import load_preferences, _SOP_CATEGORIES_EMPATH

    non_sop = "violence"  # in empath but not in SOP defaults
    assert non_sop not in _SOP_CATEGORIES_EMPATH

    w = CategoryPicker(backend="empath")
    qtbot.addWidget(w)
    w._checks[non_sop].setChecked(True)

    prefs = load_preferences()
    assert non_sop in prefs["semantic"]["categories_empath"]
    # Neither old split keys nor old buggy key should appear.
    assert "word_categories_empath" not in prefs["semantic"]
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

    from sensi.utils.preferences import load_preferences
    prefs = load_preferences()
    # After repair (document→doc) and v7 union migration, selections land
    # in the unified key.
    cats = prefs["semantic"]["categories_empath"]
    assert "violence" in cats
    assert "love" in cats
    assert "document_categories_empath" not in prefs["semantic"]
    assert "doc_categories_empath" not in prefs["semantic"]
