def test_main_window_has_expected_sidebar_items(qtbot):
    from sensi.gui.main_window import MainWindow
    w = MainWindow()
    qtbot.addWidget(w)
    labels = [w.sidebar.item(i).text() for i in range(w.sidebar.count())]
    assert labels == ["Inputs", "Word-level features",
                       "Document-level features", "Sub-word features",
                       "Sentence features", "Acoustic", "Run", "Logs"]


def test_main_window_default_section_is_inputs(qtbot):
    from sensi.gui.main_window import MainWindow
    w = MainWindow()
    qtbot.addWidget(w)
    assert w.stack.currentIndex() == 0


def test_main_window_clicking_sidebar_switches_stack(qtbot):
    from sensi.gui.main_window import MainWindow
    w = MainWindow()
    qtbot.addWidget(w)
    w.sidebar.setCurrentRow(2)  # Document-level features
    assert w.stack.currentIndex() == 2


def test_main_window_min_size(qtbot):
    from sensi.gui.main_window import MainWindow
    w = MainWindow()
    qtbot.addWidget(w)
    assert w.minimumWidth() >= 1000
    assert w.minimumHeight() >= 700


def test_main_window_title_is_sensi(qtbot):
    from sensi.gui.main_window import MainWindow
    w = MainWindow()
    qtbot.addWidget(w)
    assert w.windowTitle() == "SENSI-FeatureExtractor"


def test_reset_settings_restores_defaults_and_rebuilds(qtbot, prefs_tmp, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    from sensi.gui.main_window import MainWindow
    from sensi.utils.preferences import save_preferences, load_preferences, DEFAULT_PREFS

    save_preferences({"features": {"lexling_dep": False},
                      "shared": {"analyzer_name": "Alice"}})
    w = MainWindow()
    qtbot.addWidget(w)
    w.sidebar.setCurrentRow(2)  # Document-level features
    old_word_section = w.word_features

    # Auto-confirm the reset dialog.
    monkeypatch.setattr(QMessageBox, "question",
                        staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes))
    w._on_reset_settings()

    prefs = load_preferences()
    # Feature toggle back to default; shared (analyzer) preserved.
    assert prefs["features"]["lexling_dep"] == DEFAULT_PREFS["features"]["lexling_dep"]
    assert prefs["shared"]["analyzer_name"] == "Alice"
    # Pages rebuilt (fresh instances), count intact, selection preserved.
    assert w.stack.count() == 8
    assert w.word_features is not old_word_section
    assert w.stack.currentIndex() == 2


def test_reset_settings_cancel_leaves_prefs_untouched(qtbot, prefs_tmp, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    from sensi.gui.main_window import MainWindow
    from sensi.utils.preferences import save_preferences, load_preferences

    save_preferences({"features": {"lexling_dep": False}})
    w = MainWindow()
    qtbot.addWidget(w)
    monkeypatch.setattr(QMessageBox, "question",
                        staticmethod(lambda *a, **k: QMessageBox.StandardButton.Cancel))
    w._on_reset_settings()
    assert load_preferences()["features"]["lexling_dep"] is False
