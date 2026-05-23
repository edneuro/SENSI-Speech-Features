def test_main_window_has_expected_sidebar_items(qtbot):
    from gui.main_window import MainWindow
    w = MainWindow()
    qtbot.addWidget(w)
    labels = [w.sidebar.item(i).text() for i in range(w.sidebar.count())]
    assert labels == ["Inputs", "Word-level features",
                       "Document-level features", "Sub-word features",
                       "Sentence features", "Run", "Logs"]


def test_main_window_default_section_is_inputs(qtbot):
    from gui.main_window import MainWindow
    w = MainWindow()
    qtbot.addWidget(w)
    assert w.stack.currentIndex() == 0


def test_main_window_clicking_sidebar_switches_stack(qtbot):
    from gui.main_window import MainWindow
    w = MainWindow()
    qtbot.addWidget(w)
    w.sidebar.setCurrentRow(2)  # Document-level features
    assert w.stack.currentIndex() == 2


def test_main_window_min_size(qtbot):
    from gui.main_window import MainWindow
    w = MainWindow()
    qtbot.addWidget(w)
    assert w.minimumWidth() >= 1000
    assert w.minimumHeight() >= 700


def test_main_window_title_is_sensi(qtbot):
    from gui.main_window import MainWindow
    w = MainWindow()
    qtbot.addWidget(w)
    assert w.windowTitle() == "SENSI-FeatureExtractor"
