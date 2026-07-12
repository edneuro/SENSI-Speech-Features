"""Every section page must fit the window's minimum width without a
horizontal scrollbar (user report, 2026-07-12): labels wrap and wide rows
split so content reflows vertically. The main window is min 1000px wide;
after the sidebar (~200) and scrollbar (~15) the viewport is ~780px."""
import pytest

VIEWPORT_BUDGET = 780


def _sections():
    from gui.section_inputs import InputsSection
    from gui.section_word_features import WordFeaturesSection
    from gui.section_doc_features import DocFeaturesSection
    from gui.section_subword_features import SubwordFeaturesSection
    from gui.section_sentence_features import SentenceFeaturesSection
    from gui.section_acoustic_features import AcousticFeaturesSection
    from gui.section_run import RunSection
    from gui.section_logs import LogsSection
    return {
        "inputs": InputsSection(),
        "word": WordFeaturesSection(),
        "doc": DocFeaturesSection(),
        "subword": SubwordFeaturesSection(),
        "sentence": SentenceFeaturesSection(),
        "acoustic": AcousticFeaturesSection(),
        "run": RunSection(get_session_api_key=lambda: "", status_strip=None),
        "logs": LogsSection(),
    }


def test_every_page_fits_min_window_width(qtbot, prefs_tmp):
    over = {}
    for name, s in _sections().items():
        qtbot.addWidget(s)
        w = s.minimumSizeHint().width()
        if w > VIEWPORT_BUDGET:
            over[name] = w
    assert not over, (
        f"pages wider than the {VIEWPORT_BUDGET}px viewport budget: {over} — "
        "wrap the long label or split the wide row instead of letting a "
        "horizontal scrollbar hide content")


def test_section_scroll_areas_disable_horizontal_scrollbar(qtbot, prefs_tmp):
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QScrollArea
    from gui.main_window import MainWindow
    win = MainWindow()
    qtbot.addWidget(win)
    areas = [win.stack.widget(i) for i in range(win.stack.count())]
    scroll_areas = [a for a in areas if isinstance(a, QScrollArea)]
    assert scroll_areas, "expected section pages wrapped in QScrollAreas"
    for a in scroll_areas:
        assert a.horizontalScrollBarPolicy() == Qt.ScrollBarAlwaysOff
