def test_section_header_renders_text(qtbot):
    from gui.widgets import SectionHeader
    w = SectionHeader("Frequency", subtitle="Lexical-corpus frequency lookups.")
    qtbot.addWidget(w)
    assert w.title_label.text() == "Frequency"
    assert w.subtitle_label.text() == "Lexical-corpus frequency lookups."


def test_folder_row_emits_signal_on_text_change(qtbot):
    from gui.widgets import FolderRow
    w = FolderRow(label="Transcripts")
    qtbot.addWidget(w)
    captured = []
    w.path_changed.connect(lambda p: captured.append(p))
    w.set_path("/tmp/abc")
    assert captured == ["/tmp/abc"]


def test_toggle_row_emits_signal_on_toggle(qtbot):
    from gui.widgets import ToggleRow
    w = ToggleRow(label="Surprisal from GPT-2")
    qtbot.addWidget(w)
    captured = []
    w.toggled.connect(lambda v: captured.append(v))
    w.set_checked(True)
    assert captured == [True]


def test_amber_dot_indicator_shows_tooltip(qtbot):
    from gui.widgets import AmberDotIndicator
    w = AmberDotIndicator("EWFG parquet not found")
    qtbot.addWidget(w)
    assert w.toolTip() == "EWFG parquet not found"


def test_card_holds_child_widgets(qtbot):
    from gui.widgets import Card
    from PySide6.QtWidgets import QLabel
    c = Card(title="Frequency", subtitle="Lexical-corpus frequency.")
    qtbot.addWidget(c)
    c.add_widget(QLabel("inside"))
    # The label should be a descendant of the card frame
    found = c.findChildren(QLabel)
    assert any(lbl.text() == "inside" for lbl in found)


def test_card_has_section_header(qtbot):
    from gui.widgets import Card, SectionHeader
    c = Card(title="Acoustic", subtitle="MFA.")
    qtbot.addWidget(c)
    assert any(isinstance(child, SectionHeader)
               for child in c.findChildren(SectionHeader))
