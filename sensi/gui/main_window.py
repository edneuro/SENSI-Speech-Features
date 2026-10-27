"""Main window shell: sidebar + content stack + status strip."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QListWidget, QListWidgetItem, QStackedWidget,
    QFrame, QLabel, QScrollArea, QAbstractScrollArea,
    QPushButton, QMessageBox,
)

from sensi.gui.theme import qss_for_appearance
from sensi.gui.status_strip import StatusStrip
from sensi.gui.section_inputs import InputsSection
from sensi.gui.section_word_features import WordFeaturesSection
from sensi.gui.section_doc_features import DocFeaturesSection
from sensi.gui.section_subword_features import SubwordFeaturesSection
from sensi.gui.section_sentence_features import SentenceFeaturesSection
from sensi.gui.section_acoustic_features import AcousticFeaturesSection
from sensi.gui.section_run import RunSection
from sensi.gui.section_logs import LogsSection
from sensi.utils.preferences import load_preferences, reset_to_defaults


APP_NAME = "SENSI-FeatureExtractor"


SECTIONS = ["Inputs", "Word-level features", "Document-level features",
            "Sub-word features", "Sentence features", "Acoustic", "Run", "Logs"]


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(APP_NAME)
        self.setMinimumSize(1000, 700)
        self.resize(1200, 800)

        self._prefs = load_preferences()
        # Stylesheet is applied at the QApplication level in gui.py so it
        # cascades to all child widgets including future modal dialogs.
        # No need to set it again on MainWindow.

        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        body = QWidget()
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(0)

        # Sidebar
        sidebar_frame = QFrame()
        sidebar_frame.setObjectName("Sidebar")
        sidebar_frame.setFixedWidth(220)
        sb_layout = QVBoxLayout(sidebar_frame)
        sb_layout.setContentsMargins(8, 8, 8, 8)
        self.sidebar = QListWidget()
        self.sidebar.setFrameShape(QFrame.Shape.NoFrame)
        # Gap between nav items. Use view spacing, not QSS `::item` margin —
        # Qt 6.9 renders item margins as overlap, so the selected pill bled
        # onto its neighbours ("tabs on top of each other").
        self.sidebar.setSpacing(4)
        # No scrollbars in the sidebar — items always fit at minimum window height.
        self.sidebar.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.sidebar.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        # Match content size to items so the widget itself doesn't expand
        # past the items and create empty interior scrollable space.
        self.sidebar.setSizeAdjustPolicy(QAbstractScrollArea.SizeAdjustPolicy.AdjustToContents)
        for name in SECTIONS:
            self.sidebar.addItem(QListWidgetItem(name))
        self.sidebar.setCurrentRow(0)
        sb_layout.addWidget(self.sidebar)
        sb_layout.addStretch(1)
        self.sidebar_footer = QLabel(self._footer_text())
        self.sidebar_footer.setObjectName("Subtitle")
        self.sidebar_footer.setWordWrap(True)
        sb_layout.addWidget(self.sidebar_footer)

        # Status strip (created early so RunSection can hold a reference to it)
        self.status_strip = StatusStrip()
        # A settings save that fails even after retrying (file locked by
        # another program) is shown in the strip instead of only the console.
        from sensi.utils.preferences import set_save_error_handler
        set_save_error_handler(self.status_strip.show_message)

        # Content stack (populated by _build_and_wire_sections; rebuilt on reset)
        self.stack = QStackedWidget()
        self._build_and_wire_sections()
        # sidebar → stack connection is made once here; _build_and_wire_sections
        # only swaps the stack's contents, so this stays valid across rebuilds.
        self.sidebar.currentRowChanged.connect(self.stack.setCurrentIndex)

        body_layout.addWidget(sidebar_frame)
        body_layout.addWidget(self.stack, 1)

        # Top bar with app name (theme buttons removed — light theme only)
        top_bar = QFrame()
        top_bar.setObjectName("TopBar")
        top_bar.setFixedHeight(44)
        tb = QHBoxLayout(top_bar)
        tb.setContentsMargins(16, 4, 16, 4)
        title = QLabel(APP_NAME)
        title.setStyleSheet("font-weight: 700; font-size: 14px;")
        tb.addWidget(title, 1)
        self.reset_button = QPushButton("Reset settings")
        self.reset_button.setToolTip(
            "Reset all feature/output toggles to their defaults.\n"
            "Your input/output folders and analyzer name are kept.")
        self.reset_button.clicked.connect(self._on_reset_settings)
        tb.addWidget(self.reset_button, 0)

        root.addWidget(top_bar)
        root.addWidget(body, 1)
        root.addWidget(self.status_strip)

        self.setCentralWidget(central)

    def _build_and_wire_sections(self) -> None:
        """Create the eight section widgets, wire their cross-section signals,
        and populate the content stack. Re-runnable: called once at startup and
        again after a settings reset (widgets read prefs in __init__, so
        rebuilding is the simplest way to guarantee every page reflects the
        reset without hand-resyncing each checkbox)."""
        self.inputs_section = InputsSection()
        self.word_features = WordFeaturesSection()
        self.doc_features = DocFeaturesSection()
        self.subword_section = SubwordFeaturesSection()
        self.sentence_section = SentenceFeaturesSection()
        self.acoustic_section = AcousticFeaturesSection()
        self.run_section = RunSection(
            get_session_api_key=self.word_features.session_api_key,
            status_strip=self.status_strip,
            get_session_hf_token=self.word_features.session_hf_token,
        )
        self.logs_section = LogsSection()

        # Cross-section refresh: feature/settings changes update the Run section
        self.word_features.settings_changed.connect(self.run_section.refresh)
        self.doc_features.settings_changed.connect(self.run_section.refresh)
        self.subword_section.settings_changed.connect(self.run_section.refresh)
        self.sentence_section.settings_changed.connect(self.run_section.refresh)
        self.acoustic_section.settings_changed.connect(self.run_section.refresh)
        # Inputs (analyzer name, folder pickers) drive Run-tab readiness too.
        self.inputs_section.settings_changed.connect(self.run_section.refresh)
        # Word-level changes (backend switch, category edits) must also refresh
        # the doc-level page so its category count and psych-card availability
        # stay in sync without a restart.
        self.word_features.settings_changed.connect(self.doc_features.refresh_from_prefs)
        # Doc-level toggle changes and Run-tab output toggle changes (e.g. word_level_csv)
        # must also refresh aggregate warnings on the doc-level page.
        self.doc_features.settings_changed.connect(self.doc_features.refresh_from_prefs)
        self.run_section.settings_changed.connect(self.doc_features.refresh_from_prefs)
        # The Acoustic page's card switch and the Run tab's Acoustic level
        # switch are one setting (output.level_acoustic): mirror Run → page.
        self.run_section.settings_changed.connect(self.acoustic_section.refresh_from_prefs)
        # Sidebar footer follows the analyzer-name field in Inputs
        self.inputs_section.settings_changed.connect(self._refresh_sidebar_footer)
        # Logs tab live-tails the active run; stop on completion.
        self.run_section.run_started.connect(self.logs_section.start_live)
        self.run_section.run_finished.connect(lambda _status: self.logs_section.stop_live())

        sections = [
            self.inputs_section,
            self.word_features,
            self.doc_features,
            self.subword_section,
            self.sentence_section,
            self.acoustic_section,
            self.run_section,
            self.logs_section,
        ]
        for s in sections:
            area = QScrollArea()
            area.setWidgetResizable(True)
            # Vertical scrolling only: pages are designed to fit the window's
            # minimum width (labels wrap, wide rows split — a regression test
            # pins each section's minimum width), so a horizontal scrollbar
            # would only ever hide content.
            area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            area.setWidget(s)
            self.stack.addWidget(area)

    def _on_reset_settings(self) -> None:
        """Confirm, reset prefs to defaults (keeping I/O folders), and rebuild
        every page so the UI reflects the reset immediately."""
        resp = QMessageBox.question(
            self, "Reset settings?",
            "Reset all feature and output toggles to their defaults?\n\n"
            "Your input/output folders and analyzer name are kept. This only "
            "affects this computer (preferences are stored locally).",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel, QMessageBox.StandardButton.Cancel)
        if resp != QMessageBox.StandardButton.Yes:
            return
        reset_to_defaults(keep_shared=True)
        current_row = self.sidebar.currentRow()
        # Tear down the old pages, then rebuild from the fresh prefs.
        while self.stack.count():
            w = self.stack.widget(0)
            self.stack.removeWidget(w)
            w.deleteLater()
        self._build_and_wire_sections()
        self.stack.setCurrentIndex(max(0, current_row))
        self._refresh_sidebar_footer()

    def _footer_text(self) -> str:
        # Re-read from disk so the footer reflects the current analyzer name
        # (otherwise it stays frozen at whatever was on disk at app launch).
        prefs = load_preferences()
        analyzer = prefs["shared"].get("analyzer_name") or "(no analyzer)"
        return f"Analyzer: {analyzer}"

    def _refresh_sidebar_footer(self) -> None:
        self.sidebar_footer.setText(self._footer_text())
