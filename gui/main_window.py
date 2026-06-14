"""Main window shell: sidebar + content stack + status strip."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QListWidget, QListWidgetItem, QStackedWidget,
    QFrame, QLabel, QScrollArea, QAbstractScrollArea,
)

from gui.theme import qss_for_appearance
from gui.status_strip import StatusStrip
from gui.section_inputs import InputsSection
from gui.section_word_features import WordFeaturesSection
from gui.section_doc_features import DocFeaturesSection
from gui.section_subword_features import SubwordFeaturesSection
from gui.section_sentence_features import SentenceFeaturesSection
from gui.section_acoustic_features import AcousticFeaturesSection
from gui.section_run import RunSection
from gui.section_logs import LogsSection
from utils.preferences import load_preferences


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
        self.sidebar.setFrameShape(QFrame.NoFrame)
        # No scrollbars in the sidebar — items always fit at minimum window height.
        self.sidebar.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.sidebar.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        # Match content size to items so the widget itself doesn't expand
        # past the items and create empty interior scrollable space.
        self.sidebar.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)
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

        # Content stack
        self.stack = QStackedWidget()

        self.inputs_section = InputsSection()
        self.word_features = WordFeaturesSection()
        self.doc_features = DocFeaturesSection()
        self.subword_section = SubwordFeaturesSection()
        self.sentence_section = SentenceFeaturesSection()
        self.acoustic_section = AcousticFeaturesSection()
        self.run_section = RunSection(
            get_session_api_key=self.word_features.session_api_key,
            status_strip=self.status_strip,
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
            area.setWidget(s)
            self.stack.addWidget(area)

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

        root.addWidget(top_bar)
        root.addWidget(body, 1)
        root.addWidget(self.status_strip)

        self.setCentralWidget(central)

    def _footer_text(self) -> str:
        # Re-read from disk so the footer reflects the current analyzer name
        # (otherwise it stays frozen at whatever was on disk at app launch).
        prefs = load_preferences()
        analyzer = prefs["shared"].get("analyzer_name") or "(no analyzer)"
        return f"Analyzer: {analyzer}"

    def _refresh_sidebar_footer(self) -> None:
        self.sidebar_footer.setText(self._footer_text())
