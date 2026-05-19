"""Reusable Qt widgets used across the GUI."""

from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QVBoxLayout,
    QLabel, QLineEdit, QPushButton, QCheckBox,
    QFileDialog, QWidget, QLayout,
)


class SectionHeader(QWidget):
    def __init__(self, title: str, subtitle: str = "", parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 12, 0, 4)
        layout.setSpacing(2)
        self.title_label = QLabel(title)
        self.title_label.setObjectName("SectionHeader")
        layout.addWidget(self.title_label)
        self.subtitle_label = QLabel(subtitle)
        self.subtitle_label.setObjectName("Subtitle")
        if not subtitle:
            self.subtitle_label.hide()
        layout.addWidget(self.subtitle_label)
        divider = QFrame()
        divider.setObjectName("SectionDivider")
        divider.setFixedHeight(1)
        layout.addWidget(divider)
        layout.addSpacing(2)


class FolderRow(QWidget):
    path_changed = Signal(str)

    def __init__(self, label: str, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        lbl = QLabel(label)
        lbl.setMinimumWidth(120)
        self.line = QLineEdit()
        self.line.textChanged.connect(self._on_text)
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._on_browse)
        layout.addWidget(lbl)
        layout.addWidget(self.line, 1)
        layout.addWidget(browse)

    def set_path(self, p: str) -> None:
        self.line.setText(p)

    def path(self) -> str:
        return self.line.text()

    def set_error(self, is_error: bool) -> None:
        self.line.setProperty("error", "true" if is_error else "false")
        self.line.style().unpolish(self.line)
        self.line.style().polish(self.line)

    def _on_text(self, text: str) -> None:
        self.path_changed.emit(text)

    def _on_browse(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select folder", self.line.text())
        if path:
            self.line.setText(path)


class ToggleRow(QWidget):
    toggled = Signal(bool)

    def __init__(self, label: str, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.check = QCheckBox(label)
        self.check.toggled.connect(self.toggled)
        layout.addWidget(self.check)
        layout.addStretch(1)
        self.warning = None  # populated by set_warning

    def set_checked(self, v: bool) -> None:
        self.check.setChecked(v)

    def is_checked(self) -> bool:
        return self.check.isChecked()

    def set_warning(self, message: str | None) -> None:
        if self.warning is not None:
            self.layout().removeWidget(self.warning)
            self.warning.deleteLater()
            self.warning = None
        if message:
            self.warning = AmberDotIndicator(message)
            self.layout().insertWidget(1, self.warning)


class AmberDotIndicator(QLabel):
    def __init__(self, tooltip: str, parent=None):
        super().__init__("🟡", parent)
        self.setToolTip(tooltip)
        self.setFixedWidth(20)


class Card(QFrame):
    """A white-background, bordered container for grouping a feature subsection."""

    def __init__(self, title: str, subtitle: str = "", parent=None):
        super().__init__(parent)
        self.setObjectName("Card")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 16)
        layout.setSpacing(8)
        self.header = SectionHeader(title, subtitle)
        layout.addWidget(self.header)
        self._content_layout = layout

    def add_widget(self, w: QWidget) -> None:
        self._content_layout.addWidget(w)

    def add_layout(self, lay: QLayout) -> None:
        self._content_layout.addLayout(lay)
