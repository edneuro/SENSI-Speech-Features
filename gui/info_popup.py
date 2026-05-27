"""Info-icon button + feature-info popup dialog.

Renders per-feature documentation .md files via QTextBrowser inside a Qt
modal dialog. No browser launch, no HTML build pipeline — the Markdown is
rendered in-app.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QDialog, QHBoxLayout, QLabel, QPushButton, QTextBrowser, QVBoxLayout,
)

from gui.feature_docs import FeatureDoc, load_feature_doc


_INFO_ICON_PATH = (
    Path(__file__).resolve().parent.parent / "assets" / "info.svg"
)


class InfoIcon(QPushButton):
    """Clickable info-circle SVG that opens a FeatureInfoDialog for `toggle_key`.

    Caller is responsible for only constructing this widget when a doc
    actually exists (see gui.feature_docs.has_feature_doc).
    """

    def __init__(self, toggle_key: str, parent=None):
        super().__init__(parent)
        self._toggle_key = toggle_key
        self.setObjectName("InfoIcon")
        self.setCursor(Qt.PointingHandCursor)
        self.setFlat(True)
        self.setIcon(QIcon(str(_INFO_ICON_PATH)))
        self.setIconSize(QSize(16, 16))
        self.setFixedSize(22, 22)
        self.setToolTip("Show feature documentation")
        self.clicked.connect(self._open_dialog)

    def _open_dialog(self) -> None:
        dlg = FeatureInfoDialog(self._toggle_key, self)
        dlg.exec()


class FeatureInfoDialog(QDialog):
    """Modal dialog rendering a feature's .md content."""

    def __init__(self, toggle_key: str, parent=None):
        super().__init__(parent)
        self._toggle_key = toggle_key
        self._doc: FeatureDoc | None = load_feature_doc(toggle_key)

        title = self._doc.name if self._doc else toggle_key
        self.setWindowTitle(f"Feature: {title}")
        self.resize(760, 640)

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(8)

        # Header strip — quick-glance frontmatter facts
        header_text = self._build_header()
        if header_text:
            header = QLabel(header_text)
            header.setObjectName("Subtitle")
            header.setWordWrap(True)
            root.addWidget(header)

        self.browser = QTextBrowser()
        self.browser.setOpenExternalLinks(True)
        if self._doc is None:
            self.browser.setPlainText(
                f"No documentation file found for '{toggle_key}'.\n\n"
                f"Expected at docs/features/{toggle_key}.md"
            )
        else:
            self.browser.setMarkdown(self._doc.body)
        root.addWidget(self.browser, 1)

        # Close button row
        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        btn_row.addWidget(close_btn)
        root.addLayout(btn_row)

    def _build_header(self) -> str:
        """Render frontmatter fields (libraries, status, levels) as a header strip."""
        if self._doc is None:
            return ""
        fm = self._doc.frontmatter
        parts: list[str] = []
        if fm.get("libraries"):
            libs = fm["libraries"]
            if isinstance(libs, list):
                libs = ", ".join(libs)
            parts.append(f"Library: {libs}")
        if fm.get("levels"):
            lvls = fm["levels"]
            if isinstance(lvls, list):
                lvls = ", ".join(lvls)
            parts.append(f"Level: {lvls}")
        if fm.get("status"):
            parts.append(f"Status: {fm['status']}")
        return "    ·    ".join(parts)
