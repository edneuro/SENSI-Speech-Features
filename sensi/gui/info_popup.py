"""Info-icon button + feature-info popup dialog.

Renders per-feature documentation .md files via QTextBrowser inside a Qt
modal dialog. No browser launch, no HTML build pipeline — the Markdown is
rendered in-app.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QSize, QUrl
from PySide6.QtGui import QIcon, QDesktopServices
from PySide6.QtWidgets import (
    QDialog, QHBoxLayout, QLabel, QPushButton, QTextBrowser, QVBoxLayout,
)

from sensi.gui.feature_docs import (
    FeatureDoc, load_feature_doc, load_doc_file, resolve_doc_link,
)

_EXTERNAL_SCHEMES = {"http", "https", "mailto", "ftp"}


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
        self.setCursor(Qt.CursorShape.PointingHandCursor)
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
        self._history: list[FeatureDoc] = []   # back-stack for in-place nav

        self.resize(760, 640)

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(8)

        # Header strip — quick-glance frontmatter facts (updated on navigation)
        self._header = QLabel()
        self._header.setObjectName("Subtitle")
        self._header.setWordWrap(True)
        root.addWidget(self._header)

        self.browser = QTextBrowser()
        # Take over link handling: QTextBrowser's default behaviour navigates
        # the document on any click, which blanks the view for relative
        # `.md` links. We route clicks ourselves in _on_anchor_clicked.
        self.browser.setOpenLinks(False)
        self.browser.setOpenExternalLinks(False)
        self.browser.anchorClicked.connect(self._on_anchor_clicked)
        root.addWidget(self.browser, 1)

        # Button row: Back (in-place navigation) + Close
        btn_row = QHBoxLayout()
        self._back_btn = QPushButton("← Back")
        self._back_btn.clicked.connect(self._go_back)
        btn_row.addWidget(self._back_btn)
        btn_row.addStretch(1)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        btn_row.addWidget(close_btn)
        root.addLayout(btn_row)

        self._render(self._doc, missing_key=toggle_key)

    def _render(self, doc: FeatureDoc | None, *, missing_key: str | None = None) -> None:
        """Show `doc` (or a not-found message) and refresh title/header/back."""
        self._doc = doc
        title = doc.name if doc else (missing_key or self._toggle_key)
        self.setWindowTitle(f"Feature: {title}")
        self._header.setText(self._build_header())
        self._header.setVisible(bool(self._header.text()))
        if doc is None:
            key = missing_key or self._toggle_key
            self.browser.setPlainText(
                f"No documentation file found for '{key}'.\n\n"
                f"Expected at docs/features/{key}.md"
            )
        else:
            self.browser.setMarkdown(doc.body)
        self._back_btn.setVisible(bool(self._history))

    def _on_anchor_clicked(self, url: QUrl) -> None:
        """Route a clicked link instead of letting QTextBrowser blank itself."""
        # External resources → open in the system browser / mail client.
        if url.scheme().lower() in _EXTERNAL_SCHEMES or url.host():
            QDesktopServices.openUrl(url)
            return
        href = url.toString()
        # Cross-link to another feature doc → load it in place.
        filename = resolve_doc_link(href)
        if filename:
            target = load_doc_file(filename)
            if target is not None:
                if self._doc is not None:
                    self._history.append(self._doc)
                self._render(target)
            return
        # Pure in-page anchor (e.g. "#meaning") → scroll if the anchor exists.
        if url.fragment() and not url.path():
            self.browser.scrollToAnchor(url.fragment())
            return
        # Anything else (e.g. a source-file path) → ignore; never blank.

    def _go_back(self) -> None:
        if self._history:
            self._render(self._history.pop())

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
