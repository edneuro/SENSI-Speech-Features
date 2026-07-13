"""Reusable Qt widgets used across the GUI."""

from PySide6.QtCore import Signal, QEvent, Qt
from PySide6.QtGui import QCursor
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QVBoxLayout,
    QLabel, QLineEdit, QPushButton, QCheckBox,
    QFileDialog, QWidget, QLayout, QToolTip,
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
        self.subtitle_label.setWordWrap(True)
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


class _ClickToToggleLabel(QLabel):
    """Word-wrapping label that toggles its buddy checkbox on click, so the
    checkbox-with-text click affordance survives the label being separate."""

    def __init__(self, text: str, buddy_check: QCheckBox, parent=None):
        super().__init__(text, parent)
        self._buddy = buddy_check

    def mousePressEvent(self, event) -> None:
        if self._buddy.isEnabled():
            self._buddy.click()
        super().mousePressEvent(event)


class ToggleRow(QWidget):
    toggled = Signal(bool)

    def __init__(self, label: str, *, info_key: str | None = None, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        # The label lives on a word-wrapping QLabel, NOT on the QCheckBox:
        # QCheckBox text cannot wrap, so long feature labels used to force
        # every page's minimum width past the window (the source of the
        # horizontal scrollbar). Clicking the label still toggles the box.
        self.check = QCheckBox()
        self.check.toggled.connect(self.toggled)
        layout.addWidget(self.check)
        self.label = _ClickToToggleLabel(label, self.check)
        self.label.setWordWrap(True)
        layout.addWidget(self.label, 1)
        # Info icon appears iff a docs/features/<info_key>.md exists. Cluster
        # master toggles pass info_key=None, so they remain icon-free.
        if info_key is not None:
            from gui.feature_docs import has_feature_doc
            if has_feature_doc(info_key):
                from gui.info_popup import InfoIcon
                layout.addWidget(InfoIcon(info_key, self))
        self.warning = None  # populated by set_warning
        self._perma_disabled = False
        # Typed handle for set_warning: QWidget.layout() returns
        # QLayout | None, which lacks insertWidget.
        self._row_layout = layout

    def set_checked(self, v: bool) -> None:
        self.check.setChecked(v)

    def is_checked(self) -> bool:
        return self.check.isChecked()

    def set_permanently_disabled(self, tooltip: str = "") -> None:
        """Hard-disable this toggle regardless of any parent master state.

        Used for features whose backing data/resource is unavailable (e.g.
        EWFG, whose source is not digitized). Unchecks the box so the feature
        is also off in saved preferences, and parks it greyed-out; a
        ClusterCard master toggle will not re-enable it.
        """
        self._perma_disabled = True
        self.check.setChecked(False)
        self.check.setEnabled(False)
        self.label.setEnabled(False)
        if tooltip:
            self.check.setToolTip(tooltip)
            self.label.setToolTip(tooltip)

    def set_warning(self, message: str | None) -> None:
        if self.warning is not None:
            self._row_layout.removeWidget(self.warning)
            self.warning.deleteLater()
            self.warning = None
        if message:
            self.warning = AmberDotIndicator(message)
            self._row_layout.insertWidget(1, self.warning)


class AmberDotIndicator(QLabel):
    def __init__(self, tooltip: str, parent=None):
        super().__init__("⚠", parent)
        self._tip = tooltip
        self.setToolTip(tooltip)        # keeps toolTip() accessor working
        self.setFixedSize(26, 22)
        self.setStyleSheet("color: #C77A00; font-size: 14px;")
        # WA_Hover ensures Qt sends Enter/Leave events even inside scroll areas.
        self.setAttribute(Qt.WidgetAttribute.WA_Hover)

    def enterEvent(self, event) -> None:
        # Show immediately on hover — bypasses the tooltip-timer pipeline that
        # scroll-area viewports can swallow before it reaches child widgets.
        QToolTip.showText(QCursor.pos(), self._tip, self, self.rect())
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:
        QToolTip.hideText()
        super().leaveEvent(event)


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


class ClusterCard(Card):
    """Card with a master ToggleRow over a grid of child toggles.

    When the master toggle is off (or `available` is False), all children
    grey out and become non-interactable. The `available` predicate
    represents a hard environment gate (e.g. backend != liwc) — when False,
    the master itself is disabled.
    """
    settings_changed = Signal()

    def __init__(self, title: str, subtitle: str,
                 master_key: str, master_label: str,
                 *, master_section: str = "features",
                 available: bool = True,
                 disabled_subtitle: str | None = None,
                 parent=None):
        super().__init__(title, subtitle, parent)
        self._master_key = master_key
        self._master_section = master_section
        self._available = available
        self._children: list[QWidget] = []

        from utils.preferences import load_preferences
        prefs = load_preferences()
        initial = prefs.get(master_section, {}).get(master_key, True)

        self._master = ToggleRow(label=master_label)
        # When unavailable, force the master display to off and disable interaction.
        self._master.set_checked(initial and available)
        self._master.check.setEnabled(available)
        self._master.toggled.connect(self._on_master_toggled)
        self.add_widget(self._master)

        if not available and disabled_subtitle:
            note = QLabel(disabled_subtitle)
            note.setObjectName("Subtitle")
            self.add_widget(note)

        self._update_children_enabled(initial and available)

    def add_child(self, widget: QWidget) -> None:
        self._children.append(widget)
        self.add_widget(widget)
        widget.setEnabled(self._master.is_checked() and self._available
                          and not getattr(widget, "_perma_disabled", False))

    def add_child_layout(self, lay) -> None:
        """Add a layout (e.g. QGridLayout) and gate its direct-child widgets
        on the master.

        NOTE: nested layouts are NOT recursed — items whose `.widget()` is
        None are silently skipped. Current callers pass flat grids of
        ToggleRows, so this is sufficient. If a caller ever needs to wrap a
        nested QHBoxLayout/QVBoxLayout, prefer add_child(wrapper_widget)
        with a containing QWidget, or extend this method with recursion.
        """
        self.add_layout(lay)
        for i in range(lay.count()):
            item = lay.itemAt(i)
            w = item.widget()
            if w is not None:
                self._children.append(w)
                w.setEnabled(self._master.is_checked() and self._available
                             and not getattr(w, "_perma_disabled", False))

    def master_toggle(self) -> ToggleRow:
        """Expose the master row. Used by tests (assertions on the inner
        QCheckBox's enabled/checked state); section pages don't need it
        because the master persists itself via _on_master_toggled."""
        return self._master

    def set_available(self, available: bool) -> None:
        """Update the hard environment gate without restarting the app.

        Restores the master to its persisted value when re-enabling so the
        user's prior choice is not silently lost.
        """
        self._available = available
        self._master.check.setEnabled(available)
        if not available:
            self._master.check.blockSignals(True)
            self._master.check.setChecked(False)
            self._master.check.blockSignals(False)
            self._update_children_enabled(False)
        else:
            from utils.preferences import load_preferences
            prefs = load_preferences()
            saved = prefs.get(self._master_section, {}).get(self._master_key, True)
            self._master.check.blockSignals(True)
            self._master.check.setChecked(saved)
            self._master.check.blockSignals(False)
            self._update_children_enabled(saved)

    def _on_master_toggled(self, v: bool) -> None:
        from utils.preferences import save_preferences
        save_preferences({self._master_section: {self._master_key: v}})
        self._update_children_enabled(v and self._available)
        self.settings_changed.emit()

    def _update_children_enabled(self, enabled: bool) -> None:
        for w in self._children:
            w.setEnabled(enabled and not getattr(w, "_perma_disabled", False))
