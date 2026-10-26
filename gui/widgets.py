"""Reusable Qt widgets used across the GUI."""

from PySide6.QtCore import Signal, QEvent, Qt, QSize, QRectF
from PySide6.QtGui import QCursor, QPainter, QColor, QPen
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QVBoxLayout,
    QLabel, QLineEdit, QPushButton, QCheckBox, QAbstractButton,
    QFileDialog, QWidget, QLayout, QToolTip, QSizePolicy,
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
        self._build_frame(layout, title, subtitle)

    def _build_frame(self, layout: QVBoxLayout, title: str, subtitle: str) -> None:
        """Lay out the header and set `_content_layout`. Subclasses override
        this to change the header (ClusterCard puts a switch in it)."""
        layout.setContentsMargins(16, 12, 16, 16)
        layout.setSpacing(8)
        self.header = SectionHeader(title, subtitle)
        layout.addWidget(self.header)
        self._content_layout = layout

    def add_widget(self, w: QWidget) -> None:
        self._content_layout.addWidget(w)

    def add_layout(self, lay: QLayout) -> None:
        self._content_layout.addLayout(lay)


# Switch colours (Global Constraints): on = SENSI cyan, off = grey.
_SWITCH_ON = "#00B2D0"   # == gui.theme.ACCENT_COLOR
_SWITCH_OFF = "#b9c0c5"

TAG_INCLUDED = "Included"
TAG_EMPTY = "Included — nothing selected"
TAG_LEFT_OUT = "Left out of outputs"


class SectionSwitch(QAbstractButton):
    """Phone-style on/off pill used for section ("include in outputs")
    switches — deliberately unlike the square feature checkboxes.

    Also exposes the small ToggleRow API that callers of
    `ClusterCard.master_toggle()` use (`is_checked`, `set_checked`, `.check`),
    so code written against the old master ToggleRow keeps working.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.setAccessibleName("Include this section in the outputs")
        self.setToolTip("Include this section in the outputs")
        # Focus ring only for keyboard focus (Tab), not the initial/click
        # focus, which would otherwise read as a cyan "on" outline.
        self._keyboard_focus = False

    def focusInEvent(self, event) -> None:
        self._keyboard_focus = event.reason() in (
            Qt.FocusReason.TabFocusReason, Qt.FocusReason.BacktabFocusReason)
        super().focusInEvent(event)

    # --- ToggleRow-compatible API ---------------------------------------
    @property
    def check(self) -> "SectionSwitch":
        return self

    def is_checked(self) -> bool:
        return self.isChecked()

    def set_checked(self, v: bool) -> None:
        self.setChecked(v)

    # --- painting -------------------------------------------------------
    def sizeHint(self) -> QSize:
        return QSize(38, 20)

    def minimumSizeHint(self) -> QSize:
        return self.sizeHint()

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        if not self.isEnabled():
            p.setOpacity(0.45)
        w, h = 38.0, 20.0
        x0 = (self.width() - w) / 2
        y0 = (self.height() - h) / 2
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(_SWITCH_ON if self.isChecked() else _SWITCH_OFF))
        p.drawRoundedRect(QRectF(x0, y0, w, h), h / 2, h / 2)
        knob = 16.0
        kx = x0 + (w - knob - 2 if self.isChecked() else 2)
        ky = y0 + 2
        p.setBrush(QColor(0, 0, 0, 40))          # soft shadow
        p.drawEllipse(QRectF(kx, ky + 1, knob, knob))
        p.setBrush(QColor("#ffffff"))
        p.drawEllipse(QRectF(kx, ky, knob, knob))
        if self.hasFocus() and self._keyboard_focus:
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(QPen(QColor("#0093AB"), 1.5))
            p.drawRoundedRect(QRectF(x0 + 0.75, y0 + 0.75, w - 1.5, h - 1.5),
                              (h - 1.5) / 2, (h - 1.5) / 2)
        p.end()


def _set_state_property(w: QWidget, name: str, value: str) -> None:
    """Set a QSS dynamic property and re-polish so the style updates."""
    if w.property(name) == value:
        return
    w.setProperty(name, value)
    w.style().unpolish(w)
    w.style().polish(w)
    w.update()


class ClusterCard(Card):
    """A feature section card. Header: title, on/off `SectionSwitch`, status
    tag. The switch says whether the section is included in the outputs.

    Everything added through `add_child`, `add_child_layout` or `add_group`
    is *gated*: it lives inside a container widget that is disabled while the
    section is left out (or `available` is False), so greying reaches every
    nested control at any depth, on first paint and after a toggle. The
    switch never changes feature ticks. `add_shared` / `add_widget` content
    is never gated. `available` is a hard environment gate (e.g. backend !=
    liwc) — when False, the switch itself is disabled.
    """
    settings_changed = Signal()

    def __init__(self, title: str, subtitle: str,
                 master_key: str,
                 *, master_section: str = "features",
                 available: bool = True,
                 disabled_subtitle: str | None = None,
                 master_label: str | None = None,
                 parent=None):
        # `master_label` is accepted and ignored: the "Enable all …" master
        # row is gone. Task 5 removes the remaining callers that pass it.
        del master_label
        self._master_key = master_key
        self._master_section = master_section
        self._available = available
        self._segments: list[QWidget] = []
        self._open_segment: QWidget | None = None
        super().__init__(title, subtitle, parent)

        from utils.preferences import load_preferences
        prefs = load_preferences()
        initial = prefs.get(master_section, {}).get(master_key, True)

        # When unavailable, force the switch display to off and disable it.
        self._master.set_checked(bool(initial and available))
        self._master.setEnabled(available)
        self._master.toggled.connect(self._on_master_toggled)

        if subtitle:
            desc = QLabel(subtitle)
            desc.setObjectName("Subtitle")
            desc.setWordWrap(True)
            self._gated_layout().addWidget(desc)

        if not available and disabled_subtitle:
            note = QLabel(disabled_subtitle)
            note.setObjectName("Subtitle")
            note.setWordWrap(True)
            self.add_widget(note)

        self._apply_gate()

    # --- construction ---------------------------------------------------
    def _build_frame(self, layout: QVBoxLayout, title: str, subtitle: str) -> None:
        # The header spans the card edge to edge (inside the 1 px card
        # border); the body below gets the usual padding.
        layout.setContentsMargins(1, 1, 1, 14)
        layout.setSpacing(0)
        self.header = QFrame()
        self.header.setObjectName("SectionCardHeader")
        row = QHBoxLayout(self.header)
        row.setContentsMargins(12, 9, 14, 9)
        row.setSpacing(10)
        self.title_label = QLabel(title)
        self.title_label.setObjectName("SectionCardTitle")
        row.addWidget(self.title_label)
        self._master = SectionSwitch()
        row.addWidget(self._master)
        self.tag_label = QLabel()
        self.tag_label.setObjectName("SectionTag")
        row.addWidget(self.tag_label, 0, Qt.AlignmentFlag.AlignVCenter)
        row.addStretch(1)
        layout.addWidget(self.header)

        body = QVBoxLayout()
        body.setContentsMargins(16, 8, 16, 0)
        body.setSpacing(8)
        layout.addLayout(body)
        self._content_layout = body

    def _gated_layout(self) -> QVBoxLayout:
        """Layout of the current gated container, created on demand so gated
        and ungated content keep the order they were added in."""
        if self._open_segment is None:
            seg = QWidget()
            seg.setObjectName("SectionBody")
            lay = QVBoxLayout(seg)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.setSpacing(8)
            self._content_layout.addWidget(seg)
            seg.setEnabled(self._gate_open())
            self._segments.append(seg)
            self._open_segment = seg
        return self._open_segment.layout()

    def _watch_checkables(self, root) -> None:
        for b in _checkables_in(root):
            b.toggled.connect(self._refresh_tag)

    # --- public API -----------------------------------------------------
    def add_widget(self, w: QWidget) -> None:
        """Ungated content (e.g. a credential box)."""
        self._content_layout.addWidget(w)
        self._open_segment = None

    def add_layout(self, lay: QLayout) -> None:
        """Ungated layout."""
        self._content_layout.addLayout(lay)
        self._open_segment = None

    def add_child(self, widget: QWidget) -> None:
        """Gated widget: every control inside it, at any depth, greys out
        with the section."""
        self._gated_layout().addWidget(widget)
        self._watch_checkables(widget)
        self._refresh_tag()

    def add_child_layout(self, lay: QLayout) -> None:
        """Gated layout: nested layouts and their widgets are gated too."""
        self._gated_layout().addLayout(lay)
        self._watch_checkables(lay)
        self._refresh_tag()

    def add_group(self, label: str, content) -> None:
        """A labelled group of ticks with "All · None" links beside the
        label. The links tick/untick every checkbox in `content` (never the
        section switch); label, links and content are all gated."""
        head = QHBoxLayout()
        head.setContentsMargins(0, 4, 0, 0)
        head.setSpacing(4)
        lbl = QLabel(label)
        lbl.setObjectName("GroupLabel")
        head.addWidget(lbl)
        head.addSpacing(6)
        all_btn = QPushButton("All")
        sep = QLabel("·")
        sep.setObjectName("GroupLinkSep")
        none_btn = QPushButton("None")
        for b in (all_btn, none_btn):
            b.setObjectName("GroupLink")
            b.setFlat(True)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
        head.addWidget(all_btn)
        head.addWidget(sep)
        head.addWidget(none_btn)
        head.addStretch(1)
        gl = self._gated_layout()
        gl.addLayout(head)
        if isinstance(content, QLayout):
            gl.addLayout(content)
        else:
            gl.addWidget(content)
        all_btn.clicked.connect(lambda: _set_group_ticks(content, True))
        none_btn.clicked.connect(lambda: _set_group_ticks(content, False))
        self._watch_checkables(content)
        self._refresh_tag()

    def add_shared(self, widget: QWidget, note: str) -> None:
        """Settings other sections also use: never gated, shown in a bordered
        sub-box labelled with `note`."""
        box = QFrame()
        box.setObjectName("SharedSettings")
        lay = QVBoxLayout(box)
        lay.setContentsMargins(10, 8, 10, 8)
        lay.setSpacing(6)
        n = QLabel(note)
        n.setObjectName("SharedNote")
        n.setWordWrap(True)
        lay.addWidget(n)
        lay.addWidget(widget)
        self.add_widget(box)

    def tag_text(self) -> str:
        if not self._gate_open():
            return TAG_LEFT_OUT
        for seg in self._segments:
            if any(b.isChecked() for b in _checkables_in(seg)):
                return TAG_INCLUDED
        return TAG_EMPTY

    def master_toggle(self) -> SectionSwitch:
        """Expose the section switch (tests assert on its state)."""
        return self._master

    def set_master_checked(self, v: bool) -> None:
        """Set the switch programmatically without re-saving or emitting
        settings_changed (used to mirror a setting changed elsewhere)."""
        self._master.blockSignals(True)
        self._master.setChecked(bool(v and self._available))
        self._master.blockSignals(False)
        self._master.update()
        self._apply_gate()

    def set_available(self, available: bool) -> None:
        """Update the hard environment gate without restarting the app.

        Restores the switch to its persisted value when re-enabling so the
        user's prior choice is not silently lost.
        """
        self._available = available
        self._master.setEnabled(available)
        if available:
            from utils.preferences import load_preferences
            prefs = load_preferences()
            saved = prefs.get(self._master_section, {}).get(self._master_key, True)
        else:
            saved = False
        self._master.blockSignals(True)
        self._master.setChecked(bool(saved))
        self._master.blockSignals(False)
        self._master.update()
        self._apply_gate()

    # --- internals ------------------------------------------------------
    def _gate_open(self) -> bool:
        return self._master.isChecked() and self._available

    def _on_master_toggled(self, v: bool) -> None:
        from utils.preferences import save_preferences
        save_preferences({self._master_section: {self._master_key: v}})
        self._apply_gate()
        self.settings_changed.emit()

    def _apply_gate(self) -> None:
        # Disabling a container greys every descendant. Widgets that were
        # explicitly disabled (`_perma_disabled` rows) stay disabled when the
        # container is re-enabled — Qt keeps their own disabled flag.
        enabled = self._gate_open()
        for seg in self._segments:
            seg.setEnabled(enabled)
        _set_state_property(self.header, "included",
                            "true" if enabled else "false")
        self._refresh_tag()

    def _refresh_tag(self, *_args) -> None:
        text = self.tag_text()
        self.tag_label.setText(text)
        state = {TAG_INCLUDED: "on", TAG_EMPTY: "empty"}.get(text, "off")
        _set_state_property(self.tag_label, "state", state)


def _checkables_in(root) -> list[QAbstractButton]:
    """Every checkable button under `root` (a widget or a layout), at any
    depth — ToggleRow checkboxes, plain QCheckBoxes, radio buttons."""
    widgets: list[QWidget] = []

    def walk_layout(lay: QLayout) -> None:
        for i in range(lay.count()):
            item = lay.itemAt(i)
            if item.widget() is not None:
                widgets.append(item.widget())
            elif item.layout() is not None:
                walk_layout(item.layout())

    if isinstance(root, QLayout):
        walk_layout(root)
    else:
        widgets.append(root)
    out: list[QAbstractButton] = []
    seen: set[int] = set()
    for w in widgets:
        for b in [w, *w.findChildren(QAbstractButton)]:
            if (isinstance(b, QAbstractButton) and b.isCheckable()
                    and not isinstance(b, SectionSwitch) and id(b) not in seen):
                seen.add(id(b))
                out.append(b)
    return out


def _set_group_ticks(content, value: bool) -> None:
    """Tick/untick every checkbox in a group, skipping explicitly disabled
    (e.g. permanently unavailable) rows, which must stay as they are."""
    for b in _checkables_in(content):
        if (isinstance(b, QCheckBox)
                and not b.testAttribute(Qt.WidgetAttribute.WA_ForceDisabled)):
            b.setChecked(value)
