"""Inline category picker — replaces the modal CategoriesDialog."""

from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton,
    QLineEdit, QLabel, QCheckBox, QScrollArea, QFrame,
)

from sensi.utils.preferences import (
    load_preferences, save_preferences,
    _SOP_CATEGORIES_LIWC, _SOP_CATEGORIES_EMPATH,
)
from sensi.extractors.semantic_domain import detect_backend


def _list_backend_categories(backend: str) -> tuple[list[str], str | None]:
    """Return (categories, error_message).

    `error_message` is None on success. When non-None, `categories` is the
    SOP-default fallback and the picker should display the error prominently.
    """
    if backend == "empath":
        try:
            from empath import Empath
        except ImportError:
            return list(_SOP_CATEGORIES_EMPATH), (
                "Empath is not installed in this environment. Showing SOP "
                "defaults only. To see the full category list (~200), run:\n"
                "    pip install empath"
            )
        try:
            return sorted(Empath().cats.keys()), None
        except Exception as exc:  # pragma: no cover - defensive
            return list(_SOP_CATEGORIES_EMPATH), f"Empath loaded but failed: {exc}"
    if backend == "liwc":
        _, dict_path = detect_backend()
        if dict_path is None:
            return list(_SOP_CATEGORIES_LIWC), (
                "No LIWC dictionary found at dictionaries/*.dic. Showing SOP "
                "defaults only. LIWC-22 is license-bound — obtain a copy and "
                "place liwc-2022.dic in the dictionaries/ folder."
            )
        try:
            import liwc
            _, names = liwc.load_token_parser(str(dict_path))
            return sorted(names), None
        except Exception as exc:
            return list(_SOP_CATEGORIES_LIWC), f"Failed to parse {dict_path.name}: {exc}"
    # auto: prefer LIWC if dict available, else Empath
    b, _ = detect_backend()
    return _list_backend_categories(b)


class CategoryPicker(QWidget):
    selection_changed = Signal(list)

    def __init__(self, backend: str, parent=None):
        super().__init__(parent)
        self._backend = backend
        # v7 schema: a single categories_<backend> key shared between word and
        # doc levels. The word-level page is the canonical editor.
        self._prefs_key = f"categories_{backend}"
        self._categories, load_error = _list_backend_categories(backend)
        self._checks: dict[str, QCheckBox] = {}
        self._sop = (_SOP_CATEGORIES_LIWC if backend == "liwc"
                     else _SOP_CATEGORIES_EMPATH)

        prefs = load_preferences()
        initial = set(prefs["semantic"].get(self._prefs_key, self._sop))

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Loud error banner if backend failed to load
        if load_error:
            err = QLabel(f"⚠ {load_error}")
            err.setWordWrap(True)
            err.setStyleSheet(
                "background: #FDECEA; color: #B0312A; "
                "padding: 8px 10px; border-radius: 4px; "
                "border: 1px solid #F2C5BF;"
            )
            layout.addWidget(err)

        # Header row with count + buttons
        header = QHBoxLayout()
        self.summary_label = QLabel()
        header.addWidget(self.summary_label, 1)
        self.sop_default_button = QPushButton("SOP default")
        self.sop_default_button.clicked.connect(self._reset_to_sop)
        self.all_button = QPushButton("All")
        self.all_button.clicked.connect(self._select_all)
        self.none_button = QPushButton("None")
        self.none_button.clicked.connect(self._select_none)
        for b in (self.sop_default_button, self.all_button, self.none_button):
            header.addWidget(b)
        layout.addLayout(header)

        # Search
        self.search_field = QLineEdit()
        self.search_field.setPlaceholderText("Search categories...")
        self.search_field.textChanged.connect(self._refilter)
        layout.addWidget(self.search_field)

        # Scroll area with grid of checkboxes
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        inner = QWidget()
        grid = QGridLayout(inner)
        grid.setContentsMargins(8, 8, 8, 8)
        col_count = 2
        for i, cat in enumerate(self._categories):
            cb = QCheckBox(cat)
            cb.setChecked(cat in initial)
            cb.toggled.connect(self._on_toggle)
            self._checks[cat] = cb
            row = i // col_count
            col = i % col_count
            container = QHBoxLayout()
            container.addWidget(cb)
            if cat in self._sop:
                badge = QLabel("★ SOP")
                badge.setStyleSheet(
                    "color: #00B2D0; font-size: 10px; font-weight: 600;")
                container.addWidget(badge)
            container.addStretch(1)
            w = QWidget()
            w.setLayout(container)
            grid.addWidget(w, row, col)
        scroll.setWidget(inner)
        layout.addWidget(scroll, 1)

        self._refresh_summary()

    def total_count(self) -> int:
        return len(self._categories)

    def current_selection(self) -> list[str]:
        return [c for c, cb in self._checks.items() if cb.isChecked()]

    def set_selection(self, cats: list[str]) -> None:
        cats_set = set(cats)
        for c, cb in self._checks.items():
            cb.setChecked(c in cats_set)
        self._refresh_summary()

    def _refresh_summary(self) -> None:
        sel = self.current_selection()
        self.summary_label.setText(
            f"{len(sel)} of {len(self._categories)} selected")

    def _on_toggle(self, _checked: bool) -> None:
        sel = self.current_selection()
        save_preferences({"semantic": {self._prefs_key: sel}})
        self._refresh_summary()
        self.selection_changed.emit(sel)

    def _reset_to_sop(self) -> None:
        self.set_selection(list(self._sop))

    def _select_all(self) -> None:
        self.set_selection(list(self._categories))

    def _select_none(self) -> None:
        self.set_selection([])

    def _refilter(self, q: str) -> None:
        ql = q.lower().strip()
        for cat, cb in self._checks.items():
            container = cb.parentWidget()
            visible = (not ql) or (ql in cat.lower())
            if container is not None:
                container.setVisible(visible)
