"""Modal dialog wrapping CategoryPicker. One per level (word vs document)."""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
)

from sensi.gui.category_picker import CategoryPicker


class CategoryDialog(QDialog):
    def __init__(self, backend: str, level: str, parent=None):
        super().__init__(parent)
        assert level in ("word", "document"), level
        title = ("Configure word-level categories" if level == "word"
                 else "Configure document-level categories")
        self.setWindowTitle(title)
        self.resize(640, 600)
        self.setModal(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # `level` is no longer routed to CategoryPicker (v7 unified key);
        # it's kept on the dialog only to vary the window title.
        self.picker = CategoryPicker(backend=backend)
        layout.addWidget(self.picker, 1)

        button_row = QHBoxLayout()
        button_row.addStretch(1)
        self.reset_button = QPushButton("Reset to SOP default")
        self.reset_button.clicked.connect(self.picker._reset_to_sop)
        self.done_button = QPushButton("Done")
        self.done_button.setObjectName("Primary")
        self.done_button.clicked.connect(self.accept)
        button_row.addWidget(self.reset_button)
        button_row.addWidget(self.done_button)
        layout.addLayout(button_row)
