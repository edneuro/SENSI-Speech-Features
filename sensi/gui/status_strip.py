"""Persistent bottom status strip."""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QProgressBar, QPushButton,
)


class StatusStrip(QFrame):
    cancel_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("StatusStrip")
        self.setFixedHeight(32)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 4, 12, 4)
        self.state_label = QLabel("○  Idle")
        self.progress = QProgressBar()
        self.progress.setMaximum(100)
        self.progress.setValue(0)
        self.progress.setFixedWidth(280)
        self.detail_label = QLabel("Ready to run.")
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel_requested)
        layout.addWidget(self.state_label)
        layout.addWidget(self.progress)
        layout.addWidget(self.detail_label, 1)
        layout.addWidget(self.cancel_btn)

    def set_idle(self) -> None:
        self.state_label.setText("○  Idle")
        self.detail_label.setText("Ready to run.")
        self.progress.setValue(0)
        self.cancel_btn.setEnabled(False)

    def set_running(self, current: int, total: int, file_name: str) -> None:
        self.state_label.setText("●  Running")
        pct = int(100 * current / total) if total else 0
        self.progress.setValue(pct)
        self.detail_label.setText(f"{current}/{total}   {file_name}")
        self.cancel_btn.setEnabled(True)

    def set_done(self, total: int, elapsed_s: float) -> None:
        self.state_label.setText("✓  Done")
        self.progress.setValue(100)
        self.detail_label.setText(f"{total}/{total} files in {elapsed_s:.0f}s")
        self.cancel_btn.setEnabled(False)

    def show_message(self, text: str) -> None:
        """Show a one-off problem (e.g. a failed settings save)."""
        self.state_label.setText("⚠  Warning")
        self.detail_label.setText(text)

    def set_cancelling(self) -> None:
        self.state_label.setText("●  Cancelling")
        self.detail_label.setText("Stopping after the current file…")
        self.cancel_btn.setEnabled(False)

    def set_cancelled(self) -> None:
        self.state_label.setText("○  Cancelled")
        self.detail_label.setText("Run cancelled — files finished so far are saved; see Logs.")
        self.cancel_btn.setEnabled(False)

    def set_errored(self, n_errors: int) -> None:
        self.state_label.setText("⚠  Errored")
        self.detail_label.setText(f"{n_errors} files failed — see Logs.")
        self.cancel_btn.setEnabled(False)
