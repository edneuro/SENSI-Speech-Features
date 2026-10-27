"""Logs section: live tail of the current run + list of past runs.

The current-run viewer polls the active .log file every 250 ms and appends
any new bytes — simple, robust, no thread-safety drama. main_window wires
RunSection.run_started / run_finished signals to start/stop_live().

Past runs are listed as full paths so the user can copy them. Files from a
single GUI dispatch (word + doc batch sharing a `run_ts`) are grouped
together; groups are separated by a "--------" line in the list view.
"""

from __future__ import annotations

import re
from pathlib import Path

from PySide6.QtCore import QTimer, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem,
    QPlainTextEdit, QPushButton, QVBoxLayout, QWidget,
)

from gui.widgets import SectionHeader
from utils.preferences import load_preferences


# Matches the timestamp embedded in log filenames, e.g.
# `word_run_2026-05-12_14-30-22.log`.
_TS_RE = re.compile(r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})")
_GROUP_SEPARATOR = "─" * 40


class LogsSection(QWidget):
    """Logs tab: live tail of the active run + browsable past runs."""

    def __init__(self, output_folder: Path | None = None, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 16, 24, 16)
        layout.setSpacing(12)

        # Header
        header_row = QHBoxLayout()
        header_row.addWidget(SectionHeader("Current run"), 1)
        self.copy_btn = QPushButton("Copy")
        self.copy_btn.clicked.connect(self._copy_log)
        header_row.addWidget(self.copy_btn)
        hr_w = QWidget(); hr_w.setLayout(header_row)
        layout.addWidget(hr_w)

        # Filter / search row
        search_row = QHBoxLayout()
        search_row.addWidget(QLabel("Search:"))
        self.search = QLineEdit()
        self.search.setPlaceholderText("Filter visible lines…")
        self.search.textChanged.connect(self._apply_filter)
        search_row.addWidget(self.search, 1)
        sr_w = QWidget(); sr_w.setLayout(search_row)
        layout.addWidget(sr_w)

        # Viewer
        self.viewer = QPlainTextEdit()
        self.viewer.setReadOnly(True)
        self.viewer.setFont(QFont("JetBrains Mono", 10))
        layout.addWidget(self.viewer, 1)

        # Past runs
        layout.addWidget(SectionHeader("Past runs"))
        self.past_runs_list = QListWidget()
        self.past_runs_list.itemActivated.connect(self._load_past_run)
        layout.addWidget(self.past_runs_list, 1)

        self._all_lines: list[str] = []
        # An explicit output_folder pins the section to that folder forever
        # (used by tests). Otherwise start_live() re-reads from prefs each
        # run so the user can change the output folder between runs.
        self._explicit_output_folder = output_folder
        self._output_folder = output_folder or self._guess_output_folder()

        # Live-tail state — only active while a run is in progress.
        self._tail_timer: QTimer | None = None
        self._tail_paths: list[Path] = []     # log files we're tailing (1 or 2)
        self._tail_offsets: dict[Path, int] = {}

        self.refresh_past_runs()

    # ----------------------------------------------------------------- helpers

    def _logs_dir(self) -> Path:
        return self._output_folder / "logs"

    def _guess_output_folder(self) -> Path:
        p = load_preferences()["shared"].get("last_output_folder", "")
        return Path(p) if p else Path.cwd()

    # ----------------------------------------------------------- viewer plumbing

    def _line_matches(self, line: str) -> bool:
        q = self.search.text().lower().strip()
        return (not q) or (q in line.lower())

    def _apply_filter(self, _q: str) -> None:
        self.viewer.clear()
        for line in self._all_lines:
            if self._line_matches(line):
                self.viewer.appendPlainText(line)

    def _append_lines(self, new_lines: list[str]) -> None:
        for line in new_lines:
            self._all_lines.append(line)
            if self._line_matches(line):
                self.viewer.appendPlainText(line)

    def _copy_log(self) -> None:
        from PySide6.QtGui import QGuiApplication
        QGuiApplication.clipboard().setText("\n".join(self._all_lines))

    def _clear_viewer(self) -> None:
        self._all_lines.clear()
        self.viewer.clear()

    # --------------------------------------------------------- live-tail API

    def start_live(self) -> None:
        """Called by main_window when RunSection emits `run_started`.

        Re-reads the prefs to pick up the user's current output folder, then
        starts a 250ms QTimer that polls the logs/ dir for newly-created
        .log files and tails any that appear.
        """
        if self._explicit_output_folder is None:
            self._output_folder = self._guess_output_folder()
        self._clear_viewer()
        self._tail_paths = []
        self._tail_offsets = {}
        if self._tail_timer is None:
            self._tail_timer = QTimer(self)
            self._tail_timer.setInterval(250)
            self._tail_timer.timeout.connect(self._poll_tail)
        self._tail_timer.start()

    def stop_live(self) -> None:
        """Called by main_window when RunSection emits `run_finished`. Drains
        any final bytes from the tracked log files, stops the timer, and
        refreshes the past-runs list so the just-finished run appears."""
        if self._tail_timer is not None:
            self._tail_timer.stop()
        self._poll_tail()  # one final drain
        self._tail_paths = []
        self._tail_offsets = {}
        self.refresh_past_runs()

    def _poll_tail(self) -> None:
        """Discover any new logs in <output>/logs/ and read new bytes from
        each. Called by the QTimer every 250ms while a run is active."""
        logs_dir = self._logs_dir()
        if not logs_dir.is_dir():
            return
        # Pick up new log files created since start_live() was called.
        for p in sorted(logs_dir.glob("*_run_*.log")):
            if p not in self._tail_offsets:
                self._tail_paths.append(p)
                self._tail_offsets[p] = 0
        # Drain new bytes from each tracked file.
        for path in self._tail_paths:
            try:
                size = path.stat().st_size
            except OSError:
                continue
            offset = self._tail_offsets[path]
            if size <= offset:
                continue
            try:
                with path.open("r", encoding="utf-8", errors="replace") as f:
                    f.seek(offset)
                    new = f.read()
                    self._tail_offsets[path] = f.tell()
            except OSError:
                continue
            if new:
                # If both word and doc logs are streaming, prefix lines with
                # the source so the user can tell them apart.
                prefix = f"[{path.stem.split('_')[0]}] " if len(self._tail_paths) > 1 else ""
                lines = [prefix + line for line in new.splitlines() if line]
                self._append_lines(lines)

    # ------------------------------------------------------------- past runs

    def refresh_past_runs(self) -> None:
        """Rebuild the past-runs list. Files sharing a timestamp (i.e. from
        the same GUI dispatch) appear consecutively; groups are separated
        by a horizontal-rule line item."""
        self.past_runs_list.clear()
        logs_dir = self._logs_dir()
        if not logs_dir.is_dir():
            return

        # Sort by embedded timestamp (newest first) so same-batch logs are
        # adjacent regardless of filename prefix. Plain `sorted(reverse=True)`
        # interleaves `word_run_*` and `doc_run_*` because of the prefix.
        def _sort_key(p: Path) -> tuple[str, str]:
            m = _TS_RE.search(p.name)
            return (m.group(1) if m else "", p.name)
        all_logs = sorted(logs_dir.glob("*_run_*.log"), key=_sort_key, reverse=True)
        if not all_logs:
            return

        # Group adjacent files that share the same embedded timestamp.
        groups: list[list[Path]] = []
        current_ts: str | None = None
        for p in all_logs:
            m = _TS_RE.search(p.name)
            ts = m.group(1) if m else None
            if ts != current_ts or not groups:
                groups.append([p])
                current_ts = ts
            else:
                groups[-1].append(p)

        for i, group in enumerate(groups):
            if i > 0:
                # Separator row between batches. We leave the default Qt
                # flags (selectable etc.) — _load_past_run is a no-op for
                # items with no path data, so clicking it does nothing.
                self.past_runs_list.addItem(QListWidgetItem(_GROUP_SEPARATOR))
            for path in group:
                item = QListWidgetItem(str(path))
                item.setData(0x100, str(path))  # Qt.ItemDataRole.UserRole = 0x100
                self.past_runs_list.addItem(item)

    def _load_past_run(self, item: QListWidgetItem) -> None:
        """Replace the viewer contents with the selected log file."""
        path_str = item.data(0x100)
        if not path_str:
            return
        path = Path(path_str)
        if not path.is_file():
            return
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return
        self._clear_viewer()
        self._all_lines = text.splitlines()
        for line in self._all_lines:
            if self._line_matches(line):
                self.viewer.appendPlainText(line)
