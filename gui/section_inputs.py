"""Inputs section: analyzer name, folders, pairing preview."""

from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QTableView,
    QHeaderView,
)
from PySide6.QtGui import QStandardItemModel, QStandardItem

from gui.widgets import SectionHeader, FolderRow
from utils.preferences import load_preferences, save_preferences


TRANSCRIPT_EXTS = {".txt", ".tsv", ".md"}
AUDIO_EXTS = {".wav", ".flac", ".mp3"}


class InputsSection(QWidget):
    settings_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 16, 24, 16)
        layout.setSpacing(16)

        prefs = load_preferences()["shared"]

        # Analyzer
        layout.addWidget(SectionHeader(
            "Analyzer",
            "Recorded in every run manifest and log file.",
        ))
        self.analyzer_line = QLineEdit(prefs.get("analyzer_name", ""))
        self.analyzer_line.textChanged.connect(self._on_analyzer_change)
        layout.addWidget(self.analyzer_line)
        self._on_analyzer_change(self.analyzer_line.text())

        # Folders
        layout.addWidget(SectionHeader("Folders"))
        self.transcripts_row, self.transcripts_count_label = self._folder_block(
            layout, "Transcripts folder:", prefs.get("last_transcripts_folder", ""),
            self._on_transcripts_change)
        self.audio_row, self.audio_count_label = self._folder_block(
            layout, "Audio folder:", prefs.get("last_audio_folder", ""),
            self._on_audio_change)
        self.output_row, self.output_count_label = self._folder_block(
            layout, "Output folder:", prefs.get("last_output_folder", ""),
            self._on_output_change)

        # Pairing preview
        layout.addWidget(SectionHeader("Pairing preview"))
        self.pairing_summary = QLabel("No transcripts/audio selected yet.")
        layout.addWidget(self.pairing_summary)
        self.pairing_table = QTableView()
        self.pairing_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.pairing_table, 1)

        # Initial validation pass
        self._on_transcripts_change(self.transcripts_row.path())
        self._on_audio_change(self.audio_row.path())
        self._on_output_change(self.output_row.path())

    def _folder_block(self, parent_layout, label, initial_path, callback):
        wrapper = QWidget()
        v = QVBoxLayout(wrapper)
        v.setContentsMargins(0, 0, 0, 0)
        row = FolderRow(label=label)
        row.set_path(initial_path)
        count_label = QLabel("")
        count_label.setObjectName("Subtitle")
        v.addWidget(row)
        v.addWidget(count_label)
        parent_layout.addWidget(wrapper)
        row.path_changed.connect(callback)
        return row, count_label

    def _on_analyzer_change(self, text: str) -> None:
        is_empty = not text.strip()
        self.analyzer_line.setProperty("error", "true" if is_empty else "false")
        self.analyzer_line.style().unpolish(self.analyzer_line)
        self.analyzer_line.style().polish(self.analyzer_line)
        save_preferences({"shared": {"analyzer_name": text.strip()}})
        self.settings_changed.emit()

    def _validate_and_count(self, path_str: str, exts: set[str], count_label: QLabel,
                            row: FolderRow) -> int:
        if not path_str.strip():
            row.set_error(False)
            count_label.setText("")
            return -1
        p = Path(path_str)
        if not p.is_dir():
            row.set_error(True)
            count_label.setText("Folder does not exist.")
            return -1
        row.set_error(False)
        if not exts:
            count_label.setText("(created if missing)")
            return 0
        n = sum(1 for f in p.iterdir() if f.is_file() and f.suffix.lower() in exts)
        count_label.setText(f"✓ {n} files")
        return n

    def _on_transcripts_change(self, path: str) -> None:
        self._validate_and_count(path, TRANSCRIPT_EXTS,
                                  self.transcripts_count_label, self.transcripts_row)
        save_preferences({"shared": {"last_transcripts_folder": path.strip()}})
        self._refresh_pairing()
        self.settings_changed.emit()

    def _on_audio_change(self, path: str) -> None:
        self._validate_and_count(path, AUDIO_EXTS,
                                  self.audio_count_label, self.audio_row)
        save_preferences({"shared": {"last_audio_folder": path.strip()}})
        self._refresh_pairing()
        self.settings_changed.emit()

    def _on_output_change(self, path: str) -> None:
        self._validate_and_count(path, set(),  # any folder OK
                                  self.output_count_label, self.output_row)
        save_preferences({"shared": {"last_output_folder": path.strip()}})
        self.settings_changed.emit()

    def _refresh_pairing(self) -> None:
        t = Path(self.transcripts_row.path() or "")
        a = Path(self.audio_row.path() or "")
        if not (t.is_dir() and a.is_dir()):
            self.pairing_summary.setText("Pick both folders to see pairing.")
            self.pairing_table.setModel(QStandardItemModel(self))
            return
        ts = {p.stem.lower(): p.name for p in t.iterdir()
              if p.is_file() and p.suffix.lower() in TRANSCRIPT_EXTS}
        au = {p.stem.lower(): p.name for p in a.iterdir()
              if p.is_file() and p.suffix.lower() in AUDIO_EXTS}
        matched = sorted(set(ts) & set(au))
        unmatched_t = sorted(set(ts) - set(au))
        unmatched_a = sorted(set(au) - set(ts))
        self.pairing_summary.setText(
            f"{len(ts)} transcripts × {len(au)} audio  →  {len(matched)} matched pairs, "
            f"{len(unmatched_t) + len(unmatched_a)} unmatched"
        )
        model = QStandardItemModel(0, 3, self)
        model.setHorizontalHeaderLabels(["Transcript", "Audio", "Status"])
        for stem in matched:
            model.appendRow([QStandardItem(ts[stem]),
                             QStandardItem(au[stem]),
                             QStandardItem("paired")])
        for stem in unmatched_t:
            model.appendRow([QStandardItem(ts[stem]),
                             QStandardItem(""),
                             QStandardItem("no audio match")])
        for stem in unmatched_a:
            model.appendRow([QStandardItem(""),
                             QStandardItem(au[stem]),
                             QStandardItem("no transcript match")])
        self.pairing_table.setModel(model)
