"""Run section: one switch per output level, readiness summary, cost estimate, Run button."""

import threading
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame,
)

from gui.widgets import SectionHeader, SectionSwitch, _set_state_property
from gui.preflight import compute_warnings
from gui.run_dispatch import run as dispatch_run
from gui.planner import plan_from_prefs
from gui.planner.plan import LEVEL_ARTIFACTS
from utils.config_checks import value_errors
from utils.preferences import load_preferences, save_preferences

# Transcript extensions the batch runners pair (gui/planner/emit.py).
_TEXT_EXTS = (".txt", ".tsv", ".md")


# Run-tab level rows, in display order: (prefs output key, name, plain line).
# Each level's files come from gui.planner.plan.LEVEL_ARTIFACTS, so the rows
# can never disagree with what the planner writes.
_LEVEL_ROWS: list[tuple[str, str, str]] = [
    ("level_word", "Word level",
     "Everything ticked on the Word features page."),
    ("level_document", "Document level",
     "Everything ticked on the Document features page."),
    ("level_subword", "Sub-word level",
     "Everything ticked on the Sub-word features page."),
    ("level_sentence", "Sentence level",
     "Everything ticked on the Sentence features page."),
    ("level_acoustic", "Acoustic level",
     "Audio-only time series from the Acoustic features page; "
     "doesn't need transcripts."),
]

# Plan artifact id -> the file it writes (path template).
_ARTIFACT_PATHS: dict[str, str] = {
    "features":              "word/<stim>_features.csv",
    "word2vec":              "word/<stim>_word2vec.mat",
    "bert_word":             "word/<stim>_bert_word.mat",
    "documents_summary":     "document/documents_summary_<timestamp>.csv",
    "syllables":             "syllable/<stim>_syllables.csv",
    "phonemes":              "phoneme/<stim>_phonemes.csv",
    "phoneme_surprisal_mat": "phoneme/<stim>_phoneme_surprisal.mat",
    "sentences":             "sentence/<stim>_sentences.csv",
    "bert_sentence":         "sentence/<stim>_bert_sentence.mat",
    "acoustic":              "acoustic/<stim>_acoustic.mat",
}


def _count_tag(level_key: str, on: bool, n_live: int) -> str:
    """Count tag text for a level row."""
    if not on:
        return "Left out"
    if n_live == 0:
        return "Nothing selected"
    if level_key == "level_document":
        return "1 file for the whole run"
    if level_key == "level_acoustic":
        return "1 file per audio file"
    return f"{n_live} file{'' if n_live == 1 else 's'} per stimulus"


class _LevelRow(QWidget):
    """One Run-tab level: switch, bold name + count tag, plain line, files."""

    def __init__(self, key: str, name: str, desc: str, parent=None):
        super().__init__(parent)
        self.key = key
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 4, 0, 4)
        outer.setSpacing(12)
        self.switch = SectionSwitch()
        self.switch.setAccessibleName(f"Include the {name.lower()} in the outputs")
        self.switch.setToolTip(f"Include the {name.lower()} in the outputs")
        outer.addWidget(self.switch, 0, Qt.AlignmentFlag.AlignTop)

        body = QVBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(2)
        head = QHBoxLayout()
        head.setSpacing(6)
        self.name_label = QLabel(name)
        self.name_label.setObjectName("LevelName")
        head.addWidget(self.name_label)
        self.tag_label = QLabel()
        self.tag_label.setObjectName("SectionTag")
        head.addWidget(self.tag_label, 0, Qt.AlignmentFlag.AlignVCenter)
        head.addStretch(1)
        body.addLayout(head)
        self.desc_label = QLabel(desc)
        self.desc_label.setObjectName("LevelDesc")
        self.desc_label.setWordWrap(True)
        body.addWidget(self.desc_label)
        self.file_labels: dict[str, QLabel] = {}
        for art in LEVEL_ARTIFACTS[key]:
            lbl = QLabel(_ARTIFACT_PATHS[art])
            lbl.setObjectName("LevelFile")
            lbl.setTextFormat(Qt.TextFormat.PlainText)
            body.addWidget(lbl)
            self.file_labels[art] = lbl
        outer.addLayout(body, 1)

    def update_from(self, on: bool, live) -> None:
        """Sync the switch (without re-saving), the tag and the file lines."""
        self.switch.blockSignals(True)
        self.switch.setChecked(on)
        self.switch.blockSignals(False)
        self.switch.update()
        n_live = sum(1 for a in self.file_labels if a in live)
        self.tag_label.setText(_count_tag(self.key, on, n_live))
        state = "on" if on else "off"
        for w in (self.tag_label, self.name_label, self.desc_label):
            _set_state_property(w, "state", state)
        for art, lbl in self.file_labels.items():
            # A left-out level lists no files (mockup); an included level
            # strikes through the file types nothing ticked will produce.
            lbl.setVisible(on)
            produced = art in live
            f = lbl.font()
            f.setStrikeOut(not produced)
            lbl.setFont(f)
            _set_state_property(lbl, "state", "on" if produced else "no")


class RunSection(QWidget):
    run_started = Signal()
    run_finished = Signal(str)
    settings_changed = Signal()   # emitted when a level switch changes
    # Internal cross-thread bridges. dispatch_run invokes its progress/done
    # callbacks from a worker thread; emitting a Qt signal auto-upgrades to
    # QueuedConnection so the connected slots run on the main (GUI) thread.
    # Not part of the public RunSection contract — hence the leading underscore.
    _progress_event = Signal(int, int, str)
    _done_event = Signal(str)

    def __init__(self, get_session_api_key, status_strip, parent=None,
                 get_session_hf_token=None):
        super().__init__(parent)
        self._get_api_key = get_session_api_key
        self._get_hf_token = get_session_hf_token or (lambda: "")
        self._status_strip = status_strip
        # Wire cross-thread bridges to main-thread slots before any widget
        # creation that might cause them to fire.
        self._progress_event.connect(self._on_progress_main)
        self._done_event.connect(self._on_done_main)
        # Cancel: the status strip's button sets this; the batch runners poll
        # it between documents (main.py should_cancel), so the file in progress
        # finishes and the rest are skipped.
        self._cancel = threading.Event()
        if status_strip is not None:
            status_strip.cancel_requested.connect(self._on_cancel)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 16, 24, 16)
        layout.setSpacing(16)

        # Outputs: one switch per level.
        layout.addWidget(SectionHeader("Outputs"))
        self.level_rows: dict[str, _LevelRow] = {}
        for key, name, desc in _LEVEL_ROWS:
            row = _LevelRow(key, name, desc)
            row.switch.toggled.connect(
                lambda v, k=key: self._on_level_toggled(k, v))
            layout.addWidget(row)
            self.level_rows[key] = row

        # Readiness
        layout.addWidget(SectionHeader("Readiness"))
        self.readiness_label = QLabel("…")
        self.readiness_label.setWordWrap(True)
        layout.addWidget(self.readiness_label)
        self.warnings_label = QLabel("")
        self.warnings_label.setWordWrap(True)
        layout.addWidget(self.warnings_label)

        # Cost panel
        self.cost_panel = QFrame()
        self.cost_panel.setObjectName("Card")
        cp = QVBoxLayout(self.cost_panel)
        cp.addWidget(QLabel("<b>Cost estimate</b>"))
        self.cost_detail = QLabel("…")
        self.cost_detail.setWordWrap(True)
        cp.addWidget(self.cost_detail)
        layout.addWidget(self.cost_panel)

        # Run button
        layout.addStretch(1)
        self.run_button = QPushButton("▶  Run extraction")
        self.run_button.setObjectName("Primary")
        self.run_button.clicked.connect(self._on_run)
        layout.addWidget(self.run_button)

        self.refresh()

    def _on_level_toggled(self, key: str, value: bool) -> None:
        save_preferences({"output": {key: value}})
        self.refresh()
        self.settings_changed.emit()

    def showEvent(self, event) -> None:
        # Readiness/cost depend on prefs edited on OTHER pages (Inputs folders,
        # analyzer name, feature toggles). Those pages don't all push a refresh
        # here, and switching tabs doesn't either, so re-read current prefs every
        # time this page becomes visible. (Without this the readiness can show a
        # stale "Not ready" after the Inputs are filled.)
        super().showEvent(event)
        self.refresh()

    def refresh(self) -> None:
        prefs = load_preferences()
        plan = plan_from_prefs(prefs)

        # Level rows: switch state from prefs (the Acoustic card edits
        # output.level_acoustic too); files struck through unless the plan
        # will write them.
        for key, row in self.level_rows.items():
            row.update_from(bool(prefs["output"].get(key, False)),
                            plan.output_csvs)

        # Required inputs depend on which outputs are actually selected. The
        # acoustic level is audio-only: its runner globs the audio folder and
        # ignores transcripts, so an acoustic-only run must NOT require a
        # transcripts folder. Every other output is derived from the transcript,
        # so those still require it. Audio is required only when the run uses
        # it: the acoustic level, or any stage that needs forced alignment. A
        # text-only run (no alignment) works without an audio folder.
        shared = prefs["shared"]

        def _dir_ok(p: str) -> bool:
            # NB: Path("").is_dir() is True (empty string -> "." -> cwd), so an
            # unset folder would wrongly pass. Require a non-empty path.
            return bool(p) and Path(p).is_dir()

        analyzer_ok    = bool(shared.get("analyzer_name", "").strip())
        transcripts_ok = _dir_ok(shared.get("last_transcripts_folder", ""))
        audio_ok       = _dir_ok(shared.get("last_audio_folder", ""))
        output_ok      = _dir_ok(shared.get("last_output_folder", ""))

        has_output        = bool(plan.output_csvs)
        needs_transcripts = bool(plan.output_csvs - {"acoustic"})
        acoustic_only     = has_output and not needs_transcripts
        needs_audio       = has_output and (
            "acoustic" in plan.output_csvs or "mfa_align" in plan.stages_to_run)
        audio_path        = shared.get("last_audio_folder", "")

        missing = []
        if not has_output:
            missing.append("turn on at least one level and tick a feature on its page")
        if not analyzer_ok:
            missing.append("enter an analyzer name (Inputs)")
        if not output_ok:
            missing.append("pick an output folder (Inputs)")
        if needs_audio and not audio_ok:
            missing.append("pick an audio folder (Inputs)")
        elif audio_path and not audio_ok:
            missing.append("the audio folder in Inputs doesn't exist")
        if needs_transcripts and not transcripts_ok:
            missing.append("pick a transcripts folder (needed for word / document "
                           "/ sub-word / sentence outputs)")
        elif needs_transcripts and not any(
                p.is_file() and p.suffix.lower() in _TEXT_EXTS
                for p in Path(shared["last_transcripts_folder"]).iterdir()):
            missing.append("the transcripts folder has no .txt / .tsv / .md files")
        missing += value_errors(prefs)

        ok = not missing
        self.run_button.setEnabled(ok)

        warnings = compute_warnings(session_api_key=self._get_api_key(),
                                    session_hf_token=self._get_hf_token())
        lines = []
        if acoustic_only:
            lines.append("ℹ Acoustic level is audio-only — every audio file is "
                         "processed; transcripts are not used.")
        lines += [f"• {w.message}" for w in warnings]

        if not ok:
            self.readiness_label.setText("✗  Not ready — " + "; ".join(missing) + ".")
        elif warnings:
            self.readiness_label.setText(f"⚠  {len(warnings)} warnings — review Features.")
        else:
            self.readiness_label.setText("✓  Ready to run.")
        self.warnings_label.setText("\n".join(lines))

        # Cost panel — dead: predictability offers local HuggingFace models
        # only (see extractors/predictability.py; OpenAI backend removed),
        # so there is no paid usage to estimate. Kept hidden rather than
        # deleted so a future paid backend can be wired back in without
        # rebuilding this panel.
        self.cost_panel.setVisible(False)

    def _on_progress_main(self, i: int, total: int, name: str) -> None:
        """Main-thread slot for worker-thread progress updates."""
        if self._status_strip:
            self._status_strip.set_running(i, total, name)

    def _on_cancel(self) -> None:
        self._cancel.set()
        if self._status_strip:
            self._status_strip.set_cancelling()

    def _on_done_main(self, status: str) -> None:
        """Main-thread slot for worker-thread completion."""
        if status == "done":
            if self._status_strip:
                self._status_strip.set_done(1, 0)
        elif status == "cancelled":
            if self._status_strip:
                self._status_strip.set_cancelled()
        else:
            if self._status_strip:
                self._status_strip.set_errored(1)
        self.run_finished.emit(status)
        self.run_button.setEnabled(True)

    def _on_run(self) -> None:
        self._cancel.clear()
        self.run_button.setEnabled(False)
        self.run_started.emit()
        if self._status_strip:
            self._status_strip.set_running(0, 1, "starting…")

        # Pass signal.emit (itself a callable) as the dispatcher callbacks.
        # The worker thread will emit; the connected slots run on the main
        # thread via Qt's auto-QueuedConnection.
        dispatch_run(
            progress_cb=self._progress_event.emit,
            done_cb=self._done_event.emit,
            session_api_key=self._get_api_key(),
            session_hf_token=self._get_hf_token(),
            should_cancel=self._cancel.is_set,
        )
