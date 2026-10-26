"""Run section: one switch per output level, readiness summary, cost estimate, Run button."""

import logging
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
from gui.theme import TEXT_MUTED
from utils.config_checks import value_errors
from utils.preferences import load_preferences, save_preferences

_log = logging.getLogger(__name__)

# Live estimate workers (see RunSection._start_estimates). Module-level so
# tests (and shutdown code) can wait for them: wait_for_estimate_threads.
_ESTIMATE_THREADS: set[threading.Thread] = set()
_ESTIMATE_THREADS_LOCK = threading.Lock()

_EMPTY_ESTIMATES = {"file_sizes": {}, "level_sizes": {}, "disk": "—",
                    "ram": "—", "time": "—", "warning": None, "learned": True}


def wait_for_estimate_threads(timeout: float = 10.0) -> bool:
    """Join every running estimate worker; True when none is left alive."""
    with _ESTIMATE_THREADS_LOCK:
        threads = list(_ESTIMATE_THREADS)
    for t in threads:
        t.join(timeout)
    return not any(t.is_alive() for t in threads)

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
        self.size_label = QLabel()
        self.size_label.setObjectName("LevelSize")
        self.size_label.hide()
        head.addWidget(self.size_label, 0, Qt.AlignmentFlag.AlignVCenter)
        head.addStretch(1)
        body.addLayout(head)
        self.desc_label = QLabel(desc)
        self.desc_label.setObjectName("LevelDesc")
        self.desc_label.setWordWrap(True)
        body.addWidget(self.desc_label)
        self.file_labels: dict[str, QLabel] = {}
        self.file_size_labels: dict[str, QLabel] = {}
        for art in LEVEL_ARTIFACTS[key]:
            line = QHBoxLayout()
            line.setSpacing(6)
            lbl = QLabel(_ARTIFACT_PATHS[art])
            lbl.setObjectName("LevelFile")
            lbl.setTextFormat(Qt.TextFormat.PlainText)
            line.addWidget(lbl)
            sz = QLabel()
            sz.setObjectName("LevelFileSize")
            sz.hide()
            line.addWidget(sz)
            line.addStretch(1)
            body.addLayout(line)
            self.file_labels[art] = lbl
            self.file_size_labels[art] = sz
        outer.addLayout(body, 1)

    def update_from(self, on: bool, live, sizes: dict[str, str] | None = None,
                    level_size: str = "") -> None:
        """Sync the switch (without re-saving), the tag and the file lines.

        `sizes` maps artifact -> display size (missing = nothing to show);
        `level_size` is the level total shown after the count tag.
        """
        sizes = sizes or {}
        self.switch.blockSignals(True)
        self.switch.setChecked(on)
        self.switch.blockSignals(False)
        self.switch.update()
        n_live = sum(1 for a in self.file_labels if a in live)
        self.tag_label.setText(_count_tag(self.key, on, n_live))
        self.size_label.setText(level_size)
        self.size_label.setVisible(on and n_live > 0 and bool(level_size))
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
            sz = self.file_size_labels[art]
            text = sizes.get(art, "") if produced else ""
            sz.setText(text)
            sz.setVisible(on and bool(text))


def _gb(x: float) -> str:
    """One decimal under 10 GB (so 0.9 GB doesn't read as 1 or 0), whole GB above."""
    return f"{x:.1f}" if x < 10 else f"{x:.0f}"


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
    # Estimate worker -> main thread: (generation, estimates dict).
    _estimates_event = Signal(int, object)

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
        self._estimates_event.connect(self._on_estimates_main)
        # Estimates are computed off the GUI thread (scan_inputs may hash
        # audio files). Only the result of the latest request is applied.
        self._est_gen = 0
        self._est_busy = False
        self._est_pending = None
        self._est_applied_gen = 0              # generation shown right now
        self._est_last: dict | None = None     # last applied result
        self._est_view: tuple | None = None    # (prefs, plan) of latest refresh
        self._system_info_lock = threading.Lock()
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

        # Estimates summary (disk / peak memory / time). Display only: it
        # never touches run_button; values are "—" until the inputs are set.
        self._system_info: dict | None = None
        self.estimates_box = QFrame()
        self.estimates_box.setObjectName("EstimatesBox")
        eb = QVBoxLayout(self.estimates_box)
        eb.setContentsMargins(12, 10, 12, 10)
        eb.setSpacing(3)
        self.est_values: dict[str, QLabel] = {}
        for key, name in (("disk", "Disk space for outputs"),
                          ("ram", "Peak memory (RAM)"),
                          ("time", "Expected time")):
            row = QHBoxLayout()
            k = QLabel(name)
            k.setObjectName("EstimateKey")
            v = QLabel("—")
            v.setObjectName("EstimateValue")
            # RAM carries a lighter "this computer: N GB" span (mockup).
            v.setTextFormat(Qt.TextFormat.RichText if key == "ram"
                            else Qt.TextFormat.PlainText)
            row.addWidget(k)
            row.addStretch(1)
            row.addWidget(v)
            eb.addLayout(row)
            self.est_values[key] = v
        self.estimate_note = QLabel(
            "First estimate. It gets more precise as the app learns from "
            "runs on this computer.")
        self.estimate_note.setObjectName("EstimateNote")
        self.estimate_note.setWordWrap(True)
        eb.addWidget(self.estimate_note)
        layout.addWidget(self.estimates_box)
        self.estimate_warning = QLabel("")
        self.estimate_warning.setObjectName("EstimateWarning")
        self.estimate_warning.setWordWrap(True)
        self.estimate_warning.setTextFormat(Qt.TextFormat.PlainText)
        self.estimate_warning.hide()
        layout.addWidget(self.estimate_warning)

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
        # Estimates come from a background worker: show the previous values
        # (dashes before the first result) until the new ones arrive.
        self._est_view = (prefs, plan)
        self._apply_estimates(self._est_last or _EMPTY_ESTIMATES)
        self._start_estimates(prefs, plan)

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

    def _get_system_info(self) -> dict:
        """Hardware facts, gathered once per RunSection (imports torch lazily).
        Called from estimate workers, hence the lock."""
        with self._system_info_lock:
            if self._system_info is None:
                from utils.system_info import gather_system_info
                self._system_info = gather_system_info()
            return self._system_info

    def _start_estimates(self, prefs: dict, plan) -> None:
        """Compute estimates on a daemon thread; the result is marshalled back
        through _estimates_event (queued to the GUI thread). A newer request
        bumps the generation, so a stale result is dropped on arrival.

        One worker at a time: a request made while one is in flight is parked
        and started when it returns, so a cold first scan (which hashes the
        audio) never runs twice side by side."""
        self._est_gen += 1
        gen = self._est_gen
        if self._est_busy:
            self._est_pending = (prefs, plan, gen)
            return
        self._spawn_estimates(prefs, plan, gen)

    def _spawn_estimates(self, prefs: dict, plan, gen: int) -> None:
        self._est_busy = True
        level_keys = tuple(self.level_rows)

        def _work() -> None:
            try:
                est = self._compute_estimates(prefs, plan, level_keys)
                try:
                    self._estimates_event.emit(gen, est)
                except RuntimeError:
                    pass      # the widget was deleted while we computed
            finally:
                with _ESTIMATE_THREADS_LOCK:
                    _ESTIMATE_THREADS.discard(threading.current_thread())

        t = threading.Thread(target=_work, daemon=True,
                             name=f"run-estimates-{gen}")
        with _ESTIMATE_THREADS_LOCK:
            _ESTIMATE_THREADS.add(t)
        t.start()

    def _on_estimates_main(self, gen: int, est: dict) -> None:
        """GUI-thread slot: apply only the latest request's result."""
        self._est_busy = False
        if self._est_pending is not None:
            pending, self._est_pending = self._est_pending, None
            self._spawn_estimates(*pending)
        if gen != self._est_gen:
            return
        self._est_last = est
        self._est_applied_gen = gen
        self._apply_estimates(est)

    def _apply_estimates(self, est: dict) -> None:
        if self._est_view is None:
            return
        prefs, plan = self._est_view
        for key, row in self.level_rows.items():
            row.update_from(bool(prefs["output"].get(key, False)),
                            plan.output_csvs,
                            est["file_sizes"], est["level_sizes"].get(key, ""))
        self._show_estimates(est)

    def _compute_estimates(self, prefs: dict, plan, level_keys) -> dict:
        """Run the estimators (worker thread; touches no widget). Any failure
        degrades to dashes, never an error."""
        empty = _EMPTY_ESTIMATES
        try:
            from utils import run_estimates as re_
            from utils.timing_history import load_history
            shared = prefs["shared"]
            stats = re_.scan_inputs(
                shared.get("last_transcripts_folder", ""),
                shared.get("last_audio_folder", ""),
                output_folder=shared.get("last_output_folder", ""),
                g2p=bool((prefs.get("alignment") or {}).get("g2p_oov", False)))
            if stats is None:
                return empty
            system = self._get_system_info()
            disk = re_.estimate_disk(plan, prefs, stats)
            ram = re_.estimate_ram(plan, prefs, system)
            cuda = bool((system.get("gpu") or {}).get("name"))
            tm = re_.estimate_time(plan, prefs, stats, load_history(), cuda)
            # Audio-only inputs: no transcript figures, so text levels show "—".
            has_text = stats.n_stimuli > 0
            shown = {a: b for a, b in disk.items()
                     if has_text or a == "acoustic"}
            level_sizes = {
                key: "≈ " + re_.format_bytes(re_.level_bytes(shown, key))
                for key in level_keys
                if has_text or key == "level_acoustic"}
            return {
                "file_sizes": {a: "≈ " + re_.format_bytes(b) for a, b in shown.items()},
                "level_sizes": level_sizes,
                "disk": ("≈ " + re_.format_bytes(sum(shown.values()))
                         if shown else "—"),
                "ram": (f"≈ {_gb(ram.peak_gb)} GB&nbsp;&nbsp;<span style="
                        f"'font-weight:400;color:{TEXT_MUTED}'>this computer: "
                        f"{ram.machine_gb:.0f} GB</span>"),
                "time": "≈ " + re_.format_duration(tm.low_s, tm.high_s),
                "warning": ram.warning, "learned": tm.learned,
            }
        except Exception:
            _log.exception("Run-tab estimates failed; showing dashes")
            return empty

    def _show_estimates(self, est: dict) -> None:
        for k, v in self.est_values.items():
            v.setText(est[k])
        self.estimate_note.setVisible(not est["learned"])
        self.estimate_warning.setText(est["warning"] or "")
        self.estimate_warning.setVisible(bool(est["warning"]))

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
