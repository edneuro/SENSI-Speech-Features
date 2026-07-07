"""Run section: output checkboxes, readiness summary, cost estimate, Run button."""

from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame,
)

from gui.widgets import SectionHeader, ToggleRow
from gui.preflight import compute_warnings
from gui.cost_estimate import estimate, count_input_tokens
from gui.run_dispatch import run as dispatch_run
from gui.planner import plan_from_prefs
from utils.preferences import load_preferences, save_preferences
from extractors.llm_cloze import LLM_OPTIONS


# Maps a plan output-CSV id -> the artifact it actually writes (path template +
# format), in display order. Drives the read-only "Will produce" summary so the
# Run tab shows the true output set (CSV *and* .mat) for the current settings,
# including the embedding .mat files that are toggled on the feature pages rather
# than the Run tab. Kept next to plan._OUTPUT_CSV_MAP conceptually.
_ARTIFACT_DESC: list[tuple[str, str]] = [
    ("features",          "word/<stim>_features.csv"),
    ("word2vec",          "word/<stim>_word2vec.mat   (Word page)"),
    ("bert_word",         "word/<stim>_bert_word.mat   (Word page)"),
    ("sentences",         "sentence/<stim>_sentences.csv"),
    ("bert_sentence",     "sentence/<stim>_bert_sentence.mat   (Sentence page)"),
    ("syllables",         "syllable/<stim>_syllables.csv"),
    ("phonemes",          "phoneme/<stim>_phonemes.csv"),
    ("acoustic",          "acoustic/<stim>_acoustic.mat   (audio-only)"),
    ("phoneme_surprisal_mat", "phoneme/<stim>_phoneme_surprisal.mat   (impulse trains)"),
    ("documents_summary", "document/documents_summary_<ts>.csv   (single file, all stimuli)"),
]


class RunSection(QWidget):
    run_started = Signal()
    run_finished = Signal(str)
    settings_changed = Signal()   # emitted when output toggles change
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
        # Cache for tiktoken cost computation — keyed by (model_id, folder_path).
        # Re-tokenizing every transcript file on every feature toggle was a
        # 1-3s freeze; we recompute only when model or folder changes.
        self._cost_cache_key: tuple[str, str] | None = None
        self._cost_cache_value: tuple[int, int] | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 16, 24, 16)
        layout.setSpacing(16)

        # Output
        layout.addWidget(SectionHeader("Output"))
        prefs = load_preferences()
        self.word_csv_toggle = ToggleRow("Word-level CSVs (one per stimulus)")
        self.word_csv_toggle.set_checked(prefs["output"]["word_level_csv"])
        self.word_csv_toggle.toggled.connect(
            lambda v: [save_preferences({"output": {"word_level_csv": v}}), self.refresh(), self.settings_changed.emit()])
        layout.addWidget(self.word_csv_toggle)
        self.doc_sum_toggle = ToggleRow("Document-level summary (single CSV)")
        self.doc_sum_toggle.set_checked(prefs["output"]["document_summary"])
        self.doc_sum_toggle.toggled.connect(
            lambda v: [save_preferences({"output": {"document_summary": v}}), self.refresh(), self.settings_changed.emit()])
        layout.addWidget(self.doc_sum_toggle)
        self.syllables_csv_toggle = ToggleRow("Syllables CSV (one per stimulus)")
        self.syllables_csv_toggle.set_checked(prefs["output"]["syllables_csv"])
        self.syllables_csv_toggle.toggled.connect(
            lambda v: [save_preferences({"output": {"syllables_csv": v}}), self.refresh(), self.settings_changed.emit()])
        layout.addWidget(self.syllables_csv_toggle)
        self.phonemes_csv_toggle = ToggleRow("Phonemes CSV (one per stimulus)")
        self.phonemes_csv_toggle.set_checked(prefs["output"]["phonemes_csv"])
        self.phonemes_csv_toggle.toggled.connect(
            lambda v: [save_preferences({"output": {"phonemes_csv": v}}), self.refresh(), self.settings_changed.emit()])
        layout.addWidget(self.phonemes_csv_toggle)
        self.sentences_csv_toggle = ToggleRow("Sentence-level CSV (one per stimulus)")
        self.sentences_csv_toggle.set_checked(prefs["output"]["sentences_csv"])
        self.sentences_csv_toggle.toggled.connect(
            lambda v: [save_preferences({"output": {"sentences_csv": v}}), self.refresh(), self.settings_changed.emit()])
        layout.addWidget(self.sentences_csv_toggle)
        self.acoustic_mat_toggle = ToggleRow("Acoustic .mat (audio-only, one per stimulus)")
        self.acoustic_mat_toggle.set_checked(prefs["output"]["acoustic_ts_mat"])
        self.acoustic_mat_toggle.toggled.connect(
            lambda v: [save_preferences({"output": {"acoustic_ts_mat": v}}), self.refresh(), self.settings_changed.emit()])
        layout.addWidget(self.acoustic_mat_toggle)
        self.phoneme_surprisal_mat_toggle = ToggleRow("Phoneme surprisal .mat (impulse trains)")
        self.phoneme_surprisal_mat_toggle.set_checked(prefs["output"]["phoneme_surprisal_mat"])
        self.phoneme_surprisal_mat_toggle.toggled.connect(
            lambda v: [save_preferences({"output": {"phoneme_surprisal_mat": v}}), self.refresh(), self.settings_changed.emit()])
        layout.addWidget(self.phoneme_surprisal_mat_toggle)

        # Read-only "what will actually be written" summary (CSV + .mat), derived
        # from the plan so it stays accurate and includes outputs whose toggles
        # live on the feature pages (embeddings).
        self.will_produce_label = QLabel("")
        self.will_produce_label.setWordWrap(True)
        self.will_produce_label.setObjectName("Subtitle")
        layout.addWidget(self.will_produce_label)

        # Readiness
        layout.addWidget(SectionHeader("Readiness"))
        self.readiness_label = QLabel("…")
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

        # "Will produce" summary — exactly the files this run will write.
        produced = [desc for csv_id, desc in _ARTIFACT_DESC
                    if csv_id in plan.output_csvs]
        if produced:
            self.will_produce_label.setText(
                "Will produce (one per stimulus unless noted):\n"
                + "\n".join(f"   • {d}" for d in produced))
        else:
            self.will_produce_label.setText(
                "Will produce: nothing yet — enable an output above "
                "(or a feature on its page).")

        # Required inputs depend on which outputs are actually selected. The
        # acoustic level is audio-only: its runner globs the audio folder and
        # ignores transcripts, so an acoustic-only run must NOT require a
        # transcripts folder. Every other output is derived from the transcript,
        # so those still require it. Audio is required whenever any output is
        # selected (acoustic needs it; the word/doc batches are paired with audio).
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

        missing = []
        if not has_output:
            missing.append("select at least one output")
        if not analyzer_ok:
            missing.append("enter an analyzer name (Inputs)")
        if not output_ok:
            missing.append("pick an output folder (Inputs)")
        if has_output and not audio_ok:
            missing.append("pick an audio folder (Inputs)")
        if needs_transcripts and not transcripts_ok:
            missing.append("pick a transcripts folder (needed for word / document "
                           "/ sub-word / sentence / embedding outputs)")

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

        # Cost panel — only when planner schedules llm_predict AND the model is an OpenAI one.
        # llm_model defaults to "" (never persisted when user keeps the combo's default);
        # treat "" as LLM_OPTIONS[0][0] (gpt-4o-mini) — what the combo shows and what
        # select_model() would pick at runtime when the key is unset.
        model = prefs["predictability"].get("llm_model", "") or LLM_OPTIONS[0][0]
        if "llm_predict" in plan.stages_to_run and model.startswith("gpt-"):
            self.cost_panel.setVisible(True)
            tdir = prefs["shared"].get("last_transcripts_folder", "")
            # _dir_ok guards the empty-path case: Path("").is_dir() is True (-> "."
            # -> cwd), which would otherwise tokenize the whole project as transcripts.
            if _dir_ok(tdir):
                t = Path(tdir)
                cache_key = (model, str(t))
                if cache_key == self._cost_cache_key and self._cost_cache_value is not None:
                    n_in, n_out = self._cost_cache_value
                else:
                    paths = [p for p in t.iterdir() if p.is_file()]
                    n_in = count_input_tokens(paths, model)
                    n_out = sum(len(p.read_text(encoding="utf-8", errors="ignore").split())
                                 for p in paths if p.is_file())
                    self._cost_cache_key = cache_key
                    self._cost_cache_value = (n_in, n_out)
                est = estimate(model_id=model, n_input_tokens=n_in,
                                n_output_tokens=n_out)
                self.cost_detail.setText(
                    f"Model: {est.model_id}<br>"
                    f"Input tokens: ~ {est.n_input_tokens:,} (tiktoken)<br>"
                    f"Output tokens: ~ {est.n_output_tokens:,} (1/word)<br>"
                    f"Rates (last updated {est.rates_last_updated}): "
                    f"${est.usd_input/max(est.n_input_tokens,1)*1e6:.2f}/1M input, "
                    f"${est.usd_output/max(est.n_output_tokens,1)*1e6:.2f}/1M output<br>"
                    f"<b>Estimated: ≈ ${est.usd_total:.3f}</b><br>"
                    "⚠ OpenAI pricing changes — verify at openai.com/api/pricing"
                )
            else:
                self.cost_detail.setText("Pick a transcripts folder to compute cost.")
        else:
            self.cost_panel.setVisible(False)

    def _on_progress_main(self, i: int, total: int, name: str) -> None:
        """Main-thread slot for worker-thread progress updates."""
        if self._status_strip:
            self._status_strip.set_running(i, total, name)

    def _on_done_main(self, status: str) -> None:
        """Main-thread slot for worker-thread completion."""
        if status == "done":
            if self._status_strip:
                self._status_strip.set_done(1, 0)
        else:
            if self._status_strip:
                self._status_strip.set_errored(1)
        self.run_finished.emit(status)
        self.run_button.setEnabled(True)

    def _on_run(self) -> None:
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
        )
