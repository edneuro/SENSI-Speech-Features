"""Document-level features page (batch 3): per-column Counts + Composite psych,
modal doc-level category selection."""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QDoubleSpinBox,
)

from gui.widgets import ClusterCard, ToggleRow
from gui.planner import plan_from_prefs
from utils.preferences import load_preferences, save_preferences

# Maps each doc-aggregate output toggle to (required_stage, hover_tooltip).
# If the aggregate is enabled but the required stage is absent from the plan,
# a warning indicator is shown on that toggle row.
_AGG_STAGE_REQ: dict[str, tuple[str, str]] = {
    "output_doc_agg_zipf_freq": (
        "word_freq",
        "<b>Will output NaN</b> — corpus frequency is not in the run plan.<br>"
        "• Enable corpus frequency (wordfreq) on the Word-level features page<br>"
        "• Ensure the Word level is on (Run tab)",
    ),
    "output_doc_agg_edu_zipf": (
        "ewfg_freq",
        "<b>Will output NaN</b> — EWFG grade-band frequency is not in the run plan.<br>"
        "• Enable at least one EWFG band on the Word-level features page<br>"
        "• Ensure the Word level is on (Run tab)",
    ),
    "output_doc_agg_predictability": (
        "predictability",
        "<b>Will output NaN</b> — predictability is not in the run plan.<br>"
        "• Select at least one model and one metric in the Predictability "
        "card on the Word-level features page<br>"
        "• Ensure the Word level is on (Run tab)",
    ),
}


class DocFeaturesSection(QWidget):
    settings_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._toggles: dict[str, ToggleRow] = {}
        prefs = load_preferences()
        f = prefs["features"]
        o = prefs["output"]

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 16, 24, 16)
        layout.setSpacing(12)

        # The Document level switch lives on the Run page. Each card here
        # has a section switch plus its column ticks; the planner decides
        # what runs from those prefs.

        # --- Counts (per-column) ---
        counts_card = ClusterCard(
            "Counts",
            "Aggregate counts for the document-level summary CSV.",
            master_key="master_doc_counts",
            master_section="output",
        )
        counts_card.settings_changed.connect(self.settings_changed)
        counts_grid = QGridLayout()
        counts_keys = [
            ("output_out_count_tokens", "n_tokens"),
            ("output_out_count_content_tokens", "n_content_tokens"),
            ("output_out_count_sentences", "n_sentences"),
            ("output_out_count_paragraphs", "n_paragraphs"),
        ]
        for i, (key, label) in enumerate(counts_keys):
            counts_grid.addWidget(
                self._make_toggle(key, label, o[key[len("output_"):]], "output"),
                i // 2, i % 2,
            )
        counts_card.add_group("Columns", counts_grid)
        layout.addWidget(counts_card)

        # --- Aggregates (individual per-family toggle) ---
        agg_card = ClusterCard(
            "Aggregates", "Means of word-level metrics.",
            master_key="master_doc_aggregates",
            master_section="output",
        )
        agg_card.settings_changed.connect(self.settings_changed)
        agg_box = QVBoxLayout()
        for key, label in [
            ("output_doc_agg_zipf_freq",
             "mean_zipf_freq  (corpus-frequency mean, wordfreq)"),
            ("output_doc_agg_edu_zipf",
             "mean_edu_zipf_*  (EWFG grade-band means: K2/3–5/6–8/9–12/Overall)"),
            ("output_doc_agg_predictability",
             "mean_<model>_<metric>  (per selected model x metric — see "
             "the Predictability card)"),
            ("output_doc_agg_lexical_norms",
             "Lexical-norm aggregates (mean AoA/concreteness/…, prop_late_acquired, coverage)"),
        ]:
            agg_box.addWidget(
                self._make_toggle(key, label, o[key[len("output_"):]], "output"))
        agg_card.add_group("Means", agg_box)
        # EWFG is unavailable (source only in print; see Word-level features),
        # so its document-level aggregate has nothing to aggregate — disable it.
        self._toggles["output_doc_agg_edu_zipf"].set_permanently_disabled(
            "Unavailable: the Educator's Word Frequency Guide (Zeno et al. 1995) "
            "is not digitized, so the word-level EWFG columns it averages do not "
            "exist. Disabled until the norms are obtained."
        )
        edu_agg_note = QLabel(
            "<i>mean_edu_zipf_*</i> is <b>disabled</b>: the EWFG word-level "
            "frequencies it averages are unavailable (their source exists only "
            "in print). See the Frequency card on the Word-level features page."
        )
        edu_agg_note.setObjectName("Subtitle")
        edu_agg_note.setWordWrap(True)
        agg_card.add_child(edu_agg_note)
        # Late-acquired AoA threshold (used by prop_late_acquired in the
        # lexical-norm aggregates). Configurable here.
        thr_row = QHBoxLayout()
        thr_lbl = QLabel("Late-acquired AoA threshold (years):")
        thr_lbl.setObjectName("Subtitle")
        self._late_aoa_spin = QDoubleSpinBox()
        self._late_aoa_spin.setRange(0.0, 25.0)
        self._late_aoa_spin.setSingleStep(0.5)
        self._late_aoa_spin.setDecimals(1)
        self._late_aoa_spin.setValue(
            float(prefs.get("lexical_norms", {}).get("late_acquired_threshold", 10)))
        self._late_aoa_spin.valueChanged.connect(self._on_late_aoa_changed)
        thr_row.addWidget(thr_lbl)
        thr_row.addWidget(self._late_aoa_spin)
        thr_row.addStretch(1)
        thr_w = QWidget(); thr_w.setLayout(thr_row)
        agg_card.add_child(thr_w)
        layout.addWidget(agg_card)

        # --- Lexical diversity ---
        ld_card = ClusterCard(
            "Lexical diversity", "",
            master_key="master_doc_lex_diversity",
            master_section="output",
        )
        ld_card.settings_changed.connect(self.settings_changed)
        ld_card.add_child(self._make_toggle(
            "output_doc_lex_diversity",
            "Type-token ratio (TTR) and MTLD",
            o["doc_lex_diversity"], "output"))
        layout.addWidget(ld_card)

        # --- Readability ---
        rd_card = ClusterCard(
            "Readability", "",
            master_key="master_doc_readability",
            master_section="output",
        )
        rd_card.settings_changed.connect(self.settings_changed)
        rd_card.add_child(self._make_toggle(
            "output_doc_readability",
            "Flesch reading ease + Flesch-Kincaid grade level",
            o["doc_readability"], "output"))
        layout.addWidget(rd_card)

        # --- Category proportions ---
        cat_card = ClusterCard(
            "Category proportions",
            "Per-category count and proportion in the document.",
            master_key="master_doc_category_proportions",
            master_section="output",
        )
        cat_card.settings_changed.connect(self.settings_changed)
        cat_card.add_child(self._make_toggle(
            "output_doc_category_proportions",
            "Compute count_<cat>, prop_<cat> for each active category",
            o["doc_category_proportions"], "output"))
        self.cats_count_label = QLabel(self._cats_count_text())
        self.cats_count_label.setWordWrap(True)
        hint = QLabel("Edit categories on the Word-level features page — the selection is shared between levels.")
        hint.setObjectName("Subtitle")
        hint.setWordWrap(True)
        cat_card.add_child(self.cats_count_label)
        cat_card.add_child(hint)
        layout.addWidget(cat_card)

        # --- Composite psychological scores (per-column) ---
        psych_card = ClusterCard(
            "Composite psychological scores",
            "Pennebaker (2014) formulas computed license-clean from spaCy "
            "POS/morphology + VADER valence. Approximations, not official "
            "LIWC-22 outputs.",
            master_key="master_psych_composite",
        )
        self._psych_card = psych_card
        psych_card.settings_changed.connect(self.settings_changed)
        psych_grid = QGridLayout()
        psych_keys = [
            ("feat_psych_authentic", "psych_authentic"),
            ("feat_psych_analytic",  "psych_analytic"),
            ("feat_psych_tone",      "psych_tone"),
            ("feat_psych_clout",     "psych_clout"),
        ]
        for i, (key, label) in enumerate(psych_keys):
            psych_grid.addWidget(
                self._make_toggle(key, label, f[key], "features"),
                i // 2, i % 2,
            )
        psych_card.add_group("Columns", psych_grid)
        layout.addWidget(psych_card)

        # --- Prosody (per-column, 16 toggles) ---
        # master_doc_articulation_rate is gated by the per-feature toggle; no separate UI master needed
        pros_card = ClusterCard(
            "Prosody",
            "Document-level pitch/volume aggregates, HNR, articulation rate, "
            "and pause statistics. Requires alignment.",
            master_key="master_doc_prosody",
            master_section="output",
        )
        pros_card.settings_changed.connect(self.settings_changed)
        pros_grid = QGridLayout()
        pros_keys = [
            # Row 0: pitch summaries
            ("feat_doc_prosody_pitch_mean",  "doc_pitch_mean"),
            ("feat_doc_prosody_pitch_min",   "doc_pitch_min"),
            ("feat_doc_prosody_pitch_max",   "doc_pitch_max"),
            ("feat_doc_prosody_pitch_std",   "doc_pitch_std"),
            # Row 1: volume summaries
            ("feat_doc_prosody_volume_mean", "doc_volume_mean"),
            ("feat_doc_prosody_volume_min",  "doc_volume_min"),
            ("feat_doc_prosody_volume_max",  "doc_volume_max"),
            ("feat_doc_prosody_volume_std",  "doc_volume_std"),
            # Row 2: deltas + HNR + articulation
            ("feat_doc_prosody_pitch_delta_mean",  "doc_pitch_delta_mean"),
            ("feat_doc_prosody_volume_delta_mean", "doc_volume_delta_mean"),
            ("feat_doc_prosody_hnr",                "doc_hnr"),
            ("feat_doc_prosody_articulation_rate",  "doc_articulation_rate"),
            # Row 3: pause stats
            ("feat_doc_prosody_pause_count",         "doc_pause_count"),
            ("feat_doc_prosody_pause_mean_duration", "doc_pause_mean_duration"),
            ("feat_doc_prosody_pause_total_duration","doc_pause_total_duration"),
            ("feat_doc_prosody_pause_pct_silence",   "doc_pause_pct_silence"),
        ]
        for i, (key, label) in enumerate(pros_keys):
            # 2 columns: 4 columns of long doc_* column names forced the page
            # wider than the window (horizontal scrollbar); grow vertically.
            pros_grid.addWidget(
                self._make_toggle(key, label, f[key], "features"),
                i // 2, i % 2,
            )
        pros_card.add_group("Columns", pros_grid)

        # Pause threshold spinbox — config UI, greys with master
        prosody_prefs = prefs.get("prosody", {})
        thresh_row = QHBoxLayout()
        thresh_row.addWidget(QLabel("Pause threshold (sec):"))
        self.pause_threshold_spin = QDoubleSpinBox()
        self.pause_threshold_spin.setRange(0.05, 5.0)
        self.pause_threshold_spin.setSingleStep(0.05)
        self.pause_threshold_spin.setValue(prosody_prefs.get("pause_threshold_s", 0.25))
        self.pause_threshold_spin.valueChanged.connect(
            lambda v: save_preferences({"prosody": {"pause_threshold_s": v}}))
        thresh_row.addWidget(self.pause_threshold_spin)
        thresh_row.addStretch(1)
        thresh_w = QWidget(); thresh_w.setLayout(thresh_row)
        pros_card.add_child(thresh_w)
        layout.addWidget(pros_card)

        layout.addStretch(1)
        self._refresh_agg_warnings()

    def toggle(self, key: str): return self._toggles.get(key)

    def _make_toggle(self, key, label, initial, section):
        row = ToggleRow(label=label, info_key=key)
        row.set_checked(initial)
        row.toggled.connect(lambda v, k=key, s=section: self._on_toggle(s, k, v))
        self._toggles[key] = row
        return row

    def _on_toggle(self, section, key, value):
        prefix = "output_"
        if key.startswith(prefix):
            save_preferences({"output": {key[len(prefix):]: value}})
        else:
            save_preferences({section: {key: value}})
        self.settings_changed.emit()

    def _on_late_aoa_changed(self, value: float) -> None:
        save_preferences({"lexical_norms": {"late_acquired_threshold": float(value)}})
        self.settings_changed.emit()

    def refresh_from_prefs(self) -> None:
        """Re-read prefs and update dynamic labels, card states, and aggregate warnings."""
        self.cats_count_label.setText(self._cats_count_text())
        # Composite psych scores are license-clean (spaCy + VADER) and no
        # longer depend on the semantic backend, so the card is always
        # available.
        self._refresh_agg_warnings()

    def _refresh_agg_warnings(self) -> None:
        """Show/hide warning indicators on doc-aggregate toggles.

        A warning appears when the aggregate is enabled but the word-level
        stage it depends on is not in the current run plan (either because
        the feature toggle is off or the Word level is off).
        """
        prefs = load_preferences()
        plan = plan_from_prefs(prefs)
        in_plan = plan.stages_to_run
        o = prefs["output"]
        for toggle_key, (stage, msg) in _AGG_STAGE_REQ.items():
            row = self._toggles.get(toggle_key)
            if row is None:
                continue
            agg_key = toggle_key[len("output_"):]
            if o.get(agg_key, False) and stage not in in_plan:
                row.set_warning(msg)
            else:
                row.set_warning(None)

    def _cats_count_text(self) -> str:
        prefs = load_preferences()
        backend = prefs["semantic"].get("backend", "auto")
        if backend == "auto":
            from extractors.semantic_domain import detect_backend
            backend, _ = detect_backend()
        key = f"categories_{backend}"
        sel = prefs["semantic"].get(key, [])
        return f"{len(sel)} categories active in document-level summary."
