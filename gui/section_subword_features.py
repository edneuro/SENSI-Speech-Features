"""Sub-word features page — Syllables (3 cluster cards) and Phonemes per-column toggles."""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QWidget, QVBoxLayout, QGridLayout, QHBoxLayout, QLabel, QDoubleSpinBox

from gui.widgets import Card, ClusterCard, ToggleRow
from utils.preferences import load_preferences, save_preferences


class SubwordFeaturesSection(QWidget):
    settings_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._toggles: dict[str, ToggleRow] = {}
        prefs = load_preferences()
        f = prefs["features"]

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 16, 24, 16)
        layout.setSpacing(12)

        # --- Syllable timing ---
        syll_timing_card = ClusterCard(
            "Syllable timing",
            "Onset/End/Duration of each syllable",
            master_key="master_syllable_intervals",
            master_label="Emit syllable interval columns",
        )
        syll_timing_grid = QGridLayout()
        syll_timing_keys = [
            ("feat_syllable_onset",    "Onset"),
            ("feat_syllable_end",      "End"),
            ("feat_syllable_duration", "Duration"),
        ]
        for i, (key, label) in enumerate(syll_timing_keys):
            syll_timing_grid.addWidget(
                self._make_toggle(key, label, f[key]),
                i // 2, i % 2,
            )
        syll_timing_card.add_child_layout(syll_timing_grid)
        syll_timing_card.settings_changed.connect(self.settings_changed)
        layout.addWidget(syll_timing_card)

        # --- Syllable labels ---
        syll_labels_card = ClusterCard(
            "Syllable labels",
            "Orthographic + ARPABET labels",
            master_key="master_syllable_labels",
            master_label="Emit syllable label columns",
        )
        syll_labels_grid = QGridLayout()
        syll_labels_keys = [
            ("feat_syllable_label",   "Label (orthographic, pyphen-derived)"),
            ("feat_syllable_arpabet", "ARPABET (dash-joined, with stress)"),
        ]
        for i, (key, label) in enumerate(syll_labels_keys):
            syll_labels_grid.addWidget(
                self._make_toggle(key, label, f[key]),
                i // 2, i % 2,
            )
        syll_labels_card.add_child_layout(syll_labels_grid)
        syll_labels_card.settings_changed.connect(self.settings_changed)
        layout.addWidget(syll_labels_card)

        # --- Syllable prosody ---
        syll_pros_card = ClusterCard(
            "Syllable prosody",
            "Pitch/Volume means and deltas per syllable",
            master_key="master_syllable_prosody",
            master_label="Emit syllable prosody columns",
        )
        syll_pros_grid = QGridLayout()
        syll_pros_keys = [
            ("feat_syllable_pitch_mean",   "Pitch_Mean"),
            ("feat_syllable_volume_mean",  "Volume_Mean"),
            ("feat_syllable_pitch_delta",  "Pitch_Delta"),
            ("feat_syllable_volume_delta", "Volume_Delta"),
        ]
        for i, (key, label) in enumerate(syll_pros_keys):
            syll_pros_grid.addWidget(
                self._make_toggle(key, label, f[key]),
                i // 2, i % 2,
            )
        syll_pros_card.add_child_layout(syll_pros_grid)
        syll_pros_card.settings_changed.connect(self.settings_changed)
        layout.addWidget(syll_pros_card)

        # --- P-center ---
        pcenter_card = Card(
            "P-center",
            "Perceptual onset of each syllable (Scott 1993). "
            "Requires MFA alignment.",
        )
        pcenter_card.add_widget(
            self._make_toggle(
                "feat_syllable_pcenter",
                "P_Center (seconds from audio start)",
                f["feat_syllable_pcenter"],
            )
        )
        # k coefficient spinbox
        _pros = prefs.get("prosody", {})
        k_row = QHBoxLayout()
        k_row.addWidget(QLabel("k coefficient (0–1):"))
        self._pcenter_k_spin = QDoubleSpinBox()
        self._pcenter_k_spin.setRange(0.0, 1.0)
        self._pcenter_k_spin.setSingleStep(0.05)
        self._pcenter_k_spin.setDecimals(2)
        self._pcenter_k_spin.setValue(_pros.get("pcenter_k", 0.3))
        self._pcenter_k_spin.valueChanged.connect(
            lambda v: save_preferences({"prosody": {"pcenter_k": v}}))
        k_row.addWidget(self._pcenter_k_spin)
        k_row.addStretch(1)
        k_w = QWidget()
        k_w.setLayout(k_row)
        pcenter_card.add_widget(k_w)
        layout.addWidget(pcenter_card)

        # --- Phonemes ---
        phon_card = ClusterCard(
            "Phonemes",
            "One row per phoneme. Requires MFA alignment.",
            master_key="master_phoneme_intervals",
            master_label="Emit phoneme columns",
        )
        phon_grid = QGridLayout()
        phon_keys = [
            ("feat_phoneme_label",    "Label (ARPABET incl. stress digit)"),
            ("feat_phoneme_onset",    "Onset"),
            ("feat_phoneme_end",      "End"),
            ("feat_phoneme_duration", "Duration"),
        ]
        for i, (key, label) in enumerate(phon_keys):
            phon_grid.addWidget(
                self._make_toggle(key, label, f[key]),
                i // 2, i % 2,
            )
        phon_card.add_child_layout(phon_grid)
        phon_card.settings_changed.connect(self.settings_changed)
        layout.addWidget(phon_card)

        layout.addStretch(1)

    def toggle(self, key: str): return self._toggles.get(key)

    # Note: sibling pages (section_word_features, section_doc_features) take a
    # `section` arg here to route saves to "features" vs "output". This page
    # only carries `features` keys (sidecar master toggles live on the Run
    # page), so the simpler signature suffices. Add a `section` parameter if
    # an output toggle is ever added to this page.
    def _make_toggle(self, key, label, initial):
        row = ToggleRow(label=label)
        row.set_checked(initial)
        row.toggled.connect(lambda v, k=key: self._on_toggle(k, v))
        self._toggles[key] = row
        return row

    def _on_toggle(self, key, value):
        save_preferences({"features": {key: value}})
        self.settings_changed.emit()
