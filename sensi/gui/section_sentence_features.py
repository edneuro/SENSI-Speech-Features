"""Sentence-level features page — timing and prosody toggles."""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QWidget, QVBoxLayout, QGridLayout

from sensi.gui.widgets import ClusterCard, ToggleRow
from sensi.utils.preferences import load_preferences, save_preferences


class SentenceFeaturesSection(QWidget):
    settings_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._toggles: dict[str, ToggleRow] = {}
        prefs = load_preferences()
        f = prefs["features"]

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 16, 24, 16)
        layout.setSpacing(12)

        timing_card = ClusterCard(
            "Sentence timing",
            "Onset/End/text from MFA word alignment (min onset, max end per sentence)",
            master_key="master_sentence_timing",
        )
        timing_grid = QGridLayout()
        timing_keys = [
            ("feat_sentence_onset", "Onset"),
            ("feat_sentence_end",   "End"),
            ("feat_sentence_text",  "Sentence text"),
        ]
        for i, (key, label) in enumerate(timing_keys):
            timing_grid.addWidget(
                self._make_toggle(key, label, f[key]),
                i // 2, i % 2,
            )
        timing_card.add_group("Columns", timing_grid)
        timing_card.settings_changed.connect(self.settings_changed)
        layout.addWidget(timing_card)

        prosody_card = ClusterCard(
            "Sentence prosody",
            "Pitch, volume, and HNR aggregated over each sentence's [Onset, End] window",
            master_key="master_sentence_prosody",
        )
        prosody_grid = QGridLayout()
        prosody_keys = [
            ("feat_sentence_pitch_mean",   "Pitch Mean"),
            ("feat_sentence_pitch_min",    "Pitch Min (5th pct)"),
            ("feat_sentence_pitch_max",    "Pitch Max (95th pct)"),
            ("feat_sentence_pitch_std",    "Pitch Std"),
            ("feat_sentence_pitch_delta",  "Pitch Delta"),
            ("feat_sentence_volume_mean",  "Volume Mean"),
            ("feat_sentence_volume_min",   "Volume Min"),
            ("feat_sentence_volume_max",   "Volume Max"),
            ("feat_sentence_volume_std",   "Volume Std"),
            ("feat_sentence_volume_delta", "Volume Delta"),
            ("feat_sentence_hnr",          "HNR"),
        ]
        for i, (key, label) in enumerate(prosody_keys):
            prosody_grid.addWidget(
                self._make_toggle(key, label, f[key]),
                i // 2, i % 2,
            )
        prosody_card.add_group("Columns", prosody_grid)
        prosody_card.settings_changed.connect(self.settings_changed)
        layout.addWidget(prosody_card)

        # --- Semantic similarity (embeddings) ---
        sim_card = ClusterCard(
            "Semantic similarity",
            "Pretrained sentence embeddings saved as .mat sidecars in sentence/.",
            master_key="master_sentence_similarity",
        )
        sim_card.add_child(self._make_toggle(
            "master_bert_sentence",
            "BERT — bert-base-uncased, 12 layers × 768 d → <stim>_bert_sentence.mat",
            f.get("master_bert_sentence", True)))
        sim_card.settings_changed.connect(self.settings_changed)
        layout.addWidget(sim_card)

        layout.addStretch()

    def _make_toggle(self, key: str, label: str, initial: bool) -> ToggleRow:
        row = ToggleRow(label=label, info_key=key)
        row.set_checked(initial)
        row.toggled.connect(lambda v, k=key: self._on_toggle(k, v))
        self._toggles[key] = row
        return row

    def _on_toggle(self, key: str, value: bool) -> None:
        save_preferences({"features": {key: value}})
        self.settings_changed.emit()

    def refresh_from_prefs(self) -> None:
        prefs = load_preferences()
        f = prefs["features"]
        for key, toggle in self._toggles.items():
            toggle.blockSignals(True)
            toggle.set_checked(f.get(key, True))
            toggle.blockSignals(False)
