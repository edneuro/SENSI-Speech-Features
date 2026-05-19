"""Sub-word features page — Syllables and Phonemes per-column toggles."""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QWidget, QVBoxLayout, QGridLayout

from gui.widgets import Card, ToggleRow
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

        # --- Syllables ---
        syll_card = Card(
            "Syllables",
            "One row per syllable. Requires MFA alignment.")
        syll_grid = QGridLayout()
        syll_keys = [
            # Timing
            ("feat_syllable_onset",        "Onset"),
            ("feat_syllable_end",          "End"),
            ("feat_syllable_duration",     "Duration"),
            # Label
            ("feat_syllable_label",        "Label (orthographic, pyphen-derived)"),
            ("feat_syllable_arpabet",      "ARPABET (dash-joined, with stress)"),
            # Prosody
            ("feat_syllable_pitch_mean",   "Pitch_Mean"),
            ("feat_syllable_volume_mean",  "Volume_Mean"),
            ("feat_syllable_pitch_delta",  "Pitch_Delta"),
            ("feat_syllable_volume_delta", "Volume_Delta"),
        ]
        for i, (key, label) in enumerate(syll_keys):
            syll_grid.addWidget(
                self._make_toggle(key, label, f[key]),
                i // 2, i % 2,
            )
        syll_card.add_layout(syll_grid)
        layout.addWidget(syll_card)

        # --- Phonemes ---
        phon_card = Card(
            "Phonemes",
            "One row per phoneme. Requires MFA alignment.")
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
        phon_card.add_layout(phon_grid)
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
