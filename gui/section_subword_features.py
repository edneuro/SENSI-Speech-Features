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
            ("feat_syllable_ipa",     "IPA (dash-joined)"),
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
            ("feat_phoneme_arpabet",     "ARPABET (bare code, no stress digit)"),
            ("feat_phoneme_stress",      "Stress (0/1/2; empty on consonants)"),
            ("feat_phoneme_ipa",         "IPA"),
            ("feat_phoneme_pron_source", "Pron_Source (dictionary/g2p/oov_unresolved)"),
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

        # --- Out-of-dictionary words (MFA G2P) ---
        g2p_card = Card(
            "Out-of-dictionary words",
            "MFA aligns words missing from its dictionary (unusual names) to a "
            "single 'spn' phone with no phoneme identity. Enable this to have "
            "MFA generate a pronunciation for them instead. The pronunciation "
            "is modeled from spelling, not observed — Pron_Source marks every "
            "affected word as 'g2p'. Changing this re-runs alignment once.",
        )
        g2p_card.add_widget(
            self._make_toggle(
                "g2p_oov",
                "Generate pronunciations for out-of-dictionary words (G2P)",
                prefs.get("alignment", {}).get("g2p_oov", False),
                section="alignment",
            )
        )
        layout.addWidget(g2p_card)

        # --- Phoneme surprisal (Brodbeck/Gillis cohort model) ---
        psr_card = ClusterCard(
            "Phoneme surprisal",
            "Context-independent cohort surprisal + entropy per phoneme. "
            "Requires MFA alignment. Also emits a .mat impulse-train sidecar "
            "when enabled on the Run page.",
            master_key="master_phoneme_surprisal",
            master_label="Emit phoneme surprisal columns",
        )
        psr_grid = QGridLayout()
        psr_keys = [
            ("feat_phoneme_surprisal",      "phoneme_surprisal (bits)"),
            ("feat_phoneme_cohort_entropy", "cohort_entropy (bits)"),
        ]
        for i, (key, label) in enumerate(psr_keys):
            psr_grid.addWidget(
                self._make_toggle(key, label, f[key]),
                i // 2, i % 2,
            )
        psr_card.add_child_layout(psr_grid)

        # Analysis rate, saved directly to the phoneme_surprisal section (like
        # pcenter_k -> prosody above). It's a config value rather than a
        # feature toggle, so it is not gated by the master. strip_stress /
        # primary_only are left at their prefs defaults (still functional
        # through emit) to keep the card minimal — analysis_sr is the param
        # researchers change most.
        _psr = prefs.get("phoneme_surprisal", {})
        psr_sr_row = QHBoxLayout()
        psr_sr_row.addWidget(QLabel("Analysis rate (Hz):"))
        self._psr_sr_spin = QDoubleSpinBox()
        self._psr_sr_spin.setRange(1.0, 1000.0)
        self._psr_sr_spin.setDecimals(0)
        self._psr_sr_spin.setValue(_psr.get("analysis_sr", 128))
        self._psr_sr_spin.valueChanged.connect(
            lambda v: save_preferences({"phoneme_surprisal": {"analysis_sr": v}}))
        psr_sr_row.addWidget(self._psr_sr_spin)
        psr_sr_row.addStretch(1)
        psr_sr_w = QWidget()
        psr_sr_w.setLayout(psr_sr_row)
        psr_card.add_widget(psr_sr_w)
        psr_card.settings_changed.connect(self.settings_changed)
        layout.addWidget(psr_card)

        layout.addStretch(1)

    def toggle(self, key: str): return self._toggles.get(key)

    # Note: sibling pages (section_word_features, section_doc_features) take a
    # `section` arg here to route saves to "features" vs "output". This page
    # mostly carries `features` keys (sidecar master toggles live on the Run
    # page) but also the standalone `alignment.g2p_oov` toggle, hence the
    # `section` parameter (mirrors the P-center `k` spinbox writing directly
    # to `prefs["prosody"]` above).
    def _make_toggle(self, key, label, initial, section="features"):
        row = ToggleRow(label=label, info_key=key)
        row.set_checked(initial)
        row.toggled.connect(lambda v, k=key, s=section: self._on_toggle(k, v, s))
        self._toggles[key] = row
        return row

    def _on_toggle(self, key, value, section="features"):
        save_preferences({section: {key: value}})
        self.settings_changed.emit()
