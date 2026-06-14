"""Acoustic-level features page — audio-only time-series sidecar toggles + params."""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QGridLayout, QHBoxLayout, QLabel,
    QComboBox, QDoubleSpinBox, QSpinBox,
)

from gui.widgets import Card, ToggleRow
from utils.preferences import load_preferences, save_preferences


class AcousticFeaturesSection(QWidget):
    settings_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._toggles: dict[str, ToggleRow] = {}
        prefs = load_preferences()
        f = prefs["features"]
        a = prefs["acoustic_ts"]

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 16, 24, 16)
        layout.setSpacing(12)

        # --- Acoustic time-series features ---
        # The acoustic level is run/saved from the Run tab ("Acoustic .mat"
        # output toggle). This page only chooses WHICH features it computes — so
        # it is a plain Card (no master), matching how other levels configure
        # their features on their own page while the Run tab gates output.
        ts_card = Card(
            "Acoustic features (audio-only)",
            "Continuous acoustic time series on a shared analysis grid. Enable "
            "the output in the Run tab ('Acoustic .mat'); choose the features here.",
        )
        ts_grid = QGridLayout()
        acoustic_keys = [
            ("feat_acoustic_ts_envelope",  "Amplitude envelope (broadband)"),
            ("feat_acoustic_ts_onset_env", "Onset envelope (HWR derivative)"),
            ("feat_acoustic_ts_logmel",    "Log-mel spectrogram"),
            ("feat_acoustic_ts_mfcc",      "MFCC (off by default)"),
        ]
        for i, (key, label) in enumerate(acoustic_keys):
            ts_grid.addWidget(
                self._make_toggle(key, label, f[key]),
                i // 2, i % 2,
            )
        ts_card.add_layout(ts_grid)
        layout.addWidget(ts_card)

        # --- Parameters ---
        params_card = Card(
            "Acoustic parameters",
            "Analysis grid and feature settings shared across all acoustic time series.",
        )

        rate_row = QHBoxLayout()
        rate_row.addWidget(QLabel("Analysis rate (Hz):"))
        self._rate_spin = QDoubleSpinBox()
        self._rate_spin.setRange(1.0, 2000.0)
        self._rate_spin.setDecimals(1)
        self._rate_spin.setValue(float(a.get("analysis_rate", 128)))
        self._rate_spin.valueChanged.connect(
            lambda v: self._save_param("analysis_rate", v))
        rate_row.addWidget(self._rate_spin)
        rate_row.addStretch(1)
        _rate_w = QWidget(); _rate_w.setLayout(rate_row)
        params_card.add_widget(_rate_w)

        env_row = QHBoxLayout()
        env_row.addWidget(QLabel("Envelope method:"))
        self._env_method_combo = QComboBox()
        self._env_method_combo.addItems(["rms", "hilbert"])
        self._env_method_combo.setCurrentText(a.get("env_method", "rms"))
        self._env_method_combo.currentTextChanged.connect(
            lambda v: self._save_param("env_method", v))
        env_row.addWidget(self._env_method_combo)
        env_row.addWidget(QLabel("Compression exponent:"))
        self._env_comp_spin = QDoubleSpinBox()
        self._env_comp_spin.setRange(0.0, 2.0)
        self._env_comp_spin.setSingleStep(0.05)
        self._env_comp_spin.setDecimals(2)
        self._env_comp_spin.setValue(float(a.get("env_comp", 0.3)))
        self._env_comp_spin.valueChanged.connect(
            lambda v: self._save_param("env_comp", v))
        env_row.addWidget(self._env_comp_spin)
        env_row.addStretch(1)
        _env_w = QWidget(); _env_w.setLayout(env_row)
        params_card.add_widget(_env_w)

        mel_row = QHBoxLayout()
        mel_row.addWidget(QLabel("Mel bands (n_mels):"))
        self._n_mels_spin = QSpinBox()
        self._n_mels_spin.setRange(1, 128)
        self._n_mels_spin.setValue(int(a.get("n_mels", 16)))
        self._n_mels_spin.valueChanged.connect(
            lambda v: self._save_param("n_mels", v))
        mel_row.addWidget(self._n_mels_spin)
        mel_row.addWidget(QLabel("MFCC (n_mfcc):"))
        self._n_mfcc_spin = QSpinBox()
        self._n_mfcc_spin.setRange(1, 60)
        self._n_mfcc_spin.setValue(int(a.get("n_mfcc", 13)))
        self._n_mfcc_spin.valueChanged.connect(
            lambda v: self._save_param("n_mfcc", v))
        mel_row.addWidget(self._n_mfcc_spin)
        mel_row.addWidget(QLabel("Mel preset:"))
        self._mel_preset_combo = QComboBox()
        self._mel_preset_combo.addItems(["speech", "music", "general"])
        self._mel_preset_combo.setCurrentText(a.get("mel_preset", "speech"))
        self._mel_preset_combo.currentTextChanged.connect(
            lambda v: self._save_param("mel_preset", v))
        mel_row.addWidget(self._mel_preset_combo)
        mel_row.addStretch(1)
        _mel_w = QWidget(); _mel_w.setLayout(mel_row)
        params_card.add_widget(_mel_w)

        layout.addWidget(params_card)
        layout.addStretch(1)

    def toggle(self, key: str): return self._toggles.get(key)

    def _make_toggle(self, key: str, label: str, initial: bool) -> ToggleRow:
        row = ToggleRow(label=label, info_key=key)
        row.set_checked(initial)
        row.toggled.connect(lambda v, k=key: self._on_toggle(k, v))
        self._toggles[key] = row
        return row

    def _on_toggle(self, key: str, value: bool) -> None:
        save_preferences({"features": {key: value}})
        self.settings_changed.emit()

    def _save_param(self, key: str, value) -> None:
        save_preferences({"acoustic_ts": {key: value}})
        self.settings_changed.emit()

    def refresh_from_prefs(self) -> None:
        prefs = load_preferences()
        f = prefs["features"]
        for key, toggle in self._toggles.items():
            toggle.blockSignals(True)
            toggle.set_checked(f.get(key, True))
            toggle.blockSignals(False)
