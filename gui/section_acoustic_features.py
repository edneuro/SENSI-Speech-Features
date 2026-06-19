"""Acoustic-level features page — audio-only time-series sidecar toggles + params.

Single-card layout: a shared analysis-rate grid on top, then each feature's
toggle with its own parameters indented directly beneath it. Parameter groups
grey out when no enabled feature consumes them (see _sync_param_enabled, which
mirrors extract_acoustic() in extractors/acoustic_ts.py)."""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QFrame,
    QComboBox, QDoubleSpinBox, QSpinBox,
)

from gui.widgets import Card, ToggleRow
from utils.preferences import load_preferences, save_preferences


def _indent(inner: QWidget, left: int = 28) -> QWidget:
    """Wrap a widget in a left-indented container so a feature's params sit
    visually beneath (and subordinate to) its toggle."""
    w = QWidget()
    lay = QHBoxLayout(w)
    lay.setContentsMargins(left, 0, 0, 0)
    lay.addWidget(inner)
    lay.addStretch(1)
    return w


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

        card = Card(
            "Acoustic features (audio-only)",
            "Continuous acoustic time series on a shared analysis grid. Enable "
            "the output in the Run tab ('Acoustic .mat'); choose features and "
            "their settings here.",
        )

        # --- Shared analysis grid (applies to every series) ---
        rate_row = QHBoxLayout()
        rate_row.addWidget(QLabel("Analysis rate (Hz):"))
        self._rate_spin = QDoubleSpinBox()
        self._rate_spin.setRange(1.0, 2000.0)
        self._rate_spin.setDecimals(1)
        self._rate_spin.setValue(float(a.get("analysis_rate", 128)))
        self._rate_spin.valueChanged.connect(
            lambda v: self._save_param("analysis_rate", v))
        rate_row.addWidget(self._rate_spin)
        rate_caption = QLabel("applies to every series below")
        rate_caption.setObjectName("Subtitle")
        rate_row.addWidget(rate_caption)
        rate_row.addStretch(1)
        _rate_w = QWidget(); _rate_w.setLayout(rate_row)
        card.add_widget(_rate_w)

        divider = QFrame()
        divider.setObjectName("SectionDivider")
        divider.setFixedHeight(1)
        card.add_widget(divider)

        # --- Amplitude envelope + its params ---
        card.add_widget(self._make_toggle(
            "feat_acoustic_ts_envelope", "Amplitude envelope (broadband)",
            f["feat_acoustic_ts_envelope"]))
        env_row = QHBoxLayout(); env_row.setContentsMargins(0, 0, 0, 0)
        env_row.addWidget(QLabel("Method:"))
        self._env_method_combo = QComboBox()
        self._env_method_combo.addItems(["rms", "hilbert"])
        self._env_method_combo.setCurrentText(a.get("env_method", "rms"))
        self._env_method_combo.currentTextChanged.connect(
            lambda v: self._save_param("env_method", v))
        env_row.addWidget(self._env_method_combo)
        env_row.addWidget(QLabel("Compression:"))
        self._env_comp_spin = QDoubleSpinBox()
        self._env_comp_spin.setRange(0.0, 2.0)
        self._env_comp_spin.setSingleStep(0.05)
        self._env_comp_spin.setDecimals(2)
        self._env_comp_spin.setValue(float(a.get("env_comp", 0.3)))
        self._env_comp_spin.valueChanged.connect(
            lambda v: self._save_param("env_comp", v))
        env_row.addWidget(self._env_comp_spin)
        env_row.addWidget(QLabel("Window:"))
        self._env_window_spin = QDoubleSpinBox()
        self._env_window_spin.setRange(1.0, 16.0)
        self._env_window_spin.setSingleStep(1.0)
        self._env_window_spin.setDecimals(1)
        self._env_window_spin.setValue(float(a.get("env_window", 1.0)))
        self._env_window_spin.valueChanged.connect(
            lambda v: self._save_param("env_window", v))
        env_row.addWidget(self._env_window_spin)
        _env_inner = QWidget(); _env_inner.setLayout(env_row)
        self._env_params_w = _indent(_env_inner)
        card.add_widget(self._env_params_w)

        # --- Onset envelope (no own params) ---
        card.add_widget(self._make_toggle(
            "feat_acoustic_ts_onset_env", "Onset envelope (HWR derivative)",
            f["feat_acoustic_ts_onset_env"]))
        onset_caption = QLabel(
            "Derived from the amplitude envelope — uses its settings above.")
        onset_caption.setObjectName("Subtitle")
        card.add_widget(_indent(onset_caption))

        # --- Log-mel spectrogram + its params ---
        card.add_widget(self._make_toggle(
            "feat_acoustic_ts_logmel", "Log-mel spectrogram",
            f["feat_acoustic_ts_logmel"]))
        mel_row = QHBoxLayout(); mel_row.setContentsMargins(0, 0, 0, 0)
        mel_row.addWidget(QLabel("Mel bands (n_mels):"))
        self._n_mels_spin = QSpinBox()
        self._n_mels_spin.setRange(1, 128)
        self._n_mels_spin.setValue(int(a.get("n_mels", 16)))
        self._n_mels_spin.valueChanged.connect(
            lambda v: self._save_param("n_mels", v))
        mel_row.addWidget(self._n_mels_spin)
        mel_row.addWidget(QLabel("Preset:"))
        self._mel_preset_combo = QComboBox()
        self._mel_preset_combo.addItems(["speech", "music", "general"])
        self._mel_preset_combo.setCurrentText(a.get("mel_preset", "speech"))
        self._mel_preset_combo.currentTextChanged.connect(
            lambda v: self._save_param("mel_preset", v))
        mel_row.addWidget(self._mel_preset_combo)
        _mel_inner = QWidget(); _mel_inner.setLayout(mel_row)
        self._mel_params_w = _indent(_mel_inner)
        card.add_widget(self._mel_params_w)

        # --- MFCC + its params ---
        card.add_widget(self._make_toggle(
            "feat_acoustic_ts_mfcc", "MFCC (off by default)",
            f["feat_acoustic_ts_mfcc"]))
        mfcc_row = QHBoxLayout(); mfcc_row.setContentsMargins(0, 0, 0, 0)
        mfcc_row.addWidget(QLabel("MFCC count (n_mfcc):"))
        self._n_mfcc_spin = QSpinBox()
        self._n_mfcc_spin.setRange(1, 60)
        self._n_mfcc_spin.setValue(int(a.get("n_mfcc", 13)))
        self._n_mfcc_spin.valueChanged.connect(
            lambda v: self._save_param("n_mfcc", v))
        mfcc_row.addWidget(self._n_mfcc_spin)
        mfcc_caption = QLabel("also uses mel bands / preset above")
        mfcc_caption.setObjectName("Subtitle")
        mfcc_row.addWidget(mfcc_caption)
        _mfcc_inner = QWidget(); _mfcc_inner.setLayout(mfcc_row)
        self._mfcc_params_w = _indent(_mfcc_inner)
        card.add_widget(self._mfcc_params_w)

        layout.addWidget(card)
        layout.addStretch(1)

        self._sync_param_enabled()

    def toggle(self, key: str): return self._toggles.get(key)

    def _make_toggle(self, key: str, label: str, initial: bool) -> ToggleRow:
        row = ToggleRow(label=label, info_key=key)
        row.set_checked(initial)
        row.toggled.connect(lambda v, k=key: self._on_toggle(k, v))
        self._toggles[key] = row
        return row

    def _on_toggle(self, key: str, value: bool) -> None:
        save_preferences({"features": {key: value}})
        self._sync_param_enabled()
        self.settings_changed.emit()

    def _sync_param_enabled(self) -> None:
        """Grey each param group out unless an enabled feature consumes it.
        Mirrors extract_acoustic() in extractors/acoustic_ts.py:
        env params -> envelope OR onset; mel params -> logmel OR mfcc;
        n_mfcc -> mfcc."""
        env_on = self._toggles["feat_acoustic_ts_envelope"].is_checked()
        onset_on = self._toggles["feat_acoustic_ts_onset_env"].is_checked()
        logmel_on = self._toggles["feat_acoustic_ts_logmel"].is_checked()
        mfcc_on = self._toggles["feat_acoustic_ts_mfcc"].is_checked()
        self._env_params_w.setEnabled(env_on or onset_on)
        self._mel_params_w.setEnabled(logmel_on or mfcc_on)
        self._mfcc_params_w.setEnabled(mfcc_on)

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
        self._sync_param_enabled()
