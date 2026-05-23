"""Embeddings page — Word2Vec and BERT master toggles."""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel

from gui.widgets import Card, ToggleRow
from utils.preferences import load_preferences, save_preferences


class EmbeddingsSection(QWidget):
    settings_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._toggles: list[tuple[ToggleRow, str, str]] = []
        prefs = load_preferences()
        f = prefs["features"]

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 16, 24, 16)
        layout.setSpacing(12)

        note = QLabel(
            "Embeddings are saved as .mat files alongside the word-level CSV. "
            "Word2Vec requires a one-time manual download — see download_models.py."
        )
        note.setObjectName("Subtitle")
        note.setWordWrap(True)
        layout.addWidget(note)

        w2v_card = Card(
            "Word2Vec",
            "Google News 300-d vectors — produces <stimulus>_word2vec.mat in word/",
        )
        w2v_card.add_widget(
            self._make_toggle(
                "master_word2vec", "Extract Word2Vec embeddings",
                f.get("master_word2vec", True), "word2vec_mat",
            )
        )
        layout.addWidget(w2v_card)

        bert_card = Card(
            "BERT",
            "bert-base-uncased, all 12 transformer layers, sliding-window — "
            "produces <stimulus>_bert_word.mat and <stimulus>_bert_sentence.mat",
        )
        bert_card.add_widget(
            self._make_toggle(
                "master_bert_word", "Extract BERT word embeddings",
                f.get("master_bert_word", True), "bert_word_mat",
            )
        )
        bert_card.add_widget(
            self._make_toggle(
                "master_bert_sentence", "Extract BERT sentence embeddings",
                f.get("master_bert_sentence", True), "bert_sentence_mat",
            )
        )
        layout.addWidget(bert_card)

        layout.addStretch()

    def _make_toggle(self, feat_key: str, label: str, initial: bool, out_key: str) -> ToggleRow:
        row = ToggleRow(label=label)
        row.set_checked(initial)
        row.toggled.connect(lambda v, fk=feat_key, ok=out_key: self._on_toggle(fk, v, ok))
        self._toggles.append((row, feat_key, out_key))
        return row

    def _on_toggle(self, feat_key: str, value: bool, out_key: str) -> None:
        save_preferences({"features": {feat_key: value}, "output": {out_key: value}})
        self.settings_changed.emit()

    def refresh_from_prefs(self) -> None:
        prefs = load_preferences()
        f = prefs["features"]
        for row, feat_key, _ in self._toggles:
            row.blockSignals(True)
            row.set_checked(f.get(feat_key, True))
            row.blockSignals(False)
