"""Word-level features page (batch 3): per-column Structural + Acoustic,
modal category selection, card-wrapped subsections."""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QButtonGroup, QRadioButton, QComboBox, QLineEdit, QPushButton,
    QDoubleSpinBox,
)

from gui.widgets import Card, ClusterCard, ToggleRow
from utils.preferences import load_preferences, save_preferences
from extractors.llm_cloze import LLM_OPTIONS


class WordFeaturesSection(QWidget):
    settings_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._toggles: dict[str, ToggleRow] = {}
        prefs = load_preferences()
        f = prefs["features"]

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 16, 24, 16)
        layout.setSpacing(12)

        # Output toggle ("Word-level CSV") lives on the Run page — single
        # source of truth for what to emit. This page is purely about
        # configuring WHICH columns appear in that CSV.

        # --- Structural (per-column) ---
        struct_card = ClusterCard(
            "Structural",
            "Document position and offsets — from spaCy. "
            "Pick which columns appear in the word-level CSV.",
            master_key="master_structural",
            master_label="Enable all structural columns",
        )
        struct_card.settings_changed.connect(self.settings_changed)
        struct_grid = QGridLayout()
        struct_keys = [
            ("feat_struct_doc_id", "doc_id"),
            ("feat_struct_paragraph_id", "paragraph_id"),
            ("feat_struct_sentence_id", "sentence_id"),
            ("feat_struct_sentence_text", "sentence"),
            ("feat_struct_char_start", "start"),
            ("feat_struct_char_end", "end"),
            ("feat_struct_token_id", "token_id"),
            ("feat_struct_token_in_sentence_id", "token_in_sentence_id"),
        ]
        for i, (key, label) in enumerate(struct_keys):
            struct_grid.addWidget(
                self._make_toggle(key, label, f[key], "features"),
                i // 2, i % 2,
            )
        struct_card.add_child_layout(struct_grid)
        layout.addWidget(struct_card)

        # --- Lexical-linguistic ---
        lex_card = ClusterCard(
            "Lexical–linguistic",
            "Linguistic annotation from spaCy (en_core_web_sm).",
            master_key="master_lexling",
            master_label="Enable all lex-ling columns",
        )
        lex_card.settings_changed.connect(self.settings_changed)
        for key, label in [
            ("lexling_tokens", "Tokenization & lemmatization (token, lemma)"),
            ("lexling_pos",    "Part-of-speech tags (upos, xpos)"),
            ("lexling_morph",  "Morphological features (feats)"),
            ("lexling_dep",    "Dependency parse (dep_rel, head_token_id)"),
        ]:
            lex_card.add_child(self._make_toggle(key, label, f[key], "features"))
        layout.addWidget(lex_card)

        # --- Frequency ---
        freq_card = ClusterCard(
            "Frequency", "Lexical-corpus frequency lookups.",
            master_key="master_frequency",
            master_label="Enable all frequency columns",
        )
        freq_card.settings_changed.connect(self.settings_changed)
        freq_card.add_child(self._make_toggle(
            "freq_subtlex",
            "General-corpus frequency (wordfreq blend, Zipf scale) → Zipf_Freq",
            f["freq_subtlex"], "features"))
        ewfg_label = QLabel("Grade-banded (EWFG):")
        ewfg_label.setObjectName("Subtitle")
        freq_card.add_child(ewfg_label)
        ewfg_grid = QGridLayout()
        ewfg_keys = [
            ("freq_ewfg_k2", "K–2 → Edu_Zipf_K2"),
            ("freq_ewfg_3to5", "3–5 → Edu_Zipf_3to5"),
            ("freq_ewfg_6to8", "6–8 → Edu_Zipf_6to8"),
            ("freq_ewfg_9to12", "9–12 → Edu_Zipf_9to12"),
            ("freq_ewfg_overall", "Overall → Edu_Zipf_Overall"),
        ]
        for i, (key, label) in enumerate(ewfg_keys):
            ewfg_grid.addWidget(
                self._make_toggle(key, label, f[key], "features"),
                i // 2, i % 2,
            )
        freq_card.add_child_layout(ewfg_grid)
        # EWFG grade-band frequencies are unavailable: the source — The
        # Educator's Word Frequency Guide (Zeno et al. 1995) — exists only in
        # print and has not been digitized, so there is no lookup data. Disable
        # the toggles (the extractor code is kept for if/when the data arrives).
        _ewfg_tip = (
            "Unavailable: the Educator's Word Frequency Guide (Zeno et al. 1995) "
            "exists only in print and has not been digitized, so there is no "
            "lookup data. Disabled until the norms are obtained."
        )
        for key, _ in ewfg_keys:
            self._toggles[key].set_permanently_disabled(_ewfg_tip)
        ewfg_note = QLabel(
            "Grade-band (EWFG) frequencies are <b>unavailable</b>: their source "
            "— <i>The Educator's Word Frequency Guide</i> (Zeno et al. 1995) — "
            "exists only in print and has not been digitized. These toggles are "
            "disabled for now and may be added later. "
            '<a href="https://books.google.com/books/about/'
            'The_Educator_s_Word_Frequency_Guide.html?id=jsJ1QgAACAAJ">'
            "Book reference</a>."
        )
        ewfg_note.setObjectName("Subtitle")
        ewfg_note.setWordWrap(True)
        ewfg_note.setOpenExternalLinks(True)
        freq_card.add_widget(ewfg_note)
        layout.addWidget(freq_card)

        # --- Predictability ---
        # Feature toggles are gated by the master; model picker and API key
        # are configuration widgets added via add_widget (non-gated path) so
        # they remain accessible even when the master is off.
        pred_card = ClusterCard(
            "Predictability", "Context-conditional probability features.",
            master_key="master_predictability",
            master_label="Enable all predictability columns",
        )
        pred_card.settings_changed.connect(self.settings_changed)
        surp_label = QLabel("Surprisal — -log₂ P(word | preceding context), in bits.")
        pred_card.add_child(surp_label)
        pred_card.add_child(self._make_toggle(
            "pred_surprisal_gpt2",
            "From GPT-2 → column: Surprisal (legacy baseline)",
            f["pred_surprisal_gpt2"], "features"))
        pred_card.add_child(self._make_toggle(
            "pred_surprisal_llm",
            "From LLM → column: LLM_Surprisal",
            f["pred_surprisal_llm"], "features"))
        cloze_label = QLabel("Cloze features (LLM forward pass)")
        pred_card.add_child(cloze_label)
        pred_card.add_child(self._make_toggle(
            "pred_cloze", "Cloze probability → LLM_Cloze",
            f["pred_cloze"], "features"))
        pred_card.add_child(self._make_toggle(
            "pred_topmatch", "Top-1 match → LLM_TopMatch",
            f["pred_topmatch"], "features"))

        # Model picker — non-gated (config, not output)
        mp = QHBoxLayout()
        mp.addWidget(QLabel("LLM model:"))
        self.llm_model_combo = QComboBox()
        self.llm_model_combo.addItems([lab for _, lab in LLM_OPTIONS])
        saved_model = prefs["predictability"].get("llm_model", "")
        saved_label = next(
            (lab for mid, lab in LLM_OPTIONS if mid == saved_model),
            LLM_OPTIONS[0][1])
        self.llm_model_combo.setCurrentText(saved_label)
        self.llm_model_combo.currentTextChanged.connect(self._on_model_change)
        mp.addWidget(self.llm_model_combo, 1)
        mp_w = QWidget(); mp_w.setLayout(mp)
        pred_card.add_widget(mp_w)

        # Credentials help — text + the same blue ⓘ info-icon used by feature
        # toggles, opening the shared setup doc (OpenAI key / gated Llama
        # license + token) in the in-app feature-info dialog.
        from gui.info_popup import InfoIcon
        cred_help = QHBoxLayout()
        cred_help.setContentsMargins(0, 0, 0, 0)
        cred_help.addWidget(QLabel("How to get an API key / HuggingFace access"))
        self.credentials_help_icon = InfoIcon("llm_credentials", self)
        cred_help.addWidget(self.credentials_help_icon)
        cred_help.addStretch(1)
        cred_help_w = QWidget(); cred_help_w.setLayout(cred_help)
        pred_card.add_widget(cred_help_w)

        # API key row (only for OpenAI) — non-gated
        self.api_key_row = QWidget()
        ak = QHBoxLayout(self.api_key_row); ak.setContentsMargins(0, 0, 0, 0)
        ak.addWidget(QLabel("API key:"))
        self.api_key_line = QLineEdit()
        self.api_key_line.setEchoMode(QLineEdit.Password)
        self.api_key_line.setPlaceholderText("Paste OpenAI API key — never saved to disk")
        self.api_key_line.textChanged.connect(
            lambda txt: self.refresh_warnings(session_api_key=txt))
        ak.addWidget(self.api_key_line, 1)
        show_btn = QPushButton("Show")
        show_btn.setCheckable(True)
        show_btn.toggled.connect(lambda v: self.api_key_line.setEchoMode(
            QLineEdit.Normal if v else QLineEdit.Password))
        ak.addWidget(show_btn)
        pred_card.add_widget(self.api_key_row)

        # HF token row (only for gated local Llama, shown when no token detected)
        self.hf_token_row = QWidget()
        ht = QHBoxLayout(self.hf_token_row); ht.setContentsMargins(0, 0, 0, 0)
        ht.addWidget(QLabel("HF token:"))
        self.hf_token_line = QLineEdit()
        self.hf_token_line.setEchoMode(QLineEdit.Password)
        self.hf_token_line.setPlaceholderText("Paste HuggingFace token — never saved to disk")
        self.hf_token_line.textChanged.connect(
            lambda txt: self.refresh_warnings(
                session_api_key=self.session_api_key(), session_hf_token=txt))
        ht.addWidget(self.hf_token_line, 1)
        hf_show = QPushButton("Show"); hf_show.setCheckable(True)
        hf_show.toggled.connect(lambda v: self.hf_token_line.setEchoMode(
            QLineEdit.Normal if v else QLineEdit.Password))
        ht.addWidget(hf_show)
        pred_card.add_widget(self.hf_token_row)
        self.hf_token_detected_label = QLabel("✓ HuggingFace token detected")
        pred_card.add_widget(self.hf_token_detected_label)
        self._refresh_llm_credential_visibility()
        if not saved_model:
            # api_key_row exists now — safe to call. Persists the displayed
            # default so preflight and the cost panel see a real model ID.
            self._on_model_change(self.llm_model_combo.currentText())
        layout.addWidget(pred_card)

        # --- Semantic indicators ---
        sem_card = Card(
            "Semantic indicators",
            "Per-word category indicators from a dictionary backend (content POS only).")
        backend_row = QHBoxLayout()
        backend_row.addWidget(QLabel("Backend:"))
        self._backend_group = QButtonGroup(self)
        saved_backend = prefs["semantic"].get("backend", "empath")
        if saved_backend == "liwc":
            # LIWC is disabled; never show a disabled radio as the active one.
            saved_backend = "empath"
        _liwc_tooltip = (
            "LIWC is disabled. The LIWC-22 license agreement forbids using its "
            "dictionary outside LIWC's own software, so SensiExtractor cannot "
            "ship it. Empath is the open-source default. See the ⓘ info for how "
            "to license LIWC for your own offline use."
        )
        for value, label in [("auto", "Auto"), ("liwc", "LIWC"), ("empath", "Empath")]:
            rb = QRadioButton(label)
            if value == "liwc":
                rb.setEnabled(False)
                rb.setToolTip(_liwc_tooltip)
            rb.setChecked(value == saved_backend)
            rb.toggled.connect(
                lambda checked, v=value: checked and self._on_backend_change(v))
            self._backend_group.addButton(rb)
            backend_row.addWidget(rb)
        backend_row.addStretch(1)
        br_w = QWidget(); br_w.setLayout(backend_row)
        sem_card.add_widget(br_w)
        liwc_note = QLabel(
            "LIWC backend removed due to its license agreement. Empath (free, "
            "open-source) is the default. Click the ⓘ next to “Per-word category "
            "indicators” for details and where to obtain a LIWC license."
        )
        liwc_note.setObjectName("Subtitle")
        liwc_note.setWordWrap(True)
        sem_card.add_widget(liwc_note)
        sem_card.add_widget(self._make_toggle(
            "sem_indicators", "Per-word category indicators",
            f["sem_indicators"], "features"))

        # Configure-categories button + count label
        cats_row = QHBoxLayout()
        self.cats_count_label = QLabel(self._cats_count_text())
        self.configure_cats_button = QPushButton("Configure word-level categories…")
        self.configure_cats_button.clicked.connect(self._open_category_dialog)
        cats_row.addWidget(self.cats_count_label, 1)
        cats_row.addWidget(self.configure_cats_button)
        cats_w = QWidget(); cats_w.setLayout(cats_row)
        sem_card.add_widget(cats_w)
        layout.addWidget(sem_card)

        # --- Semantic similarity (embeddings) ---
        sim_card = Card(
            "Semantic similarity",
            "Pretrained word embeddings saved as .mat sidecars in word/. "
            "Word2Vec requires a one-time manual download — see utils/download_models.py.",
        )
        sim_card.add_widget(self._make_embed_toggle(
            "master_word2vec", "Word2Vec — Google News 300-d → <stim>_word2vec.mat",
            f.get("master_word2vec", True), "word2vec_mat",
        ))
        sim_card.add_widget(self._make_embed_toggle(
            "master_bert_word",
            "BERT — bert-base-uncased, 12 layers × 768 d → <stim>_bert_word.mat",
            f.get("master_bert_word", True), "bert_word_mat",
        ))
        layout.addWidget(sim_card)

        # --- Lexical norms (per-column; user-supplied data) ---
        norms_card = ClusterCard(
            "Lexical norms",
            "Per-word psycholinguistic norms from user-supplied tables (no data "
            "ships — see the ⓘ for the expected file schemas): age of "
            "acquisition, concreteness, imageability, familiarity, and "
            "valence/arousal/dominance.",
            master_key="master_lexical_norms",
            master_label="Enable lexical norms",
        )
        norms_card.settings_changed.connect(self.settings_changed)
        norms_grid = QGridLayout()
        norms_keys = [
            ("feat_norm_aoa_mean",     "AoA_Mean"),
            ("feat_norm_aoa_sd",       "AoA_SD"),
            ("feat_norm_concreteness", "Concreteness"),
            ("feat_norm_imageability", "Imageability"),
            ("feat_norm_familiarity",  "Familiarity"),
            ("feat_norm_valence",      "Valence"),
            ("feat_norm_arousal",      "Arousal"),
            ("feat_norm_dominance",    "Dominance"),
        ]
        for i, (key, label) in enumerate(norms_keys):
            norms_grid.addWidget(
                self._make_toggle(key, label, f[key], "features"),
                i // 2, i % 2,
            )
        norms_card.add_child_layout(norms_grid)
        layout.addWidget(norms_card)

        # --- Acoustic (per-column) ---
        ac_card = ClusterCard(
            "Acoustic",
            "True acoustic word boundaries from audio (forced alignment, MFA).",
            master_key="master_acoustic",
            master_label="Enable all acoustic columns",
        )
        ac_card.settings_changed.connect(self.settings_changed)
        ac_grid = QGridLayout()
        ac_keys = [
            ("feat_acoustic_onset", "Onset"),
            ("feat_acoustic_end", "End"),
            ("feat_acoustic_word_duration", "Word_Duration"),
            ("feat_acoustic_word_gap", "Word_Gap"),
        ]
        for i, (key, label) in enumerate(ac_keys):
            ac_grid.addWidget(
                self._make_toggle(key, label, f[key], "features"),
                i // 2, i % 2,
            )
        ac_card.add_child_layout(ac_grid)
        layout.addWidget(ac_card)

        # --- Prosody (per-column) ---
        pros_card = ClusterCard(
            "Prosody",
            "Pitch and volume summaries within each MFA word interval. "
            "Requires alignment.",
            master_key="master_word_prosody",
            master_label="Enable word prosody",
        )
        pros_card.settings_changed.connect(self.settings_changed)
        # 2×5 toggle grid (pitch row, volume row)
        pros_grid = QGridLayout()
        pros_keys = [
            ("feat_prosody_pitch_mean",  "Pitch_Mean"),
            ("feat_prosody_pitch_min",   "Pitch_Min"),
            ("feat_prosody_pitch_max",   "Pitch_Max"),
            ("feat_prosody_pitch_std",   "Pitch_Std"),
            ("feat_prosody_pitch_delta", "Pitch_Delta"),
            ("feat_prosody_volume_mean",  "Volume_Mean"),
            ("feat_prosody_volume_min",   "Volume_Min"),
            ("feat_prosody_volume_max",   "Volume_Max"),
            ("feat_prosody_volume_std",   "Volume_Std"),
            ("feat_prosody_volume_delta", "Volume_Delta"),
        ]
        for i, (key, label) in enumerate(pros_keys):
            pros_grid.addWidget(
                self._make_toggle(key, label, f[key], "features"),
                i // 5, i % 5,
            )
        pros_card.add_child_layout(pros_grid)

        # Pitch units combo + floor/ceiling spinboxes — config UI, greys with master
        prosody_prefs = prefs.get("prosody", {})
        cfg_row = QHBoxLayout()
        cfg_row.addWidget(QLabel("Pitch units:"))
        self.pitch_units_combo = QComboBox()
        self.pitch_units_combo.addItems(["Semitones (rel speaker median)", "Hz"])
        if prosody_prefs.get("pitch_units", "semitones") == "hz":
            self.pitch_units_combo.setCurrentText("Hz")
        self.pitch_units_combo.currentTextChanged.connect(self._on_pitch_units_change)
        cfg_row.addWidget(self.pitch_units_combo, 1)
        cfg_row.addWidget(QLabel("Pitch floor (Hz):"))
        self.pitch_floor_spin = QDoubleSpinBox()
        self.pitch_floor_spin.setRange(20.0, 400.0)
        self.pitch_floor_spin.setValue(prosody_prefs.get("pitch_floor_hz", 75.0))
        self.pitch_floor_spin.valueChanged.connect(
            lambda v: save_preferences({"prosody": {"pitch_floor_hz": v}}))
        cfg_row.addWidget(self.pitch_floor_spin)
        cfg_row.addWidget(QLabel("ceiling:"))
        self.pitch_ceiling_spin = QDoubleSpinBox()
        self.pitch_ceiling_spin.setRange(100.0, 1200.0)
        self.pitch_ceiling_spin.setValue(prosody_prefs.get("pitch_ceiling_hz", 500.0))
        self.pitch_ceiling_spin.valueChanged.connect(
            lambda v: save_preferences({"prosody": {"pitch_ceiling_hz": v}}))
        cfg_row.addWidget(self.pitch_ceiling_spin)
        cfg_w = QWidget(); cfg_w.setLayout(cfg_row)
        pros_card.add_child(cfg_w)
        layout.addWidget(pros_card)

        layout.addStretch(1)
        self.refresh_warnings()
        self.settings_changed.connect(self.refresh_warnings)

    def toggle(self, key: str): return self._toggles.get(key)

    def session_api_key(self) -> str:
        return self.api_key_line.text() if self.api_key_row.isVisible() else ""

    def session_hf_token(self) -> str:
        return self.hf_token_line.text() if self.hf_token_row.isVisible() else ""

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

    def _make_embed_toggle(self, feat_key: str, label: str, initial: bool,
                           out_key: str) -> ToggleRow:
        """Embedding master toggles drive both a feature flag and an output flag
        (the same value), so the planner master switch and the .mat sidecar
        emission stay in lockstep."""
        row = ToggleRow(label=label, info_key=feat_key)
        row.set_checked(initial)
        row.toggled.connect(
            lambda v, fk=feat_key, ok=out_key: self._on_embed_toggle(fk, v, ok))
        self._toggles[feat_key] = row
        return row

    def _on_embed_toggle(self, feat_key: str, value: bool, out_key: str) -> None:
        save_preferences({"features": {feat_key: value}, "output": {out_key: value}})
        self.settings_changed.emit()

    def refresh_warnings(self, session_api_key: str = "", session_hf_token: str = "") -> None:
        from gui.preflight import compute_warnings
        warnings = compute_warnings(session_api_key=session_api_key,
                                    session_hf_token=session_hf_token)
        for row in self._toggles.values():
            row.set_warning(None)
        for w in warnings:
            row = self._toggles.get(w.toggle_key)
            if row is not None:
                row.set_warning(w.message)

    def _on_model_change(self, label):
        model_id = next((mid for mid, lab in LLM_OPTIONS if lab == label), label)
        save_preferences({"predictability": {"llm_model": model_id}})
        self._refresh_llm_credential_visibility()
        self.settings_changed.emit()

    def _on_pitch_units_change(self, label: str) -> None:
        value = "hz" if label.lower().startswith("hz") else "semitones"
        save_preferences({"prosody": {"pitch_units": value}})
        self.settings_changed.emit()

    def _refresh_llm_credential_visibility(self):
        from extractors.llm_cloze import is_gated_local_model, has_hf_token
        label = self.llm_model_combo.currentText()
        model_id = next((mid for mid, lab in LLM_OPTIONS if lab == label), label)
        is_openai = model_id.startswith("gpt-")
        gated = is_gated_local_model(model_id)
        token_present = gated and has_hf_token()
        self.api_key_row.setVisible(is_openai)
        self.hf_token_row.setVisible(gated and not token_present)
        self.hf_token_detected_label.setVisible(token_present)

    def _on_backend_change(self, backend):
        save_preferences({"semantic": {"backend": backend}})
        self._update_cats_count_text()
        self.settings_changed.emit()

    def _cats_count_text(self) -> str:
        prefs = load_preferences()
        backend = prefs["semantic"].get("backend", "auto")
        if backend == "auto":
            from extractors.semantic_domain import detect_backend
            backend, _ = detect_backend()
        key = f"categories_{backend}"
        sel = prefs["semantic"].get(key, [])
        return f"{len(sel)} categories selected (shared with document level)."

    def _update_cats_count_text(self):
        self.cats_count_label.setText(self._cats_count_text())

    def _open_category_dialog(self):
        from gui.category_dialog import CategoryDialog
        prefs = load_preferences()
        backend = prefs["semantic"].get("backend", "auto")
        if backend == "auto":
            from extractors.semantic_domain import detect_backend
            backend, _ = detect_backend()
        dlg = CategoryDialog(backend=backend, level="word", parent=self)
        dlg.exec()
        self._update_cats_count_text()
        self.settings_changed.emit()
