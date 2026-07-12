"""Word-level features page (batch 3): per-column Structural + Acoustic,
modal category selection, card-wrapped subsections."""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QButtonGroup, QRadioButton, QComboBox, QLineEdit, QPushButton,
    QDoubleSpinBox, QSpinBox, QCheckBox,
)

from gui.widgets import Card, ClusterCard, ToggleRow
from utils.preferences import load_preferences, save_preferences
from extractors.predictability_models import (
    all_specs, is_gated, spec_for, FLOOR_CONTEXT,
)
# Imported as a module (not `from ... import has_hf_token`) so that tests can
# monkeypatch `extractors.predictability.has_hf_token` and have it take
# effect here — a direct-name import would bind a stale reference at
# module-load time (mirrors gui/preflight.py's `from extractors import
# llm_cloze` pattern).
from extractors import predictability as _predictability_mod


class WordFeaturesSection(QWidget):
    settings_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._toggles: dict[str, ToggleRow] = {}
        self._force_show_token_row = False
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
        # Model + metric selectors are configuration widgets added via
        # add_widget (non-gated path) so they remain accessible even when
        # the master is off — mirrors the old model-picker/credentials rows.
        # There are no more per-column feature toggles here: every
        # <slug>_<Metric> word column is produced dynamically from the
        # models x metrics selection (see extractors/predictability.py and
        # gui/planner/plan.py's _predictability_columns).
        pred_card = ClusterCard(
            "Predictability",
            "Context-conditional probability features. Each selected model "
            "is scored once per stimulus; each selected metric is emitted "
            "as a <slug>_<Metric> column per model (e.g. gpt2_Surprisal).",
            master_key="master_predictability",
            master_label="Enable all predictability columns",
        )
        pred_card.settings_changed.connect(self.settings_changed)

        pred_prefs = prefs["predictability"]
        saved_models = set(pred_prefs.get("models") or [])
        saved_metrics = pred_prefs.get("metrics") or {}

        models_label = QLabel("Models:")
        models_label.setObjectName("Subtitle")
        pred_card.add_widget(models_label)
        saved_context = pred_prefs.get("context") or {}
        from gui.widgets import _ClickToToggleLabel
        self.model_checkboxes: dict[str, QCheckBox] = {}
        self.model_labels: dict[str, QLabel] = {}
        self.context_spins: dict[str, QSpinBox] = {}
        for spec in all_specs():
            row = QHBoxLayout(); row.setContentsMargins(0, 0, 0, 0)
            # Text lives on a wrapping label, not the checkbox (QCheckBox
            # text can't wrap and long labels forced a horizontal scrollbar).
            cb = QCheckBox()
            cb.setChecked(spec.slug in saved_models)
            cb.toggled.connect(lambda _v, s=spec.slug: self._on_model_toggled(s, _v))
            self.model_checkboxes[spec.slug] = cb
            row.addWidget(cb)
            lbl = _ClickToToggleLabel(f"{spec.label} — {spec.note}", cb)
            lbl.setWordWrap(True)
            self.model_labels[spec.slug] = lbl
            row.addWidget(lbl, 1)
            # Per-model context cap (Project B). Range [FLOOR, native]; the
            # value defaults to native (= uncapped, stored as absence).
            # Always editable — greying it until the model is checked read as
            # "broken" (same discoverability lesson as the HF-token box); a
            # value set here simply takes effect whenever the model runs.
            row.addWidget(QLabel("context:"))
            spin = QSpinBox()
            spin.setRange(FLOOR_CONTEXT, spec.native_ctx)
            # Adaptive stepping: a fixed 128 step was invisible against a
            # 131072-token range (one arrow click changed the value by 0.1%,
            # which read as "the box does nothing"). Adaptive steps scale
            # with the current value (~10%), so clicks move it meaningfully
            # at any magnitude; exact values can still be typed.
            spin.setStepType(QSpinBox.StepType.AdaptiveDecimalStepType)
            spin.setValue(int(saved_context.get(spec.slug, spec.native_ctx)))
            spin.setSuffix(f" / {spec.native_ctx} tok")
            spin.setToolTip(
                "Max context (subwords) fed to this model per forward pass. "
                "Lower = faster / less memory on long stimuli, but each word "
                "sees less preceding context. Default is the model's full "
                f"window ({spec.native_ctx}). Takes effect when this model "
                "is selected.")
            spin.valueChanged.connect(
                lambda v, s=spec.slug: self._on_context_changed(s, v))
            self.context_spins[spec.slug] = spin
            row.addWidget(spin)
            row_w = QWidget(); row_w.setLayout(row)
            pred_card.add_widget(row_w)

        metrics_label = QLabel("Metrics:")
        metrics_label.setObjectName("Subtitle")
        pred_card.add_widget(metrics_label)
        from gui.info_popup import InfoIcon
        self.metric_checkboxes: dict[str, QCheckBox] = {}
        self.metric_labels: dict[str, QLabel] = {}
        self.metric_info_icons: dict[str, InfoIcon] = {}
        for key, doc_key, label in [
            ("surprisal", "pred_surprisal",
             "Surprisal — -log₂ P(word | preceding context), in bits"),
            ("cloze", "pred_cloze", "Cloze — P(word | preceding context)"),
            ("topmatch", "pred_topmatch",
             "Top-match — 1 if the model's top predicted next "
             "token equals the actual word"),
        ]:
            mrow = QHBoxLayout(); mrow.setContentsMargins(0, 0, 0, 0)
            cb = QCheckBox()
            cb.setChecked(bool(saved_metrics.get(key, False)))
            cb.toggled.connect(lambda _v, k=key: self._on_metric_toggled(k, _v))
            self.metric_checkboxes[key] = cb
            mrow.addWidget(cb)
            lbl = _ClickToToggleLabel(label, cb)
            lbl.setWordWrap(True)
            self.metric_labels[key] = lbl
            mrow.addWidget(lbl, 1)
            # Same blue ⓘ the old per-column ToggleRows auto-rendered — the
            # per-metric docs (theory, EEG rationale, citations) must stay
            # reachable from the GUI.
            icon = InfoIcon(doc_key, self)
            self.metric_info_icons[key] = icon
            mrow.addWidget(icon)
            mrow_w = QWidget(); mrow_w.setLayout(mrow)
            pred_card.add_widget(mrow_w)

        # Credentials help — text + the same blue ⓘ info-icon used by feature
        # toggles, opening the shared setup doc (gated Llama license + HF
        # token) in the in-app feature-info dialog. OpenAI is no longer
        # offered (see extractors/predictability.py), so this is HF-only.
        from gui.info_popup import InfoIcon
        cred_help = QHBoxLayout()
        cred_help.setContentsMargins(0, 0, 0, 0)
        cred_help.addWidget(QLabel("How to get HuggingFace access"))
        self.credentials_help_icon = InfoIcon("llm_credentials", self)
        cred_help.addWidget(self.credentials_help_icon)
        cred_help.addStretch(1)
        cred_help_w = QWidget(); cred_help_w.setLayout(cred_help)
        pred_card.add_widget(cred_help_w)

        # HF token row. Shown whenever no token is already detected (from the
        # environment or a prior `hf auth login`), regardless of which models
        # are checked — so there is always a discoverable place to paste it.
        # The gated Llama models can't run without it; GPT-2 ignores it.
        hf_hint = QLabel("HuggingFace token — required for the gated Llama "
                         "models (GPT-2 is free and needs none):")
        hf_hint.setObjectName("Subtitle")
        hf_hint.setWordWrap(True)
        self.hf_token_hint = hf_hint
        pred_card.add_widget(hf_hint)
        self.hf_token_row = QWidget()
        ht = QHBoxLayout(self.hf_token_row); ht.setContentsMargins(0, 0, 0, 0)
        ht.addWidget(QLabel("HF token:"))
        self.hf_token_line = QLineEdit()
        self.hf_token_line.setEchoMode(QLineEdit.Password)
        self.hf_token_line.setPlaceholderText(
            "Paste HuggingFace token — Save stores it in config/hf_token.txt")
        self.hf_token_line.textChanged.connect(
            lambda txt: self.refresh_warnings(session_hf_token=txt))
        ht.addWidget(self.hf_token_line, 1)
        hf_show = QPushButton("Show"); hf_show.setCheckable(True)
        hf_show.toggled.connect(lambda v: self.hf_token_line.setEchoMode(
            QLineEdit.Normal if v else QLineEdit.Password))
        ht.addWidget(hf_show)
        self.hf_token_save_btn = QPushButton("Save")
        self.hf_token_save_btn.setToolTip(
            "Store the token in config/hf_token.txt (gitignored) so future "
            "runs — GUI or headless — find it automatically.")
        self.hf_token_save_btn.clicked.connect(self._save_hf_token)
        ht.addWidget(self.hf_token_save_btn)
        pred_card.add_widget(self.hf_token_row)
        self.hf_token_detected_row = QWidget()
        hd = QHBoxLayout(self.hf_token_detected_row); hd.setContentsMargins(0, 0, 0, 0)
        self.hf_token_detected_label = QLabel("✓ HuggingFace token detected")
        hd.addWidget(self.hf_token_detected_label)
        self.hf_token_replace_btn = QPushButton("Replace…")
        self.hf_token_replace_btn.setToolTip(
            "A saved token can go stale (mistyped or revoked) — this reveals "
            "the input again so you can paste and save a new one.")
        self.hf_token_replace_btn.clicked.connect(self._on_hf_token_replace_clicked)
        hd.addWidget(self.hf_token_replace_btn)
        hd.addStretch(1)
        pred_card.add_widget(self.hf_token_detected_row)
        self._refresh_llm_credential_visibility()
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
        self.cats_count_label.setWordWrap(True)
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
            # 5 rows x 2 columns (pitch column, volume column): a 5-column
            # grid forced the page wider than the window (horizontal
            # scrollbar); vertical growth is fine.
            pros_grid.addWidget(
                self._make_toggle(key, label, f[key], "features"),
                i % 5, i // 5,
            )
        pros_card.add_child_layout(pros_grid)

        # Pitch config — two rows (units; floor/ceiling): a single long row
        # forced the page wider than the window (horizontal scrollbar).
        prosody_prefs = prefs.get("prosody", {})
        units_row = QHBoxLayout()
        units_row.addWidget(QLabel("Pitch units:"))
        self.pitch_units_combo = QComboBox()
        self.pitch_units_combo.addItems(["Semitones (rel speaker median)", "Hz"])
        if prosody_prefs.get("pitch_units", "semitones") == "hz":
            self.pitch_units_combo.setCurrentText("Hz")
        self.pitch_units_combo.currentTextChanged.connect(self._on_pitch_units_change)
        units_row.addWidget(self.pitch_units_combo, 1)
        units_w = QWidget(); units_w.setLayout(units_row)
        pros_card.add_child(units_w)
        cfg_row = QHBoxLayout()
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
        cfg_row.addStretch(1)
        cfg_w = QWidget(); cfg_w.setLayout(cfg_row)
        pros_card.add_child(cfg_w)
        layout.addWidget(pros_card)

        layout.addStretch(1)
        self.refresh_warnings()
        self.settings_changed.connect(self.refresh_warnings)

    def toggle(self, key: str): return self._toggles.get(key)

    def session_api_key(self) -> str:
        """Inert no-op — the OpenAI backend was removed (local HuggingFace
        models only; see extractors/predictability.py). Kept so
        gui/main_window.py's existing `get_session_api_key=self.word_features
        .session_api_key` wiring stays valid without a main_window.py edit."""
        return ""

    def session_hf_token(self) -> str:
        return self.hf_token_line.text() if self.hf_token_row.isVisible() else ""

    def _save_hf_token(self) -> None:
        token = self.hf_token_line.text().strip()
        if not token:
            return
        from utils.credentials import save_hf_token
        save_hf_token(token)
        self.hf_token_line.clear()
        self._force_show_token_row = False
        self._refresh_llm_credential_visibility()
        self.refresh_warnings()

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

    def _on_model_toggled(self, slug: str, checked: bool) -> None:
        selected = [s for s, cb in self.model_checkboxes.items() if cb.isChecked()]
        save_preferences({"predictability": {"models": selected}})
        self._refresh_llm_credential_visibility()
        self.settings_changed.emit()

    def _on_context_changed(self, slug: str, value: int) -> None:
        # Store the cap only when it's below the model's native window; a value
        # at (or above) native means "uncapped", represented as the slug's
        # ABSENCE from the context map (so the default stays native).
        context = dict(load_preferences()["predictability"].get("context") or {})
        native = spec_for(slug).native_ctx
        if int(value) >= native:
            context.pop(slug, None)
        else:
            context[slug] = int(value)
        save_preferences({"predictability": {"context": context}})
        self.settings_changed.emit()

    def _on_metric_toggled(self, _key: str, _checked: bool) -> None:
        metrics = {k: cb.isChecked() for k, cb in self.metric_checkboxes.items()}
        save_preferences({"predictability": {"metrics": metrics}})
        self.settings_changed.emit()

    def _on_pitch_units_change(self, label: str) -> None:
        value = "hz" if label.lower().startswith("hz") else "semitones"
        save_preferences({"prosody": {"pitch_units": value}})
        self.settings_changed.emit()

    def _on_hf_token_replace_clicked(self) -> None:
        self._force_show_token_row = True
        self._refresh_llm_credential_visibility()

    def _refresh_llm_credential_visibility(self):
        # A token can come from the env (HF_TOKEN / HUGGING_FACE_HUB_TOKEN), a
        # prior `hf auth login`, or the token file saved by this batch's Save
        # button (config/hf_token.txt). When one is present, show the
        # "detected" confirmation with a Replace… button; otherwise (or after
        # Replace… is clicked) show the hint + input so the user can paste a
        # new one — independent of which models are checked, so there is
        # always a discoverable place to paste it.
        token_present = _predictability_mod.has_hf_token()
        show_row = (not token_present) or self._force_show_token_row
        self.hf_token_hint.setVisible(show_row)
        self.hf_token_row.setVisible(show_row)
        self.hf_token_detected_row.setVisible(token_present)

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
