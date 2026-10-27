# Acoustic Page Redesign + Lazy Extractor Imports — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the acoustic features GUI page show each feature's parameters directly beneath it (with shared params on top and contextual greying), and stop the GUI from importing torch/transformers/spaCy at startup.

**Architecture:** Two independent changes. (B) `gui/planner/registry.py` keeps its `Stage` metadata but binds each `run` callable to a lazy wrapper that imports its extractor on first call instead of at module load — planning reads only metadata, so the heavy ML stack now loads only when a stage actually runs. (A) `gui/section_acoustic_features.py` is rewritten from a two-card "toggles up top / params below" layout into a single card: a shared analysis-rate grid, then each feature toggle followed by its own indented params, greyed when no enabled feature consumes them.

**Tech Stack:** PySide6, pytest / pytest-qt, importlib.

**Testing note:** Per repo TDD scope (`memory/feedback-tdd-scope.md`), extractor *math* is TDD; GUI and refactors are **test-after**. Both tasks here are GUI/refactor, so each implements first, verifies by running, then adds a guardrail test.

---

### Task 1: Lazy extractor imports in the planner registry

Removes the startup tax: `import gui.planner` currently takes ~51 s because `registry.py` eagerly imports all 18 extractors (torch, transformers, spaCy). The `Stage` definitions below reference the extractor callables by local names (`_spacy_parse`, `add_surprisal`, …); we keep those exact names but bind them to lazy wrappers, so **no Stage definition changes**.

**Files:**
- Modify: `gui/planner/registry.py:22-41` (replace the comment + 18 `from extractors.* import …` lines)
- Test: `tests/gui/test_planner_lazy_imports.py` (create)

- [ ] **Step 1: Replace the eager import block with lazy bindings**

In `gui/planner/registry.py`, replace lines 22–41 (the `# Extractor function imports …` comment through the `from extractors.acoustic_ts import …` line) with:

```python
# Extractor callables are bound lazily: planning reads only Stage metadata
# (name/consumes/produces), so importing this module must NOT drag in the
# heavy ML stack (torch/transformers/spaCy). Each name below imports its
# extractor on first *call* (i.e. when a stage actually runs), caching it.
from importlib import import_module


def _lazy(module_path: str, attr: str):
    """Return a callable that imports `module_path`.`attr` on first call and
    caches it, then delegates. Keeps GUI startup free of torch/spaCy/etc."""
    cache: dict = {}

    def call(*args, **kwargs):
        fn = cache.get("fn")
        if fn is None:
            fn = getattr(import_module(module_path), attr)
            cache["fn"] = fn
        return fn(*args, **kwargs)

    return call


_spacy_parse           = _lazy("extractors.text_parser", "parse")
_mfa_align             = _lazy("extractors.aligner", "align_with_phones")
_praat_extract         = _lazy("extractors.audio_features", "extract")
add_word_freq          = _lazy("extractors.word_freq", "add_word_freq")
add_edu_freq           = _lazy("extractors.educator_freq", "add_edu_freq")
add_surprisal          = _lazy("extractors.surprisal", "add_surprisal")
add_llm_cloze          = _lazy("extractors.llm_cloze", "add_llm_cloze")
add_semantic_domain    = _lazy("extractors.semantic_domain", "add_semantic_domain")
_add_lexical_norms     = _lazy("extractors.lexical_norms", "add_lexical_norms")
add_word_prosody       = _lazy("extractors.prosody", "add_word_prosody")
add_timing_columns     = _lazy("extractors.timing", "add_timing_columns")
syllabify_pairs        = _lazy("extractors.syllabify", "syllabify_pairs")
add_syllable_prosody   = _lazy("extractors.syllable_prosody", "add_syllable_prosody")
_add_pcenter           = _lazy("extractors.pcenter", "add_pcenter")
_add_sentence_prosody  = _lazy("extractors.sentence_prosody", "add_sentence_prosody")
_embed_word2vec        = _lazy("extractors.word2vec_embed", "embed_word2vec")
_embed_bert            = _lazy("extractors.bert_embed", "embed_bert_words_and_sentences")
_extract_acoustic      = _lazy("extractors.acoustic_ts", "extract_acoustic")
```

Leave everything from `STAGES: list[Stage] = [` onward unchanged — every `run=` reference resolves to these same names.

- [ ] **Step 2: Grep for any code relying on `Stage.run` identity**

Run:
```bash
grep -rn "\.run is \|run is add_\|run is _" tests/ gui/ extractors/
grep -rn "from gui.planner.registry import" tests/ gui/ main.py
```
Expected: no test asserts `stage.run is <extractor_fn>`. If any do, change them to call through (`stage.run(...)`) or compare on `stage.name`. Note findings; do not skip this step.

- [ ] **Step 3: Verify the GUI-relevant import paths still resolve and planning works**

Run:
```bash
conda activate ./env
python -c "import gui.planner; from gui.planner import plan_from_prefs; print('ok', len(__import__('gui.planner.registry', fromlist=['STAGES']).STAGES), 'stages')"
```
Expected: prints `ok <N> stages` with no error (N is the stage count, currently ~20).

- [ ] **Step 4: Measure the startup win**

Run:
```bash
python -c "import time,sys; t=time.time(); import gui.planner; dt=time.time()-t; print('import gui.planner: %.2fs' % dt); print('heavy loaded:', [m for m in ('torch','transformers','spacy') if m in sys.modules])"
```
Expected: import time drops from ~51 s to well under ~2 s, and `heavy loaded: []`.

- [ ] **Step 5: Write the regression test**

Create `tests/gui/test_planner_lazy_imports.py`:

```python
"""Guardrail: importing the planner must not pull in the heavy ML stack.

Regression test for the ~51 s GUI startup caused by registry.py eagerly
importing torch/transformers/spaCy. Planning needs only Stage metadata; the
extractor callables import lazily on first run.
"""
import importlib
import sys


def test_importing_planner_does_not_load_heavy_ml():
    # Start from a clean slate for the modules we care about.
    for mod in ("torch", "transformers", "spacy"):
        sys.modules.pop(mod, None)
    for name in [m for m in sys.modules if m.startswith("gui.planner")]:
        del sys.modules[name]

    importlib.import_module("gui.planner")

    for mod in ("torch", "transformers", "spacy"):
        assert mod not in sys.modules, f"{mod} was imported at planner import time"


def test_stages_metadata_available_without_running():
    from gui.planner.registry import STAGES
    assert len(STAGES) > 0
    # Every stage exposes its metadata and a callable run, but calling it is
    # what would trigger the import — we never call it here.
    for stage in STAGES:
        assert stage.name
        assert callable(stage.run)
```

- [ ] **Step 6: Run the test**

Run: `pytest tests/gui/test_planner_lazy_imports.py -v`
Expected: PASS (2 passed).

- [ ] **Step 7: Commit**

```bash
git add gui/planner/registry.py tests/gui/test_planner_lazy_imports.py
git commit -m "perf(planner): import extractors lazily so GUI startup skips torch/spaCy"
```

---

### Task 2: Acoustic page indented-list layout with contextual greying

Rewrites `gui/section_acoustic_features.py` into one card: shared analysis rate on top, then each feature toggle followed by its own indented params. Preserves every preference key and the public attributes existing tests use (`_toggles`, `_rate_spin`, `_env_method_combo`, `_env_comp_spin`, `_env_window_spin`, `_n_mels_spin`, `_n_mfcc_spin`, `_mel_preset_combo`, `toggle()`, `refresh_from_prefs()`).

**Files:**
- Modify: `gui/section_acoustic_features.py` (full rewrite)
- Test: `tests/gui/test_section_acoustic_features.py` (append one test)

- [ ] **Step 1: Replace the file with the new single-card layout**

Overwrite `gui/section_acoustic_features.py` with:

```python
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
```

- [ ] **Step 2: Run the existing section tests (must still pass unchanged)**

Run: `pytest tests/gui/test_section_acoustic_features.py -v`
Expected: the two existing tests (`test_section_constructs_with_four_toggles`, `test_rate_change_persists`) PASS — proves the preserved keys/attributes (`_toggles`, `_rate_spin`) and persistence still work.

- [ ] **Step 3: Append the greying test**

Add to `tests/gui/test_section_acoustic_features.py`:

```python
def test_param_greying_follows_feature_toggles(qtbot, tmp_path, monkeypatch):
    monkeypatch.setenv("SENSI_PREFS_PATH", str(tmp_path / "preferences.json"))
    sec = AcousticFeaturesSection()
    qtbot.addWidget(sec)

    # Envelope + onset off -> env params disabled.
    sec._toggles["feat_acoustic_ts_envelope"].set_checked(False)
    sec._toggles["feat_acoustic_ts_onset_env"].set_checked(False)
    assert not sec._env_params_w.isEnabled()

    # Onset back on -> env params re-enabled (onset consumes them).
    sec._toggles["feat_acoustic_ts_onset_env"].set_checked(True)
    assert sec._env_params_w.isEnabled()

    # Log-mel on, MFCC off -> mel params enabled, n_mfcc disabled.
    sec._toggles["feat_acoustic_ts_logmel"].set_checked(True)
    sec._toggles["feat_acoustic_ts_mfcc"].set_checked(False)
    assert sec._mel_params_w.isEnabled()
    assert not sec._mfcc_params_w.isEnabled()

    # Both log-mel and MFCC off -> mel params disabled.
    sec._toggles["feat_acoustic_ts_logmel"].set_checked(False)
    assert not sec._mel_params_w.isEnabled()
```

- [ ] **Step 4: Run the greying test**

Run: `pytest tests/gui/test_section_acoustic_features.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Launch the real GUI and eyeball the acoustic page**

Run: `python gui.py`
Expected (per `memory/project-mkl-threading-entrypoints.md`, tests passing ≠ feature runs): GUI starts quickly (Task 1 win), and the Acoustic page shows the analysis-rate grid on top, then each feature with its params indented below; toggling Log-mel/MFCC/envelope greys the matching params live. Close the window when satisfied.

- [ ] **Step 6: Commit**

```bash
git add gui/section_acoustic_features.py tests/gui/test_section_acoustic_features.py
git commit -m "feat(gui): acoustic page lists each feature with its params; grey unused params"
```

---

### Task 3: Full suite + master_plan note

- [ ] **Step 1: Run the full test suite**

Run: `pytest -q`
Expected: all pass (baseline per `memory/project-num2words-env-gap.md` is 490 passed, 2 skipped; this adds 3 tests → ~493 passed, 2 skipped). If a collection error mentions a missing package, the `./env` is out of sync — run `pip install -r requirements.txt` and re-run.

- [ ] **Step 2: Update master_plan acoustic row**

In `docs/master_plan.md`, find the acoustic-batch row/entry and add a brief note that the acoustic page now uses the per-feature indented layout and that planner imports are lazy (startup no longer loads torch/spaCy). Keep it to one or two sentences consistent with the surrounding table style.

- [ ] **Step 3: Commit**

```bash
git add docs/master_plan.md
git commit -m "docs(master_plan): note acoustic page relayout + lazy planner imports"
```

---

## Self-Review

- **Spec coverage:** Part A (indented list, shared analysis grid, contextual greying per the dependency table) → Task 2. Part B (lazy `_lazy` wrappers, remove eager imports, identity-risk grep) → Task 1. Testing (greying test; planner-no-heavy-import regression test) → Tasks 2 & 1. "No behavior change / preserved keys" → Task 2 keeps keys and is verified by the unchanged existing tests passing (Step 2). Out-of-scope items (Prosody card, extractor math) are untouched. Covered.
- **Placeholder scan:** none — all steps carry exact code/commands.
- **Type/name consistency:** `_env_params_w` / `_mel_params_w` / `_mfcc_params_w`, `_sync_param_enabled`, `_indent`, and preserved attrs (`_rate_spin`, `_env_method_combo`, `_env_comp_spin`, `_env_window_spin`, `_n_mels_spin`, `_n_mfcc_spin`, `_mel_preset_combo`) are used identically across the rewrite, the greying test, and `_sync_param_enabled`. Lazy local names in Task 1 (`_spacy_parse`, `add_surprisal`, `_add_lexical_norms`, …) match the existing `run=` references in the untouched `STAGES`.
```
