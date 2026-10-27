# GUI HuggingFace-token support for local Llama — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a user running the GUI locally supply (or auto-detect) a HuggingFace token so the gated local Llama LLM-cloze backend works, with clear guidance when something is missing.

**Architecture:** Mirror the existing OpenAI-API-key pattern end to end — a session-only password field shown by model selection, injected as an env var for the run, surfaced by preflight. Add token detection + an actionable backend load error. Sherlock/HPC is out of scope.

**Tech Stack:** Python 3.10, PySide6, pytest + pytest-qt, huggingface_hub, transformers.

## Global Constraints

- Token is **session-only** — never passed to `save_preferences`, never written to disk.
- Gated-model rule is centralized: `model_id.startswith("meta-llama/")`.
- Preflight warnings must **never instruct the user to enable a toggle** (invariant enforced by `tests/gui/test_preflight_invariants.py`).
- Follow the OpenAI-key precedent verbatim; do not invent new UI idioms.
- Run tests with the project env: `./env/python.exe -m pytest ...`.

Spec: `docs/superpowers/specs/2026-07-07-gui-hf-token-llama-design.md`.

---

### Task 1: Token-detection helpers

**Files:**
- Modify: `extractors/llm_cloze.py` (add two module-level functions near `select_model`)
- Test: `tests/extractors/test_llm_cloze.py`

**Interfaces:**
- Produces: `is_gated_local_model(model_id: str) -> bool`, `has_hf_token() -> bool`

- [ ] **Step 1: Write the failing tests**

Add to `tests/extractors/test_llm_cloze.py`:
```python
def test_is_gated_local_model():
    from extractors.llm_cloze import is_gated_local_model
    assert is_gated_local_model('meta-llama/Meta-Llama-3-8B') is True
    assert is_gated_local_model('meta-llama/Llama-3.2-3B') is True
    assert is_gated_local_model('gpt-4o-mini') is False


def test_has_hf_token_true_when_get_token_returns(monkeypatch):
    import extractors.llm_cloze as m
    monkeypatch.setattr('huggingface_hub.get_token', lambda: 'hf_abc', raising=False)
    assert m.has_hf_token() is True


def test_has_hf_token_false_when_no_token(monkeypatch):
    import extractors.llm_cloze as m
    monkeypatch.setattr('huggingface_hub.get_token', lambda: None, raising=False)
    assert m.has_hf_token() is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./env/python.exe -m pytest tests/extractors/test_llm_cloze.py -k "gated or hf_token" -v`
Expected: FAIL with `ImportError: cannot import name 'is_gated_local_model'`.

- [ ] **Step 3: Implement the helpers**

In `extractors/llm_cloze.py`, immediately after the `select_model()` function, add:
```python
def is_gated_local_model(model_id: str) -> bool:
    """True for gated local HF models (the meta-llama/* family) that need a token."""
    return model_id.startswith("meta-llama/")


def has_hf_token() -> bool:
    """True if a HuggingFace token is already available — HF_TOKEN /
    HUGGING_FACE_HUB_TOKEN env, or a token saved by a prior `hf auth login`.
    Lazy import so importing this module (and starting the GUI) stays cheap."""
    try:
        from huggingface_hub import get_token
    except Exception:
        return False
    return bool(get_token())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./env/python.exe -m pytest tests/extractors/test_llm_cloze.py -k "gated or hf_token" -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add extractors/llm_cloze.py tests/extractors/test_llm_cloze.py
git commit -m "feat(llm-cloze): add is_gated_local_model + has_hf_token helpers"
```

---

### Task 2: Actionable backend load error

**Files:**
- Modify: `extractors/llm_cloze.py:151-167` (`_HuggingFaceBackend.__init__`)
- Test: `tests/extractors/test_llm_cloze.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `_HuggingFaceBackend.__init__` raises `RuntimeError` (with license/token/download guidance) on gated/not-found/offline load failure; still silently falls back to fp16 when only bitsandbytes is unavailable.

- [ ] **Step 1: Write the failing tests**

Add to `tests/extractors/test_llm_cloze.py`:
```python
import types
import pytest


def test_backend_raises_actionable_error_on_gated(monkeypatch):
    import extractors.llm_cloze as m

    def raise_gated(*a, **k):
        raise OSError("401 Client Error: gated repo")

    monkeypatch.setattr('transformers.AutoTokenizer.from_pretrained', raise_gated)
    with pytest.raises(RuntimeError) as ei:
        m._HuggingFaceBackend('meta-llama/Meta-Llama-3-8B')
    msg = str(ei.value)
    assert 'license' in msg and 'token' in msg
    assert 'meta-llama/Meta-Llama-3-8B' in msg


def test_backend_falls_back_to_fp16_on_bnb_error(monkeypatch):
    import extractors.llm_cloze as m
    calls = []

    class DummyModel:
        config = types.SimpleNamespace(max_position_embeddings=1024)
        def eval(self): pass

    def fake_model(model_id, **kwargs):
        calls.append(kwargs)
        if 'quantization_config' in kwargs:
            raise RuntimeError("bitsandbytes not available on this platform")
        return DummyModel()

    monkeypatch.setattr('transformers.AutoTokenizer.from_pretrained', lambda *a, **k: object())
    monkeypatch.setattr('transformers.AutoModelForCausalLM.from_pretrained', fake_model)
    monkeypatch.setattr('transformers.BitsAndBytesConfig', lambda **k: object())

    backend = m._HuggingFaceBackend('meta-llama/Meta-Llama-3-8B')
    assert any('quantization_config' in c for c in calls)      # tried quantized
    assert any('quantization_config' not in c for c in calls)  # fell back to fp16
    assert backend._context_limit == 1024
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./env/python.exe -m pytest tests/extractors/test_llm_cloze.py -k "backend_raises or fp16" -v`
Expected: FAIL — currently the gated error propagates as `OSError` (not `RuntimeError`), and the bare `except` masks intent.

- [ ] **Step 3: Rewrite `_HuggingFaceBackend.__init__`**

Replace the body of `_HuggingFaceBackend.__init__` (currently `extractors/llm_cloze.py:151-167`) with:
```python
    def __init__(self, model_id: str):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

        self._torch = torch
        self._model_id = model_id
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(model_id)
            try:
                quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
                self._model = AutoModelForCausalLM.from_pretrained(
                    model_id, quantization_config=quant, device_map='auto',
                )
            except Exception:
                # bitsandbytes unavailable/unsupported on this platform — retry
                # without 4-bit quantization (fp16/cpu). If the *repo* is the
                # problem (gated / not found / offline), this retry raises too and
                # is caught by the outer handler below with actionable guidance.
                self._model = AutoModelForCausalLM.from_pretrained(model_id)
        except Exception as exc:
            raise RuntimeError(_model_load_hint(model_id, exc)) from exc
        self._model.eval()
        self._context_limit = getattr(self._model.config, 'max_position_embeddings', None)
```

Add this module-level helper just above the `class _HuggingFaceBackend` line:
```python
def _model_load_hint(model_id: str, exc: Exception) -> str:
    return (
        f"Could not load local LLM {model_id!r}: {type(exc).__name__}: {exc}\n"
        f"To fix: (1) accept the license at https://huggingface.co/{model_id} ; "
        f"(2) provide a HuggingFace token (GUI token box, HF_TOKEN env var, or "
        f"`hf auth login`); (3) on an offline machine, pre-download the weights "
        f"first. Or select an OpenAI model (set OPENAI_API_KEY) — no gating or download."
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./env/python.exe -m pytest tests/extractors/test_llm_cloze.py -v`
Expected: all pass (the two new tests plus the existing 15).

- [ ] **Step 5: Commit**

```bash
git add extractors/llm_cloze.py tests/extractors/test_llm_cloze.py
git commit -m "feat(llm-cloze): actionable error on gated/offline model load; scope bnb fallback"
```

---

### Task 3: GUI HF-token box + visibility + getter

**Files:**
- Modify: `gui/section_word_features.py` (add rows after the `api_key_row` block ~:192; rename `_refresh_api_key_visibility`; add `session_hf_token`; extend `refresh_warnings`)
- Test: `tests/gui/test_section_word_features.py`

**Interfaces:**
- Consumes: `is_gated_local_model`, `has_hf_token` (Task 1).
- Produces: `WordFeaturesSection.session_hf_token() -> str`; renamed method `_refresh_llm_credential_visibility(self)`; widgets `hf_token_row`, `hf_token_line`, `hf_token_detected_label`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/gui/test_section_word_features.py`:
```python
def _label_for(model_id):
    from extractors.llm_cloze import LLM_OPTIONS
    return next(lab for mid, lab in LLM_OPTIONS if mid == model_id)


def test_openai_model_shows_api_key_hides_hf(qtbot, monkeypatch):
    from gui.section_word_features import WordFeaturesSection
    w = WordFeaturesSection(); qtbot.addWidget(w)
    w.llm_model_combo.setCurrentText(_label_for('gpt-4o-mini'))
    assert w.api_key_row.isVisible() is True
    assert w.hf_token_row.isVisible() is False


def test_gated_model_no_token_shows_hf_box(qtbot, monkeypatch):
    import extractors.llm_cloze as m
    monkeypatch.setattr(m, 'has_hf_token', lambda: False)
    from gui.section_word_features import WordFeaturesSection
    w = WordFeaturesSection(); qtbot.addWidget(w)
    w.llm_model_combo.setCurrentText(_label_for('meta-llama/Meta-Llama-3-8B'))
    assert w.hf_token_row.isVisible() is True
    assert w.hf_token_detected_label.isVisible() is False
    assert w.api_key_row.isVisible() is False


def test_gated_model_with_token_shows_detected_label(qtbot, monkeypatch):
    import extractors.llm_cloze as m
    monkeypatch.setattr(m, 'has_hf_token', lambda: True)
    from gui.section_word_features import WordFeaturesSection
    w = WordFeaturesSection(); qtbot.addWidget(w)
    w.llm_model_combo.setCurrentText(_label_for('meta-llama/Meta-Llama-3-8B'))
    assert w.hf_token_row.isVisible() is False
    assert w.hf_token_detected_label.isVisible() is True


def test_session_hf_token_returns_text_only_when_visible(qtbot, monkeypatch):
    import extractors.llm_cloze as m
    monkeypatch.setattr(m, 'has_hf_token', lambda: False)
    from gui.section_word_features import WordFeaturesSection
    w = WordFeaturesSection(); qtbot.addWidget(w)
    w.llm_model_combo.setCurrentText(_label_for('meta-llama/Meta-Llama-3-8B'))
    w.hf_token_line.setText('hf_secret')
    assert w.session_hf_token() == 'hf_secret'
    w.llm_model_combo.setCurrentText(_label_for('gpt-4o-mini'))
    assert w.session_hf_token() == ''   # hidden → empty
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./env/python.exe -m pytest tests/gui/test_section_word_features.py -k "hf or api_key or detected or credential" -v`
Expected: FAIL with `AttributeError: 'WordFeaturesSection' object has no attribute 'hf_token_row'`.

- [ ] **Step 3: Add the HF-token widgets**

In `gui/section_word_features.py`, immediately after `self._refresh_api_key_visibility()` at ~:192 (before `if not saved_model:`), insert:
```python
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
```

- [ ] **Step 4: Replace the visibility method and update its call sites**

Replace `_refresh_api_key_visibility` (currently `gui/section_word_features.py:441-444`) with:
```python
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
```

Update the two existing call sites to the new name:
- In `__init__` (~:192): `self._refresh_api_key_visibility()` → `self._refresh_llm_credential_visibility()`. This call must move to run **after** the HF-token widgets are created (i.e., place it right after the `hf_token_detected_label` block from Step 3), since it now references `hf_token_row`.
- In `_on_model_change` (~:433): `self._refresh_api_key_visibility()` → `self._refresh_llm_credential_visibility()`.

- [ ] **Step 5: Add `session_hf_token` and extend `refresh_warnings`**

After `session_api_key` (`gui/section_word_features.py:386-387`) add:
```python
    def session_hf_token(self) -> str:
        return self.hf_token_line.text() if self.hf_token_row.isVisible() else ""
```

Change `refresh_warnings` (currently `:420-428`) signature and forwarding:
```python
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
```
(`compute_warnings` gains the `session_hf_token` param in Task 5; if running Task 3 before Task 5, that param already has a default there once Task 5 lands. Implement Task 5 before running the full suite.)

- [ ] **Step 6: Run tests to verify they pass**

Run: `./env/python.exe -m pytest tests/gui/test_section_word_features.py -k "hf or api_key or detected or credential" -v`
Expected: 4 passed.

- [ ] **Step 7: Commit**

```bash
git add gui/section_word_features.py tests/gui/test_section_word_features.py
git commit -m "feat(gui): HF-token box for gated local Llama (mirrors OpenAI-key box)"
```

---

### Task 4: Wire the token into the run

**Files:**
- Modify: `gui/run_dispatch.py:48-102` (`run` signature + env set/pop)
- Modify: `gui/section_run.py:49-51,204,283-287` (accept + forward the getter)
- Modify: `gui/main_window.py:125` (pass the getter)
- Test: `tests/gui/test_run_dispatch.py`

**Interfaces:**
- Consumes: `WordFeaturesSection.session_hf_token` (Task 3).
- Produces: `run(..., session_hf_token: str = "", ...)` sets/pops `os.environ["HF_TOKEN"]`.

- [ ] **Step 1: Write the failing test**

Add to `tests/gui/test_run_dispatch.py`:
```python
def test_dispatcher_sets_and_pops_hf_token(monkeypatch, tmp_path):
    monkeypatch.setenv("SENSI_PREFS_PATH", str(tmp_path / "prefs.json"))
    import os
    from utils.preferences import save_preferences
    save_preferences({"output": {"word_level_csv": True}})
    monkeypatch.delenv("HF_TOKEN", raising=False)

    seen = {}
    def fake_word(args, **kwargs): seen["during"] = os.environ.get("HF_TOKEN")
    monkeypatch.setattr("main.run_word_level_batch", fake_word, raising=False)

    from gui.run_dispatch import run
    run(progress_cb=lambda *a: None, done_cb=lambda *a: None,
        session_api_key="", session_hf_token="hf_test123", in_thread=False)

    assert seen["during"] == "hf_test123"          # set for the run
    assert os.environ.get("HF_TOKEN") is None       # popped afterward
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./env/python.exe -m pytest tests/gui/test_run_dispatch.py::test_dispatcher_sets_and_pops_hf_token -v`
Expected: FAIL — `run()` has no `session_hf_token` parameter (`TypeError`).

- [ ] **Step 3: Add the param and env handling in `run_dispatch.py`**

Change the `run` signature (`gui/run_dispatch.py:48-51`) to add `session_hf_token`:
```python
def run(progress_cb: Callable[[int, int, str], None],
        done_cb: Callable[[str], None],
        session_api_key: str,
        session_hf_token: str = "",
        in_thread: bool = True) -> None:
```
In `_go()`, right after the OpenAI block (`:66-67`), add:
```python
        if session_hf_token:
            os.environ["HF_TOKEN"] = session_hf_token
```
In the `finally` (`:100-102`), after the OpenAI pop, add:
```python
            if session_hf_token:
                os.environ.pop("HF_TOKEN", None)
```

- [ ] **Step 4: Forward the getter from `section_run.py`**

- `__init__` (`gui/section_run.py:49-51`): add a keyword param and store it:
```python
    def __init__(self, get_session_api_key, status_strip, parent=None,
                 get_session_hf_token=None):
        super().__init__(parent)
        self._get_api_key = get_session_api_key
        self._get_hf_token = get_session_hf_token or (lambda: "")
```
- `compute_warnings` call (`:204`): add the token:
```python
        warnings = compute_warnings(session_api_key=self._get_api_key(),
                                    session_hf_token=self._get_hf_token())
```
- `dispatch_run` call (`:283-287`): add the token:
```python
        dispatch_run(
            progress_cb=self._progress_event.emit,
            done_cb=self._done_event.emit,
            session_api_key=self._get_api_key(),
            session_hf_token=self._get_hf_token(),
        )
```

- [ ] **Step 5: Pass the getter from `main_window.py`**

At `gui/main_window.py:125` (the `SectionRun`/`section_run` construction that already passes `get_session_api_key=self.word_features.session_api_key`), add:
```python
            get_session_hf_token=self.word_features.session_hf_token,
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `./env/python.exe -m pytest tests/gui/test_run_dispatch.py tests/gui/test_section_run.py tests/gui/test_main_window.py -v`
Expected: all pass (new HF_TOKEN test + existing dispatch/run/main-window tests).

- [ ] **Step 7: Commit**

```bash
git add gui/run_dispatch.py gui/section_run.py gui/main_window.py tests/gui/test_run_dispatch.py
git commit -m "feat(gui): inject session HF_TOKEN for the run (mirrors OPENAI_API_KEY)"
```

---

### Task 5: Preflight warning for gated model without token

**Files:**
- Modify: `gui/preflight.py:69,93-102`
- Test: `tests/gui/test_preflight.py`

**Interfaces:**
- Consumes: `is_gated_local_model`, `has_hf_token` (Task 1).
- Produces: `compute_warnings(session_api_key="", session_hf_token="")`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/gui/test_preflight.py` (mirror the existing OpenAI-key tests at :38-63):
```python
def test_preflight_warns_on_gated_local_without_token(prefs_tmp, monkeypatch):
    import extractors.llm_cloze as m
    monkeypatch.setattr(m, 'has_hf_token', lambda: False)
    from utils.preferences import save_preferences
    from gui.preflight import compute_warnings
    save_preferences({
        "output": {"word_level_csv": True},
        "predictability": {"llm_model": "meta-llama/Meta-Llama-3-8B"},
        "features": {"pred_cloze": True},
    })
    warnings = compute_warnings(session_api_key="", session_hf_token="")
    msgs = [w.message for w in warnings if w.toggle_key == "pred_cloze"]
    assert any("HuggingFace token" in msg for msg in msgs)


def test_preflight_no_gated_warning_when_token_present(prefs_tmp, monkeypatch):
    import extractors.llm_cloze as m
    monkeypatch.setattr(m, 'has_hf_token', lambda: True)
    from utils.preferences import save_preferences
    from gui.preflight import compute_warnings
    save_preferences({
        "output": {"word_level_csv": True},
        "predictability": {"llm_model": "meta-llama/Meta-Llama-3-8B"},
        "features": {"pred_cloze": True},
    })
    warnings = compute_warnings(session_api_key="", session_hf_token="")
    assert not any("HuggingFace token" in w.message for w in warnings)


def test_preflight_no_gated_warning_when_session_token_pasted(prefs_tmp, monkeypatch):
    import extractors.llm_cloze as m
    monkeypatch.setattr(m, 'has_hf_token', lambda: False)
    from utils.preferences import save_preferences
    from gui.preflight import compute_warnings
    save_preferences({
        "output": {"word_level_csv": True},
        "predictability": {"llm_model": "meta-llama/Meta-Llama-3-8B"},
        "features": {"pred_cloze": True},
    })
    warnings = compute_warnings(session_api_key="", session_hf_token="hf_pasted")
    assert not any("HuggingFace token" in w.message for w in warnings)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./env/python.exe -m pytest tests/gui/test_preflight.py -k "gated or token_present or session_token" -v`
Expected: FAIL — `compute_warnings()` has no `session_hf_token` param (`TypeError`).

- [ ] **Step 3: Extend `compute_warnings`**

Change the signature (`gui/preflight.py:69`):
```python
def compute_warnings(session_api_key: str = "", session_hf_token: str = "") -> list[Warning]:
```
In the `llm_predict` block (`:94-102`), turn the OpenAI `if` into `if/elif` by adding, right after the existing OpenAI loop:
```python
        elif is_gated_local_model(llm_model) and not session_hf_token and not has_hf_token():
            for k in ("pred_surprisal_llm", "pred_cloze", "pred_topmatch"):
                if prefs["features"].get(k, False):
                    out.append(Warning(
                        toggle_key=k, severity="amber",
                        message=f"Local model '{llm_model}' needs a HuggingFace token "
                                "+ accepted license — paste a token in Predictability "
                                "or run `hf auth login`."))
```
Add the import at the top of `gui/preflight.py` (near `:19`):
```python
from extractors.llm_cloze import LLM_OPTIONS, is_gated_local_model, has_hf_token
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./env/python.exe -m pytest tests/gui/test_preflight.py tests/gui/test_preflight_invariants.py -v`
Expected: all pass — including the invariant test (the message points at the token, not a toggle).

- [ ] **Step 5: Commit**

```bash
git add gui/preflight.py tests/gui/test_preflight.py
git commit -m "feat(gui): preflight warns when gated local Llama selected without a token"
```

---

### Task 6: Local-run documentation

**Files:**
- Modify: `docs/features/pred_cloze.md` (add a setup section)
- Modify: `docs/features/pred_surprisal_llm.md` (one-line cross-link)

**Interfaces:** none (prose).

- [ ] **Step 1: Add the setup section to `pred_cloze.md`**

Append this section to `docs/features/pred_cloze.md`:
```markdown
## Setting up the local Llama backend (local runs)

The `meta-llama/*` models are **gated and free**, but require one-time setup:

1. **Accept the license.** Sign in at huggingface.co and open the model page
   (e.g. <https://huggingface.co/meta-llama/Meta-Llama-3-8B>); accept the terms
   and wait for "access granted".
2. **Provide a token.** Create a *Read* token (Settings → Access Tokens). Either
   paste it into the **HF token** box that appears in Predictability when a Llama
   model is selected (session-only, never saved to disk), or run `hf auth login`
   once so it's auto-detected (the box is then replaced by "✓ token detected").
3. **GPU.** An 8B model needs roughly ≥12 GB VRAM; on CPU it runs but is slow.

No API key or download is needed for the **OpenAI** backend — set
`OPENAI_API_KEY` and pick a `gpt-*` model for a zero-setup alternative.

*(Running on Sherlock/HPC is documented separately.)*
```

- [ ] **Step 2: Cross-link from `pred_surprisal_llm.md`**

Add near the top of `docs/features/pred_surprisal_llm.md`:
```markdown
> Local Llama backends need one-time token/license setup — see
> **[pred_cloze.md → Setting up the local Llama backend](pred_cloze.md#setting-up-the-local-llama-backend-local-runs)**.
```

- [ ] **Step 3: Commit**

```bash
git add docs/features/pred_cloze.md docs/features/pred_surprisal_llm.md
git commit -m "docs(llm-cloze): local-run setup for the gated Llama backend"
```

---

## Self-Review

**Spec coverage:** A→Task 1; B→Task 3; C→Task 4; D→Task 5; E→Task 2; F→Task 6. All six components covered. (The model-ID correction the spec lists under problem #1 is already committed on `main`.)

**Placeholder scan:** No TBD/TODO; every code step shows complete code and exact commands.

**Type consistency:** `is_gated_local_model` / `has_hf_token` (Task 1) are consumed with those exact names in Tasks 3 and 5. `session_hf_token()` (Task 3) is consumed in Task 4. `compute_warnings(session_api_key, session_hf_token)` (Task 5) matches the call in Task 3's `refresh_warnings`. `run(..., session_hf_token="")` (Task 4) matches Task 4's test. Method rename `_refresh_api_key_visibility`→`_refresh_llm_credential_visibility` updated at both call sites.

**Ordering note:** Task 3's `refresh_warnings` passes `session_hf_token` to `compute_warnings`, which only accepts it after Task 5. Implement in numeric order but run the *full* suite only after Task 5 lands.
