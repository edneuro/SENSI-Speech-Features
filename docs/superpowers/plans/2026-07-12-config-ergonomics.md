# Config Ergonomics (templates + persistent HF token) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship committed, generated config templates (`config/features_defaults.yaml` + `config/run_defaults.yaml`) for GUI-skipping users, and make the HuggingFace token persistable at `config/hf_token.txt` (GUI saves it; headless runs read it by default).

**Architecture:** A new `utils/credentials.py` owns token-file resolution (relative paths resolve against `get_config_dir()`), reading, exporting to `HF_TOKEN`, and saving; `run.py`, `gui/run_dispatch.py`, and `extractors.predictability.has_hf_token` all consume it. Templates are build artifacts generated from `DEFAULT_PREFS` by `scripts/export_default_config.py`, pinned to the code by a semantic drift-guard test.

**Tech Stack:** Python 3, pyyaml, PySide6, pytest (+pytest-qt). All test commands (bash): `PATH="$(pwd)/env/Library/bin:$PATH" ./env/python.exe -m pytest <targets> -v`.

**Spec (user-approved):** `docs/superpowers/specs/2026-07-12-config-ergonomics-design.md`

## Global Constraints

- Branch `config-ergonomics` off `main`. One commit per task. Suite baseline: **755 passed, 1 skipped**; full suite green at the end of every task.
- **No prefs schema bump.** The only default change is `DEFAULT_PREFS["shared"]["hf_token_file"]`: `""` → `"hf_token.txt"` (v19 literal only). A stored `""` in an existing run.yaml is the user's value and is left alone — the GUI self-heals it when a token is saved.
- **Relative `hf_token_file` values resolve against `get_config_dir()`**; absolute paths and `~`/`$VARS` keep working. Missing/unreadable/empty file ⇒ no token (never an exception).
- Session-token behavior is preserved: a token pasted but NOT saved still flows through `session_hf_token` → `HF_TOKEN` env for that run, exactly as today.
- Templates are **committed** and carry a template banner; the drift guard compares **parsed YAML** (`read_yaml(template) == the matching half of split_prefs(DEFAULT_PREFS)`), not bytes, so the banner is allowed. `config/hf_token.txt` is **gitignored**; the templates are not.
- GUI copy must say the token is saved to `config/hf_token.txt` — remove every "never saved to disk" claim.
- Do not touch `unrelated/`, master toggles, output toggles, or the Run tab.
- Windows dev machine; also runs on Linux (Sherlock) and Apple Silicon — no OS-specific paths outside `pathlib`/`os.path`.

---

### Task 1: `utils/credentials.py` + default change + consumers (run.py, has_hf_token)

**Files:**
- Create: `utils/credentials.py`
- Modify: `utils/preferences.py` (the `DEFAULT_PREFS_V19` literal's `shared` line), `run.py` (use the helper; delete its local `load_credentials`), `extractors/predictability.py:286-294` (`has_hf_token`)
- Test: `tests/utils/test_credentials.py` (new, TDD); update `tests/utils/test_preferences_yaml.py::test_default_shared_has_hf_token_file` (asserts `== ""` today); update the two `load_credentials` tests in `tests/test_run_py.py`

**Interfaces:**
- Consumes: `utils.preferences.get_config_dir`, `load_preferences`, `save_preferences`.
- Produces (used by Tasks 2–3):
  - `resolve_hf_token_path(shared: dict) -> Path | None` — `None` when `hf_token_file` is empty; expands `~`/`$VARS`; a relative result resolves against `get_config_dir()`.
  - `read_hf_token(shared: dict) -> str` — token text stripped; `""` on missing/unreadable/empty.
  - `export_hf_token(shared: dict) -> bool` — sets `os.environ["HF_TOKEN"]` from the file when a token is found; returns True iff it set the env var. Never overwrites an existing `HF_TOKEN`.
  - `save_hf_token(token: str) -> Path` — writes `get_config_dir()/hf_token.txt` (strip + trailing newline), calls `save_preferences({"shared": {"hf_token_file": "hf_token.txt"}})`, returns the path.

- [ ] **Step 1: Write the failing tests** (`tests/utils/test_credentials.py`)

```python
import os
from pathlib import Path

from utils.credentials import (
    resolve_hf_token_path, read_hf_token, export_hf_token, save_hf_token,
)
from utils.preferences import DEFAULT_PREFS, load_preferences


def test_default_hf_token_file_is_relative_conventional_name():
    assert DEFAULT_PREFS["shared"]["hf_token_file"] == "hf_token.txt"


def test_resolve_relative_against_config_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("SENSI_CONFIG_DIR", str(tmp_path))
    assert resolve_hf_token_path({"hf_token_file": "hf_token.txt"}) == tmp_path / "hf_token.txt"


def test_resolve_absolute_and_vars_and_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("SENSI_CONFIG_DIR", str(tmp_path))
    absolute = tmp_path / "elsewhere" / "tok.txt"
    assert resolve_hf_token_path({"hf_token_file": str(absolute)}) == absolute
    monkeypatch.setenv("TOKDIR", str(tmp_path / "vardir"))
    assert resolve_hf_token_path({"hf_token_file": "$TOKDIR/t.txt"}) == tmp_path / "vardir" / "t.txt"
    assert resolve_hf_token_path({"hf_token_file": ""}) is None
    assert resolve_hf_token_path({}) is None


def test_read_hf_token_missing_unreadable_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("SENSI_CONFIG_DIR", str(tmp_path))
    shared = {"hf_token_file": "hf_token.txt"}
    assert read_hf_token(shared) == ""                       # missing file
    (tmp_path / "hf_token.txt").write_text("  \n", encoding="utf-8")
    assert read_hf_token(shared) == ""                       # whitespace only
    (tmp_path / "hf_token.txt").write_text("hf_secret\n", encoding="utf-8")
    assert read_hf_token(shared) == "hf_secret"


def test_export_hf_token_sets_env_and_never_clobbers(tmp_path, monkeypatch):
    monkeypatch.setenv("SENSI_CONFIG_DIR", str(tmp_path))
    monkeypatch.delenv("HF_TOKEN", raising=False)
    shared = {"hf_token_file": "hf_token.txt"}
    assert export_hf_token(shared) is False                  # no file → no export
    assert "HF_TOKEN" not in os.environ
    (tmp_path / "hf_token.txt").write_text("hf_file\n", encoding="utf-8")
    assert export_hf_token(shared) is True
    assert os.environ["HF_TOKEN"] == "hf_file"
    monkeypatch.setenv("HF_TOKEN", "hf_session")
    assert export_hf_token(shared) is False                  # existing env wins
    assert os.environ["HF_TOKEN"] == "hf_session"


def test_save_hf_token_writes_file_and_points_pref_at_it(tmp_path, monkeypatch):
    monkeypatch.setenv("SENSI_CONFIG_DIR", str(tmp_path))
    p = save_hf_token("  hf_new_token  ")
    assert p == tmp_path / "hf_token.txt"
    assert p.read_text(encoding="utf-8") == "hf_new_token\n"
    assert load_preferences()["shared"]["hf_token_file"] == "hf_token.txt"


def test_has_hf_token_sees_configured_file(tmp_path, monkeypatch):
    # has_hf_token() must treat a readable, non-empty configured token file
    # as "token present" (in addition to env / `hf auth login`).
    import extractors.predictability as m
    monkeypatch.setenv("SENSI_CONFIG_DIR", str(tmp_path))
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr("huggingface_hub.get_token", lambda: None)
    assert m.has_hf_token() is False
    (tmp_path / "hf_token.txt").write_text("hf_x\n", encoding="utf-8")
    assert m.has_hf_token() is True
```

- [ ] **Step 2: Run to verify failure**

Run: `PATH="$(pwd)/env/Library/bin:$PATH" ./env/python.exe -m pytest tests/utils/test_credentials.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'utils.credentials'`

- [ ] **Step 3: Implement**

`utils/credentials.py` (complete file):

```python
"""HuggingFace token-file handling shared by run.py, the GUI dispatcher, and
token detection.

The conventional location is <config_dir>/hf_token.txt (gitignored), named by
prefs `shared.hf_token_file`. A relative value resolves against the config
dir (utils.preferences.get_config_dir), so the shipped default "hf_token.txt"
is correct on every machine; absolute paths and ~/$VARS also work.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

HF_TOKEN_BASENAME = "hf_token.txt"


def resolve_hf_token_path(shared: dict[str, Any]) -> Path | None:
    """Path named by shared.hf_token_file, or None when unset/empty."""
    raw = (shared or {}).get("hf_token_file") or ""
    if not raw:
        return None
    expanded = os.path.expanduser(os.path.expandvars(str(raw)))
    path = Path(expanded)
    if not path.is_absolute():
        from utils.preferences import get_config_dir
        path = get_config_dir() / path
    return path


def read_hf_token(shared: dict[str, Any]) -> str:
    """Token text from the configured file; '' when missing/unreadable/empty."""
    path = resolve_hf_token_path(shared)
    if path is None or not path.is_file():
        return ""
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def export_hf_token(shared: dict[str, Any]) -> bool:
    """Set HF_TOKEN from the configured file. Never overwrites an existing
    HF_TOKEN (a session-pasted token wins). Returns True iff it set the var."""
    if os.environ.get("HF_TOKEN"):
        return False
    token = read_hf_token(shared)
    if not token:
        return False
    os.environ["HF_TOKEN"] = token
    return True


def save_hf_token(token: str) -> Path:
    """Persist a token to <config_dir>/hf_token.txt and point the pref at it."""
    from utils.preferences import get_config_dir, save_preferences
    path = get_config_dir() / HF_TOKEN_BASENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(token.strip() + "\n", encoding="utf-8")
    save_preferences({"shared": {"hf_token_file": HF_TOKEN_BASENAME}})
    return path
```

`utils/preferences.py` — in the `DEFAULT_PREFS_V19` literal change the shared line to
`"shared": {**DEFAULT_PREFS_V18["shared"], "hf_token_file": "hf_token.txt"},` and update its comment (conventional gitignored token file; relative → config dir).

`extractors/predictability.py` — `has_hf_token` becomes:

```python
def has_hf_token() -> bool:
    """True if a HuggingFace token is already available — HF_TOKEN /
    HUGGING_FACE_HUB_TOKEN env, a token saved by a prior `hf auth login`,
    or the token file named by prefs shared.hf_token_file (default
    config/hf_token.txt). Lazy imports keep GUI startup cheap."""
    try:
        from huggingface_hub import get_token
    except Exception:
        return False
    if bool(get_token()):
        return True
    try:
        from utils.credentials import read_hf_token
        from utils.preferences import load_preferences
        return bool(read_hf_token(load_preferences().get("shared", {})))
    except Exception:
        return False
```

`run.py` — delete its `load_credentials` function; in `main()` replace the `load_credentials(shared)` call with:

```python
    from utils.credentials import export_hf_token
    export_hf_token(shared)
```

`tests/test_run_py.py` — the two `load_credentials` tests now target the util (keep them here as run.py-level behavior tests, importing `export_hf_token` from `utils.credentials`, or delete them since `tests/utils/test_credentials.py` covers the behavior — prefer delete + a one-line comment is NOT needed; just delete them). `tests/utils/test_preferences_yaml.py::test_default_shared_has_hf_token_file` — change the assertion to `== "hf_token.txt"`.

- [ ] **Step 4: Run tests**

Run: `PATH="$(pwd)/env/Library/bin:$PATH" ./env/python.exe -m pytest tests/utils/test_credentials.py tests/utils/test_preferences_yaml.py tests/test_run_py.py tests/gui/test_preflight.py -v` → all pass.
Then the full suite once → green (some GUI tests monkeypatch `has_hf_token` wholesale, so they are unaffected; if a GUI test fails because the REAL `has_hf_token` now reads prefs, the conftest `SENSI_CONFIG_DIR` isolation gives it an empty tmp dir → still False — investigate anything that contradicts this).

- [ ] **Step 5: Commit**

```bash
git add utils/credentials.py utils/preferences.py extractors/predictability.py run.py tests/
git commit -m "feat(config): shared HF token-file helper; conventional config/hf_token.txt default"
```

---

### Task 2: GUI — save the pasted token; dispatcher exports the file token

**Files:**
- Modify: `gui/section_word_features.py:260-287` (token row), `gui/run_dispatch.py:65-70` (`_go`)
- Test: `tests/gui/test_section_word_features.py` (append), `tests/gui/test_run_dispatch.py` (append) — test-after (GUI wiring)

**Interfaces:**
- Consumes (Task 1): `utils.credentials.save_hf_token`, `export_hf_token`.
- Produces: a `Save` button (`self.hf_token_save_btn`) on the token row; `run_dispatch.run(...)` exports the configured file token into `HF_TOKEN` when no session token was pasted (and pops it afterward only if it set it).

- [ ] **Step 1: Implement the GUI changes**

In `gui/section_word_features.py`, replace the placeholder text and add a Save button after the Show button (keep every existing attribute name — tests use them):

```python
        self.hf_token_line.setPlaceholderText(
            "Paste HuggingFace token — Save stores it in config/hf_token.txt")
        ...
        ht.addWidget(hf_show)
        self.hf_token_save_btn = QPushButton("Save")
        self.hf_token_save_btn.setToolTip(
            "Store the token in config/hf_token.txt (gitignored) so future "
            "runs — GUI or headless — find it automatically.")
        self.hf_token_save_btn.clicked.connect(self._save_hf_token)
        ht.addWidget(self.hf_token_save_btn)
```

and update the detected label text where it is created (line ~284):
`self.hf_token_detected_label = QLabel("✓ HuggingFace token detected")` stays, but `_save_hf_token` refreshes visibility so it appears right after a save. Add the slot next to `session_hf_token()`:

```python
    def _save_hf_token(self) -> None:
        token = self.hf_token_line.text().strip()
        if not token:
            return
        from utils.credentials import save_hf_token
        save_hf_token(token)
        self.hf_token_line.clear()
        self._refresh_llm_credential_visibility()
        self.refresh_warnings()
```

(`_refresh_llm_credential_visibility` already flips hint/row/detected-label off `has_hf_token()`, which now sees the file — no change needed there.)

In `gui/run_dispatch.py` `_go()`, after the session-token export (line ~70):

```python
        if session_hf_token:
            os.environ["HF_TOKEN"] = session_hf_token
        prefs = load_preferences()
        from utils.credentials import export_hf_token
        exported_file_token = export_hf_token(prefs.get("shared", {}))
```

(note: move the existing `prefs = load_preferences()` up if needed so it precedes the export — it already sits right after the session-token block) and in the `finally` block:

```python
            if session_hf_token:
                os.environ.pop("HF_TOKEN", None)
            elif exported_file_token:
                os.environ.pop("HF_TOKEN", None)
```

- [ ] **Step 2: Write the tests**

Append to `tests/gui/test_section_word_features.py`:

```python
def test_save_hf_token_button_persists_and_hides_row(qtbot, monkeypatch, tmp_path):
    import extractors.predictability as m
    monkeypatch.setenv("SENSI_CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr("huggingface_hub.get_token", lambda: None)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    w = _make_widget(qtbot)          # use this file's existing widget factory/fixture idiom
    assert w.hf_token_row.isVisible() is True
    w.hf_token_line.setText("hf_gui_token")
    w.hf_token_save_btn.click()
    assert (tmp_path / "hf_token.txt").read_text(encoding="utf-8") == "hf_gui_token\n"
    from utils.preferences import load_preferences
    assert load_preferences()["shared"]["hf_token_file"] == "hf_token.txt"
    assert w.hf_token_line.text() == ""              # field cleared after save
    assert w.hf_token_row.isVisible() is False       # real has_hf_token now sees the file
    assert w.hf_token_detected_label.isVisible() is True


def test_save_hf_token_button_ignores_empty(qtbot, monkeypatch, tmp_path):
    import extractors.predictability as m
    monkeypatch.setenv("SENSI_CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(m, "has_hf_token", lambda: False)
    w = _make_widget(qtbot)
    w.hf_token_line.setText("   ")
    w.hf_token_save_btn.click()
    assert not (tmp_path / "hf_token.txt").exists()
```

(Adapt `_make_widget` to however this test file already constructs the page — reuse its existing helper/fixture verbatim; do not invent a new construction path. The first test must NOT monkeypatch `has_hf_token` — it exercises the real file-detection.)

Append to `tests/gui/test_run_dispatch.py` (mirror the existing `test_dispatcher_sets_and_pops_hf_token` at line ~554 for style):

```python
def test_dispatcher_exports_file_token_when_no_session_token(monkeypatch, tmp_path):
    import os
    import gui.run_dispatch as rd
    monkeypatch.setenv("SENSI_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("SENSI_PREFS_PATH", str(tmp_path / "prefs.json"))
    monkeypatch.delenv("HF_TOKEN", raising=False)
    (tmp_path / "hf_token.txt").write_text("hf_from_file\n", encoding="utf-8")
    seen = {}
    def fake_word(args, **kwargs): seen["during"] = os.environ.get("HF_TOKEN")
    monkeypatch.setattr("main.run_word_level_batch", fake_word, raising=False)
    rd.run(progress_cb=lambda *a: None, done_cb=lambda s: None,
           session_api_key="", session_hf_token="", in_thread=False)
    assert seen["during"] == "hf_from_file"          # exported for the run
    assert os.environ.get("HF_TOKEN") is None        # popped afterward
```

(This test needs word-level output on — check how the neighboring dispatcher tests arrange prefs and copy that arrangement; the default `word_level_csv` is already True.)

- [ ] **Step 3: Run tests**

Run: `PATH="$(pwd)/env/Library/bin:$PATH" ./env/python.exe -m pytest tests/gui/test_section_word_features.py tests/gui/test_run_dispatch.py -v` → pass. Full suite once → green. Also grep the repo for `never saved to disk` — zero hits after this task.

- [ ] **Step 4: Commit**

```bash
git add gui/section_word_features.py gui/run_dispatch.py tests/gui/
git commit -m "feat(gui): Save button persists HF token to config/hf_token.txt; dispatcher reads it"
```

---

### Task 3: Committed defaults templates + generator + drift guard

**Files:**
- Create: `scripts/export_default_config.py`, `config/features_defaults.yaml` (generated), `config/run_defaults.yaml` (generated)
- Modify: `run.py` (missing-features.yaml warning text), `.gitignore` (add `config/hf_token.txt`)
- Test: `tests/utils/test_config_templates.py` (new)

**Interfaces:**
- Consumes: `split_prefs`, `write_features_yaml`, `write_run_yaml`, `read_yaml` (utils.config_yaml); `DEFAULT_PREFS`.
- Produces: `scripts/export_default_config.py` with `main() -> None` and `TEMPLATE_BANNER_FEATURES` / `TEMPLATE_BANNER_RUN` constants; regenerating is `python scripts/export_default_config.py`.

- [ ] **Step 1: Write the generator** (`scripts/export_default_config.py`, complete file)

```python
#!/usr/bin/env python3
"""Regenerate the committed config templates from the code defaults.

config/features_defaults.yaml and config/run_defaults.yaml are build
artifacts of utils.preferences.DEFAULT_PREFS — the code is the source of
truth. A drift-guard test (tests/utils/test_config_templates.py) fails
whenever the schema changes without rerunning this script:

    python scripts/export_default_config.py

Headless users copy the templates to features.yaml / run.yaml and edit.
"""
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from utils.preferences import DEFAULT_PREFS  # noqa: E402
from utils.config_yaml import (  # noqa: E402
    split_prefs, write_features_yaml, write_run_yaml,
)

TEMPLATE_BANNER_FEATURES = (
    "# ============================ TEMPLATE ============================\n"
    "# Copy this file to config/features.yaml and edit — it is the shipped\n"
    "# defaults for \"what to extract\". Regenerate after a schema change:\n"
    "#     python scripts/export_default_config.py\n"
    "# ==================================================================\n"
)
TEMPLATE_BANNER_RUN = (
    "# ============================ TEMPLATE ============================\n"
    "# Copy this file to config/run.yaml and edit — machine-specific run\n"
    "# settings (folders, analyzer name, token-file path). Regenerate:\n"
    "#     python scripts/export_default_config.py\n"
    "# ==================================================================\n"
)


def main() -> None:
    config_dir = os.path.join(_REPO_ROOT, "config")
    features_data, run_data = split_prefs(DEFAULT_PREFS)
    f_path = os.path.join(config_dir, "features_defaults.yaml")
    r_path = os.path.join(config_dir, "run_defaults.yaml")
    write_features_yaml(f_path, features_data)
    write_run_yaml(r_path, run_data)
    for path, banner in ((f_path, TEMPLATE_BANNER_FEATURES),
                         (r_path, TEMPLATE_BANNER_RUN)):
        with open(path, "r", encoding="utf-8") as fh:
            body = fh.read()
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(banner + body)
    print(f"Wrote {f_path} and {r_path} from DEFAULT_PREFS "
          f"(schema v{DEFAULT_PREFS['schema_version']}).")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Generate the templates and write the drift guard**

Run: `PATH="$(pwd)/env/Library/bin:$PATH" ./env/python.exe scripts/export_default_config.py`
Expected: both files appear under `config/`; open them and eyeball the banner + sections.

`tests/utils/test_config_templates.py`:

```python
"""Drift guard: the committed config templates are build artifacts of
DEFAULT_PREFS. If this fails, rerun: python scripts/export_default_config.py"""
from pathlib import Path

from utils.preferences import DEFAULT_PREFS
from utils.config_yaml import read_yaml, split_prefs

_CONFIG = Path(__file__).resolve().parents[2] / "config"
_REGEN = "templates out of sync with DEFAULT_PREFS — rerun: python scripts/export_default_config.py"


def test_features_template_matches_code_defaults():
    features_data, _ = split_prefs(DEFAULT_PREFS)
    assert read_yaml(_CONFIG / "features_defaults.yaml") == features_data, _REGEN


def test_run_template_matches_code_defaults():
    _, run_data = split_prefs(DEFAULT_PREFS)
    assert read_yaml(_CONFIG / "run_defaults.yaml") == run_data, _REGEN


def test_templates_carry_the_template_banner():
    for name in ("features_defaults.yaml", "run_defaults.yaml"):
        text = (_CONFIG / name).read_text(encoding="utf-8")
        assert text.startswith("# ======"), name
        assert "TEMPLATE" in text.splitlines()[0], name
```

- [ ] **Step 3: run.py warning + .gitignore**

In `run.py`, extend the missing-features.yaml warning (keep the existing `features.yaml not found` substring — a test matches it):

```python
        print(f"[warn] features.yaml not found in {get_features_yaml_path().parent} "
              f"— running with shipped defaults (minimal feature set). To start "
              f"from the documented defaults: copy config/features_defaults.yaml "
              f"to config/features.yaml and edit.", file=sys.stderr)
```

`.gitignore` — add `config/hf_token.txt` to the runtime-config block (the templates match no ignore rule; verify with `git check-ignore -v config/features_defaults.yaml` returning nothing).

- [ ] **Step 4: Run tests**

Run: `PATH="$(pwd)/env/Library/bin:$PATH" ./env/python.exe -m pytest tests/utils/test_config_templates.py tests/test_run_py.py -v` → pass. Full suite once → green.

- [ ] **Step 5: Commit**

```bash
git add scripts/export_default_config.py config/features_defaults.yaml config/run_defaults.yaml tests/utils/test_config_templates.py run.py .gitignore
git commit -m "feat(config): committed defaults templates + generator + drift guard"
```

---

### Task 4: Docs — CLAUDE.md, README, RUNBOOK, llm_credentials.md, master plan

**Files:**
- Modify: `CLAUDE.md`, `README.md`, `sherlock/RUNBOOK.md`, `docs/features/llm_credentials.md`, `docs/master_plan.md`

**Interfaces:** Consumes the shipped behavior of Tasks 1–3 (describe what IS).

- [ ] **Step 1: Edits**

- `CLAUDE.md`: the `config/` tree line gains the templates + token file: mention `features_defaults.yaml`/`run_defaults.yaml` (committed templates — copy to the live names to skip the GUI) and `hf_token.txt` (gitignored token file). Keep the one-line-per-entry convention.
- `README.md`: wherever headless running / gated models are described, add the copy-the-template quickstart and the `config/hf_token.txt` convention (grep for the Llama/token passage near line 192).
- `sherlock/RUNBOOK.md`: the create-run.yaml step becomes "copy `config/run_defaults.yaml` → `config/run.yaml` and edit"; the HF-token step says to put the token in `$GROUP_HOME/SensiExtractor/config/hf_token.txt` (the default `hf_token_file: hf_token.txt` resolves against the repo's `config/`).
- `docs/features/llm_credentials.md`: token is now persistable — GUI Save button writes `config/hf_token.txt` (gitignored); headless runs read it automatically; env var / `hf auth login` still work. Keep citations untouched.
- `docs/master_plan.md`: §3 row (2026-07-12, "Config ergonomics: defaults templates + persistent HF token", ~6–10 lines: templates + drift guard, `utils/credentials.py`, GUI Save button, dispatcher/file-token export, `hf_token_file` default `hf_token.txt` resolved against the config dir, suite count); §4 roadmap gains two parked items with their constraints — (a) "Gating simplification (masters + output toggles + Run-tab summary)" [deferred until after manual validation; v20 removal-migration + planner + preflight + GUI + docs blast radius; Run tab becomes read-only expected-output summary + readiness], (b) "Move one-time CLI tools (`utils/download_models.py`, `download_lexical_norms.py`, `validate_cloze.py`) → `scripts/`" [neatness only; touches setup.sh:29 / setup.bat:40 / CLAUDE.md / README + any importing tests].

- [ ] **Step 2: Verify + commit**

Run: `PATH="$(pwd)/env/Library/bin:$PATH" ./env/python.exe -m pytest --collect-only -q 2>&1 | tail -2` (sanity, no errors). Grep the five edited files for `never saved to disk` (0 hits) and for `run_extract|export_config.py` (0 new hits; the generator script is `export_default_config.py` — don't confuse them).

```bash
git add CLAUDE.md README.md sherlock/RUNBOOK.md docs/features/llm_credentials.md docs/master_plan.md
git commit -m "docs: defaults templates + config/hf_token.txt conventions; park gating redesign in roadmap"
```

---

## Self-review notes (already applied)

- Spec coverage: templates+guard (T3), reset unchanged (no task — no code touches reset), token file+GUI save+fallbacks (T1–T2), deferred items to roadmap (T4). ✔
- Type consistency: `save_hf_token(token) -> Path`, `export_hf_token(shared) -> bool` used identically in T1/T2. ✔
- The GUI visibility test relies on the REAL `has_hf_token` reading prefs via `SENSI_CONFIG_DIR` isolation — that is the point of the test; do not monkeypatch it there.
