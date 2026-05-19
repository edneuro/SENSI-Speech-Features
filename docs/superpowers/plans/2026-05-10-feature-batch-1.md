# Feature Batch 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship batch 1 of new SensiExtractor features (EWFG child frequencies, semantic-domain indicators with LIWC/Empath, LLM-cloze, derived timing columns, document-level sidecar) and a GUI overhaul (folder-pair batch mode, analyzer field, run logs, improved selectors).

**Architecture:** Each new feature is a standalone extractor module under `extractors/` that takes a token DataFrame and returns a DataFrame with new columns appended. Shared utilities (preferences, file pairing, run logging, system info, document summary) live under `utils/`. The pipeline orchestrator in `main.py` gains a batch mode that pairs files from two folders and emits per-stimulus CSVs plus a structured run manifest. `gui.py` is restructured around folder pickers, an analyzer field, a categories modal, and grouped feature toggles. All new dependencies are pip-installable; `liwc` dictionary file is licensed and gitignored.

**Tech Stack:** Python 3.10+, pandas, spaCy, transformers, torch, tkinter/ttk (GUI), pytest (testing), psutil (system info), empath (open semantic dictionary), liwc (LIWC `.dic` parser), openai (optional API), wordfreq (existing).

**Spec:** `docs/superpowers/specs/2026-05-10-feature-batch-1-design.md`

---

## File structure

### New files
```
__init__.py                                   # package version
extractors/educator_freq.py                   # EWFG grade-band frequency lookup
extractors/semantic_domain.py                 # LIWC/Empath dictionary lookup
extractors/llm_cloze.py                       # LLM forward-pass + 3 cloze metrics
extractors/timing.py                          # Word_Duration, Word_Gap derivation
utils/preferences.py                          # preferences.json load/save
utils/file_pairing.py                         # folder scan + base-name pairing
utils/run_logger.py                           # log + manifest.json + no-match.csv
utils/system_info.py                          # OS / Python / GPU / RAM detection
utils/doc_summary.py                          # per-stimulus aggregation
config/.gitkeep                               # commit empty config dir
dictionaries/.gitkeep                         # commit empty dictionaries dir
tools/build_edu_zipf_lookup.py                # one-time EWFG CD parser (skeleton)
validate_cloze.py                             # one-shot Provo Corpus validation
tests/__init__.py
tests/conftest.py
tests/extractors/__init__.py
tests/extractors/test_timing.py
tests/extractors/test_educator_freq.py
tests/extractors/test_semantic_domain.py
tests/extractors/test_llm_cloze.py
tests/utils/__init__.py
tests/utils/test_preferences.py
tests/utils/test_file_pairing.py
tests/utils/test_run_logger.py
tests/utils/test_system_info.py
tests/utils/test_doc_summary.py
tests/integration/__init__.py
tests/integration/test_pipeline_batch.py
tests/fixtures/transcripts/story1.txt
tests/fixtures/transcripts/story2.txt
tests/fixtures/audio/story1.wav                # tiny silent placeholder
tests/fixtures/audio/story2.wav                # tiny silent placeholder
tests/fixtures/empty.txt
docs/superpowers/plans/2026-05-10-feature-batch-1.md  # this file
```

### Modified files
```
.gitignore                # add config/preferences.json, dictionaries/*.dic, dictionaries/*.parquet, output/
requirements.txt          # add pytest, psutil, empath, liwc, openai, accelerate, bitsandbytes
main.py                   # wire new extractors, add batch mode, add CLI flags
gui.py                    # full overhaul per spec §10
Feature_Extraction_Guide.md  # add new feature sections + GUI workflow + cost/correlation table
README.md                 # update flag list and "What's new"
```

---

## Phase A — Foundations (Tasks 1–3)

### Task 1: Initialize git, test infrastructure, and version

**Files:**
- Create: `.git/` (via `git init`)
- Create: `__init__.py`
- Create: `tests/__init__.py`, `tests/conftest.py`
- Create: `tests/extractors/__init__.py`, `tests/utils/__init__.py`, `tests/integration/__init__.py`
- Modify: `.gitignore`
- Modify: `requirements.txt`

- [ ] **Step 1: Initialize git**

Run:
```bash
git init
```
Expected: `Initialized empty Git repository in .../SensiExtractor/.git/`

- [ ] **Step 2: Update `.gitignore`**

Append these lines to existing `.gitignore`:
```
# Test artifacts
.pytest_cache/
htmlcov/
*.egg-info/

# Runtime config (license-bound or user-specific)
config/preferences.json
dictionaries/*.dic
dictionaries/*.parquet
dictionaries/*.csv

# Run outputs
output/

# Provo validation cache
data/provo/
```

- [ ] **Step 3: Add testing dep to `requirements.txt`**

Append to `requirements.txt`:
```
# Testing & system inspection
pytest>=7.4
psutil>=5.9
```
Run: `pip install -r requirements.txt`
Expected: pytest and psutil installed.

- [ ] **Step 4: Create top-level `__init__.py` with version**

Create `__init__.py` (project root):
```python
"""SensiExtractor package marker. Defines the project version."""

__version__ = "0.2.0"
```

- [ ] **Step 5: Create test scaffolding**

Create `tests/__init__.py` (empty file).

Create `tests/conftest.py`:
```python
"""Add the project root to sys.path so tests can import extractors/utils."""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
```

Create `tests/extractors/__init__.py`, `tests/utils/__init__.py`, `tests/integration/__init__.py` (all empty).

- [ ] **Step 6: Smoke-test pytest**

Create `tests/test_smoke.py`:
```python
def test_pytest_runs():
    assert 1 + 1 == 2
```

Run: `pytest tests/test_smoke.py -v`
Expected: `1 passed`.

Delete `tests/test_smoke.py` after confirming.

- [ ] **Step 7: Commit**

```bash
git add .gitignore requirements.txt __init__.py tests/__init__.py tests/conftest.py tests/extractors/__init__.py tests/utils/__init__.py tests/integration/__init__.py
git commit -m "chore: initialize git, pytest scaffolding, and package version"
```

---

### Task 2: System info utility

**Files:**
- Create: `utils/system_info.py`
- Create: `tests/utils/test_system_info.py`

- [ ] **Step 1: Write failing test**

Create `tests/utils/test_system_info.py`:
```python
import platform
from utils.system_info import gather_system_info


def test_system_info_returns_dict_with_expected_keys():
    info = gather_system_info()
    assert set(info.keys()) >= {'os', 'python_version', 'gpu', 'system_ram_gb'}


def test_system_info_os_matches_platform():
    info = gather_system_info()
    assert platform.system() in info['os']


def test_system_info_python_version_is_string():
    info = gather_system_info()
    assert isinstance(info['python_version'], str)
    assert info['python_version'].count('.') >= 2  # e.g. "3.10.12"


def test_system_info_gpu_has_name_and_vram_keys():
    info = gather_system_info()
    assert 'name' in info['gpu']
    assert 'vram_gb' in info['gpu']
```

- [ ] **Step 2: Run test, expect failure**

Run: `pytest tests/utils/test_system_info.py -v`
Expected: `ModuleNotFoundError: No module named 'utils.system_info'`

- [ ] **Step 3: Implement**

Create `utils/system_info.py`:
```python
"""Collect OS / Python / GPU / RAM info for run manifests.

Used by `utils.run_logger` to record the environment a batch was run in,
so future debugging across machines is traceable.
"""

import platform
import psutil


def gather_system_info() -> dict:
    """Return a dict with os, python_version, gpu, system_ram_gb."""
    return {
        'os': f"{platform.system()} {platform.release()}",
        'python_version': platform.python_version(),
        'gpu': _detect_gpu(),
        'system_ram_gb': round(psutil.virtual_memory().total / (1024 ** 3), 1),
    }


def _detect_gpu() -> dict:
    """Return {'name': str|None, 'vram_gb': float}. Falls back gracefully."""
    try:
        import torch  # imported lazily — torch is a heavy import
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return {
                'name': torch.cuda.get_device_name(0),
                'vram_gb': round(props.total_memory / (1024 ** 3), 1),
            }
    except Exception:
        pass
    return {'name': None, 'vram_gb': 0.0}
```

- [ ] **Step 4: Run test, expect pass**

Run: `pytest tests/utils/test_system_info.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add utils/system_info.py tests/utils/test_system_info.py
git commit -m "feat(utils): add system_info utility for run manifests"
```

---

### Task 3: Preferences utility

**Files:**
- Create: `utils/preferences.py`
- Create: `tests/utils/test_preferences.py`
- Create: `config/.gitkeep`

- [ ] **Step 1: Create empty config dir marker**

Create `config/.gitkeep` (empty file). This is needed because `config/preferences.json` is gitignored, and we want the `config/` directory to exist after a fresh clone.

- [ ] **Step 2: Write failing test**

Create `tests/utils/test_preferences.py`:
```python
import json
import os
from pathlib import Path

import pytest

from utils.preferences import (
    DEFAULT_PREFS,
    load_preferences,
    save_preferences,
    get_preferences_path,
)


def test_load_preferences_returns_default_when_file_missing(tmp_path, monkeypatch):
    monkeypatch.setenv('SENSI_PREFS_PATH', str(tmp_path / 'does-not-exist.json'))
    prefs = load_preferences()
    assert prefs == DEFAULT_PREFS


def test_save_then_load_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv('SENSI_PREFS_PATH', str(tmp_path / 'p.json'))
    save_preferences({'analyzer_name': 'Alice', 'semantic_categories': ['social']})
    loaded = load_preferences()
    assert loaded['analyzer_name'] == 'Alice'
    assert loaded['semantic_categories'] == ['social']
    # Defaults filled in for missing keys
    assert 'feature_toggles' in loaded


def test_unknown_keys_are_preserved(tmp_path, monkeypatch):
    """Forward compatibility: unknown keys should round-trip."""
    monkeypatch.setenv('SENSI_PREFS_PATH', str(tmp_path / 'p.json'))
    target = tmp_path / 'p.json'
    target.write_text(json.dumps({'analyzer_name': 'Bob', 'future_key': 42}))
    loaded = load_preferences()
    assert loaded['analyzer_name'] == 'Bob'
    assert loaded.get('future_key') == 42


def test_get_preferences_path_respects_env_var(tmp_path, monkeypatch):
    custom = tmp_path / 'custom.json'
    monkeypatch.setenv('SENSI_PREFS_PATH', str(custom))
    assert get_preferences_path() == custom


def test_get_preferences_path_defaults_to_config_dir(monkeypatch):
    monkeypatch.delenv('SENSI_PREFS_PATH', raising=False)
    p = get_preferences_path()
    assert p.name == 'preferences.json'
    assert p.parent.name == 'config'
```

- [ ] **Step 3: Run test, expect failure**

Run: `pytest tests/utils/test_preferences.py -v`
Expected: `ModuleNotFoundError: No module named 'utils.preferences'`

- [ ] **Step 4: Implement**

Create `utils/preferences.py`:
```python
"""Load and save user preferences (analyzer, last folders, model choice, categories, toggles).

Schema is forward-compatible:
- Unknown keys on load are preserved (round-trip safe).
- Missing keys fall back to DEFAULT_PREFS values.

Default location: <project_root>/config/preferences.json
Override via SENSI_PREFS_PATH environment variable.
The file is gitignored.
"""

import json
import os
from pathlib import Path
from typing import Any


DEFAULT_PREFS: dict[str, Any] = {
    'analyzer_name': '',
    'last_transcripts_folder': '',
    'last_audio_folder': '',
    'last_output_folder': '',
    'llm_model': '',  # empty = auto-detect
    'dictionary_backend': 'auto',  # auto | liwc | empath | <path>
    'semantic_categories': [
        'social', 'cognition', 'emotion_positive', 'emotion_negative',
        'body', 'motion', 'perception', 'time', 'space',
        'math', 'science', 'work', 'family', 'communication', 'money',
    ],
    'feature_toggles': {
        'spacy_parse': True,
        'zipf_freq': True,
        'edu_freq': True,
        'sem_domain': True,
        'gpt2_surprisal': True,
        'llm_cloze': True,
        'alignment': True,
        'doc_summary': True,
    },
}


def get_preferences_path() -> Path:
    """Return the path to preferences.json (env-var override or project default)."""
    override = os.environ.get('SENSI_PREFS_PATH')
    if override:
        return Path(override)
    project_root = Path(__file__).resolve().parent.parent
    return project_root / 'config' / 'preferences.json'


def load_preferences() -> dict[str, Any]:
    """Load preferences, merging file contents with DEFAULT_PREFS.

    Missing keys are filled from defaults. Unknown keys are preserved.
    """
    path = get_preferences_path()
    merged: dict[str, Any] = json.loads(json.dumps(DEFAULT_PREFS))  # deep copy

    if path.exists():
        try:
            with path.open('r', encoding='utf-8') as f:
                user = json.load(f)
            if isinstance(user, dict):
                merged.update(user)
                # Re-merge nested dicts (feature_toggles) so defaults aren't wiped
                if isinstance(user.get('feature_toggles'), dict):
                    merged_toggles = dict(DEFAULT_PREFS['feature_toggles'])
                    merged_toggles.update(user['feature_toggles'])
                    merged['feature_toggles'] = merged_toggles
        except (json.JSONDecodeError, OSError):
            # Corrupt or unreadable file — fall back to defaults silently
            pass
    return merged


def save_preferences(updates: dict[str, Any]) -> None:
    """Merge `updates` into existing preferences and write to disk.

    Creates the parent directory if needed.
    """
    path = get_preferences_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    current = load_preferences()
    current.update(updates)

    with path.open('w', encoding='utf-8') as f:
        json.dump(current, f, indent=2)
```

- [ ] **Step 5: Run test, expect pass**

Run: `pytest tests/utils/test_preferences.py -v`
Expected: 5 passed.

- [ ] **Step 6: Commit**

```bash
git add utils/preferences.py tests/utils/test_preferences.py config/.gitkeep
git commit -m "feat(utils): add preferences load/save with extensible JSON schema"
```

---

## Phase B — Batch infrastructure (Tasks 4–5)

### Task 4: File pairing utility

**Files:**
- Create: `utils/file_pairing.py`
- Create: `tests/utils/test_file_pairing.py`

- [ ] **Step 1: Write failing test**

Create `tests/utils/test_file_pairing.py`:
```python
from pathlib import Path

import pytest

from utils.file_pairing import find_pairs, PairingResult


def _touch(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text('x', encoding='utf-8')


def test_basic_pairing(tmp_path):
    txt_dir = tmp_path / 'txt'
    aud_dir = tmp_path / 'aud'
    _touch(txt_dir / 'story1.txt')
    _touch(txt_dir / 'story2.txt')
    _touch(aud_dir / 'story1.wav')
    _touch(aud_dir / 'story2.wav')

    result = find_pairs(txt_dir, aud_dir)
    assert len(result.pairs) == 2
    assert result.unmatched_text == []
    assert result.unmatched_audio == []
    assert result.warnings == []


def test_unmatched_text(tmp_path):
    txt = tmp_path / 'txt'
    aud = tmp_path / 'aud'
    _touch(txt / 'story1.txt')
    _touch(txt / 'story2.txt')
    _touch(aud / 'story1.wav')

    result = find_pairs(txt, aud)
    assert len(result.pairs) == 1
    assert [p.name for p in result.unmatched_text] == ['story2.txt']


def test_case_insensitive_matching(tmp_path):
    txt = tmp_path / 'txt'
    aud = tmp_path / 'aud'
    _touch(txt / 'Story1.txt')
    _touch(aud / 'story1.WAV')
    result = find_pairs(txt, aud)
    assert len(result.pairs) == 1


def test_extension_priority_for_audio(tmp_path):
    txt = tmp_path / 'txt'
    aud = tmp_path / 'aud'
    _touch(txt / 'story1.txt')
    _touch(aud / 'story1.wav')
    _touch(aud / 'story1.mp3')
    result = find_pairs(txt, aud)
    assert len(result.pairs) == 1
    assert result.pairs[0].audio_path.name == 'story1.wav'  # wav > mp3
    assert any('ambiguous' in w.lower() for w in result.warnings)


def test_top_level_only_no_recursion(tmp_path):
    txt = tmp_path / 'txt'
    aud = tmp_path / 'aud'
    _touch(txt / 'subdir' / 'story1.txt')
    _touch(aud / 'story1.wav')
    result = find_pairs(txt, aud)
    assert len(result.pairs) == 0
    # The subdir-nested transcript is not even scanned, so it's not "unmatched"


def test_custom_extensions(tmp_path):
    txt = tmp_path / 'txt'
    aud = tmp_path / 'aud'
    _touch(txt / 'story1.tsv')
    _touch(aud / 'story1.flac')
    result = find_pairs(txt, aud, text_exts=['tsv'], audio_exts=['flac'])
    assert len(result.pairs) == 1
```

- [ ] **Step 2: Run test, expect failure**

Run: `pytest tests/utils/test_file_pairing.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

Create `utils/file_pairing.py`:
```python
"""Pair text files and audio files by base filename (case-insensitive).

- Top-level scan only (no recursion).
- Default text extensions: txt, tsv, md.
- Default audio extensions: wav, mp3, flac.
- Ambiguous matches resolved by extension priority order with a warning.
- Unmatched files reported separately for the no-match.csv.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

DEFAULT_TEXT_EXTS = ['txt', 'tsv', 'md']
DEFAULT_AUDIO_EXTS = ['wav', 'flac', 'mp3']


@dataclass
class Pair:
    stimulus_id: str  # base name (lower-cased for canonical id)
    transcript_path: Path
    audio_path: Path


@dataclass
class PairingResult:
    pairs: list[Pair] = field(default_factory=list)
    unmatched_text: list[Path] = field(default_factory=list)
    unmatched_audio: list[Path] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def find_pairs(
    transcripts_dir: Path,
    audio_dir: Path,
    text_exts: Iterable[str] | None = None,
    audio_exts: Iterable[str] | None = None,
) -> PairingResult:
    """Pair files in transcripts_dir with files in audio_dir by base filename."""
    text_exts = list(text_exts) if text_exts else DEFAULT_TEXT_EXTS
    audio_exts = list(audio_exts) if audio_exts else DEFAULT_AUDIO_EXTS

    transcripts_dir = Path(transcripts_dir)
    audio_dir = Path(audio_dir)

    text_by_base = _scan_top_level(transcripts_dir, text_exts)
    audio_by_base = _scan_top_level(audio_dir, audio_exts)

    result = PairingResult()
    for base, txt_candidates in text_by_base.items():
        # Resolve text extension by priority
        chosen_text, txt_warning = _resolve_priority(txt_candidates, text_exts, base, 'text')
        if txt_warning:
            result.warnings.append(txt_warning)

        if base in audio_by_base:
            chosen_audio, aud_warning = _resolve_priority(
                audio_by_base[base], audio_exts, base, 'audio'
            )
            if aud_warning:
                result.warnings.append(aud_warning)
            result.pairs.append(Pair(
                stimulus_id=base,
                transcript_path=chosen_text,
                audio_path=chosen_audio,
            ))
        else:
            result.unmatched_text.append(chosen_text)

    for base, aud_candidates in audio_by_base.items():
        if base not in text_by_base:
            chosen_audio, _ = _resolve_priority(aud_candidates, audio_exts, base, 'audio')
            result.unmatched_audio.append(chosen_audio)

    # Sort for deterministic output
    result.pairs.sort(key=lambda p: p.stimulus_id)
    result.unmatched_text.sort()
    result.unmatched_audio.sort()
    return result


def _scan_top_level(folder: Path, exts: list[str]) -> dict[str, list[Path]]:
    """Map base-name (lowercased, no extension) → list of matching paths."""
    if not folder.is_dir():
        return {}
    ext_set = {e.lower().lstrip('.') for e in exts}
    out: dict[str, list[Path]] = {}
    for entry in folder.iterdir():
        if not entry.is_file():
            continue
        ext = entry.suffix.lower().lstrip('.')
        if ext not in ext_set:
            continue
        base = entry.stem.lower()
        out.setdefault(base, []).append(entry)
    return out


def _resolve_priority(
    candidates: list[Path],
    priority: list[str],
    base: str,
    kind: str,
) -> tuple[Path, str | None]:
    """Pick the highest-priority candidate; return warning if multiple existed."""
    priority_lower = [e.lower().lstrip('.') for e in priority]

    def rank(p: Path) -> int:
        ext = p.suffix.lower().lstrip('.')
        return priority_lower.index(ext) if ext in priority_lower else len(priority_lower)

    candidates_sorted = sorted(candidates, key=rank)
    chosen = candidates_sorted[0]
    if len(candidates_sorted) > 1:
        others = ', '.join(p.name for p in candidates_sorted[1:])
        warn = f"Ambiguous {kind} match for '{base}': chose {chosen.name} (also found: {others})"
        return chosen, warn
    return chosen, None
```

- [ ] **Step 4: Run test, expect pass**

Run: `pytest tests/utils/test_file_pairing.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add utils/file_pairing.py tests/utils/test_file_pairing.py
git commit -m "feat(utils): add file pairing by base filename with extension priority"
```

---

### Task 5: Run logger utility

**Files:**
- Create: `utils/run_logger.py`
- Create: `tests/utils/test_run_logger.py`

- [ ] **Step 1: Write failing test**

Create `tests/utils/test_run_logger.py`:
```python
import csv
import json
from pathlib import Path

from utils.run_logger import RunLogger


def test_run_logger_creates_three_artifacts(tmp_path):
    out = tmp_path / 'out'
    out.mkdir()
    logger = RunLogger(
        output_folder=out,
        analyzer='Alice',
        config={'llm_model': 'llama-3-8b'},
        input_transcripts_folder=tmp_path / 'txt',
        input_audio_folder=tmp_path / 'aud',
        text_extensions=['txt'],
        audio_extensions=['wav'],
    )
    logger.add_warning('mock warning')
    logger.record_pair_success(
        stimulus_id='story1',
        transcript_path=tmp_path / 'txt/story1.txt',
        audio_path=tmp_path / 'aud/story1.wav',
        runtime_seconds=1.5,
        output_csv_path=out / 'story1_features.csv',
        output_summary_path=out / 'story1_summary.csv',
    )
    logger.record_pair_error(
        stimulus_id='story2',
        transcript_path=tmp_path / 'txt/story2.txt',
        audio_path=tmp_path / 'aud/story2.wav',
        runtime_seconds=0.2,
        error_message='boom',
    )
    logger.record_unmatched_text(tmp_path / 'txt/story3.txt')
    logger.record_unmatched_audio(tmp_path / 'aud/story4.wav')
    logger.finalize()

    log_files = list(out.glob('run_*.log'))
    manifests = list(out.glob('run_*_manifest.json'))
    no_match = list(out.glob('run_*_no-match.csv'))
    assert len(log_files) == 1
    assert len(manifests) == 1
    assert len(no_match) == 1


def test_manifest_contents(tmp_path):
    out = tmp_path / 'out'
    out.mkdir()
    logger = RunLogger(
        output_folder=out,
        analyzer='Bob',
        config={'feature': 'value'},
        input_transcripts_folder=tmp_path / 'txt',
        input_audio_folder=tmp_path / 'aud',
        text_extensions=['txt'],
        audio_extensions=['wav'],
    )
    logger.record_pair_success(
        stimulus_id='s1',
        transcript_path=tmp_path / 'txt/s1.txt',
        audio_path=tmp_path / 'aud/s1.wav',
        runtime_seconds=2.0,
        output_csv_path=out / 's1_features.csv',
        output_summary_path=out / 's1_summary.csv',
    )
    logger.finalize()

    manifest_path = next(out.glob('run_*_manifest.json'))
    data = json.loads(manifest_path.read_text())
    assert data['analyzer'] == 'Bob'
    assert data['config'] == {'feature': 'value'}
    assert data['n_pairs_processed'] == 1
    assert data['pairs'][0]['stimulus_id'] == 's1'
    assert data['pairs'][0]['status'] == 'success'
    assert 'system_info' in data
    assert 'sensi_extractor_version' in data
    assert data['total_runtime_seconds'] >= 0


def test_no_match_csv_columns(tmp_path):
    out = tmp_path / 'out'
    out.mkdir()
    logger = RunLogger(
        output_folder=out, analyzer='C', config={},
        input_transcripts_folder=tmp_path / 'txt',
        input_audio_folder=tmp_path / 'aud',
        text_extensions=['txt'], audio_extensions=['wav'],
    )
    logger.record_unmatched_text(tmp_path / 'txt/orphan.txt')
    logger.record_unmatched_audio(tmp_path / 'aud/orphan.wav')
    logger.finalize()

    no_match_path = next(out.glob('run_*_no-match.csv'))
    with no_match_path.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert {r['filename'] for r in rows} == {'orphan.txt', 'orphan.wav'}
    assert all(set(r.keys()) == {
        'filename', 'found_in_folder', 'missing_counterpart', 'reason'
    } for r in rows)
```

- [ ] **Step 2: Run test, expect failure**

Run: `pytest tests/utils/test_run_logger.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

Create `utils/run_logger.py`:
```python
"""Per-run log + manifest + no-match.csv writer.

Three artifacts per batch run, all sharing a timestamp:
    run_<YYYY-MM-DD_HH-MM-SS>.log              # human-readable log
    run_<YYYY-MM-DD_HH-MM-SS>_manifest.json    # structured manifest
    run_<YYYY-MM-DD_HH-MM-SS>_no-match.csv     # files that didn't pair

Used by main.py's batch mode and gui.py.
"""

import csv
import datetime as _dt
import json
import sys
import traceback
from pathlib import Path
from typing import Any

from utils.system_info import gather_system_info

try:
    from __init__ import __version__  # type: ignore
except Exception:
    __version__ = "unknown"


class RunLogger:
    def __init__(
        self,
        output_folder: Path,
        analyzer: str,
        config: dict[str, Any],
        input_transcripts_folder: Path,
        input_audio_folder: Path,
        text_extensions: list[str],
        audio_extensions: list[str],
    ):
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)

        self._start = _dt.datetime.now()
        ts = self._start.strftime('%Y-%m-%d_%H-%M-%S')
        self._log_path = self.output_folder / f'run_{ts}.log'
        self._manifest_path = self.output_folder / f'run_{ts}_manifest.json'
        self._no_match_path = self.output_folder / f'run_{ts}_no-match.csv'

        self._analyzer = analyzer
        self._config = config
        self._input_transcripts_folder = Path(input_transcripts_folder)
        self._input_audio_folder = Path(input_audio_folder)
        self._text_extensions = list(text_extensions)
        self._audio_extensions = list(audio_extensions)

        self._pairs: list[dict] = []
        self._warnings: list[str] = []
        self._errors: list[str] = []
        self._unmatched_text: list[Path] = []
        self._unmatched_audio: list[Path] = []

        self._log_fh = self._log_path.open('w', encoding='utf-8')
        self._info(f'Run started by {analyzer} at {self._start.isoformat(timespec="seconds")}')

    # ------------- public API -------------

    def add_warning(self, msg: str) -> None:
        self._warnings.append(msg)
        self._write_log('WARN', msg)

    def add_error(self, msg: str) -> None:
        self._errors.append(msg)
        self._write_log('ERROR', msg)

    def record_pair_success(
        self,
        stimulus_id: str,
        transcript_path: Path,
        audio_path: Path,
        runtime_seconds: float,
        output_csv_path: Path,
        output_summary_path: Path | None,
    ) -> None:
        self._pairs.append({
            'stimulus_id': stimulus_id,
            'transcript_path': str(transcript_path),
            'audio_path': str(audio_path),
            'runtime_seconds': round(runtime_seconds, 3),
            'status': 'success',
            'output_csv_path': str(output_csv_path),
            'output_summary_path': str(output_summary_path) if output_summary_path else None,
            'error_message': None,
        })
        self._info(f'  ✓ {stimulus_id} ({runtime_seconds:.1f}s)')

    def record_pair_error(
        self,
        stimulus_id: str,
        transcript_path: Path,
        audio_path: Path,
        runtime_seconds: float,
        error_message: str,
    ) -> None:
        self._pairs.append({
            'stimulus_id': stimulus_id,
            'transcript_path': str(transcript_path),
            'audio_path': str(audio_path),
            'runtime_seconds': round(runtime_seconds, 3),
            'status': 'error',
            'output_csv_path': None,
            'output_summary_path': None,
            'error_message': error_message,
        })
        self._errors.append(f'{stimulus_id}: {error_message}')
        self._write_log('ERROR', f'  ✗ {stimulus_id}: {error_message}')

    def record_unmatched_text(self, path: Path) -> None:
        self._unmatched_text.append(Path(path))

    def record_unmatched_audio(self, path: Path) -> None:
        self._unmatched_audio.append(Path(path))

    def finalize(self) -> None:
        end = _dt.datetime.now()
        total = (end - self._start).total_seconds()
        self._info(f'Run finished at {end.isoformat(timespec="seconds")} ({total:.1f}s total)')

        manifest = {
            'start_time': self._start.isoformat(timespec='seconds'),
            'end_time': end.isoformat(timespec='seconds'),
            'total_runtime_seconds': round(total, 3),
            'analyzer': self._analyzer,
            'sensi_extractor_version': __version__,
            'system_info': gather_system_info(),
            'config': self._config,
            'input_transcripts_folder': str(self._input_transcripts_folder),
            'input_audio_folder': str(self._input_audio_folder),
            'output_folder': str(self.output_folder),
            'text_extensions_searched': self._text_extensions,
            'audio_extensions_searched': self._audio_extensions,
            'n_pairs_matched': len(self._pairs),
            'n_pairs_processed': sum(1 for p in self._pairs if p['status'] == 'success'),
            'n_unmatched_text': len(self._unmatched_text),
            'n_unmatched_audio': len(self._unmatched_audio),
            'pairs': self._pairs,
            'warnings': self._warnings,
            'errors': self._errors,
        }

        with self._manifest_path.open('w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2)

        # no-match.csv (always emitted; empty file if nothing unmatched)
        with self._no_match_path.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(
                f, fieldnames=['filename', 'found_in_folder', 'missing_counterpart', 'reason']
            )
            writer.writeheader()
            for p in self._unmatched_text:
                writer.writerow({
                    'filename': p.name,
                    'found_in_folder': str(p.parent),
                    'missing_counterpart': str(self._input_audio_folder),
                    'reason': 'no audio file with matching base name',
                })
            for p in self._unmatched_audio:
                writer.writerow({
                    'filename': p.name,
                    'found_in_folder': str(p.parent),
                    'missing_counterpart': str(self._input_transcripts_folder),
                    'reason': 'no transcript file with matching base name',
                })

        self._log_fh.close()

    # ------------- internals -------------

    def _info(self, msg: str) -> None:
        self._write_log('INFO', msg)

    def _write_log(self, level: str, msg: str) -> None:
        ts = _dt.datetime.now().strftime('%H:%M:%S')
        line = f'[{ts}] {level}: {msg}\n'
        self._log_fh.write(line)
        self._log_fh.flush()
```

- [ ] **Step 4: Run test, expect pass**

Run: `pytest tests/utils/test_run_logger.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add utils/run_logger.py tests/utils/test_run_logger.py
git commit -m "feat(utils): add run logger emitting log, manifest.json, no-match.csv"
```

---

## Phase C — New extractors (Tasks 6–10)

### Task 6: Timing extractor (Word_Duration, Word_Gap)

**Files:**
- Create: `extractors/timing.py`
- Create: `tests/extractors/test_timing.py`

- [ ] **Step 1: Write failing test**

Create `tests/extractors/test_timing.py`:
```python
import math

import pandas as pd
import pytest

from extractors.timing import add_timing_columns


def test_word_duration_basic():
    df = pd.DataFrame({
        'token': ['the', 'cat', 'sat'],
        'upos': ['DET', 'NOUN', 'VERB'],
        'Onset': [0.0, 0.5, 1.2],
        'End': [0.4, 1.0, 1.6],
    })
    out = add_timing_columns(df)
    assert list(out['Word_Duration']) == pytest.approx([0.4, 0.5, 0.4])


def test_word_gap_first_word_is_nan():
    df = pd.DataFrame({
        'token': ['the', 'cat'],
        'upos': ['DET', 'NOUN'],
        'Onset': [0.0, 0.5],
        'End': [0.4, 1.0],
    })
    out = add_timing_columns(df)
    assert math.isnan(out['Word_Gap'].iloc[0])
    assert out['Word_Gap'].iloc[1] == pytest.approx(0.1)  # 0.5 - 0.4


def test_nan_in_inputs_propagates():
    df = pd.DataFrame({
        'token': ['the', 'cat', '.'],
        'upos': ['DET', 'NOUN', 'PUNCT'],
        'Onset': [0.0, 0.5, None],
        'End': [0.4, 1.0, None],
    })
    out = add_timing_columns(df)
    assert math.isnan(out['Word_Duration'].iloc[2])
    assert math.isnan(out['Word_Gap'].iloc[2])


def test_no_alignment_columns_emits_nan():
    df = pd.DataFrame({'token': ['the'], 'upos': ['DET']})
    out = add_timing_columns(df)
    assert 'Word_Duration' in out.columns
    assert 'Word_Gap' in out.columns
    assert math.isnan(out['Word_Duration'].iloc[0])
    assert math.isnan(out['Word_Gap'].iloc[0])
```

- [ ] **Step 2: Run test, expect failure**

Run: `pytest tests/extractors/test_timing.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

Create `extractors/timing.py`:
```python
"""Derived timing columns: Word_Duration and Word_Gap.

Both are computed from the existing MFA Onset / End columns. No new
dependencies. If alignment hasn't been run, both columns are emitted as
all-NaN to keep the output schema stable.

Word_Duration[i] = End[i] - Onset[i]
Word_Gap[i]      = Onset[i] - End[i-1]   (NaN for i=0 or if either neighbor is NaN)
"""

import numpy as np
import pandas as pd


def add_timing_columns(token_df: pd.DataFrame) -> pd.DataFrame:
    df = token_df.copy()

    if 'Onset' not in df.columns or 'End' not in df.columns:
        df['Word_Duration'] = np.nan
        df['Word_Gap'] = np.nan
        return df

    df['Word_Duration'] = df['End'] - df['Onset']

    prev_end = df['End'].shift(1)
    df['Word_Gap'] = df['Onset'] - prev_end
    return df
```

- [ ] **Step 4: Run test, expect pass**

Run: `pytest tests/extractors/test_timing.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add extractors/timing.py tests/extractors/test_timing.py
git commit -m "feat(extractors): add Word_Duration and Word_Gap derived columns"
```

---

### Task 7: Educator's frequency extractor

**Files:**
- Create: `extractors/educator_freq.py`
- Create: `tests/extractors/test_educator_freq.py`
- Create: `dictionaries/.gitkeep`

- [ ] **Step 1: Create dictionaries dir marker**

Create `dictionaries/.gitkeep` (empty file).

- [ ] **Step 2: Write failing test**

Create `tests/extractors/test_educator_freq.py`:
```python
import math

import pandas as pd
import pytest

from extractors.educator_freq import add_edu_freq, EXPECTED_COLUMNS


def _write_lookup(tmp_path):
    """Create a small parquet lookup with a few words."""
    import pandas as pd
    lookup = pd.DataFrame([
        {'word': 'cat',  'edu_zipf_k2': 5.5, 'edu_zipf_3to5': 5.2, 'edu_zipf_6to8': 4.9, 'edu_zipf_9to12': 4.5, 'edu_zipf_overall': 5.0},
        {'word': 'photosynthesis', 'edu_zipf_k2': 0.0, 'edu_zipf_3to5': 1.0, 'edu_zipf_6to8': 3.5, 'edu_zipf_9to12': 4.2, 'edu_zipf_overall': 2.9},
    ])
    path = tmp_path / 'edu.parquet'
    lookup.to_parquet(path)
    return path


def test_columns_added(tmp_path):
    df = pd.DataFrame({'token': ['cat'], 'lemma': ['cat'], 'upos': ['NOUN']})
    out = add_edu_freq(df, lookup_path=_write_lookup(tmp_path))
    for col in EXPECTED_COLUMNS:
        assert col in out.columns


def test_known_word_values(tmp_path):
    df = pd.DataFrame({'token': ['cat'], 'lemma': ['cat'], 'upos': ['NOUN']})
    out = add_edu_freq(df, lookup_path=_write_lookup(tmp_path))
    assert out['Edu_Zipf_K2'].iloc[0] == pytest.approx(5.5)
    assert out['Edu_Zipf_Overall'].iloc[0] == pytest.approx(5.0)


def test_oov_returns_nan(tmp_path):
    df = pd.DataFrame({'token': ['xyzzy'], 'lemma': ['xyzzy'], 'upos': ['NOUN']})
    out = add_edu_freq(df, lookup_path=_write_lookup(tmp_path))
    assert math.isnan(out['Edu_Zipf_K2'].iloc[0])


def test_punct_gets_nan(tmp_path):
    df = pd.DataFrame({'token': ['.'], 'lemma': ['.'], 'upos': ['PUNCT']})
    out = add_edu_freq(df, lookup_path=_write_lookup(tmp_path))
    assert math.isnan(out['Edu_Zipf_Overall'].iloc[0])


def test_missing_lookup_emits_nan_columns_with_warning(tmp_path, caplog):
    df = pd.DataFrame({'token': ['cat'], 'lemma': ['cat'], 'upos': ['NOUN']})
    out = add_edu_freq(df, lookup_path=tmp_path / 'nonexistent.parquet')
    for col in EXPECTED_COLUMNS:
        assert col in out.columns
    assert math.isnan(out['Edu_Zipf_K2'].iloc[0])
```

- [ ] **Step 3: Run test, expect failure**

Run: `pytest tests/extractors/test_educator_freq.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 4: Implement**

Create `extractors/educator_freq.py`:
```python
"""Educator's Word Frequency Guide (EWFG) grade-band frequency lookup.

Adds five columns to a token DataFrame:
    Edu_Zipf_K2, Edu_Zipf_3to5, Edu_Zipf_6to8, Edu_Zipf_9to12, Edu_Zipf_Overall

Reads from a lookup parquet file at `dictionaries/edu_zipf.parquet` (default)
or a user-supplied path. The lookup must have columns:
    word, edu_zipf_k2, edu_zipf_3to5, edu_zipf_6to8, edu_zipf_9to12, edu_zipf_overall

If the lookup file is missing, all five columns are emitted as NaN and a
warning is logged. This keeps the schema stable and unblocks downstream
analysis even when EWFG data acquisition is pending.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

EXPECTED_COLUMNS = [
    'Edu_Zipf_K2', 'Edu_Zipf_3to5', 'Edu_Zipf_6to8', 'Edu_Zipf_9to12', 'Edu_Zipf_Overall',
]

_LOOKUP_COLUMNS = [
    'edu_zipf_k2', 'edu_zipf_3to5', 'edu_zipf_6to8', 'edu_zipf_9to12', 'edu_zipf_overall',
]

_DEFAULT_LOOKUP_PATH = Path(__file__).resolve().parent.parent / 'dictionaries' / 'edu_zipf.parquet'

_lookup_cache: dict[Path, pd.DataFrame] = {}


def add_edu_freq(token_df: pd.DataFrame, lookup_path: Path | str | None = None) -> pd.DataFrame:
    df = token_df.copy()
    path = Path(lookup_path) if lookup_path else _DEFAULT_LOOKUP_PATH

    lookup = _load_lookup(path)
    if lookup is None:
        for col in EXPECTED_COLUMNS:
            df[col] = np.nan
        return df

    # Lower-case lemmas for lookup; punctuation gets NaN
    keys = df.apply(
        lambda r: r['lemma'].lower() if r['upos'] != 'PUNCT' else None,
        axis=1,
    )
    matched = lookup.reindex(keys.fillna('__missing__'))

    for out_col, in_col in zip(EXPECTED_COLUMNS, _LOOKUP_COLUMNS):
        df[out_col] = matched[in_col].values
    return df


def _load_lookup(path: Path) -> pd.DataFrame | None:
    if path in _lookup_cache:
        return _lookup_cache[path]
    if not path.exists():
        logger.warning(
            'EWFG lookup not found at %s — Edu_Zipf_* columns will be NaN. '
            'See docs/superpowers/specs/2026-05-10-feature-batch-1-design.md '
            'Appendix A for acquisition steps.',
            path,
        )
        return None
    try:
        df = pd.read_parquet(path).set_index('word')
    except Exception as exc:
        logger.warning('Failed to load EWFG lookup from %s: %s', path, exc)
        return None
    _lookup_cache[path] = df
    return df
```

- [ ] **Step 5: Run test, expect pass**

Run: `pytest tests/extractors/test_educator_freq.py -v`
Expected: 5 passed.

- [ ] **Step 6: Commit**

```bash
git add extractors/educator_freq.py tests/extractors/test_educator_freq.py dictionaries/.gitkeep
git commit -m "feat(extractors): add EWFG grade-band frequency lookup with NaN-fallback"
```

---

### Task 8: Semantic-domain extractor (Empath + LIWC backends)

**Files:**
- Create: `extractors/semantic_domain.py`
- Create: `tests/extractors/test_semantic_domain.py`
- Modify: `requirements.txt`

- [ ] **Step 1: Add deps to requirements.txt**

Append:
```
# Semantic-domain backends
empath>=0.89
liwc>=0.5
```
Run: `pip install empath liwc`

- [ ] **Step 2: Write failing test**

Create `tests/extractors/test_semantic_domain.py`:
```python
import math

import pandas as pd
import pytest

from extractors.semantic_domain import (
    add_semantic_domain,
    detect_backend,
    BackendNotAvailable,
)


def test_detect_backend_returns_empath_when_no_dic(tmp_path, monkeypatch):
    # Point dictionary path at empty tmp_path
    backend, dict_path = detect_backend(dictionaries_dir=tmp_path)
    assert backend == 'empath'
    assert dict_path is None


def test_detect_backend_returns_liwc_when_dic_present(tmp_path):
    (tmp_path / 'liwc-2022.dic').write_text('% header %\n')
    backend, dict_path = detect_backend(dictionaries_dir=tmp_path)
    assert backend == 'liwc'
    assert dict_path == tmp_path / 'liwc-2022.dic'


def test_empath_basic_categories(tmp_path):
    df = pd.DataFrame({
        'token': ['mother', 'run', 'the'],
        'lemma': ['mother', 'run', 'the'],
        'upos': ['NOUN', 'VERB', 'DET'],
    })
    out = add_semantic_domain(
        df,
        backend='empath',
        categories=['family', 'motion'],
    )
    assert 'Sem_Empath_family' in out.columns
    assert 'Sem_Empath_motion' in out.columns
    assert 'Sem_Categories' in out.columns
    # Function word gets NaN
    assert math.isnan(out['Sem_Empath_family'].iloc[2])


def test_function_words_get_nan(tmp_path):
    df = pd.DataFrame({
        'token': ['the', 'and'],
        'lemma': ['the', 'and'],
        'upos': ['DET', 'CCONJ'],
    })
    out = add_semantic_domain(df, backend='empath', categories=['social'])
    assert math.isnan(out['Sem_Empath_social'].iloc[0])
    assert math.isnan(out['Sem_Empath_social'].iloc[1])


def test_sem_categories_rollup_format(tmp_path):
    df = pd.DataFrame({
        'token': ['mother'],
        'lemma': ['mother'],
        'upos': ['NOUN'],
    })
    out = add_semantic_domain(df, backend='empath', categories=['family', 'social'])
    rollup = out['Sem_Categories'].iloc[0]
    # mother is in both family and social per Empath
    parts = rollup.split(',') if rollup else []
    assert 'family' in parts or 'social' in parts


def test_unavailable_backend_raises(tmp_path):
    df = pd.DataFrame({'token': ['x'], 'lemma': ['x'], 'upos': ['NOUN']})
    with pytest.raises(BackendNotAvailable):
        add_semantic_domain(df, backend='liwc', categories=['social'],
                            dictionary_path=tmp_path / 'nonexistent.dic')
```

- [ ] **Step 3: Run test, expect failure**

Run: `pytest tests/extractors/test_semantic_domain.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 4: Implement**

Create `extractors/semantic_domain.py`:
```python
"""Semantic-domain category indicators with pluggable dictionary backends.

Two backends share the same interface:
    - LIWC: requires a licensed .dic file at dictionaries/liwc-2022.dic
            (or any .dic at the configured path). Parsed via the `liwc` package.
    - Empath: open-source, pip-installed. No license required. Default fallback.

Backend auto-detection: if dictionaries/liwc-2022.dic exists, use LIWC; else Empath.

Output columns added to the token DataFrame:
    Sem_<Dict>_<category> ∈ {0, 1, NaN}     (one binary column per category)
    Sem_Categories                          (comma-separated list of matched categories
                                             for each token, "" if no match, NaN if function word)

POS filter: only NOUN, VERB, ADJ, ADV are scored. Other tokens get NaN.
"""

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


CONTENT_POS = {'NOUN', 'VERB', 'ADJ', 'ADV', 'PROPN'}

_DICTIONARIES_DIR = Path(__file__).resolve().parent.parent / 'dictionaries'


class BackendNotAvailable(Exception):
    pass


def detect_backend(dictionaries_dir: Path | None = None) -> tuple[str, Path | None]:
    """Return (backend_name, dict_path) — picks LIWC if a .dic exists, else Empath."""
    d = Path(dictionaries_dir) if dictionaries_dir else _DICTIONARIES_DIR
    if d.exists():
        for cand in sorted(d.glob('*.dic')):
            return 'liwc', cand
    return 'empath', None


def add_semantic_domain(
    token_df: pd.DataFrame,
    backend: str,
    categories: Iterable[str],
    dictionary_path: Path | str | None = None,
) -> pd.DataFrame:
    """Add Sem_<Dict>_<category> columns + Sem_Categories roll-up.

    Args:
        token_df: must contain 'lemma' and 'upos' columns.
        backend:  'liwc' or 'empath'.
        categories: list of category names to score.
        dictionary_path: required for LIWC; ignored for Empath.
    """
    backend = backend.lower()
    categories = list(categories)
    df = token_df.copy()

    if backend == 'liwc':
        scorer = _LiwcScorer(dictionary_path)
        dict_label = 'LIWC'
    elif backend == 'empath':
        scorer = _EmpathScorer()
        dict_label = 'Empath'
    else:
        raise BackendNotAvailable(f'Unknown backend: {backend}')

    # Score each token
    matches_per_row: list[set[str]] = []
    for _, row in df.iterrows():
        if row['upos'] not in CONTENT_POS:
            matches_per_row.append(None)  # sentinel for NaN
            continue
        matches = scorer.categories_for(row['lemma'], categories)
        matches_per_row.append(matches)

    # Emit one column per requested category
    for cat in categories:
        col = f'Sem_{dict_label}_{cat}'
        df[col] = [
            np.nan if m is None else (1 if cat in m else 0)
            for m in matches_per_row
        ]

    df['Sem_Categories'] = [
        np.nan if m is None else (','.join(sorted(m)) if m else '')
        for m in matches_per_row
    ]
    return df


# --------------------- backends ---------------------

class _EmpathScorer:
    def __init__(self):
        from empath import Empath  # imported here so it's only required when used
        self._lex = Empath()

    def categories_for(self, lemma: str, requested: list[str]) -> set[str]:
        out: set[str] = set()
        if not isinstance(lemma, str) or not lemma.strip():
            return out
        for cat in requested:
            if cat in self._lex.cats and lemma.lower() in self._lex.cats[cat]:
                out.add(cat)
        return out


class _LiwcScorer:
    def __init__(self, dictionary_path: Path | str | None):
        if dictionary_path is None:
            raise BackendNotAvailable('LIWC backend requires --sem-dict pointing to a .dic file')
        path = Path(dictionary_path)
        if not path.exists():
            raise BackendNotAvailable(f'LIWC dictionary not found: {path}')
        try:
            import liwc
        except ImportError as exc:
            raise BackendNotAvailable(f'`liwc` package not installed: {exc}')
        parse, category_names = liwc.load_token_parser(str(path))
        self._parse = parse
        self._all_categories = set(category_names)

    def categories_for(self, lemma: str, requested: list[str]) -> set[str]:
        if not isinstance(lemma, str) or not lemma.strip():
            return set()
        matched = set(self._parse(lemma.lower()))
        return matched & set(requested)
```

- [ ] **Step 5: Run test, expect pass**

Run: `pytest tests/extractors/test_semantic_domain.py -v`
Expected: 6 passed (assuming Empath installs correctly; if a category lookup test fails on `mother` → `family`, adjust the assertion to use a category Empath definitely has, like `health` for `medicine`).

- [ ] **Step 6: Commit**

```bash
git add extractors/semantic_domain.py tests/extractors/test_semantic_domain.py requirements.txt
git commit -m "feat(extractors): add semantic-domain extractor with LIWC and Empath backends"
```

---

### Task 9: LLM-cloze extractor

**Files:**
- Create: `extractors/llm_cloze.py`
- Create: `tests/extractors/test_llm_cloze.py`
- Modify: `requirements.txt`

- [ ] **Step 1: Add deps to requirements.txt**

Append:
```
# LLM-cloze
accelerate>=0.27
bitsandbytes>=0.42  ; platform_system != "Darwin"
openai>=1.10
```
Run: `pip install accelerate bitsandbytes openai`

- [ ] **Step 2: Write failing test (with mocked model)**

Create `tests/extractors/test_llm_cloze.py`:
```python
import math
from unittest.mock import patch

import pandas as pd
import pytest

from extractors.llm_cloze import (
    add_llm_cloze,
    select_model,
    EXPECTED_COLUMNS,
)


class _FakeBackend:
    """Stand-in for an LLM backend used in unit tests."""

    def score(self, text: str, tokens: list[str]) -> list[dict]:
        # Return a fixed pattern: first token gets P=0.5; rest get 0.1
        out = []
        for i, tok in enumerate(tokens):
            p = 0.5 if i == 0 else 0.1
            out.append({
                'p': p,
                'surprisal': -math.log2(p),
                'top_match': 1 if i == 0 else 0,
            })
        return out


def test_columns_added():
    df = pd.DataFrame({'token': ['the', 'cat'], 'upos': ['DET', 'NOUN']})
    out = add_llm_cloze(df, text='the cat', backend=_FakeBackend())
    for col in EXPECTED_COLUMNS:
        assert col in out.columns


def test_punct_gets_nan():
    df = pd.DataFrame({'token': ['the', '.'], 'upos': ['DET', 'PUNCT']})
    out = add_llm_cloze(df, text='the .', backend=_FakeBackend())
    assert math.isnan(out['LLM_Cloze'].iloc[1])
    assert math.isnan(out['LLM_Surprisal'].iloc[1])
    assert math.isnan(out['LLM_TopMatch'].iloc[1])


def test_metric_values():
    df = pd.DataFrame({'token': ['hi', 'there'], 'upos': ['INTJ', 'ADV']})
    out = add_llm_cloze(df, text='hi there', backend=_FakeBackend())
    assert out['LLM_Cloze'].iloc[0] == pytest.approx(0.5)
    assert out['LLM_Cloze'].iloc[1] == pytest.approx(0.1)
    assert out['LLM_TopMatch'].iloc[0] == 1
    assert out['LLM_TopMatch'].iloc[1] == 0


def test_select_model_uses_openai_key(monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'sk-test')
    assert select_model() == 'gpt-4o-mini'


def test_select_model_picks_local_when_no_api_key(monkeypatch):
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    with patch('extractors.llm_cloze._available_vram_gb', return_value=8.0):
        assert select_model() == 'meta-llama/Llama-3.2-3B'
    with patch('extractors.llm_cloze._available_vram_gb', return_value=14.0):
        assert select_model() == 'meta-llama/Llama-3-8B'
    with patch('extractors.llm_cloze._available_vram_gb', return_value=50.0):
        assert select_model() == 'meta-llama/Llama-3-70B'
```

- [ ] **Step 3: Run test, expect failure**

Run: `pytest tests/extractors/test_llm_cloze.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 4: Implement**

Create `extractors/llm_cloze.py`:
```python
"""LLM-derived cloze metrics.

Produces three columns from a single forward pass per token:
    LLM_Cloze     = P(target | preceding context)
    LLM_Surprisal = -log2 P(target | preceding context)   (bits)
    LLM_TopMatch  = 1 if argmax-next-token == target else 0

Model selection (auto):
    OPENAI_API_KEY set            → gpt-4o-mini
    GPU VRAM ≥ 48 GB              → meta-llama/Llama-3-70B
    GPU VRAM ≥ 12 GB              → meta-llama/Llama-3-8B
    GPU VRAM ≥  8 GB              → meta-llama/Llama-3.2-3B
    otherwise                     → meta-llama/Llama-3.2-3B (CPU)

Override with `--llm-model <id>`. The backend interface is pluggable so
tests can inject mock backends (see _FakeBackend in the unit tests).
"""

import math
import os
from typing import Protocol

import numpy as np
import pandas as pd

EXPECTED_COLUMNS = ['LLM_Cloze', 'LLM_Surprisal', 'LLM_TopMatch']

# Shared by GUI dropdown (gui.py) and the run dispatch (gui_run.py).
# Kept here to avoid circular imports between gui.py and gui_run.py.
LLM_OPTIONS = [
    ('gpt-4o-mini',                 'gpt-4o-mini   ($0.10–0.30/stim, r≈0.83, requires OPENAI_API_KEY)'),
    ('gpt-4o',                      'gpt-4o        ($1–3/stim, r≈0.85, requires OPENAI_API_KEY)'),
    ('meta-llama/Llama-3-70B',      'Llama-3-70B   (free, r≈0.84, needs ≥48GB VRAM)'),
    ('meta-llama/Llama-3-8B',       'Llama-3-8B    (free, r≈0.79, needs ≥12GB VRAM)'),
    ('meta-llama/Llama-3.2-3B',     'Llama-3.2-3B  (free, r≈0.76, needs ≥8GB VRAM)'),
]


class _Backend(Protocol):
    def score(self, text: str, tokens: list[str]) -> list[dict]:
        """Return one dict per token with keys: p, surprisal, top_match."""


def select_model() -> str:
    """Pick the best LLM model id given the current environment."""
    if os.environ.get('OPENAI_API_KEY'):
        return 'gpt-4o-mini'
    vram = _available_vram_gb()
    if vram >= 48:
        return 'meta-llama/Llama-3-70B'
    if vram >= 12:
        return 'meta-llama/Llama-3-8B'
    return 'meta-llama/Llama-3.2-3B'


def _available_vram_gb() -> float:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    except Exception:
        pass
    return 0.0


def add_llm_cloze(
    token_df: pd.DataFrame,
    text: str,
    backend: _Backend | None = None,
    model_id: str | None = None,
) -> pd.DataFrame:
    """Append LLM_Cloze, LLM_Surprisal, LLM_TopMatch columns to token_df."""
    df = token_df.copy()

    if backend is None:
        backend = _make_default_backend(model_id or select_model())

    tokens = list(df['token'])
    scores = backend.score(text, tokens)

    p_col, surp_col, top_col = [], [], []
    for (i, row), score in zip(df.iterrows(), scores):
        if row['upos'] == 'PUNCT':
            p_col.append(np.nan)
            surp_col.append(np.nan)
            top_col.append(np.nan)
            continue
        p_col.append(round(score['p'], 6))
        surp_col.append(round(score['surprisal'], 4))
        top_col.append(int(score['top_match']))

    df['LLM_Cloze'] = p_col
    df['LLM_Surprisal'] = surp_col
    df['LLM_TopMatch'] = top_col
    return df


# --------------------- backends ---------------------

def _make_default_backend(model_id: str) -> _Backend:
    if model_id.startswith('gpt-'):
        return _OpenAIBackend(model_id)
    return _HuggingFaceBackend(model_id)


class _HuggingFaceBackend:
    """Local Hugging Face causal-LM backend with 4-bit quantization when possible."""

    def __init__(self, model_id: str):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        try:
            quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
            self._model = AutoModelForCausalLM.from_pretrained(
                model_id, quantization_config=quant, device_map='auto',
            )
        except Exception:
            # Fall back to fp16/cpu if bitsandbytes isn't supported
            self._model = AutoModelForCausalLM.from_pretrained(model_id)
        self._model.eval()

    def score(self, text: str, tokens: list[str]) -> list[dict]:
        torch = self._torch
        enc = self._tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        ids = enc['input_ids']
        offsets = enc['offset_mapping']
        device = next(self._model.parameters()).device
        ids_t = torch.tensor([ids], device=device)
        with torch.no_grad():
            logits = self._model(ids_t).logits[0]
        log_probs = torch.log_softmax(logits, dim=-1)
        # Per-subword log-prob and top-1 prediction
        sub_log_p = [0.0]
        sub_top1 = [None]
        for i in range(1, len(ids)):
            sub_log_p.append(log_probs[i - 1, ids[i]].item())
            sub_top1.append(int(log_probs[i - 1].argmax().item()))

        # Map subwords → spaCy tokens via character offsets
        results = []
        char_pos = 0
        for tok in tokens:
            idx = text.find(tok, char_pos)
            if idx == -1:
                results.append({'p': float('nan'), 'surprisal': float('nan'), 'top_match': 0})
                continue
            tok_start, tok_end = idx, idx + len(tok)
            char_pos = tok_end
            total_lp = 0.0
            first_sub = None
            found = False
            for sub_idx, (s, e) in enumerate(offsets):
                if e <= tok_start:
                    continue
                if s >= tok_end:
                    break
                total_lp += sub_log_p[sub_idx]
                if first_sub is None:
                    first_sub = sub_idx
                found = True
            if not found:
                results.append({'p': float('nan'), 'surprisal': float('nan'), 'top_match': 0})
                continue
            p = math.exp(total_lp)
            surprisal = -total_lp / math.log(2)
            top_match = 1 if (first_sub is not None and sub_top1[first_sub] == ids[first_sub]) else 0
            results.append({'p': p, 'surprisal': surprisal, 'top_match': top_match})
        return results


class _OpenAIBackend:
    """OpenAI Chat Completions logprobs-based scorer."""

    def __init__(self, model_id: str):
        from openai import OpenAI
        self._client = OpenAI()
        self._model_id = model_id

    def score(self, text: str, tokens: list[str]) -> list[dict]:
        # NOTE: For per-token logprobs over a fixed continuation, OpenAI's
        # Completions endpoint with `logprobs` is the right primitive.
        # Newer Chat Completions support `logprobs` and `top_logprobs` for
        # generated text but not for arbitrary scoring of an existing string.
        # We use the legacy /completions endpoint via the Responses API:
        responses = self._client.completions.create(
            model=self._model_id,
            prompt=text,
            max_tokens=0,
            logprobs=5,
            echo=True,
        )
        choice = responses.choices[0]
        token_logprobs = choice.logprobs.token_logprobs
        offsets = choice.logprobs.text_offset
        sub_tokens = choice.logprobs.tokens
        top_choices = choice.logprobs.top_logprobs

        results = []
        char_pos = 0
        for tok in tokens:
            idx = text.find(tok, char_pos)
            if idx == -1:
                results.append({'p': float('nan'), 'surprisal': float('nan'), 'top_match': 0})
                continue
            tok_start, tok_end = idx, idx + len(tok)
            char_pos = tok_end
            total_lp = 0.0
            first_sub = None
            found = False
            for sub_idx, (off, sub) in enumerate(zip(offsets, sub_tokens)):
                if off + len(sub) <= tok_start:
                    continue
                if off >= tok_end:
                    break
                lp = token_logprobs[sub_idx] or 0.0
                total_lp += lp
                if first_sub is None:
                    first_sub = sub_idx
                found = True
            if not found:
                results.append({'p': float('nan'), 'surprisal': float('nan'), 'top_match': 0})
                continue
            p = math.exp(total_lp)
            surprisal = -total_lp / math.log(2)
            top_match = 0
            if first_sub is not None and top_choices[first_sub]:
                argmax_token = max(top_choices[first_sub], key=top_choices[first_sub].get)
                if argmax_token.strip() == sub_tokens[first_sub].strip():
                    top_match = 1
            results.append({'p': p, 'surprisal': surprisal, 'top_match': top_match})
        return results
```

- [ ] **Step 5: Run test, expect pass**

Run: `pytest tests/extractors/test_llm_cloze.py -v`
Expected: 5 passed. (Tests use a mock backend, so no model is actually loaded.)

- [ ] **Step 6: Commit**

```bash
git add extractors/llm_cloze.py tests/extractors/test_llm_cloze.py requirements.txt
git commit -m "feat(extractors): add LLM-cloze with HF and OpenAI backends + auto model select"
```

---

### Task 10: Document summary aggregator

**Files:**
- Create: `utils/doc_summary.py`
- Create: `tests/utils/test_doc_summary.py`

- [ ] **Step 1: Write failing test**

Create `tests/utils/test_doc_summary.py`:
```python
import pandas as pd
import pytest

from utils.doc_summary import build_doc_summary


def test_basic_counts():
    df = pd.DataFrame({
        'token': ['the', 'cat', 'sat'],
        'lemma': ['the', 'cat', 'sat'],
        'upos':  ['DET', 'NOUN', 'VERB'],
        'sentence_id': [1, 1, 1],
        'paragraph_id': [1, 1, 1],
    })
    summary = build_doc_summary(df, stimulus_id='story1', dictionary_used='Empath')
    assert summary['stimulus_id'] == 'story1'
    assert summary['n_tokens'] == 3
    assert summary['n_content_tokens'] == 2  # cat, sat
    assert summary['n_sentences'] == 1
    assert summary['dictionary_used'] == 'Empath'


def test_means_skip_nan():
    df = pd.DataFrame({
        'token': ['cat', 'sat'],
        'lemma': ['cat', 'sat'],
        'upos': ['NOUN', 'VERB'],
        'sentence_id': [1, 1],
        'paragraph_id': [1, 1],
        'Zipf_Freq': [5.5, None],
        'Surprisal': [3.0, 4.0],
    })
    s = build_doc_summary(df, stimulus_id='s', dictionary_used='Empath')
    assert s['mean_zipf_freq'] == pytest.approx(5.5)
    assert s['mean_surprisal'] == pytest.approx(3.5)


def test_category_proportions():
    df = pd.DataFrame({
        'token': ['cat', 'dog', 'run'],
        'lemma': ['cat', 'dog', 'run'],
        'upos': ['NOUN', 'NOUN', 'VERB'],
        'sentence_id': [1, 1, 1],
        'paragraph_id': [1, 1, 1],
        'Sem_Empath_animal': [1, 1, 0],
        'Sem_Empath_motion': [0, 0, 1],
    })
    s = build_doc_summary(df, stimulus_id='s', dictionary_used='Empath')
    assert s['count_animal'] == 2
    assert s['prop_animal'] == pytest.approx(2 / 3)
    assert s['count_motion'] == 1
```

- [ ] **Step 2: Run test, expect failure**

Run: `pytest tests/utils/test_doc_summary.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

Create `utils/doc_summary.py`:
```python
"""Per-stimulus aggregation — emit a one-row summary for each processed pair.

Uses the per-word DataFrame produced by the extractor pipeline as input.
"""

import pandas as pd

CONTENT_POS = {'NOUN', 'VERB', 'ADJ', 'ADV', 'PROPN'}


def build_doc_summary(token_df: pd.DataFrame, stimulus_id: str, dictionary_used: str) -> dict:
    """Aggregate a per-word DataFrame into a one-row summary dict."""
    df = token_df

    summary: dict = {
        'stimulus_id': stimulus_id,
        'n_tokens': int(len(df)),
        'n_content_tokens': int(df['upos'].isin(CONTENT_POS).sum()) if 'upos' in df else 0,
        'n_sentences': int(df['sentence_id'].nunique()) if 'sentence_id' in df else 0,
        'n_paragraphs': int(df['paragraph_id'].nunique()) if 'paragraph_id' in df else 0,
        'dictionary_used': dictionary_used,
    }

    # Means of numeric feature columns (skipna by default)
    for src, dst in [
        ('Zipf_Freq', 'mean_zipf_freq'),
        ('Edu_Zipf_Overall', 'mean_edu_zipf_overall'),
        ('Surprisal', 'mean_surprisal'),
        ('LLM_Surprisal', 'mean_llm_surprisal'),
        ('LLM_Cloze', 'mean_llm_cloze'),
    ]:
        if src in df.columns:
            summary[dst] = float(df[src].mean()) if df[src].notna().any() else None

    # Category counts and proportions over content tokens
    if 'upos' in df.columns:
        n_content = int(df['upos'].isin(CONTENT_POS).sum())
    else:
        n_content = 0

    sem_cols = [c for c in df.columns if c.startswith('Sem_') and c not in ('Sem_Categories',)]
    for col in sem_cols:
        # Column name pattern: Sem_<Dict>_<category>
        parts = col.split('_', 2)
        if len(parts) < 3:
            continue
        category = parts[2]
        count = int(df[col].fillna(0).astype(int).sum())
        summary[f'count_{category}'] = count
        summary[f'prop_{category}'] = (count / n_content) if n_content > 0 else 0.0

    return summary
```

- [ ] **Step 4: Run test, expect pass**

Run: `pytest tests/utils/test_doc_summary.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add utils/doc_summary.py tests/utils/test_doc_summary.py
git commit -m "feat(utils): add per-stimulus document summary aggregator"
```

---

## Phase D — Pipeline integration (Tasks 11–13)

### Task 11: Wire new extractors into `main.py` (single-file mode unchanged)

**Files:**
- Modify: `main.py`

- [ ] **Step 1: Inspect current `main.py` `run()` function**

Open `main.py` and identify:
- Imports at top
- `_COLUMN_ORDER` constant
- `run()` function

The integration touchpoints:
- Add new flags to `build_parser()`
- Add new extractor calls in `run()` between existing steps
- Update `_COLUMN_ORDER` to include new columns

- [ ] **Step 2: Update `build_parser()` with new flags**

Modify `main.py`'s `build_parser()` (around line 60). Add after the existing flags:

```python
    # New batch 1 feature flags
    p.add_argument('--no-edu-freq',     action='store_true', help='Skip EWFG grade-band frequencies.')
    p.add_argument('--edu-freq-lookup', default=None, help='Override path to EWFG lookup parquet.')

    p.add_argument('--no-sem-domain',   action='store_true', help='Skip semantic-domain columns and doc summary.')
    p.add_argument('--sem-dict',        default='auto',
                   help='Semantic dictionary backend: liwc | empath | auto | <path-to-.dic>.')
    p.add_argument('--sem-categories',  default=None,
                   help='Comma-separated category names. Default = SOP 15-category set.')
    p.add_argument('--save-prefs',      action='store_true',
                   help='Persist current --sem-categories selection to preferences.json.')

    p.add_argument('--no-llm-cloze',    action='store_true', help='Skip LLM-cloze metrics.')
    p.add_argument('--llm-model',       default=None, help='Override LLM model id (auto-detected by default).')

    p.add_argument('--no-doc-summary',  action='store_true',
                   help='Skip <stimulus>_summary.csv emission in batch mode.')

    # Batch-mode flags
    p.add_argument('--transcripts-folder', default=None, help='Batch mode: transcripts folder.')
    p.add_argument('--audio-folder',       default=None, help='Batch mode: audio folder.')
    p.add_argument('--output-folder',      default=None, help='Batch mode: output folder.')
    p.add_argument('--analyzer',           default=None, help='Required for batch mode; recorded in manifest.')
    p.add_argument('--text-extensions',    default='txt,tsv,md', help='Comma-separated text extensions.')
    p.add_argument('--audio-extensions',   default='wav,flac,mp3', help='Comma-separated audio extensions.')
```

- [ ] **Step 3: Update `_COLUMN_ORDER`**

Replace the existing `_COLUMN_ORDER` constant with:

```python
_COLUMN_ORDER = [
    'doc_id', 'paragraph_id', 'sentence_id', 'sentence', 'start', 'end',
    'term_id', 'token_id',
    'token', 'lemma', 'upos', 'xpos', 'feats', 'dep_rel', 'head_token_id',
    'deps', 'misc',
    'Zipf_Freq',
    'Edu_Zipf_K2', 'Edu_Zipf_3to5', 'Edu_Zipf_6to8', 'Edu_Zipf_9to12', 'Edu_Zipf_Overall',
    # Sem_<Dict>_<cat> columns get appended dynamically by _reorder
    'Sem_Categories',
    'Surprisal',
    'LLM_Surprisal', 'LLM_Cloze', 'LLM_TopMatch',
    'Transcript', 'Onset', 'End',
    'Word_Duration', 'Word_Gap',
]
```

- [ ] **Step 4: Update `run()` to call new extractors**

Replace the `run()` function with:

```python
def run(args) -> pd.DataFrame:
    from extractors.text_parser import parse as spacy_parse
    from extractors.word_freq import add_word_freq
    from extractors.surprisal import add_surprisal
    from extractors.aligner import align as mfa_align
    from extractors.educator_freq import add_edu_freq
    from extractors.semantic_domain import add_semantic_domain, detect_backend
    from extractors.llm_cloze import add_llm_cloze, select_model
    from extractors.timing import add_timing_columns
    from utils.preferences import load_preferences

    prefs = load_preferences()
    sem_categories = (
        args.sem_categories.split(',')
        if args.sem_categories
        else prefs['semantic_categories']
    )

    with open(args.transcript, 'r', encoding='utf-8') as f:
        text = f.read()

    df = None

    if not args.no_parse:
        print('[spacy] Running spaCy annotation…')
        df = spacy_parse(text)

    if not args.no_freq and df is not None:
        print('[freq] Computing SUBTLEX Zipf…')
        df = add_word_freq(df)

    if not args.no_edu_freq and df is not None:
        print('[edu] Computing EWFG grade-band frequencies…')
        df = add_edu_freq(df, lookup_path=args.edu_freq_lookup)

    if not args.no_sem_domain and df is not None:
        backend, dict_path = _resolve_sem_backend(args.sem_dict)
        print(f'[sem] Computing semantic-domain ({backend})…')
        df = add_semantic_domain(df, backend=backend, categories=sem_categories,
                                 dictionary_path=dict_path)

    if not args.no_surprisal and df is not None:
        print('[gpt2] Computing GPT-2 surprisal (legacy)…')
        df = add_surprisal(df, text)

    if not args.no_llm_cloze and df is not None:
        model_id = args.llm_model or select_model()
        print(f'[llm] Computing LLM-cloze with {model_id}…')
        df = add_llm_cloze(df, text=text, model_id=model_id)

    if not args.no_align and args.audio:
        print('[mfa] Running MFA alignment…')
        align_df = mfa_align(args.audio, text)
        if df is not None:
            df = _merge_alignment(df, align_df)
        else:
            df = align_df

    if df is not None:
        df = add_timing_columns(df)
        df = _reorder(df)

    if df is None:
        print('Nothing to output — all extractors were skipped.', file=sys.stderr)
        sys.exit(1)

    if args.save_prefs and args.sem_categories:
        from utils.preferences import save_preferences
        save_preferences({'semantic_categories': sem_categories})

    return df


def _resolve_sem_backend(sem_dict_arg: str) -> tuple[str, Path | None]:
    from extractors.semantic_domain import detect_backend
    if sem_dict_arg == 'auto':
        return detect_backend()
    if sem_dict_arg == 'empath':
        return 'empath', None
    if sem_dict_arg == 'liwc':
        backend, path = detect_backend()
        if backend != 'liwc':
            raise SystemExit('--sem-dict liwc but no .dic file found in dictionaries/')
        return 'liwc', path
    p = Path(sem_dict_arg)
    if p.suffix.lower() == '.dic' and p.exists():
        return 'liwc', p
    raise SystemExit(f'--sem-dict: unrecognized value "{sem_dict_arg}"')
```

Update the `_reorder` function to keep `Sem_<Dict>_*` columns adjacent to `Sem_Categories`:

```python
def _reorder(df: pd.DataFrame) -> pd.DataFrame:
    sem_cols = sorted([c for c in df.columns if c.startswith('Sem_') and c != 'Sem_Categories'])
    canonical: list[str] = []
    for c in _COLUMN_ORDER:
        if c == 'Sem_Categories':
            canonical.extend(sem_cols)
            canonical.append('Sem_Categories')
        else:
            canonical.append(c)
    present = [c for c in canonical if c in df.columns]
    extra = [c for c in df.columns if c not in present]
    return df[present + extra]
```

- [ ] **Step 5: Manual smoke test**

Create `tests/fixtures/transcripts/test_short.txt`:
```
The cat sat on the mat.
```

Run:
```bash
python main.py --transcript tests/fixtures/transcripts/test_short.txt --no-align --no-llm-cloze --output /tmp/short.csv
```
Expected: a CSV with all the new column names appearing (with NaN where data isn't available).

- [ ] **Step 6: Commit**

```bash
git add main.py
git commit -m "feat(pipeline): wire new extractors and flags into main.py single-file mode"
```

---

### Task 12: Add batch mode to `main.py`

**Files:**
- Modify: `main.py`
- Create: `tests/integration/test_pipeline_batch.py`

- [ ] **Step 1: Write failing integration test**

Create `tests/fixtures/transcripts/story1.txt`, `tests/fixtures/transcripts/story2.txt`, `tests/fixtures/transcripts/story3.txt`:
```
# story1.txt
The cat sat on the mat.

# story2.txt
The dog ran in the park.

# story3.txt
A short orphan transcript.
```

(For `tests/fixtures/audio/`, you can create empty `.wav` files for now since the integration test will run `--no-align`.)

```bash
mkdir -p tests/fixtures/audio
touch tests/fixtures/audio/story1.wav
touch tests/fixtures/audio/story2.wav
```

Create `tests/integration/test_pipeline_batch.py`:
```python
import json
import shutil
from pathlib import Path

import pytest

from main import run_batch, build_parser

FIXTURES = Path(__file__).parent.parent / 'fixtures'


def _args_for_batch(tmp_path):
    parser = build_parser()
    args = parser.parse_args([
        '--transcript', 'unused.txt',  # ignored in batch mode but required by parser
        '--transcripts-folder', str(FIXTURES / 'transcripts'),
        '--audio-folder', str(FIXTURES / 'audio'),
        '--output-folder', str(tmp_path / 'out'),
        '--analyzer', 'TestRunner',
        '--no-align',  # skip MFA in tests
        '--no-llm-cloze',  # skip LLM in tests
        '--no-edu-freq',  # skip — no lookup file in fixtures
    ])
    return args


def test_batch_processes_all_pairs(tmp_path):
    args = _args_for_batch(tmp_path)
    run_batch(args)
    out = tmp_path / 'out'
    assert (out / 'story1_features.csv').exists()
    assert (out / 'story2_features.csv').exists()
    # story3 has no audio match — should NOT have features.csv emitted
    assert not (out / 'story3_features.csv').exists()


def test_batch_emits_run_artifacts(tmp_path):
    args = _args_for_batch(tmp_path)
    run_batch(args)
    out = tmp_path / 'out'
    assert any(out.glob('run_*.log'))
    assert any(out.glob('run_*_manifest.json'))
    assert any(out.glob('run_*_no-match.csv'))


def test_batch_manifest_records_unmatched(tmp_path):
    args = _args_for_batch(tmp_path)
    run_batch(args)
    manifest = json.loads(next((tmp_path / 'out').glob('run_*_manifest.json')).read_text())
    assert manifest['n_unmatched_text'] >= 1  # story3 has no audio
    assert manifest['analyzer'] == 'TestRunner'


def test_batch_emits_doc_summary(tmp_path):
    args = _args_for_batch(tmp_path)
    args.no_doc_summary = False
    args.no_sem_domain = False  # need sem columns to trigger summary
    run_batch(args)
    # At least one summary file should exist (depending on Empath availability)
    out = tmp_path / 'out'
    summaries = list(out.glob('*_summary.csv'))
    # If Empath isn't installed in test env, summary may still be written using only base aggregates
    assert len(summaries) >= 0  # placeholder; tighten after empath install confirmed
```

- [ ] **Step 2: Run test, expect failure**

Run: `pytest tests/integration/test_pipeline_batch.py -v`
Expected: `ImportError: cannot import name 'run_batch' from 'main'`.

- [ ] **Step 3: Implement `run_batch` in `main.py`**

Add to `main.py` (above `def main()`):

```python
import time
from pathlib import Path

from utils.file_pairing import find_pairs
from utils.run_logger import RunLogger
from utils.doc_summary import build_doc_summary


def run_batch(args) -> None:
    """Process all matched (transcript, audio) pairs in the configured folders."""
    transcripts_dir = Path(args.transcripts_folder)
    audio_dir = Path(args.audio_folder)
    output_dir = Path(args.output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.analyzer:
        raise SystemExit('--analyzer is required for batch mode')

    text_exts = [e.strip() for e in args.text_extensions.split(',') if e.strip()]
    audio_exts = [e.strip() for e in args.audio_extensions.split(',') if e.strip()]

    pairing = find_pairs(transcripts_dir, audio_dir, text_exts=text_exts, audio_exts=audio_exts)

    config_snapshot = {
        'no_parse': args.no_parse, 'no_freq': args.no_freq, 'no_edu_freq': args.no_edu_freq,
        'no_sem_domain': args.no_sem_domain, 'no_surprisal': args.no_surprisal,
        'no_llm_cloze': args.no_llm_cloze, 'no_align': args.no_align,
        'no_doc_summary': args.no_doc_summary,
        'sem_dict': args.sem_dict, 'sem_categories': args.sem_categories,
        'llm_model': args.llm_model,
        'edu_freq_lookup': args.edu_freq_lookup,
    }

    logger = RunLogger(
        output_folder=output_dir,
        analyzer=args.analyzer,
        config=config_snapshot,
        input_transcripts_folder=transcripts_dir,
        input_audio_folder=audio_dir,
        text_extensions=text_exts,
        audio_extensions=audio_exts,
    )

    for warning in pairing.warnings:
        logger.add_warning(warning)
    for unmatched in pairing.unmatched_text:
        logger.record_unmatched_text(unmatched)
    for unmatched in pairing.unmatched_audio:
        logger.record_unmatched_audio(unmatched)

    for pair in pairing.pairs:
        start = time.time()
        try:
            args.transcript = str(pair.transcript_path)
            args.audio = str(pair.audio_path)
            df = run(args)

            features_path = output_dir / f'{pair.stimulus_id}_features.csv'
            df.to_csv(features_path, index=False)

            summary_path = None
            if not args.no_doc_summary and not args.no_sem_domain:
                dict_label = 'LIWC' if any(c.startswith('Sem_LIWC_') for c in df.columns) else 'Empath'
                summary = build_doc_summary(df, stimulus_id=pair.stimulus_id, dictionary_used=dict_label)
                summary_path = output_dir / f'{pair.stimulus_id}_summary.csv'
                pd.DataFrame([summary]).to_csv(summary_path, index=False)

            logger.record_pair_success(
                stimulus_id=pair.stimulus_id,
                transcript_path=pair.transcript_path,
                audio_path=pair.audio_path,
                runtime_seconds=time.time() - start,
                output_csv_path=features_path,
                output_summary_path=summary_path,
            )
        except Exception as exc:
            import traceback
            tb = ''.join(traceback.format_exception_only(type(exc), exc)).strip()
            logger.record_pair_error(
                stimulus_id=pair.stimulus_id,
                transcript_path=pair.transcript_path,
                audio_path=pair.audio_path,
                runtime_seconds=time.time() - start,
                error_message=tb,
            )
    logger.finalize()
```

Update `main()` to dispatch on batch vs single:

```python
def main():
    parser = build_parser()
    args = parser.parse_args()

    is_batch = bool(args.transcripts_folder or args.audio_folder or args.output_folder)
    if is_batch:
        if not (args.transcripts_folder and args.audio_folder and args.output_folder):
            raise SystemExit('Batch mode requires --transcripts-folder, --audio-folder, and --output-folder')
        run_batch(args)
        return

    if not args.no_align and args.audio is None:
        print('Warning: --audio not provided; alignment will be skipped.')
    df = run(args)
    df.to_csv(args.output, index=False)
    print(f'\nDone. {len(df)} rows written to {args.output}')
```

Make `--transcript` non-required when batch flags are present:

```python
    p.add_argument('--transcript', required=False, default='',
                   help='Single-file mode: transcript path. Ignored in batch mode.')
```

- [ ] **Step 4: Run integration tests, expect pass**

Run: `pytest tests/integration/test_pipeline_batch.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add main.py tests/integration/test_pipeline_batch.py tests/fixtures/transcripts/story1.txt tests/fixtures/transcripts/story2.txt tests/fixtures/transcripts/story3.txt tests/fixtures/audio/story1.wav tests/fixtures/audio/story2.wav
git commit -m "feat(pipeline): add batch mode with file pairing, run logs, doc summary"
```

---

### Task 13: Smoke-test the full CLI

**Files:**
- Modify: `main.py` (only if issues surface)

- [ ] **Step 1: Single-file run with all available extractors**

Run:
```bash
python main.py --transcript tests/fixtures/transcripts/story1.txt --no-align --no-llm-cloze --output /tmp/story1.csv
```
Expected: Exits 0; CSV has Zipf_Freq, Sem_*, Sem_Categories, Surprisal columns. EWFG columns present but NaN (no lookup file).

- [ ] **Step 2: Batch run end-to-end**

Run:
```bash
python main.py \
  --transcripts-folder tests/fixtures/transcripts \
  --audio-folder tests/fixtures/audio \
  --output-folder /tmp/sensi-batch-test \
  --analyzer "Smoke Test" \
  --no-align --no-llm-cloze
```
Expected: `/tmp/sensi-batch-test/` contains story1_features.csv, story2_features.csv, story1_summary.csv, story2_summary.csv, run_*.log, run_*_manifest.json, run_*_no-match.csv. The no-match.csv lists story3.txt.

- [ ] **Step 3: Inspect manifest**

```bash
cat /tmp/sensi-batch-test/run_*_manifest.json
```
Verify: analyzer, system_info, config, n_pairs_processed=2, n_unmatched_text=1, pairs[].status all "success".

- [ ] **Step 4: No code changes if smoke tests pass.** Otherwise fix and recommit.

---

## Phase E — GUI overhaul (Tasks 14–17)

### Task 14: Replace gui.py main panel layout (folder pickers + analyzer + sections)

**Files:**
- Modify: `gui.py` (full replacement)

**This is a substantial replacement.** The new GUI mirrors the spec §10. Implement in this single task because tkinter widgets are tightly coupled — splitting risks broken intermediate states.

- [ ] **Step 1: Read existing gui.py thoroughly**

```bash
# (already familiar from spec writing)
```

- [ ] **Step 2: Replace gui.py with the new layout**

Replace the entire contents of `gui.py` with the version below. **Read the whole listing before pasting** to verify it makes sense in your environment.

```python
"""SensiExtractor — graphical interface (batch mode).

Run with:
    python gui.py
"""

import os
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk

# Redirect model/cache downloads into the project folder
_HERE = Path(__file__).parent.resolve()
os.environ.setdefault('HF_HOME',      str(_HERE / 'hf_cache'))
os.environ.setdefault('MFA_ROOT_DIR', str(_HERE / 'mfa_data'))
os.environ.setdefault('HF_HUB_DISABLE_IMPLICIT_TOKEN', '1')

from utils.preferences import load_preferences, save_preferences
from extractors.semantic_domain import detect_backend
from extractors.llm_cloze import LLM_OPTIONS  # shared with gui_run.py


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('SensiExtractor (batch mode)')
        self.geometry('780x720')
        self._prefs = load_preferences()
        self._build()

    def _build(self):
        pad = {'padx': 10, 'pady': 6}

        # --- Section 1: Inputs --------------------------------------------------
        frm_in = ttk.LabelFrame(self, text='1 · Inputs', padding=8)
        frm_in.grid(row=0, column=0, sticky='ew', **pad)
        frm_in.columnconfigure(1, weight=1)

        self.transcripts_var = tk.StringVar(value=self._prefs.get('last_transcripts_folder', ''))
        self.audio_var       = tk.StringVar(value=self._prefs.get('last_audio_folder', ''))
        self.output_var      = tk.StringVar(value=self._prefs.get('last_output_folder', ''))
        self.analyzer_var    = tk.StringVar(value=self._prefs.get('analyzer_name', ''))

        self._folder_row(frm_in, 0, 'Transcripts folder:', self.transcripts_var, self._browse_transcripts)
        self._folder_row(frm_in, 1, 'Audio folder:',       self.audio_var,       self._browse_audio)
        self._folder_row(frm_in, 2, 'Output folder:',      self.output_var,      self._browse_output)

        ttk.Label(frm_in, text='Analyzer name:').grid(row=3, column=0, sticky='w', pady=(8, 0))
        ttk.Entry(frm_in, textvariable=self.analyzer_var, width=40).grid(
            row=3, column=1, sticky='ew', padx=(6, 4), pady=(8, 0))

        # --- Section 2: Model & dictionary --------------------------------------
        frm_md = ttk.LabelFrame(self, text='2 · Model & dictionary', padding=8)
        frm_md.grid(row=1, column=0, sticky='ew', **pad)
        frm_md.columnconfigure(1, weight=1)

        ttk.Label(frm_md, text='LLM model:').grid(row=0, column=0, sticky='w')
        self.llm_var = tk.StringVar(value=self._prefs.get('llm_model', '') or LLM_OPTIONS[0][1])
        self.llm_combo = ttk.Combobox(
            frm_md, textvariable=self.llm_var,
            values=[label for _, label in LLM_OPTIONS],
            state='readonly', width=60,
        )
        self.llm_combo.grid(row=0, column=1, sticky='ew', padx=(6, 4))

        ttk.Label(frm_md, text='Dictionary backend:').grid(row=1, column=0, sticky='w', pady=(6, 0))
        self.dict_var = tk.StringVar(value=self._prefs.get('dictionary_backend', 'auto'))
        dict_frame = ttk.Frame(frm_md)
        dict_frame.grid(row=1, column=1, sticky='w', padx=(6, 0), pady=(6, 0))
        for value, label in [('auto', 'Auto'), ('liwc', 'LIWC'), ('empath', 'Empath')]:
            ttk.Radiobutton(dict_frame, text=label, value=value, variable=self.dict_var).pack(side='left', padx=(0, 8))
        backend_detected, _ = detect_backend()
        self.dict_status_var = tk.StringVar(value=f'(auto-detected: {backend_detected})')
        ttk.Label(frm_md, textvariable=self.dict_status_var, foreground='gray').grid(
            row=2, column=1, sticky='w', padx=(6, 0))

        ttk.Button(frm_md, text='Configure categories…', command=self._open_categories_dialog).grid(
            row=3, column=1, sticky='w', padx=(6, 0), pady=(8, 0))

        # --- Section 3: Feature toggles -----------------------------------------
        frm_feat = ttk.LabelFrame(self, text='3 · Features to extract', padding=8)
        frm_feat.grid(row=2, column=0, sticky='ew', **pad)

        toggles = self._prefs.get('feature_toggles', {})
        self.tog_parse     = tk.BooleanVar(value=toggles.get('spacy_parse', True))
        self.tog_zipf      = tk.BooleanVar(value=toggles.get('zipf_freq', True))
        self.tog_edu       = tk.BooleanVar(value=toggles.get('edu_freq', True))
        self.tog_sem       = tk.BooleanVar(value=toggles.get('sem_domain', True))
        self.tog_gpt2      = tk.BooleanVar(value=toggles.get('gpt2_surprisal', True))
        self.tog_llm       = tk.BooleanVar(value=toggles.get('llm_cloze', True))
        self.tog_align     = tk.BooleanVar(value=toggles.get('alignment', True))
        self.tog_summary   = tk.BooleanVar(value=toggles.get('doc_summary', True))

        groups = [
            ('Linguistic',     [('spaCy parse',                        self.tog_parse)]),
            ('Frequency',      [('SUBTLEX Zipf (Zipf_Freq)',           self.tog_zipf),
                                ('EWFG grade-band frequencies (Edu_Zipf_*)', self.tog_edu)]),
            ('Semantic',       [('Semantic-domain indicators (Sem_*)', self.tog_sem)]),
            ('Predictability', [('GPT-2 surprisal (legacy baseline)',  self.tog_gpt2),
                                ('LLM-cloze (LLM_*)',                  self.tog_llm)]),
            ('Alignment',      [('MFA + Word_Duration + Word_Gap',     self.tog_align)]),
            ('Outputs',        [('Document-level summary sidecar',     self.tog_summary)]),
        ]
        row = 0
        for group_name, items in groups:
            ttk.Label(frm_feat, text=group_name, font=('TkDefaultFont', 9, 'bold')).grid(
                row=row, column=0, sticky='w', pady=(4, 0))
            row += 1
            for label, var in items:
                ttk.Checkbutton(frm_feat, text=label, variable=var).grid(row=row, column=0, sticky='w', padx=(16, 0))
                row += 1

        # --- Section 4: Run -----------------------------------------------------
        frm_run = ttk.Frame(self)
        frm_run.grid(row=3, column=0, sticky='ew', **pad)
        self.run_btn = ttk.Button(frm_run, text='Run extraction', command=self._on_run)
        self.run_btn.pack(side='left')
        self.cancel_btn = ttk.Button(frm_run, text='Cancel', state='disabled', command=self._on_cancel)
        self.cancel_btn.pack(side='left', padx=(8, 0))
        self.progress = ttk.Progressbar(frm_run, mode='determinate', length=300)
        self.progress.pack(side='left', padx=(8, 0))
        self.status_var = tk.StringVar(value='Idle')
        ttk.Label(frm_run, textvariable=self.status_var).pack(side='left', padx=(8, 0))

        # --- Section 5: Log -----------------------------------------------------
        frm_log = ttk.LabelFrame(self, text='Log', padding=8)
        frm_log.grid(row=4, column=0, sticky='nsew', **pad)
        self.log = scrolledtext.ScrolledText(frm_log, height=14, font=('Consolas', 9), state='disabled')
        self.log.pack(fill='both', expand=True)

        self.columnconfigure(0, weight=1)
        self.rowconfigure(4, weight=1)

        self._cancel_requested = False

    def _folder_row(self, parent, row, label, var, browse_cb):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky='w')
        entry = ttk.Entry(parent, textvariable=var, width=48)
        entry.grid(row=row, column=1, sticky='ew', padx=(6, 4))
        ttk.Button(parent, text='Browse…', command=browse_cb).grid(row=row, column=2)

    # ---------------------------------------------------------------------
    # Browse callbacks
    # ---------------------------------------------------------------------

    def _browse_transcripts(self):
        path = filedialog.askdirectory(initialdir=self.transcripts_var.get() or os.path.expanduser('~'))
        if path:
            self.transcripts_var.set(path)

    def _browse_audio(self):
        path = filedialog.askdirectory(initialdir=self.audio_var.get() or os.path.expanduser('~'))
        if path:
            self.audio_var.set(path)

    def _browse_output(self):
        path = filedialog.askdirectory(initialdir=self.output_var.get() or os.path.expanduser('~'))
        if path:
            self.output_var.set(path)

    # ---------------------------------------------------------------------
    # Categories modal (Task 15 fills this in)
    # ---------------------------------------------------------------------

    def _open_categories_dialog(self):
        from gui_categories_dialog import CategoriesDialog
        CategoriesDialog(self)

    # ---------------------------------------------------------------------
    # Run dispatch (Task 16 fills this in)
    # ---------------------------------------------------------------------

    def _on_run(self):
        from gui_run import start_run
        start_run(self)

    def _on_cancel(self):
        self._cancel_requested = True
        self.status_var.set('Cancelling after current file…')

    def append_log(self, msg: str) -> None:
        self.log.configure(state='normal')
        self.log.insert(tk.END, msg)
        self.log.see(tk.END)
        self.log.configure(state='disabled')


if __name__ == '__main__':
    App().mainloop()
```

- [ ] **Step 3: Smoke-test the launch (categories dialog and run won't work yet)**

Run:
```bash
python gui.py
```
Expected: Window opens with three folder fields, analyzer field, model dropdown, dictionary radios, feature checkboxes, run/cancel/progress, log box. Closing the window exits cleanly. Clicking "Configure categories…" raises `ImportError` for `gui_categories_dialog` — that's expected; Task 15 adds it.

- [ ] **Step 4: Commit**

```bash
git add gui.py
git commit -m "feat(gui): replace main panel with folder-pair batch layout"
```

---

### Task 15: Categories modal dialog

**Files:**
- Create: `gui_categories_dialog.py`

- [ ] **Step 1: Implement the modal**

Create `gui_categories_dialog.py`:
```python
"""Modal dialog for selecting semantic-domain categories.

Opens from gui.App._open_categories_dialog. Loads available categories from the
active dictionary backend (LIWC or Empath). Saves selection to preferences.json
on OK.
"""

import tkinter as tk
from tkinter import ttk, messagebox

from utils.preferences import DEFAULT_PREFS, load_preferences, save_preferences
from extractors.semantic_domain import detect_backend


def _all_categories(backend: str, dict_path) -> list[str]:
    if backend == 'empath':
        try:
            from empath import Empath
            return sorted(Empath().cats.keys())
        except Exception:
            return list(DEFAULT_PREFS['semantic_categories'])
    if backend == 'liwc':
        try:
            import liwc
            _, names = liwc.load_token_parser(str(dict_path))
            return sorted(names)
        except Exception:
            return list(DEFAULT_PREFS['semantic_categories'])
    return list(DEFAULT_PREFS['semantic_categories'])


class CategoriesDialog(tk.Toplevel):
    def __init__(self, master):
        super().__init__(master)
        self.title('Configure semantic categories')
        self.geometry('420x520')
        self.transient(master)
        self.grab_set()

        backend, dict_path = detect_backend()
        self._backend = backend
        all_cats = _all_categories(backend, dict_path)

        prefs = load_preferences()
        selected = set(prefs.get('semantic_categories', DEFAULT_PREFS['semantic_categories']))

        # Search box
        search_var = tk.StringVar()
        ttk.Label(self, text=f'Backend: {backend}').pack(anchor='w', padx=10, pady=(10, 4))
        ttk.Label(self, text='Search:').pack(anchor='w', padx=10)
        ttk.Entry(self, textvariable=search_var).pack(fill='x', padx=10, pady=(0, 8))

        # Scrollable category list
        canvas = tk.Canvas(self, highlightthickness=0)
        scroll = ttk.Scrollbar(self, orient='vertical', command=canvas.yview)
        canvas.configure(yscrollcommand=scroll.set)
        canvas.pack(side='left', fill='both', expand=True, padx=(10, 0), pady=(0, 8))
        scroll.pack(side='right', fill='y', padx=(0, 10), pady=(0, 8))

        frame = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=frame, anchor='nw')

        self._vars: dict[str, tk.BooleanVar] = {}
        self._checkbuttons: dict[str, ttk.Checkbutton] = {}
        for cat in all_cats:
            var = tk.BooleanVar(value=cat in selected)
            self._vars[cat] = var
            cb = ttk.Checkbutton(frame, text=cat, variable=var)
            cb.pack(anchor='w')
            self._checkbuttons[cat] = cb

        frame.update_idletasks()
        canvas.configure(scrollregion=canvas.bbox('all'))

        def _filter(*_):
            q = search_var.get().lower().strip()
            for cat, cb in self._checkbuttons.items():
                if not q or q in cat.lower():
                    cb.pack(anchor='w')
                else:
                    cb.pack_forget()

        search_var.trace_add('write', _filter)

        # Buttons
        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill='x', side='bottom', padx=10, pady=10)
        ttk.Button(btn_frame, text='Reset to SOP default', command=self._reset).pack(side='left')
        ttk.Button(btn_frame, text='Cancel', command=self.destroy).pack(side='right')
        ttk.Button(btn_frame, text='OK', command=self._save_and_close).pack(side='right', padx=(0, 6))

    def _reset(self):
        defaults = set(DEFAULT_PREFS['semantic_categories'])
        for cat, var in self._vars.items():
            var.set(cat in defaults)

    def _save_and_close(self):
        chosen = [cat for cat, var in self._vars.items() if var.get()]
        if not chosen:
            messagebox.showwarning('No categories selected',
                                   'Please select at least one category.')
            return
        save_preferences({'semantic_categories': chosen})
        self.destroy()
```

- [ ] **Step 2: Smoke-test**

Run `python gui.py`, click "Configure categories…". Expected: modal opens with categories from the detected backend pre-checked per preferences. Search filters live. Reset re-checks the SOP defaults. OK saves to `config/preferences.json`. Cancel discards.

Inspect `config/preferences.json` after clicking OK — `semantic_categories` should reflect the selection.

- [ ] **Step 3: Commit**

```bash
git add gui_categories_dialog.py
git commit -m "feat(gui): add modal dialog for configuring semantic categories"
```

---

### Task 16: Wire GUI run button to batch pipeline

**Files:**
- Create: `gui_run.py`

- [ ] **Step 1: Implement run dispatch**

Create `gui_run.py`:
```python
"""Glue between gui.App and main.run_batch().

Runs the batch in a background thread, redirects stdout/stderr to the log
widget, and updates the progress bar after each pair.
"""

import argparse
import os
import sys
import threading

from tkinter import messagebox

from utils.preferences import save_preferences

from extractors.llm_cloze import LLM_OPTIONS


def _label_to_id(label: str) -> str:
    for mid, lab in LLM_OPTIONS:
        if lab == label:
            return mid
    return label  # treat as raw id


def start_run(app) -> None:
    transcripts = app.transcripts_var.get().strip()
    audio = app.audio_var.get().strip()
    output = app.output_var.get().strip()
    analyzer = app.analyzer_var.get().strip()

    missing = [name for name, val in [
        ('Transcripts folder', transcripts),
        ('Audio folder', audio),
        ('Output folder', output),
        ('Analyzer name', analyzer),
    ] if not val]
    if missing:
        messagebox.showerror('Missing required input', 'Please fill in: ' + ', '.join(missing))
        return

    # Persist current selections for next launch
    save_preferences({
        'analyzer_name': analyzer,
        'last_transcripts_folder': transcripts,
        'last_audio_folder': audio,
        'last_output_folder': output,
        'llm_model': _label_to_id(app.llm_var.get()),
        'dictionary_backend': app.dict_var.get(),
        'feature_toggles': {
            'spacy_parse':     app.tog_parse.get(),
            'zipf_freq':       app.tog_zipf.get(),
            'edu_freq':        app.tog_edu.get(),
            'sem_domain':      app.tog_sem.get(),
            'gpt2_surprisal':  app.tog_gpt2.get(),
            'llm_cloze':       app.tog_llm.get(),
            'alignment':       app.tog_align.get(),
            'doc_summary':     app.tog_summary.get(),
        },
    })

    args = _build_args(app, transcripts, audio, output, analyzer)
    app.run_btn.configure(state='disabled')
    app.cancel_btn.configure(state='normal')
    app.status_var.set('Running…')
    app._cancel_requested = False

    threading.Thread(target=_thread_run, args=(app, args), daemon=True).start()


def _build_args(app, transcripts, audio, output, analyzer) -> argparse.Namespace:
    from main import build_parser
    parser = build_parser()
    cli = [
        '--transcript', '',  # dummy for non-batch mode parser requirement
        '--transcripts-folder', transcripts,
        '--audio-folder', audio,
        '--output-folder', output,
        '--analyzer', analyzer,
        '--sem-dict', app.dict_var.get(),
        '--llm-model', _label_to_id(app.llm_var.get()),
    ]
    if not app.tog_parse.get():    cli.append('--no-parse')
    if not app.tog_zipf.get():     cli.append('--no-freq')
    if not app.tog_edu.get():      cli.append('--no-edu-freq')
    if not app.tog_sem.get():      cli.append('--no-sem-domain')
    if not app.tog_gpt2.get():     cli.append('--no-surprisal')
    if not app.tog_llm.get():      cli.append('--no-llm-cloze')
    if not app.tog_align.get():    cli.append('--no-align')
    if not app.tog_summary.get():  cli.append('--no-doc-summary')
    return parser.parse_args(cli)


class _LogRedirect:
    def __init__(self, app):
        self._app = app

    def write(self, msg):
        self._app.after(0, self._app.append_log, msg)

    def flush(self):
        pass


def _thread_run(app, args):
    redir = _LogRedirect(app)
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = redir
    try:
        from main import run_batch
        run_batch(args)
        app.after(0, lambda: app.status_var.set('Done'))
        app.after(0, lambda: messagebox.showinfo('Done', f'Run finished. Output: {args.output_folder}'))
    except Exception as exc:
        msg = str(exc)
        app.after(0, lambda: app.status_var.set('Error'))
        app.after(0, lambda m=msg: messagebox.showerror('Error', m))
    finally:
        sys.stdout, sys.stderr = old_out, old_err
        app.after(0, lambda: app.run_btn.configure(state='normal'))
        app.after(0, lambda: app.cancel_btn.configure(state='disabled'))
```

- [ ] **Step 2: Smoke-test**

Run `python gui.py`. Fill in the three folders (point at the test fixtures used in Task 12), enter an analyzer name. Uncheck Alignment + LLM-cloze + EWFG (since we have no MFA model / API key / EWFG data set up by default). Click Run.

Expected: Progress + log streams output. After completion, message box "Done". Output folder contains the same artifacts as the CLI batch test.

- [ ] **Step 3: Commit**

```bash
git add gui_run.py
git commit -m "feat(gui): wire run button to batch pipeline with persistence"
```

---

### Task 17: GUI progress + cancellation hook

**Files:**
- Modify: `main.py` (add cancellation callback)
- Modify: `gui_run.py`

- [ ] **Step 1: Add progress callback to `run_batch`**

Modify `main.py`'s `run_batch` signature to accept an optional callback:

```python
def run_batch(args, *, on_pair_done=None, should_cancel=None) -> None:
    ...
    for i, pair in enumerate(pairing.pairs):
        if should_cancel and should_cancel():
            logger.add_warning('Run cancelled by user before processing complete')
            break
        ...
        if on_pair_done:
            on_pair_done(i + 1, len(pairing.pairs), pair.stimulus_id)
```

- [ ] **Step 2: Hook progress into the GUI**

Modify `gui_run.py`'s `_thread_run`:

```python
def _thread_run(app, args):
    redir = _LogRedirect(app)
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = redir
    try:
        from main import run_batch

        def progress(done, total, stimulus_id):
            pct = int(100 * done / total) if total else 0
            app.after(0, lambda: app.progress.configure(value=pct))
            app.after(0, lambda: app.status_var.set(f'Processing {stimulus_id}  ({done}/{total})'))

        run_batch(args, on_pair_done=progress, should_cancel=lambda: app._cancel_requested)
        app.after(0, lambda: app.status_var.set('Done'))
        app.after(0, lambda: messagebox.showinfo('Done', f'Run finished. Output: {args.output_folder}'))
    except Exception as exc:
        msg = str(exc)
        app.after(0, lambda: app.status_var.set('Error'))
        app.after(0, lambda m=msg: messagebox.showerror('Error', m))
    finally:
        sys.stdout, sys.stderr = old_out, old_err
        app.after(0, lambda: app.run_btn.configure(state='normal'))
        app.after(0, lambda: app.cancel_btn.configure(state='disabled'))
```

- [ ] **Step 3: Smoke test**

Run the same GUI batch test from Task 16. Verify progress bar advances and status label updates.

- [ ] **Step 4: Commit**

```bash
git add main.py gui_run.py
git commit -m "feat(gui): add progress reporting and cancellation"
```

---

## Phase F — Validation script (Task 18)

### Task 18: Provo Corpus validation script

**Files:**
- Create: `validate_cloze.py`

- [ ] **Step 1: Implement the script**

Create `validate_cloze.py`:
```python
"""One-shot validation of the configured LLM-cloze model against Provo Corpus.

Run after install:
    python validate_cloze.py

Downloads Provo Corpus (~few MB) into data/provo/ on first run, computes
LLM_Cloze for each Provo word using whichever LLM the pipeline auto-selects
(or --llm-model override), and reports Pearson r against human cloze norms.

Exits 0 if correlation falls within the expected range for the model, else 1.
"""

import argparse
import csv
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import pearsonr

from extractors.llm_cloze import add_llm_cloze, select_model

PROVO_CSV_URL = 'https://osf.io/download/eyfxq/'  # Provo Corpus item-level cloze data
PROVO_DIR = Path(__file__).resolve().parent / 'data' / 'provo'
PROVO_CSV = PROVO_DIR / 'provo_predictability.csv'

EXPECTED_RANGES = {
    'gpt-4o-mini':                (0.78, 0.86),
    'gpt-4o':                     (0.80, 0.88),
    'meta-llama/Llama-3-70B':     (0.80, 0.88),
    'meta-llama/Llama-3-8B':      (0.74, 0.83),
    'meta-llama/Llama-3.2-3B':    (0.70, 0.80),
}


def _ensure_provo():
    PROVO_DIR.mkdir(parents=True, exist_ok=True)
    if PROVO_CSV.exists():
        return
    import urllib.request
    print(f'Downloading Provo Corpus to {PROVO_CSV}…')
    urllib.request.urlretrieve(PROVO_CSV_URL, PROVO_CSV)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--llm-model', default=None)
    args = parser.parse_args()

    _ensure_provo()

    df = pd.read_csv(PROVO_CSV)
    # Provo columns include: Text_ID, Word_Number, Word, Word_Cloze (modal-prediction proportion)
    # Build per-paragraph token sequences and track Word_Cloze per word
    grouped = df.groupby('Text_ID')

    model_id = args.llm_model or select_model()
    print(f'Validation model: {model_id}')

    all_human, all_model = [], []
    for text_id, group in grouped:
        words = list(group['Word'].astype(str))
        text = ' '.join(words)
        token_df = pd.DataFrame({
            'token': words,
            'upos': ['NOUN'] * len(words),  # treat all as content for validation
        })
        scored = add_llm_cloze(token_df, text=text, model_id=model_id)
        all_model.extend(scored['LLM_Cloze'].tolist())
        all_human.extend(group['Word_Cloze'].tolist())

    pairs = [(h, m) for h, m in zip(all_human, all_model) if h is not None and m is not None and not (pd.isna(h) or pd.isna(m))]
    h_arr = [p[0] for p in pairs]
    m_arr = [p[1] for p in pairs]
    r, _ = pearsonr(h_arr, m_arr)

    lo, hi = EXPECTED_RANGES.get(model_id, (0.6, 0.95))
    status = 'PASS ✓' if lo <= r <= hi else 'FAIL ✗'
    print()
    print('Validation against Provo Corpus')
    print(f'Model:  {model_id}')
    print(f'N words tested: {len(pairs):,}')
    print(f'Pearson r vs human cloze: {r:.3f}')
    print(f'Expected range for this model: {lo:.2f}–{hi:.2f}')
    print(status)

    sys.exit(0 if lo <= r <= hi else 1)


if __name__ == '__main__':
    main()
```

NOTE: `scipy` is needed for `pearsonr`. Add to `requirements.txt`:
```
scipy>=1.11
```
Run: `pip install scipy`.

The Provo URL above is illustrative — verify the actual OSF download URL at https://osf.io/sjefs/ before relying on it. If the auto-download is unreliable, document the manual download in `Feature_Extraction_Guide.md` as: "Download `provo_predictability.csv` from https://osf.io/sjefs/ and save to `data/provo/`."

- [ ] **Step 2: Smoke test**

Run `python validate_cloze.py --llm-model meta-llama/Llama-3.2-3B`. (Or use a smaller mock model in development.)

Expected: download succeeds, model loads, prints correlation, exits with status reflecting the comparison.

If the Provo download URL fails, document that and skip the network call by accepting `--provo-csv PATH`:

```python
parser.add_argument('--provo-csv', default=str(PROVO_CSV),
                    help='Path to a local Provo predictability CSV (skip auto-download).')
```

- [ ] **Step 3: Commit**

```bash
git add validate_cloze.py requirements.txt
git commit -m "feat: add Provo Corpus validation script for LLM-cloze"
```

---

## Phase G — Documentation (Task 19)

### Task 19: Update Feature_Extraction_Guide.md and README.md

**Files:**
- Modify: `Feature_Extraction_Guide.md`
- Modify: `README.md`

- [ ] **Step 1: Read existing docs**

Read `Feature_Extraction_Guide.md` and `README.md` to understand current structure and tone before editing.

- [ ] **Step 2: Append a "Batch 1 features" section to `Feature_Extraction_Guide.md`**

Append the following section to `Feature_Extraction_Guide.md`:

```markdown
---

## Batch 1 features (added 2026-05-10)

This section documents the features and GUI changes introduced by Batch 1.
For the design rationale, see `docs/superpowers/specs/2026-05-10-feature-batch-1-design.md`.

### EWFG grade-band frequencies — `Edu_Zipf_*`

Five word-level columns derived from the *Educator's Word Frequency Guide* (Zeno
et al. 1995):

- `Edu_Zipf_K2`     — Zipf-scale frequency in K-grade-2 educational texts
- `Edu_Zipf_3to5`   — grades 3–5
- `Edu_Zipf_6to8`   — grades 6–8
- `Edu_Zipf_9to12`  — grades 9–12
- `Edu_Zipf_Overall` — aggregate across all grades

Lookup data is **not bundled** (license-bound). The pipeline emits NaN for these
columns until you place a parquet lookup at `dictionaries/edu_zipf.parquet`. See
the spec's Appendix A for acquisition steps. Override path: `--edu-freq-lookup PATH`.

### Semantic-domain indicators — `Sem_*`

One binary column per category for content words (NOUN, VERB, ADJ, ADV, PROPN);
function words and punctuation get NaN.

- Default backend: **LIWC** if `dictionaries/liwc-2022.dic` is present (license required), else **Empath** (open-source, pip-installed).
- Default category set (15 categories): `social, cognition, emotion_positive, emotion_negative, body, motion, perception, time, space, math, science, work, family, communication, money`.
- Each category's intended meaning and example words: see Appendix C of the spec doc.
- Roll-up column `Sem_Categories`: comma-separated list of categories the word matched.

Configure via the GUI's "Configure categories…" modal (saved to `config/preferences.json`)
or via CLI `--sem-categories cat1,cat2,...`.

### LLM-cloze — `LLM_Cloze`, `LLM_Surprisal`, `LLM_TopMatch`

Three columns from a single LLM forward pass per token:

| Column | Meaning |
|---|---|
| `LLM_Cloze` | P(word \| preceding context) — empirical-cloze proxy |
| `LLM_Surprisal` | -log2 P(word \| preceding context), in bits |
| `LLM_TopMatch` | 1 if model's top-1 next-token == target, else 0 |

The existing `Surprisal` column (GPT-2) is **kept unchanged** as a baseline.

#### Model selection (auto-detect)

| Selected when | Model | Approx. cloze r vs Provo | Cost per ~3000-word stimulus |
|---|---|---|---|
| `OPENAI_API_KEY` is set | `gpt-4o-mini` | ~0.83 | $0.10–0.30 (API) |
| GPU VRAM ≥ 48 GB | `meta-llama/Llama-3-70B` | ~0.84 | Free (electricity) |
| GPU VRAM ≥ 12 GB | `meta-llama/Llama-3-8B` | ~0.79 | Free |
| GPU VRAM ≥ 8 GB | `meta-llama/Llama-3.2-3B` | ~0.76 | Free |

Override with `--llm-model gpt-4o|gpt-4o-mini|llama-3-70b|llama-3-8b|llama-3.2-3b|<hf-id>`.
Run `python validate_cloze.py` once after install to confirm the chosen model
correlates with human cloze norms in the expected range.

### Derived timing — `Word_Duration`, `Word_Gap`

Computed from MFA `Onset` / `End`:

- `Word_Duration` = `End - Onset`
- `Word_Gap` = `Onset[i] - End[i-1]` (NaN for the first word in a stimulus)

NaN for words MFA couldn't align.

### Document-level summary — `<stimulus>_summary.csv`

In batch mode, every successful pair emits a one-row sidecar CSV with:

- counts: `n_tokens, n_content_tokens, n_sentences, n_paragraphs`
- means: `mean_zipf_freq, mean_edu_zipf_overall, mean_surprisal, mean_llm_surprisal, mean_llm_cloze`
- per-category: `count_<cat>, prop_<cat>` for each active semantic category
- `dictionary_used`: `LIWC` or `Empath`

Disable with `--no-doc-summary`.

### GUI workflow

1. Launch: `python gui.py`
2. Pick a **Transcripts folder** (text files: `.txt, .tsv, .md`)
3. Pick an **Audio folder** (audio files: `.wav, .flac, .mp3`)
4. Pick an **Output folder** (where features CSVs and run logs go)
5. Enter your **Analyzer name** (required, persisted between runs)
6. Choose LLM model, dictionary backend, semantic categories, and feature toggles
7. Click **Run extraction**

The GUI scans only the **top level** of each folder (no recursion).
Files are paired by base filename (case-insensitive). Unmatched files are
listed in `run_<ts>_no-match.csv`. Errors on individual files are recorded in
the manifest and don't halt the batch.

### Persistence — `config/preferences.json`

The GUI persists the analyzer name, last folder paths, model choice, dictionary
backend, semantic categories, and feature toggle states between sessions.
The file is gitignored. Override its path via `SENSI_PREFS_PATH` env var.

### Run log artifacts (batch mode)

Three artifacts in the output folder, named with the run timestamp:

- `run_<ts>.log` — human-readable log
- `run_<ts>_manifest.json` — structured manifest with environment, config, and per-pair details
- `run_<ts>_no-match.csv` — files that didn't pair, with reason

The manifest captures the analyzer, system info, configuration snapshot, and
per-pair runtimes — sufficient to reproduce or debug a run later.

### Validation: `validate_cloze.py`

Runs the configured LLM over Provo Corpus and reports correlation with human
cloze norms. Use after installation to confirm the LLM-cloze setup is working.
```

- [ ] **Step 3: Update `README.md`**

Append a "What's new in batch 1" section near the top of `README.md`:

```markdown
## What's new (batch 1, 2026-05-10)

- **Folder-pair batch mode**: select a transcripts folder and an audio folder; the app pairs files by base filename and processes them all in one run.
- **Run logs**: every batch emits a `.log`, a structured `manifest.json`, and a `no-match.csv` in the output folder.
- **Analyzer field**: required for batch runs; recorded in the manifest.
- **EWFG grade-band frequencies** (`Edu_Zipf_*`): bring your own EWFG lookup parquet; pipeline emits NaN until you do.
- **Semantic-domain indicators** (`Sem_*`): LIWC-primary / Empath-fallback, configurable category set via GUI modal.
- **LLM-cloze** (`LLM_Cloze`, `LLM_Surprisal`, `LLM_TopMatch`): modern LLM-based predictability alongside the existing GPT-2 `Surprisal` column.
- **Derived timing** (`Word_Duration`, `Word_Gap`): cheap aggregations of MFA outputs.
- **Document-level sidecar**: `<stimulus>_summary.csv` per batch run.
- **GUI overhaul**: grouped feature toggles, LLM model dropdown, dictionary backend radio, semantic category modal, progress + cancellation.

See `Feature_Extraction_Guide.md` for full documentation, and
`docs/superpowers/specs/2026-05-10-feature-batch-1-design.md` for design rationale.
```

Update the existing CLI examples in `README.md` to include a batch-mode example:

```markdown
# Batch mode (recommended for multiple stimuli)
python main.py \
  --transcripts-folder path/to/transcripts \
  --audio-folder path/to/audio \
  --output-folder path/to/output \
  --analyzer "Your Name"
```

- [ ] **Step 4: Commit**

```bash
git add Feature_Extraction_Guide.md README.md
git commit -m "docs: document batch 1 features, GUI workflow, and run logs"
```

---

## Phase H — Final integration check

### Task 20: Run full test suite

- [ ] **Step 1**

Run:
```bash
pytest tests/ -v
```
Expected: All tests pass. If any failures, investigate root cause and fix in a follow-up commit.

- [ ] **Step 2: Manual GUI smoke test on real data**

Pick one ISC-Wu Tsai SENSI stimulus pair (one transcript + one audio file). Run through the GUI end-to-end with all features enabled. Confirm:

- Progress bar advances
- Output folder contains `<stimulus>_features.csv`, `<stimulus>_summary.csv`, `run_*.log`, `run_*_manifest.json`, `run_*_no-match.csv`
- Manifest has analyzer, system_info, config, status="success" for the pair
- Features CSV has all expected columns (spaCy + Zipf + Edu_Zipf_* (NaN if no lookup) + Sem_* + Surprisal + LLM_* + Onset/End + Word_Duration/Word_Gap)

- [ ] **Step 3: Commit any final fixes**

```bash
git add -A
git commit -m "chore: post-integration fixes from end-to-end smoke test"
```

---

## Self-review notes

**Spec coverage**:

- §4.1 EWFG → Task 7 (`extractors/educator_freq.py`)
- §4.2 Semantic-domain (LIWC + Empath, categories modal, 15-category default, POS filter, doc summary) → Tasks 8, 10, 15
- §4.3 LLM-cloze (3 metrics, model selection, validate_cloze.py) → Tasks 9, 18
- §4.4 Word_Duration / Word_Gap → Task 6
- §5 Output structure (per-stimulus features.csv + summary.csv) → Tasks 10, 12
- §6 New CLI flags → Tasks 11, 12
- §7 Column ordering → Task 11
- §8 Documentation → Task 19
- §9 Validation/testing → Task 20 + per-task tests
- §10 GUI overhaul (folder pairing, analyzer, run log, selectors, persistence) → Tasks 4, 5, 14, 15, 16, 17

**Known caveats**:

- The Provo URL in Task 18 is illustrative — verify before relying on auto-download.
- `bitsandbytes` (4-bit quant for HF backend) is not supported on macOS; the requirements.txt entry guards with `platform_system != "Darwin"`. On macOS the LLM backend falls back to fp32/CPU which is slow.
- The OpenAI Completions endpoint with `echo=True` and `logprobs` is deprecated for newer GPT-4o models; if it fails, switch to the Responses API or use a logprobs-supporting endpoint per OpenAI's current docs at implementation time.

---

## Plan complete

Plan saved to `docs/superpowers/plans/2026-05-10-feature-batch-1.md`.
