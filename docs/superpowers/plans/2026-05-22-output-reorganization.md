# Output Folder Reorganization — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the flat output folder with per-level subfolders (`word/`, `document/`, `syllable/`, `phoneme/`) using lazy mkdir; move session_units CSVs into the existing `logs/` subfolder.

**Architecture:** A single private helper `_level_dir(output_dir, level)` in `main.py` creates and returns the appropriate subfolder on first write. All six path-construction sites in `run_word_level_batch` and `run_doc_level_batch` are updated. `run_logger.py` and `gui/run_dispatch.py` need no changes — RunLogger already writes to `logs/`, and the dispatcher constructs no paths of its own.

**Tech Stack:** Python `pathlib`, pytest. No new dependencies.

---

## File map

| File | Action | Responsibility |
|---|---|---|
| `tests/integration/test_output_paths.py` | **Create** | 7 new tests asserting new per-level paths |
| `main.py` | Modify | `_level_dir` helper + 6 path substitutions |
| `tests/integration/test_pipeline_batch.py` | Modify | Update 4 assertions to new paths |
| `tests/integration/test_pipeline_batch_doc.py` | Modify | Update 2 assertions to new paths |
| `docs/master_plan.md` | Modify | Mark batch active |

---

## Task 1 — New failing tests

**Files:**
- Create: `tests/integration/test_output_paths.py`

- [ ] **Step 1: Create the test file**

```python
"""Per-level output folder structure — lazy mkdir contract.

Each test verifies that a level subfolder (word/, syllable/, phoneme/,
document/) is created only when output is actually written there, and that
session_units CSVs land in logs/ not the output root.

All tests fail until Task 2 and Task 3 update main.py.
"""
from pathlib import Path

import pandas as pd
import pytest

import main
from main import build_parser, run_word_level_batch, run_doc_level_batch

FIXTURES = Path(__file__).parent.parent / 'fixtures'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _word_args(tmp_path):
    return build_parser().parse_args([
        '--transcripts-folder', str(FIXTURES / 'transcripts'),
        '--audio-folder', str(FIXTURES / 'audio'),
        '--output-folder', str(tmp_path / 'out'),
        '--analyzer', 'TestRunner',
        '--no-align',
        '--no-llm-cloze',
        '--no-edu-freq',
    ])


def _doc_args(tmp_path):
    return build_parser().parse_args([
        '--transcript', '',
        '--transcripts-folder', str(FIXTURES / 'transcripts'),
        '--audio-folder', str(FIXTURES / 'audio'),
        '--output-folder', str(tmp_path / 'out'),
        '--analyzer', 'TestRunner',
        '--level', 'document',
        '--no-align',
        '--no-llm-cloze',
        '--no-doc-mean-llm-cloze',
        '--no-edu-freq',
        '--no-doc-mean-edu-zipf',
    ])


# ---------------------------------------------------------------------------
# Word level
# ---------------------------------------------------------------------------

def test_word_features_land_in_word_subfolder(tmp_path):
    run_word_level_batch(_word_args(tmp_path))
    out = tmp_path / 'out'
    assert (out / 'word' / 'story1_features.csv').exists()
    assert (out / 'word' / 'story2_features.csv').exists()
    assert not list(out.glob('*_features.csv')), 'features CSV found at root'


def test_word_csv_disabled_no_word_folder_not_created(tmp_path):
    """When word CSV is off and --no-align (so no sidecars), word/ must not exist."""
    args = _word_args(tmp_path)
    args.no_word_csv = True   # not a CLI flag; set directly
    run_word_level_batch(args)
    assert not (tmp_path / 'out' / 'word').exists()


def test_sidecars_land_in_subfolders(tmp_path, monkeypatch):
    """syllables → syllable/, phonemes → phoneme/."""
    fake_word_df = pd.DataFrame({
        'doc_id': [0, 0], 'sentence_id': [0, 0], 'token_id': [0, 1],
        'token': ['hello', 'world'],
        'Onset': [0.0, 0.5], 'End': [0.4, 0.9],
    })
    fake_phone_df = pd.DataFrame({
        'phone':  ['HH', 'EH1', 'L',  'OW1', 'W',  'ER1', 'L',  'D'],
        'onset':  [0.00, 0.10, 0.20, 0.30, 0.50, 0.60, 0.75, 0.85],
        'offset': [0.10, 0.20, 0.30, 0.40, 0.60, 0.75, 0.85, 0.90],
    })

    def _fake_run(args):
        args._phones_df = fake_phone_df
        args._contours = None
        return fake_word_df.copy()

    monkeypatch.setattr(main, 'run', _fake_run)

    txt_dir = tmp_path / 'txt'
    txt_dir.mkdir()
    audio_dir = tmp_path / 'audio'
    audio_dir.mkdir()
    out_dir = tmp_path / 'out'
    (txt_dir / 'tone.txt').write_text('hello world')
    (audio_dir / 'tone.wav').write_bytes(b'RIFF\x00\x00\x00\x00WAVE')

    args = build_parser().parse_args([
        '--transcripts-folder', str(txt_dir),
        '--audio-folder', str(audio_dir),
        '--output-folder', str(out_dir),
        '--analyzer', 'tester',
        '--no-align', '--no-llm-cloze', '--no-edu-freq',
    ])
    args.no_syllables = False
    args.no_phonemes = False
    run_word_level_batch(args)

    assert list((out_dir / 'syllable').glob('*_syllables.csv')), 'syllables not in syllable/'
    assert list((out_dir / 'phoneme').glob('*_phonemes.csv')), 'phonemes not in phoneme/'
    assert not list(out_dir.glob('*_syllables.csv')), 'syllables found at root'
    assert not list(out_dir.glob('*_phonemes.csv')), 'phonemes found at root'


def test_sidecars_disabled_folders_not_created(tmp_path):
    """With --no-align (phones_df=None) and sidecars off, no sidecar folder created."""
    args = _word_args(tmp_path)
    args.no_syllables = True
    args.no_phonemes = True
    run_word_level_batch(args)
    out = tmp_path / 'out'
    assert not (out / 'syllable').exists()
    assert not (out / 'phoneme').exists()


def test_session_units_not_at_word_output_root(tmp_path):
    """word_session_units_*.csv must NOT appear directly in the output folder."""
    run_word_level_batch(_word_args(tmp_path))
    assert not list((tmp_path / 'out').glob('word_session_units_*.csv')), \
        'word_session_units found at root — must be in logs/'


# ---------------------------------------------------------------------------
# Document level
# ---------------------------------------------------------------------------

def test_doc_summary_lands_in_document_subfolder(tmp_path):
    run_doc_level_batch(_doc_args(tmp_path))
    out = tmp_path / 'out'
    assert list((out / 'document').glob('documents_summary_*.csv')), \
        'documents_summary not in document/'
    assert not list(out.glob('documents_summary_*.csv')), \
        'documents_summary found at root'


def test_session_units_not_at_doc_output_root(tmp_path):
    """doc_session_units_*.csv must NOT appear directly in the output folder."""
    run_doc_level_batch(_doc_args(tmp_path))
    assert not list((tmp_path / 'out').glob('doc_session_units_*.csv')), \
        'doc_session_units found at root — must be in logs/'
```

- [ ] **Step 2: Run the new tests to confirm they all fail**

```
python -m pytest tests/integration/test_output_paths.py -v
```

Expected: all 7 tests **FAIL** (features CSV still lands at root).

---

## Task 2 — `_level_dir` helper + word-level path changes

**Files:**
- Modify: `main.py`

- [ ] **Step 1: Add `_level_dir` helper after `run_doc_level_batch` imports block (~line 385)**

Insert this function immediately before `run_word_level_batch` (between the column-map block and `run_word_level_batch`):

```python
def _level_dir(output_dir: Path, level: str) -> Path:
    """Return output_dir/level, creating it if needed (lazy mkdir)."""
    d = output_dir / level
    d.mkdir(parents=True, exist_ok=True)
    return d
```

- [ ] **Step 2: Update `syll_path` inside `run_word_level_batch`**

Find (around line 597):
```python
                    syll_path = output_dir / f'{pair.stimulus_id}_syllables.csv'
```

Replace with:
```python
                    syll_path = _level_dir(output_dir, 'syllable') / f'{pair.stimulus_id}_syllables.csv'
```

- [ ] **Step 3: Update `phon_path` inside `run_word_level_batch`**

Find (around line 604):
```python
                    phon_path = output_dir / f'{pair.stimulus_id}_phonemes.csv'
```

Replace with:
```python
                    phon_path = _level_dir(output_dir, 'phoneme') / f'{pair.stimulus_id}_phonemes.csv'
```

- [ ] **Step 4: Update `features_path` inside `run_word_level_batch`**

Find (around line 618):
```python
            if not getattr(args, 'no_word_csv', False):
                features_path = output_dir / f'{pair.stimulus_id}_features.csv'
                df.to_csv(features_path, index=False)
```

Replace with:
```python
            if not getattr(args, 'no_word_csv', False):
                features_path = _level_dir(output_dir, 'word') / f'{pair.stimulus_id}_features.csv'
                df.to_csv(features_path, index=False)
```

- [ ] **Step 5: Update `units_path` inside `run_word_level_batch`**

Find (around line 655):
```python
            units_path = output_dir / f'word_session_units_{logger.run_ts}.csv'
```

Replace with:
```python
            units_path = logger.logs_dir / f'word_session_units_{logger.run_ts}.csv'
```

- [ ] **Step 6: Run new tests — word-level ones should now pass**

```
python -m pytest tests/integration/test_output_paths.py -v -k "word or sidecar or session_units_not_at_word"
```

Expected: `test_word_features_land_in_word_subfolder`, `test_word_csv_disabled_no_word_folder_not_created`, `test_sidecars_land_in_subfolders`, `test_sidecars_disabled_folders_not_created`, `test_session_units_not_at_word_output_root` all **PASS**.

---

## Task 3 — Doc-level path changes

**Files:**
- Modify: `main.py`

- [ ] **Step 1: Update `master_path` inside `run_doc_level_batch`**

Find (around line 723):
```python
    master_path = output_dir / f'documents_summary_{logger.run_ts}.csv'
```

Replace with:
```python
    master_path = _level_dir(output_dir, 'document') / f'documents_summary_{logger.run_ts}.csv'
```

- [ ] **Step 2: Update `units_path` inside `run_doc_level_batch`**

Find (around line 828):
```python
            units_path = output_dir / f'doc_session_units_{logger.run_ts}.csv'
```

Replace with:
```python
            units_path = logger.logs_dir / f'doc_session_units_{logger.run_ts}.csv'
```

- [ ] **Step 3: Run all new tests — all 7 should now pass**

```
python -m pytest tests/integration/test_output_paths.py -v
```

Expected: all 7 tests **PASS**.

---

## Task 4 — Update existing integration tests to new paths

**Files:**
- Modify: `tests/integration/test_pipeline_batch.py`
- Modify: `tests/integration/test_pipeline_batch_doc.py`

These tests assert on old flat paths. They now fail because files moved to subfolders.

- [ ] **Step 1: Confirm which old tests now fail**

```
python -m pytest tests/integration/ -v
```

Expected failures: `test_batch_processes_all_pairs`, `test_batch_emits_syllables_and_phonemes_sidecars`, `test_word_level_batch_emits_prosody_columns`, `test_sidecars_emit_when_struct_and_acoustic_per_column_toggles_off`, `test_doc_level_emits_master_csv`, `test_doc_level_master_csv_columns`, `test_doc_level_does_not_emit_word_level_csvs`.

- [ ] **Step 2: Update `test_pipeline_batch.py`**

In `test_batch_processes_all_pairs`, replace:
```python
    assert (out / 'story1_features.csv').exists()
    assert (out / 'story2_features.csv').exists()
    assert not (out / 'story3_features.csv').exists()
    assert not (out / 'story1_summary.csv').exists()
    assert not (out / 'story2_summary.csv').exists()
```
with:
```python
    assert (out / 'word' / 'story1_features.csv').exists()
    assert (out / 'word' / 'story2_features.csv').exists()
    assert not (out / 'word' / 'story3_features.csv').exists()
    assert not (out / 'story1_summary.csv').exists()
    assert not (out / 'story2_summary.csv').exists()
```

In `test_batch_emits_syllables_and_phonemes_sidecars`, replace:
```python
    csv_path = out_dir / 'tone_features.csv'
```
with:
```python
    csv_path = out_dir / 'word' / 'tone_features.csv'
```

And replace:
```python
    syll = list(out_dir.glob('*_syllables.csv'))
    phon = list(out_dir.glob('*_phonemes.csv'))
```
with:
```python
    syll = list((out_dir / 'syllable').glob('*_syllables.csv'))
    phon = list((out_dir / 'phoneme').glob('*_phonemes.csv'))
```

In `test_word_level_batch_emits_prosody_columns`, replace:
```python
    csv_path = out_dir / 'tone_features.csv'
```
with:
```python
    csv_path = out_dir / 'word' / 'tone_features.csv'
```

In `test_sidecars_emit_when_struct_and_acoustic_per_column_toggles_off`, replace:
```python
    syll_files = list(out_dir.glob('*_syllables.csv'))
    phon_files = list(out_dir.glob('*_phonemes.csv'))
```
with:
```python
    syll_files = list((out_dir / 'syllable').glob('*_syllables.csv'))
    phon_files = list((out_dir / 'phoneme').glob('*_phonemes.csv'))
```

And replace:
```python
    assert not list(out_dir.glob('*_features.csv'))
```
with:
```python
    assert not (out_dir / 'word').exists()
```

- [ ] **Step 3: Update `test_pipeline_batch_doc.py`**

In `test_doc_level_emits_master_csv`, replace:
```python
    masters = list(out.glob('documents_summary_*.csv'))
```
with:
```python
    masters = list((out / 'document').glob('documents_summary_*.csv'))
```

In `test_doc_level_master_csv_columns`, replace:
```python
    master = next(out.glob('documents_summary_*.csv'))
```
with:
```python
    master = next((out / 'document').glob('documents_summary_*.csv'))
```

In `test_doc_level_does_not_emit_word_level_csvs`, replace:
```python
    assert not list(out.glob('*_features.csv'))
```
with:
```python
    assert not (out / 'word').exists()
```

- [ ] **Step 4: Run the full integration suite**

```
python -m pytest tests/integration/ -v
```

Expected: all tests **PASS**.

- [ ] **Step 5: Run the full test suite**

```
python -m pytest -v
```

Expected: all tests **PASS** (same count as before, just updated paths).

- [ ] **Step 6: Commit**

```bash
git add main.py \
        tests/integration/test_output_paths.py \
        tests/integration/test_pipeline_batch.py \
        tests/integration/test_pipeline_batch_doc.py
git commit -m "$(cat <<'EOF'
feat: reorganize outputs into per-level subfolders

word/ → features CSV, syllable/ → syllables sidecar,
phoneme/ → phonemes sidecar, document/ → documents_summary CSV.
Folders created lazily on first write; session_units CSVs
move from output root to logs/.

Refs: docs/superpowers/specs/2026-05-22-output-reorganization-design.md

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5 — Update master_plan.md

**Files:**
- Modify: `docs/master_plan.md`

- [ ] **Step 1: Mark the batch as active in §2**

In `docs/master_plan.md`, replace the current §2 active batch text with:

```markdown
**Active batch:** Output reorganization — **in progress.**

- Spec: [`docs/superpowers/specs/2026-05-22-output-reorganization-design.md`](superpowers/specs/2026-05-22-output-reorganization-design.md)
- Plan: [`docs/superpowers/plans/2026-05-22-output-reorganization.md`](superpowers/plans/2026-05-22-output-reorganization.md)
- Status: Implementation in progress.
- Open: Run full test suite green; validate with a real ISC-Wu Tsai stimulus that the new folder structure appears.
```

- [ ] **Step 2: Commit**

```bash
git add docs/master_plan.md
git commit -m "$(cat <<'EOF'
docs: update master_plan — output reorganization batch active

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```
