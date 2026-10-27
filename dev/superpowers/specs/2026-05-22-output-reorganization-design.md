# Output Folder Reorganization — Design Spec

**Date:** 2026-05-22
**Batch:** Output reorganization (batch 1 of 3 in the sentence-level + embeddings roadmap)
**Status:** Approved for implementation

---

## 1. Motivation

The output folder is currently flat — all CSVs land directly in `<output_folder>/`. Adding sentence-level CSVs, embedding `.mat` files, and future sound-level outputs would make this unmanageable. This batch establishes a per-level subfolder structure before any new output types land.

---

## 2. New folder structure

```
<output_folder>/
├── word/
│   └── <stim>_features.csv
├── sentence/               ← created by batch 2 (sentence-level tier)
├── document/
│   └── documents_summary_<ts>.csv
├── syllable/
│   └── <stim>_syllables.csv
├── phoneme/
│   └── <stim>_phonemes.csv
├── sound/                  ← reserved for future sound-level features
└── logs/                   ← already exists; expands to include session_units
    ├── word_run_<ts>.log
    ├── word_run_<ts>_manifest.json
    ├── word_run_<ts>_no-match.csv     (only if unmatched files exist)
    ├── doc_run_<ts>.log
    ├── doc_run_<ts>_manifest.json
    ├── word_session_units_<ts>.csv    (moved from output root)
    └── doc_session_units_<ts>.csv     (moved from output root)
```

**Lazy creation rule:** each subfolder is created only when output is actually written to it. `logs/` is always created (RunLogger already does this). `sentence/` and `sound/` are not created until their respective batches land.

**Embeddings (batch 3)** will land in `word/` and `sentence/` alongside CSVs — the folder is level-based, not format-based.

---

## 3. Old → new path mapping

| Artifact | Old path | New path |
|---|---|---|
| Word features CSV | `<out>/<stim>_features.csv` | `<out>/word/<stim>_features.csv` |
| Syllables CSV | `<out>/<stim>_syllables.csv` | `<out>/syllable/<stim>_syllables.csv` |
| Phonemes CSV | `<out>/<stim>_phonemes.csv` | `<out>/phoneme/<stim>_phonemes.csv` |
| Document summary CSV | `<out>/documents_summary_<ts>.csv` | `<out>/document/documents_summary_<ts>.csv` |
| Word session units | `<out>/word_session_units_<ts>.csv` | `<out>/logs/word_session_units_<ts>.csv` |
| Doc session units | `<out>/doc_session_units_<ts>.csv` | `<out>/logs/doc_session_units_<ts>.csv` |
| Manifests, logs, no-match | `<out>/logs/` | `<out>/logs/` ← unchanged |

---

## 4. Code changes

### 4.1 `main.py` — new helper

Add a private helper for lazy subfolder creation:

```python
def _level_dir(output_dir: Path, level: str) -> Path:
    d = output_dir / level
    d.mkdir(parents=True, exist_ok=True)
    return d
```

### 4.2 `main.py` — `run_word_level_batch()`

Four path substitutions:

| Variable | Old | New |
|---|---|---|
| `features_path` | `output_dir / f'{stim}_features.csv'` | `_level_dir(output_dir, 'word') / f'{stim}_features.csv'` |
| `syll_path` | `output_dir / f'{stim}_syllables.csv'` | `_level_dir(output_dir, 'syllable') / f'{stim}_syllables.csv'` |
| `phon_path` | `output_dir / f'{stim}_phonemes.csv'` | `_level_dir(output_dir, 'phoneme') / f'{stim}_phonemes.csv'` |
| `units_path` | `output_dir / f'word_session_units_{ts}.csv'` | `logger.logs_dir / f'word_session_units_{ts}.csv'` |

`_level_dir` for each level is called lazily inside its existing write guard:
- `word/` — inside `if not getattr(args, 'no_word_csv', False)`
- `syllable/` — inside `if not getattr(args, 'no_syllables', False) and len(syll_df) > 0`
- `phoneme/` — inside `if not getattr(args, 'no_phonemes', False) and len(phon_df) > 0`

No folder is created unless its output is actually written.

### 4.3 `main.py` — `run_doc_level_batch()`

Two path substitutions:

| Variable | Old | New |
|---|---|---|
| `master_path` | `output_dir / f'documents_summary_{ts}.csv'` | `_level_dir(output_dir, 'document') / f'documents_summary_{ts}.csv'` |
| `units_path` | `output_dir / f'doc_session_units_{ts}.csv'` | `logger.logs_dir / f'doc_session_units_{ts}.csv'` |

### 4.4 No other files change

`utils/run_logger.py` — already writes all artifacts (manifest, `.log`, `no-match.csv`) to `<output>/logs/`. No changes needed.

`gui/run_dispatch.py` — passes `output_folder` from prefs; constructs no paths of its own. No changes needed.

`gui/section_run.py` — "open output folder" action points at `output_dir` root, which now shows the level subfolders. No change needed; this is the intended UX.

---

## 5. Prefs schema

No schema version bump — no prefs keys change. The output folder path stored in `shared.last_output_folder` remains the root; subfolders are an internal implementation detail.

---

## 6. Backward compatibility

**Breaking change.** Any downstream script pointing at `<output>/<stim>_features.csv` (or the other flat paths) must be updated to the new paths in §3. Since SensiExtractor is an internal research tool with no published API, a clean break is appropriate. Downstream consumers (e.g., scripts in `../Data Curation/`) should be updated alongside this batch.

---

## 7. Testing

### Update existing tests
Any test that asserts on output file paths must switch to the new subfolder paths.

### New tests (in `tests/test_output_paths.py` or the existing batch test file)

| Test | Assertion |
|---|---|
| Word batch, all features enabled | `word/` exists; `syllable/` and `phoneme/` exist |
| Word batch, syllables disabled | `syllable/` does not exist |
| Word batch, phonemes disabled | `phoneme/` does not exist |
| Word batch, word CSV disabled | `word/` does not exist |
| Doc batch | `document/` exists |
| Both batches | `logs/` contains both `word_session_units_<ts>.csv` and `doc_session_units_<ts>.csv` |
| Neither batch | Only `logs/` exists in output root |

All tests use `tmp_path` fixtures with synthetic CSV inputs; no real audio needed.
