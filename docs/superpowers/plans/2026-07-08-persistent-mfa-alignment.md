# Persistent, batch-once MFA alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist MFA TextGrids to `<output_folder>/Montreal Forced Alignment/`, reuse them across runs via a content-hash freshness key, and run the `mfa align` subprocess at most once per dispatch.

**Architecture:** A new batch pre-pass (`ensure_alignments`) classifies each stimulus fresh vs stale, deletes stale artifacts, and aligns the whole stale set in one `mfa align` invocation, writing `<stem>.TextGrid` + `<stem>.json` sidecar per stimulus. `align_with_phones` gains disk-read + a `run_if_missing` switch; in batch mode `run()` reads from disk only and never spawns MFA. CLI single-run is unchanged.

**Tech Stack:** Python 3.10, MFA (Kaldi) as a subprocess, pandas, pytest. No new dependencies (`hashlib`, `json`, `pathlib` are stdlib).

## Global Constraints

- The on-disk subfolder name is exactly `Montreal Forced Alignment` (spaces included), under `args.output_folder`.
- MFA acoustic model / dictionary names are `english_us_arpa` / `english_us_arpa` (constants `MFA_ACOUSTIC_MODEL` / `MFA_DICTIONARY` already in `extractors/aligner.py`).
- Sidecar freshness key = `sha256(prepare_for_aligner(transcript))` + `sha256(audio bytes)` + `audio_size` (cheap pre-filter) + both model names + `version: 1`.
- Audio identity uses a **content hash**, never mtime (stimuli get rsync'd/copied, which resets mtime).
- The `mfa align` subprocess must run **at most once per dispatch**. `ensure_alignments` must spawn zero subprocesses when the stale set is empty.
- Stem is always derived from the **audio basename** (`os.path.splitext(os.path.basename(audio_path))[0]`) on both the pre-pass and the `run()` side, so disk keys always agree.
- Environment: path-based conda prefix env. Run tests with `conda activate ./env` first (or the env's `python -m pytest`).
- TDD applies to `extractors/aligner.py` (math-adjacent). `main.py` batch wiring is test-after + a real-CLI smoke check (tests passing ≠ feature runs).

---

### Task 1: Freshness primitives in `aligner.py`

**Files:**
- Modify: `extractors/aligner.py` (add imports + helpers near the top, after the existing `MFA_ACOUSTIC_MODEL` / `MFA_DICTIONARY` constants at lines 38-39)
- Test: `tests/extractors/test_aligner_persist.py` (new)

**Interfaces:**
- Consumes: `prepare_for_aligner` (already imported at `extractors/aligner.py:35`), `MFA_ACOUSTIC_MODEL`, `MFA_DICTIONARY`.
- Produces:
  - `ALIGNMENT_SUBDIR: str = 'Montreal Forced Alignment'`
  - `SIDECAR_VERSION: int = 1`
  - `alignment_dir(output_folder) -> pathlib.Path` → `Path(output_folder) / ALIGNMENT_SUBDIR`
  - `_stem(audio_path: str) -> str`
  - `_sha256_text(s: str) -> str`
  - `_sha256_file(path: str) -> str` (streamed, 1 MiB chunks)
  - `_freshness_dict(audio_path: str, transcript_text: str) -> dict`
  - `_write_sidecar(persist_dir, stem: str, audio_path: str, transcript_text: str) -> None`
  - `_is_fresh(persist_dir, stem: str, audio_path: str, transcript_text: str) -> bool`

- [ ] **Step 1: Write the failing tests**

Create `tests/extractors/test_aligner_persist.py`:

```python
"""Tests for persistent on-disk MFA alignment (freshness + batch pre-pass).

The MFA subprocess is never invoked here: freshness tests use real temp files,
and the batch tests monkeypatch _run_mfa to synthesize TextGrids.
"""
import json
import os

import pandas as pd
import pytest

from extractors import aligner


def _make_audio(tmp_path, name="stim.wav", data=b"RIFFfake-audio-bytes"):
    p = tmp_path / name
    p.write_bytes(data)
    return str(p)


def test_alignment_dir_appends_named_subfolder(tmp_path):
    d = aligner.alignment_dir(tmp_path)
    assert d.name == "Montreal Forced Alignment"
    assert d.parent == tmp_path


def test_stem_from_audio_basename():
    assert aligner._stem("/a/b/story_01.wav") == "story_01"


def test_sha256_text_is_stable_and_prepared():
    # Hash is over the prepared (whitespace-normalized, numbers-spelled) form.
    a = aligner._sha256_text(aligner_prepared("There were 2  dogs."))
    b = aligner._sha256_text(aligner_prepared("There were two dogs."))
    assert a == b


def aligner_prepared(t):
    from utils.transcript_prep import prepare_for_aligner
    return prepare_for_aligner(t)


def test_write_and_is_fresh_roundtrip(tmp_path):
    persist = tmp_path / "align"
    persist.mkdir()
    audio = _make_audio(tmp_path)
    (persist / "stim.TextGrid").write_text("fake tg", encoding="utf-8")
    aligner._write_sidecar(persist, "stim", audio, "hello world")
    assert aligner._is_fresh(persist, "stim", audio, "hello world") is True


def test_is_fresh_false_when_textgrid_missing(tmp_path):
    persist = tmp_path / "align"
    persist.mkdir()
    audio = _make_audio(tmp_path)
    aligner._write_sidecar(persist, "stim", audio, "hello")  # sidecar only
    assert aligner._is_fresh(persist, "stim", audio, "hello") is False


def test_is_fresh_false_on_transcript_change(tmp_path):
    persist = tmp_path / "align"
    persist.mkdir()
    audio = _make_audio(tmp_path)
    (persist / "stim.TextGrid").write_text("fake tg", encoding="utf-8")
    aligner._write_sidecar(persist, "stim", audio, "hello")
    assert aligner._is_fresh(persist, "stim", audio, "goodbye") is False


def test_is_fresh_false_on_audio_change(tmp_path):
    persist = tmp_path / "align"
    persist.mkdir()
    audio = _make_audio(tmp_path)
    (persist / "stim.TextGrid").write_text("fake tg", encoding="utf-8")
    aligner._write_sidecar(persist, "stim", audio, "hello")
    # Rewrite audio bytes (same length => exercises the content hash, not size).
    with open(audio, "wb") as f:
        f.write(b"RIFFdifferent-bytes!")
    assert aligner._is_fresh(persist, "stim", audio, "hello") is False


def test_is_fresh_false_on_model_change(tmp_path, monkeypatch):
    persist = tmp_path / "align"
    persist.mkdir()
    audio = _make_audio(tmp_path)
    (persist / "stim.TextGrid").write_text("fake tg", encoding="utf-8")
    aligner._write_sidecar(persist, "stim", audio, "hello")
    monkeypatch.setattr(aligner, "MFA_ACOUSTIC_MODEL", "some_other_model")
    assert aligner._is_fresh(persist, "stim", audio, "hello") is False
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/extractors/test_aligner_persist.py -v`
Expected: FAIL — `AttributeError: module 'extractors.aligner' has no attribute 'alignment_dir'` (and similar).

- [ ] **Step 3: Add imports and helpers to `extractors/aligner.py`**

Add to the import block (after `import tempfile` at line 31):

```python
import hashlib
import json
from pathlib import Path
```

Add after the model-name constants (`MFA_DICTIONARY = 'english_us_arpa'`, line 39):

```python
ALIGNMENT_SUBDIR = 'Montreal Forced Alignment'
SIDECAR_VERSION = 1


def alignment_dir(output_folder) -> Path:
    """Return <output_folder>/Montreal Forced Alignment (not created here)."""
    return Path(output_folder) / ALIGNMENT_SUBDIR


def _stem(audio_path: str) -> str:
    """Stimulus id used for on-disk artifact names: the audio basename sans ext.

    Both the batch pre-pass and align_with_phones derive the stem this way, so
    the TextGrid a pass writes is the one a later pass looks up.
    """
    return os.path.splitext(os.path.basename(audio_path))[0]


def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode('utf-8')).hexdigest()


def _sha256_file(path: str, _chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(_chunk), b''):
            h.update(block)
    return h.hexdigest()


def _freshness_dict(audio_path: str, transcript_text: str) -> dict:
    """The sidecar contents identifying the inputs a TextGrid was aligned from."""
    return {
        'transcript_sha256': _sha256_text(prepare_for_aligner(transcript_text)),
        'audio_sha256': _sha256_file(audio_path),
        'audio_size': os.path.getsize(audio_path),
        'mfa_acoustic': MFA_ACOUSTIC_MODEL,
        'mfa_dictionary': MFA_DICTIONARY,
        'version': SIDECAR_VERSION,
    }


def _write_sidecar(persist_dir, stem: str, audio_path: str,
                   transcript_text: str) -> None:
    meta = _freshness_dict(audio_path, transcript_text)
    (Path(persist_dir) / f'{stem}.json').write_text(
        json.dumps(meta, indent=2, sort_keys=True), encoding='utf-8')


def _is_fresh(persist_dir, stem: str, audio_path: str,
              transcript_text: str) -> bool:
    """True iff a reusable, current TextGrid exists for this stimulus.

    Requires the TextGrid and sidecar to exist and every key in the sidecar to
    match the current inputs. audio_size is a cheap pre-filter: a size mismatch
    short-circuits to stale without hashing the audio.
    """
    persist_dir = Path(persist_dir)
    tg = persist_dir / f'{stem}.TextGrid'
    side = persist_dir / f'{stem}.json'
    if not tg.exists() or not side.exists():
        return False
    try:
        meta = json.loads(side.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return False
    if meta.get('version') != SIDECAR_VERSION:
        return False
    if meta.get('mfa_acoustic') != MFA_ACOUSTIC_MODEL:
        return False
    if meta.get('mfa_dictionary') != MFA_DICTIONARY:
        return False
    if meta.get('transcript_sha256') != _sha256_text(
            prepare_for_aligner(transcript_text)):
        return False
    try:
        if meta.get('audio_size') != os.path.getsize(audio_path):
            return False
    except OSError:
        return False
    return meta.get('audio_sha256') == _sha256_file(audio_path)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/extractors/test_aligner_persist.py -v`
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
git add extractors/aligner.py tests/extractors/test_aligner_persist.py
git commit -m "feat(aligner): freshness primitives for persistent MFA cache"
```

---

### Task 2: Disk-read + `run_if_missing` in `align_with_phones`

**Files:**
- Modify: `extractors/aligner.py` — extract `_parse_textgrid`, refactor `_align_uncached` tail (lines 100-116) to use it, extend `align_with_phones` (lines 42-75).
- Test: `tests/extractors/test_aligner_persist.py` (append)

**Interfaces:**
- Consumes: `_stem`, `alignment_dir` (Task 1); `parse_words`/`parse_phones` (imported at `extractors/aligner.py:36`).
- Produces:
  - `_parse_textgrid(tg_path: str) -> tuple[pd.DataFrame, pd.DataFrame]` — parses + renames columns to `Transcript`/`Onset`/`End`; identical schema to `_align_uncached`'s return.
  - `align_with_phones(audio_path, transcript, cache=None, persist_dir=None, run_if_missing=True) -> tuple[pd.DataFrame, pd.DataFrame]` — lookup order: in-memory `cache` → disk TextGrid under `persist_dir` (parse, no MFA) → `_align_uncached` if `run_if_missing`, else raise. **Never writes to disk** (only `ensure_alignments` writes).

- [ ] **Step 1: Write the failing tests** (append to `tests/extractors/test_aligner_persist.py`)

```python
def _fake_frames():
    words = pd.DataFrame({"Transcript": ["a"], "Onset": [0.0], "End": [0.1]})
    phones = pd.DataFrame({"phone": ["AH0"], "onset": [0.0], "offset": [0.1]})
    return words, phones


def test_disk_hit_parses_without_running_mfa(tmp_path, monkeypatch):
    persist = tmp_path / "align"
    persist.mkdir()
    (persist / "stim.TextGrid").write_text("fake tg", encoding="utf-8")
    called = []
    monkeypatch.setattr(aligner, "_align_uncached",
                        lambda a, t: called.append(1))
    monkeypatch.setattr(aligner, "_parse_textgrid", lambda p: _fake_frames())
    w, p = aligner.align_with_phones(
        str(tmp_path / "stim.wav"), "hello", persist_dir=persist)
    assert called == [], "disk hit must not invoke MFA"
    assert list(w["Onset"]) == [0.0]


def test_run_if_missing_false_raises_on_disk_miss(tmp_path, monkeypatch):
    persist = tmp_path / "align"
    persist.mkdir()  # empty: no TextGrid for this stem
    monkeypatch.setattr(aligner, "_align_uncached",
                        lambda a, t: pytest.fail("MFA must not run"))
    with pytest.raises(RuntimeError):
        aligner.align_with_phones(
            str(tmp_path / "stim.wav"), "hello",
            persist_dir=persist, run_if_missing=False)


def test_run_if_missing_true_falls_back_to_mfa(tmp_path, monkeypatch):
    persist = tmp_path / "align"
    persist.mkdir()
    monkeypatch.setattr(aligner, "_align_uncached", lambda a, t: _fake_frames())
    w, _ = aligner.align_with_phones(
        str(tmp_path / "stim.wav"), "hello",
        persist_dir=persist, run_if_missing=True)
    assert list(w["Onset"]) == [0.0]
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/extractors/test_aligner_persist.py -k "disk_hit or run_if_missing" -v`
Expected: FAIL — `TypeError: align_with_phones() got an unexpected keyword argument 'persist_dir'`.

- [ ] **Step 3: Extract `_parse_textgrid` and refactor `_align_uncached`**

In `extractors/aligner.py`, replace the tail of `_align_uncached` (current lines 100-116, from `tg_path = ...` through `return word_df, phone_df`) with:

```python
        tg_path = os.path.join(output_dir, f'{stem}.TextGrid')
        if not os.path.exists(tg_path):
            raise RuntimeError(
                f"MFA did not produce a TextGrid for {stem}. "
                "Check that the audio and transcript match, and that MFA models are downloaded."
            )
        return _parse_textgrid(tg_path)


def _parse_textgrid(tg_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse an MFA TextGrid into (word_df, phone_df) with pipeline columns.

    word_df columns: Transcript / Onset / End. phone_df columns: phone / onset /
    offset. Shared by both the fresh-MFA path and the disk-reuse path so the two
    return identical schemas.
    """
    word_df = parse_words(tg_path)
    phone_df = parse_phones(tg_path)
    word_df = word_df.rename(columns={
        'word': 'Transcript',
        'onset': 'Onset',
        'offset': 'End',
    })
    return word_df, phone_df
```

Note: the `word_df, phone_df = parse_words(...); parse_phones(...)` block that previously lived *inside* the `with tempfile...` context now returns inside the context via `_parse_textgrid` — that is correct, the TextGrid is read before the temp dir is cleaned.

- [ ] **Step 4: Rewrite `align_with_phones`**

Replace the body of `align_with_phones` (current lines 69-75, the `if cache is None: ...` block) — keep the docstring, update the signature. New signature + body:

```python
def align_with_phones(
    audio_path: str,
    transcript: str,
    cache: dict | None = None,
    persist_dir=None,
    run_if_missing: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # (docstring: keep existing text, add persist_dir / run_if_missing notes)
    key = (os.path.abspath(audio_path), transcript)
    if cache is not None and key in cache:
        word_df, phone_df = cache[key]
        return word_df.copy(), phone_df.copy()

    result = None
    if persist_dir is not None:
        tg = Path(persist_dir) / f'{_stem(audio_path)}.TextGrid'
        if tg.exists():
            result = _parse_textgrid(str(tg))
    if result is None:
        if not run_if_missing:
            raise RuntimeError(
                f"No alignment on disk for {_stem(audio_path)!r} under "
                f"{persist_dir} and MFA is disabled for this pass. The batch "
                "pre-alignment step should have produced it; this stimulus "
                "likely failed to align."
            )
        result = _align_uncached(audio_path, transcript)

    if cache is not None:
        cache[key] = result
        word_df, phone_df = result
        return word_df.copy(), phone_df.copy()
    return result
```

Also update the docstring `Args:` section to document `persist_dir` (optional dir holding reusable `<stem>.TextGrid`; read-only here) and `run_if_missing` (when False, a disk miss raises instead of invoking MFA — used by batch passes after pre-alignment).

- [ ] **Step 5: Run the new tests + the existing aligner tests to verify all pass**

Run: `python -m pytest tests/extractors/test_aligner.py tests/extractors/test_aligner_persist.py -v`
Expected: PASS — the 4 pre-existing `test_aligner.py` tests still pass (new params default to `None`/`True`), plus the new disk/run_if_missing tests.

- [ ] **Step 6: Commit**

```bash
git add extractors/aligner.py tests/extractors/test_aligner_persist.py
git commit -m "feat(aligner): disk-reuse + run_if_missing in align_with_phones"
```

---

### Task 3: `ensure_alignments` batch pre-pass

**Files:**
- Modify: `extractors/aligner.py` (add `ensure_alignments`)
- Test: `tests/extractors/test_aligner_persist.py` (append)

**Interfaces:**
- Consumes: `_stem`, `_is_fresh`, `_write_sidecar` (Task 1); `_run_mfa` (existing, `extractors/aligner.py:133`); `prepare_for_aligner`.
- Produces:
  - `ensure_alignments(items, persist_dir) -> dict` — `items` is an iterable of `(audio_path, transcript_text)`. Ensures a fresh `<stem>.TextGrid` (+ sidecar) exists under `persist_dir` for every alignable item, running `_run_mfa` **at most once** for the whole stale set. Deletes stale artifacts before aligning so a failed realign leaves no file. Returns `{'cached': [stem...], 'aligned': [stem...], 'failed': [stem...]}`.

- [ ] **Step 1: Write the failing tests** (append)

```python
def _install_fake_mfa(monkeypatch, aligned_stems=None):
    """Replace _run_mfa with one that writes a TextGrid per staged .lab.

    aligned_stems=None => align everything staged; otherwise only those stems
    get a TextGrid (simulating per-stimulus alignment failure).
    """
    runs = {"count": 0}

    def fake(corpus_dir, output_dir):
        runs["count"] += 1
        for name in os.listdir(corpus_dir):
            if not name.endswith(".lab"):
                continue
            stem = name[:-4]
            if aligned_stems is not None and stem not in aligned_stems:
                continue
            with open(os.path.join(output_dir, f"{stem}.TextGrid"), "w") as f:
                f.write(f"tg for {stem}")

    monkeypatch.setattr(aligner, "_run_mfa", fake)
    return runs


def test_ensure_aligns_all_missing_in_one_run(tmp_path, monkeypatch):
    persist = tmp_path / "align"
    a1 = _make_audio(tmp_path, "one.wav", b"aaaa")
    a2 = _make_audio(tmp_path, "two.wav", b"bbbb")
    runs = _install_fake_mfa(monkeypatch)
    res = aligner.ensure_alignments([(a1, "hi one"), (a2, "hi two")], persist)
    assert runs["count"] == 1, "one MFA invocation for the whole batch"
    assert set(res["aligned"]) == {"one", "two"}
    assert (persist / "one.TextGrid").exists()
    assert (persist / "two.json").exists()


def test_ensure_skips_mfa_when_all_fresh(tmp_path, monkeypatch):
    persist = tmp_path / "align"
    a1 = _make_audio(tmp_path, "one.wav", b"aaaa")
    _install_fake_mfa(monkeypatch)  # first call populates disk
    aligner.ensure_alignments([(a1, "hi one")], persist)
    runs2 = _install_fake_mfa(monkeypatch)  # fresh counter
    res = aligner.ensure_alignments([(a1, "hi one")], persist)
    assert runs2["count"] == 0, "no MFA when everything is fresh"
    assert res["cached"] == ["one"] and res["aligned"] == []


def test_ensure_deletes_stale_before_realign(tmp_path, monkeypatch):
    persist = tmp_path / "align"
    a1 = _make_audio(tmp_path, "one.wav", b"aaaa")
    _install_fake_mfa(monkeypatch)
    aligner.ensure_alignments([(a1, "old transcript")], persist)
    # New transcript => stale. MFA now "fails" for this stem (produces nothing).
    _install_fake_mfa(monkeypatch, aligned_stems=set())
    res = aligner.ensure_alignments([(a1, "new transcript")], persist)
    assert res["failed"] == ["one"]
    assert not (persist / "one.TextGrid").exists(), \
        "stale TextGrid must be removed, not left for reuse"


def test_ensure_only_realigns_stale_subset(tmp_path, monkeypatch):
    persist = tmp_path / "align"
    a1 = _make_audio(tmp_path, "one.wav", b"aaaa")
    a2 = _make_audio(tmp_path, "two.wav", b"bbbb")
    _install_fake_mfa(monkeypatch)
    aligner.ensure_alignments([(a1, "hi one"), (a2, "hi two")], persist)
    # Change only two's transcript.
    staged = []
    orig = aligner._run_mfa

    def spy(corpus_dir, output_dir):
        staged.extend(n[:-4] for n in os.listdir(corpus_dir)
                      if n.endswith(".lab"))
        for s in staged:
            with open(os.path.join(output_dir, f"{s}.TextGrid"), "w") as f:
                f.write("tg")
    monkeypatch.setattr(aligner, "_run_mfa", spy)
    res = aligner.ensure_alignments([(a1, "hi one"), (a2, "CHANGED")], persist)
    assert staged == ["two"], "only the stale stimulus is re-aligned"
    assert res["cached"] == ["one"] and res["aligned"] == ["two"]
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/extractors/test_aligner_persist.py -k ensure -v`
Expected: FAIL — `AttributeError: module 'extractors.aligner' has no attribute 'ensure_alignments'`.

- [ ] **Step 3: Implement `ensure_alignments`**

Add to `extractors/aligner.py` (after `_align_uncached` / `_parse_textgrid`):

```python
def ensure_alignments(items, persist_dir) -> dict:
    """Ensure a fresh <stem>.TextGrid exists under persist_dir for each item.

    items: iterable of (audio_path, transcript_text). The stem is derived from
    the audio basename. Fresh stimuli (matching sidecar) are left untouched;
    stale/missing ones have their old artifacts deleted, then the whole stale
    set is aligned in a SINGLE `mfa align` invocation and each result is moved
    into persist_dir with a sidecar. If nothing is stale, no subprocess runs.

    Returns {'cached': [...], 'aligned': [...], 'failed': [...]} of stems.
    """
    persist_dir = Path(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    cached: list[str] = []
    # Dedup by stem (unique stimulus ids expected); last transcript wins.
    stale_by_stem: dict[str, tuple[str, str]] = {}
    for audio_path, transcript_text in items:
        stem = _stem(audio_path)
        if _is_fresh(persist_dir, stem, audio_path, transcript_text):
            cached.append(stem)
        else:
            if stem in stale_by_stem:
                print(f"[mfa] WARNING: duplicate stem {stem!r} in batch; "
                      "last transcript wins")
            stale_by_stem[stem] = (audio_path, transcript_text)

    if not stale_by_stem:
        print(f"[mfa] {len(cached)} stimuli: all cached, 0 aligned")
        return {'cached': cached, 'aligned': [], 'failed': []}

    # Delete stale artifacts first so a failed realign can't leave a stale file
    # for run() to pick up.
    for stem in stale_by_stem:
        (persist_dir / f'{stem}.TextGrid').unlink(missing_ok=True)
        (persist_dir / f'{stem}.json').unlink(missing_ok=True)

    aligned: list[str] = []
    failed: list[str] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        corpus_dir = os.path.join(tmpdir, 'corpus')
        output_dir = os.path.join(tmpdir, 'output')
        os.makedirs(corpus_dir)
        os.makedirs(output_dir)
        for stem, (audio_path, transcript_text) in stale_by_stem.items():
            shutil.copy(audio_path, os.path.join(corpus_dir, f'{stem}.wav'))
            with open(os.path.join(corpus_dir, f'{stem}.lab'), 'w',
                      encoding='utf-8') as f:
                f.write(prepare_for_aligner(transcript_text))

        _run_mfa(corpus_dir, output_dir)   # single invocation for the batch

        for stem, (audio_path, transcript_text) in stale_by_stem.items():
            produced = os.path.join(output_dir, f'{stem}.TextGrid')
            if os.path.exists(produced):
                shutil.move(produced, str(persist_dir / f'{stem}.TextGrid'))
                _write_sidecar(persist_dir, stem, audio_path, transcript_text)
                aligned.append(stem)
            else:
                failed.append(stem)

    msg = (f"[mfa] {len(cached) + len(stale_by_stem)} stimuli: "
           f"{len(cached)} cached, {len(aligned)} aligned in 1 run")
    if failed:
        msg += f", {len(failed)} failed ({', '.join(failed)})"
    print(msg)
    return {'cached': cached, 'aligned': aligned, 'failed': failed}
```

- [ ] **Step 4: Run to verify all pass**

Run: `python -m pytest tests/extractors/test_aligner_persist.py -v`
Expected: PASS (all Task 1–3 tests).

- [ ] **Step 5: Commit**

```bash
git add extractors/aligner.py tests/extractors/test_aligner_persist.py
git commit -m "feat(aligner): ensure_alignments batch pre-pass (one MFA run)"
```

---

### Task 4: Wire the pre-pass into `run()` and both batch functions

**Files:**
- Modify: `main.py` — `run()` alignment block (lines 402-412); `run_word_level_batch` (after `work_items = _batch_work_items(pairing)` at line 779); `run_doc_level_batch` (after `work_items = _batch_work_items(pairing)` at line 1091).
- Test: `tests/test_run_prealigned.py` (new, light test-after)

**Interfaces:**
- Consumes: `alignment_dir`, `ensure_alignments`, `align_with_phones` (Tasks 1–3).
- Produces: batch passes set `<args>._batch_prealigned = True` and call `ensure_alignments` once; `run()` passes `persist_dir` + `run_if_missing` into `align_with_phones`.

- [ ] **Step 1: Update `run()`'s alignment block**

Replace `main.py` lines 402-412 (the `if not args.no_align and args.audio:` block up to and including the `align_df, phones_df = mfa_align(...)` line) with:

```python
    if not args.no_align and args.audio:
        from extractors.aligner import (
            align_with_phones as mfa_align, alignment_dir,
        )
        # Reuse a per-document alignment across passes (word + doc batches in
        # one GUI run share `_align_cache`), so MFA runs at most once per doc.
        _align_cache = getattr(args, '_align_cache', None)
        # In batch mode the pre-pass (run_*_level_batch) has already aligned
        # every stimulus to <output_folder>/Montreal Forced Alignment/, so run()
        # reads TextGrids from disk and must NOT spawn MFA (run_if_missing=False).
        # CLI single-run has no output_folder: persist_dir=None, MFA runs once.
        persist_dir = (alignment_dir(args.output_folder)
                       if getattr(args, 'output_folder', None) else None)
        run_if_missing = not getattr(args, '_batch_prealigned', False)
        if _align_cache is not None and (
                os.path.abspath(args.audio), text) in _align_cache:
            print('[mfa] Reusing cached alignment…')
        else:
            print('[mfa] Loading alignment…')
        align_df, phones_df = mfa_align(
            args.audio, text, cache=_align_cache,
            persist_dir=persist_dir, run_if_missing=run_if_missing)
```

(Leave lines 413-417 — `args._phones_df = phones_df` etc. — unchanged.)

- [ ] **Step 2: Add the pre-pass to `run_word_level_batch`**

In `main.py`, immediately after `work_items = _batch_work_items(pairing)` (line 779), before the `for i, pair in enumerate(work_items):` loop, insert:

```python
    _prealign_batch(args, work_items, output_dir)
```

- [ ] **Step 3: Add the pre-pass to `run_doc_level_batch`**

In `main.py`, immediately after `work_items = _batch_work_items(pairing)` (line 1091), insert (note: doc batch runs `run(extract_args)`, so flag `extract_args`):

```python
    _prealign_batch(extract_args, work_items, output_dir)
```

- [ ] **Step 4: Add the shared `_prealign_batch` helper**

Add near the other module-level batch helpers in `main.py` (e.g. just above `def run_word_level_batch`):

```python
def _prealign_batch(run_args, work_items, output_dir) -> None:
    """Align every audio-bearing stimulus in the batch once, up front.

    Populates <output_dir>/Montreal Forced Alignment/ so the per-document run()
    calls read TextGrids from disk instead of each spawning MFA. Sets
    run_args._batch_prealigned so run() knows not to fall back to MFA on a disk
    miss (a miss then means that stimulus failed to align).

    No-op when alignment is disabled or no pair has audio.
    """
    if getattr(run_args, 'no_align', False):
        return
    from extractors.aligner import alignment_dir, ensure_alignments
    items = []
    for pair in work_items:
        if not pair.audio_path:
            continue
        with open(pair.transcript_path, 'r', encoding='utf-8') as f:
            transcript_text = f.read()
        items.append((str(pair.audio_path), transcript_text))
    if items:
        ensure_alignments(items, alignment_dir(output_dir))
    run_args._batch_prealigned = True
```

- [ ] **Step 5: Write a light test for the disk-only run path** (`tests/test_run_prealigned.py`, new)

```python
"""run() in a pre-aligned batch reads TextGrids from disk, never spawns MFA."""
import types

import pandas as pd
import pytest

import main
from extractors import aligner


def test_run_reads_disk_and_never_calls_mfa(tmp_path, monkeypatch):
    out = tmp_path / "out"
    persist = aligner.alignment_dir(out)
    persist.mkdir(parents=True)
    (persist / "stim.TextGrid").write_text("tg", encoding="utf-8")

    transcript = tmp_path / "stim.txt"
    transcript.write_text("hello world", encoding="utf-8")
    audio = tmp_path / "stim.wav"
    audio.write_bytes(b"RIFFfake")

    # Disk parse returns canned frames; MFA must never be invoked.
    monkeypatch.setattr(
        aligner, "_parse_textgrid",
        lambda p: (pd.DataFrame({"Transcript": ["hello", "world"],
                                 "Onset": [0.0, 0.5], "End": [0.4, 0.9]}),
                   pd.DataFrame({"phone": ["HH"], "onset": [0.0],
                                 "offset": [0.1]})))
    monkeypatch.setattr(aligner, "_align_uncached",
                        lambda a, t: pytest.fail("MFA must not run"))

    args = types.SimpleNamespace(
        transcript=str(transcript), audio=str(audio), output_folder=str(out),
        _batch_prealigned=True, doc_id=1,
        no_parse=True, no_freq=True, no_edu_freq=True, no_lexical_norms=True,
        no_sem_domain=True, no_surprisal=True, no_llm_cloze=True,
        no_align=False, no_prosody=True, sem_categories="",
    )
    df = main.run(args)
    assert "Onset" in df.columns
    assert list(df["Onset"]) == [0.0, 0.5]
```

- [ ] **Step 6: Run the test**

Run: `python -m pytest tests/test_run_prealigned.py -v`
Expected: PASS. If it fails on a missing `args` attribute, add that attribute (default disabling the extractor) to the namespace — the test only exercises the alignment path.

- [ ] **Step 7: Commit**

```bash
git add main.py tests/test_run_prealigned.py
git commit -m "feat: batch pre-align once, run() reads TextGrids from disk"
```

---

### Task 5: Full-suite regression + real CLI smoke check

**Files:** none (verification only). Per the env-sync and MKL memories: a green test suite does not prove the feature runs — exercise the real CLI.

- [ ] **Step 1: Run the full test suite**

Run: `python -m pytest -q`
Expected: PASS with no new failures (baseline was 490 passed, 2 skipped; this plan adds tests). If it dies on `ModuleNotFoundError`, the `./env` is out of sync — run `pip install -r requirements.txt` inside the activated env, then re-run.

- [ ] **Step 2: Real CLI batch smoke — first run aligns, writes artifacts**

Pick a small audio+transcript folder (an existing test stimulus set) and run a batch (e.g. via the GUI, or a CLI batch invocation with `--audio-folder`/`--transcripts-folder`/`--output-folder`). Confirm:
- `<output_folder>/Montreal Forced Alignment/<stem>.TextGrid` and `<stem>.json` are created.
- The log shows `[mfa] N stimuli: 0 cached, N aligned in 1 run`.

- [ ] **Step 3: Real CLI batch smoke — second run reuses, no MFA**

Re-run the identical batch. Confirm:
- The log shows `[mfa] N stimuli: N cached, 0 aligned` (no `mfa align` subprocess; noticeably faster startup).
- Output CSVs are byte-identical / feature-identical to the first run.

- [ ] **Step 4: Real CLI staleness check**

Edit one transcript, re-run. Confirm only that stimulus re-aligns (`1 aligned`, rest `cached`) and its TextGrid/sidecar are rewritten.

- [ ] **Step 5: Update project docs**

Add a one-line note to `docs/master_plan.md`'s completed-batches area recording that MFA output now persists to `<output_folder>/Montreal Forced Alignment/` and is reused across runs. Commit:

```bash
git add docs/master_plan.md
git commit -m "docs: record persistent MFA alignment cache"
```

---

## Self-Review

**Spec coverage:**
- On-disk layout + subfolder name → Task 1 (`alignment_dir`, `ALIGNMENT_SUBDIR`), Task 3 (writes artifacts). ✓
- Sidecar freshness key (transcript hash + audio hash + size pre-filter + model names + version) → Task 1 (`_freshness_dict`, `_is_fresh`). ✓
- Content hash not mtime → Task 1 (`_sha256_file`; no mtime anywhere). ✓
- `ensure_alignments`: fresh vs stale, delete-stale-first, one invocation, move+sidecar, empty-set no-op, failure leaves no file → Task 3 (all covered by tests). ✓
- `align_with_phones` gains `persist_dir` + `run_if_missing`; disk-read, no disk-write, raise on disk-miss → Task 2. ✓
- Batch functions call pre-pass, set `_batch_prealigned`; `run()` passes params; one MFA per dispatch (word populates, doc reuses) → Task 4. ✓
- Acoustic pass untouched; CLI single-run unchanged (`persist_dir=None`, `run_if_missing=True`) → Task 4 (`run()` gates on `output_folder`/`_batch_prealigned`). ✓
- Stem derived from audio basename on both sides → Task 1 (`_stem`), used by Task 2/3/`run()`. ✓
- In-memory `_align_cache` preserved → Task 2 (cache checked first, populated). ✓

**Placeholder scan:** No TBD/TODO; every code step shows complete code. ✓

**Type consistency:** `alignment_dir`→`Path`; `ensure_alignments` returns `{'cached','aligned','failed'}` (used in tests); `align_with_phones(audio_path, transcript, cache, persist_dir, run_if_missing)` signature consistent across `run()` call site and tests; `_parse_textgrid` returns the same `(word_df, phone_df)` schema used by `_align_uncached` and the disk-read path. ✓
