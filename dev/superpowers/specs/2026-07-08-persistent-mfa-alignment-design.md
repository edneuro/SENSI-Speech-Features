# Persistent, batch-once MFA alignment

**Date:** 2026-07-08
**Status:** Design approved — ready for implementation plan

## Problem

Forced alignment is the slow step in the pipeline, and today it is slower than it
needs to be for two compounding reasons:

1. **One MFA subprocess per document.** `run()` in `main.py` calls
   `align_with_phones` once per document, and `_align_uncached`
   (`extractors/aligner.py`) spins up a fresh `mfa align` subprocess each time.
   Every invocation reloads the Kaldi acoustic model + dictionary from scratch —
   that model reload, repeated N times, is the dominant cost.

2. **Results are discarded after every run.** The only cache is an in-memory
   dict (`align_cache` in `gui/run_dispatch.py`, threaded through as
   `args._align_cache`) shared across the word + doc passes of a *single*
   dispatch. When the run ends the dict is gone, so the next run realigns
   everything from zero.

## Goal

- Persist MFA output to disk so alignment survives across runs.
- Run the `mfa align` subprocess **at most once per dispatch** — collect every
  stimulus that needs alignment and align them together in a single invocation,
  rather than one-per-document.
- Never silently reuse a stale alignment (transcript or audio changed).
- Leave CLI single-run behavior and all downstream stages unchanged.

## On-disk layout

```
<output_folder>/Montreal Forced Alignment/
    <stem>.TextGrid      MFA's native output, re-parsed on reuse
    <stem>.json          sidecar: freshness key
```

`<output_folder>` is `args.output_folder` (the batch output root). The subfolder
name is `Montreal Forced Alignment` verbatim (spaces included).

### Sidecar freshness key

```json
{
  "transcript_sha256": "sha256(prepare_for_aligner(transcript))",
  "audio_sha256": "sha256 of the audio file bytes",
  "audio_size": 12345678,
  "mfa_acoustic": "english_us_arpa",
  "mfa_dictionary": "english_us_arpa",
  "version": 1
}
```

A stimulus is **fresh** iff `<stem>.TextGrid` exists AND the sidecar's
`transcript_sha256`, `audio_sha256`, `mfa_acoustic`, and `mfa_dictionary` all
match the current inputs. `audio_size` is a cheap pre-filter only: if size
differs we mark stale without hashing; if size matches we compute and compare
`audio_sha256`.

**Why content hash, not mtime:** stimuli get copied/rsync'd between machines
(e.g. the Sherlock workflow), which resets mtime and would trigger false
staleness — needless MFA reruns, defeating the cache. A content hash is stable
across copies. Hashing costs seconds of streamed I/O; a model reload costs
minutes, so the trade favors hashing. Deliberate residual gap: a same-size,
same-hash swap is treated as identical (cryptographically impossible in
practice).

## Components

### 1. `extractors/aligner.py` — `ensure_alignments(items, persist_dir)` (new)

Called once at the top of a batch, before the per-document loop. `items` is a
list of `(audio_path, transcript_text, stem)`.

For each item, classify fresh vs stale/missing using the freshness key above.
Then, for the stale/missing set only:

1. **Delete** any existing `<stem>.TextGrid` + `<stem>.json` for those stems
   first, so a failed realign can never leave a stale file behind for `run()` to
   pick up.
2. Stage each stale item into **one** temp corpus dir (`<stem>.wav` +
   `<stem>.lab` with `prepare_for_aligner(transcript)` content).
3. Run `mfa align` **once** over that corpus dir (reuse `_run_mfa`).
4. For each stem MFA produced a TextGrid for, move it into `persist_dir` and
   write its sidecar. Stems MFA failed to align get **no** file (correctly
   surfaced as an alignment failure downstream).

If the stale/missing set is empty, **no subprocess is spawned**. Emits a summary
log, e.g. `[mfa] 7 stimuli: 5 cached, 2 aligned in 1 run`.

Unique-stem assumption: stems within a batch are stimulus IDs and are unique. If
a collision is detected while staging, log a warning (last-writer-wins is
acceptable; collisions shouldn't occur in practice).

### 2. `extractors/aligner.py` — `align_with_phones(...)` gains two params

New signature: `align_with_phones(audio_path, transcript, cache=None,
persist_dir=None, run_if_missing=True)`.

Lookup order:

1. In-memory `cache` (unchanged innermost layer — avoids re-parsing across
   passes).
2. If `persist_dir` set and `<stem>.TextGrid` exists there → parse it
   (`parse_words`/`parse_phones`), populate `cache`, return. **No MFA.**
3. Else if `run_if_missing` → run single-file MFA via `_align_uncached`; if
   `persist_dir` is set, save the TextGrid + sidecar there; populate `cache`;
   return.
4. Else (`run_if_missing=False`, disk miss) → raise the existing
   "MFA did not produce a TextGrid" `RuntimeError`.

Step 2 uses existence-by-stem, not the hash — freshness was already decided by
`ensure_alignments`, which guarantees only fresh TextGrids remain on disk. This
avoids a second, independently-computed hash check in `run()` that could
disagree with the pre-pass.

`align(...)` (the thin word-only wrapper) is unaffected.

### 3. `main.py` — batch functions call the pre-pass

`run_word_level_batch` and `run_doc_level_batch`, right after building
`work_items`:

- Compute `persist_dir = alignment_dir(args.output_folder)` (helper returning
  `<output_folder>/Montreal Forced Alignment`).
- Build `items` from `work_items` (each pair's audio path + loaded/normalized
  transcript text + stem), skipping pairs with no audio.
- Call `ensure_alignments(items, persist_dir)`.
- Set `args._batch_prealigned = True`.

`run()` (line ~402) then:

- `persist_dir = alignment_dir(args.output_folder)` if `args.output_folder` set,
  else `None`.
- `run_if_missing = not getattr(args, '_batch_prealigned', False)`.
- Calls `align_with_phones(args.audio, text, cache=_align_cache,
  persist_dir=persist_dir, run_if_missing=run_if_missing)`.

Because the word pass populates the folder first, the doc pass's
`ensure_alignments` finds everything fresh and spawns zero subprocesses. Net:
**one MFA subprocess per dispatch.** A dispatch that runs only the doc pass gets
its single run there. Either way, ≤1 per dispatch.

Transcript-text consistency: `ensure_alignments` and `run()` must derive the
`transcript_text` for a stimulus the same way (same file load + normalization
run() already uses), so the hash written by the pre-pass matches what run()
would send to MFA. The plan will route both through one loader.

## Out of scope / unchanged

- **Acoustic pass** (`run_acoustic_level_batch`) — audio-only, never invokes MFA.
  Untouched.
- **CLI single-run** (`--audio`/`--transcript`/`--output`, no `output_folder`) —
  `persist_dir=None`, `run_if_missing=True` (default): one MFA run for the one
  file, no persistence. Behavior identical to today.
- TextGrid parsing, column renaming, `_merge_alignment`, prosody, timing — all
  untouched. The persisted artifact is exactly what MFA already produced.
- In-memory `_align_cache` and its plumbing in `run_dispatch.py` stay as-is.

## Testing

Per the TDD-scope convention, `aligner.py` is `extractors/` math-adjacent →
test-first there (mock the `mfa` subprocess; no real Kaldi in tests):

- Freshness classification: fresh / stale-transcript / stale-audio /
  missing-TextGrid / model-name-change.
- Delete-stale-before-realign: a stale TextGrid is removed prior to the batch
  MFA run.
- Single-invocation batching: N stale items → exactly one `_run_mfa` call over a
  corpus dir containing all N.
- Empty stale set → zero `_run_mfa` calls.
- `align_with_phones` disk-hit path (no MFA) and `run_if_missing=False` disk-miss
  → raises.
- Sidecar round-trip (write then read matches).

GUI/batch wiring (`main.py` pre-pass call, `run()` param passing) gets lighter
test-after.

## Risks

- **Cross-consistency of transcript text** between pre-pass and `run()` — handled
  by a single shared loader (see above); a mismatch would show as an unexpected
  realign, not wrong data.
- **Stem collisions** in a batch — logged; not expected with unique stimulus IDs.
- **Partial batch failure** — a stem MFA can't align leaves no file; downstream
  treats it as an alignment failure exactly as today.
