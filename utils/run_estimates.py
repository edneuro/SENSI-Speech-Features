"""Run-tab estimates: input statistics, disk size per output artifact, peak
memory and run time.

Pure functions over (Plan, prefs, InputStats); the only I/O is scan_inputs,
which reads the input folders once and caches per-file results by
(path, size, mtime). Constants come from utils/estimate_constants.py.
See docs/superpowers/specs/2026-10-26-run-estimates-design.md.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from gui.planner.plan import LEVEL_ARTIFACTS, _active_clusters, _active_output_csvs
from gui.planner.stages import Plan
from utils import estimate_constants as K

MAT_HEADER_BYTES = 1024          # fixed .mat header/metadata allowance per file
_F32, _F64 = 4, 8
_WORD2VEC_DIM = 300
_BERT_LAYERS, _BERT_DIM = 12, 768
# extractors/phoneme_surprisal.phoneme_surprisal_mat_vars: four impulse trains
# (phoneme_surprisal, cohort_entropy, phoneme_onset, word_onset), each N x 1.
_PHONEME_SURPRISAL_TRAINS = 4

# Columns every row of an artifact carries that no FeatureCluster emits
# (identity/provenance columns written unconditionally; checked against the
# all-on validation outputs in audit/out_allon).
_FIXED_COLUMNS: dict[str, int] = {
    "features": 1,            # Transcript
    "syllables": 5,           # doc_id, sentence_id, token_in_sentence_id, token, syllable_id
    "phonemes": 6,            # ... + phoneme_id
    "sentences": 2,           # doc_id, sentence_id
    "documents_summary": 4,   # stimulus_id, dictionary_used, transcript_path, audio_path
}

# Doc lexical-norm aggregates (utils/doc_summary._lexical_norm_aggregates) per
# word-level norm toggle: AoA gives mean/median/max/sd/prop_late + coverage,
# every other norm a mean + coverage. AoA_SD has no doc column.
_DOC_NORM_COLUMNS: dict[str, int] = {
    "feat_norm_aoa_mean": 6, "feat_norm_concreteness": 2,
    "feat_norm_imageability": 2, "feat_norm_familiarity": 2,
    "feat_norm_valence": 2, "feat_norm_arousal": 2, "feat_norm_dominance": 2,
}


@dataclass
class InputStats:
    n_stimuli: int            # transcripts (each becomes one word-level item)
    n_with_audio: int         # transcripts with a paired audio file
    words: int                # raw-transcript words (timing_history.count_words)
    words_with_audio: int     # the same over the paired transcripts
    audio_seconds: float      # duration of the paired audio files
    # The acoustic level processes every audio file under the audio folder
    # (main.run_acoustic_level_batch, recursive), paired or not. None = same
    # as the paired figures above.
    n_audio_files: int | None = None
    all_audio_seconds: float | None = None
    # Paired audio seconds whose saved alignment is not fresh, i.e. the part
    # MFA will actually run on (extractors.aligner.is_alignment_fresh). None =
    # unknown (no output folder given): charge every paired audio second.
    audio_seconds_to_align: float | None = None


# --- scanning ----------------------------------------------------------------

_FILE_CACHE: dict[tuple[str, int, int], float] = {}


def _cached(path: Path, compute) -> float:
    try:
        st = path.stat()
    except OSError:
        return 0
    key = (str(path.resolve()), st.st_size, st.st_mtime_ns)
    if key not in _FILE_CACHE:
        try:
            _FILE_CACHE[key] = compute(path)
        except Exception:
            _FILE_CACHE[key] = 0
    return _FILE_CACHE[key]


_FRESH_CACHE: dict[tuple, bool] = {}


def _stat_key(path: Path) -> tuple | None:
    try:
        st = path.stat()
    except OSError:
        return None
    return (str(path), st.st_size, st.st_mtime_ns)


def _alignment_fresh(output_folder: str, transcript: Path, audio: Path,
                     g2p: bool) -> bool:
    """aligner.is_alignment_fresh, cached by (path, size, mtime) of the
    transcript, the audio, the saved TextGrid and its sidecar (+ g2p)."""
    from extractors import aligner
    tg, side = aligner.saved_alignment_files(output_folder, str(audio))
    if not tg.exists() or not side.exists():
        return False                  # cheap miss: no hashing, no reading
    key = (_stat_key(transcript), _stat_key(audio), _stat_key(tg),
           _stat_key(side), bool(g2p))
    if key not in _FRESH_CACHE:
        try:
            with open(transcript, encoding="utf-8") as f:
                text = f.read()
            _FRESH_CACHE[key] = aligner.is_alignment_fresh(
                output_folder, str(audio), text, g2p=g2p)
        except Exception:
            _FRESH_CACHE[key] = False
    return _FRESH_CACHE[key]


def _count_words(path: Path) -> int:
    """Same unit main.py records text-step samples in (count_words on the raw
    transcript), so learned seconds-per-word apply unchanged."""
    from utils.timing_history import count_words
    with open(path, encoding="utf-8", errors="replace") as f:
        return count_words(f.read())


def _audio_seconds(path: Path) -> float:
    import soundfile as sf
    info = sf.info(str(path))
    return info.frames / info.samplerate if info.samplerate else 0.0


def _all_audio(audio_folder: str, audio_exts) -> tuple[int, float]:
    if not audio_folder or not os.path.isdir(audio_folder):
        return 0, 0.0
    exts = {e.lower().lstrip(".") for e in audio_exts}
    files = [p for p in Path(audio_folder).rglob("*")
             if p.is_file() and p.suffix.lower().lstrip(".") in exts]
    return len(files), float(sum(_cached(p, _audio_seconds) for p in files))


def scan_inputs(transcripts_folder: str, audio_folder: str,
                text_exts=("txt", "tsv", "md"),
                audio_exts=("wav", "flac", "mp3"), *,
                output_folder: str = "", g2p: bool = False) -> InputStats | None:
    """Count stimuli, words and audio seconds in the input folders.

    With `output_folder` (the run's shared.last_output_folder) and `g2p`
    (alignment.g2p_oov), also the paired audio seconds MFA will still align:
    stimuli whose saved alignment is fresh are reused by the run, not
    re-aligned.

    None when neither folder is usable. With only the audio folder (an
    acoustic-only plan) the text figures are zero and only the audio-folder
    figures are filled in.
    """
    n_audio, all_secs = _all_audio(audio_folder, audio_exts)
    if not transcripts_folder or not os.path.isdir(transcripts_folder):
        if not n_audio:
            return None
        return InputStats(0, 0, 0, 0, 0.0, n_audio, all_secs)

    from utils.file_pairing import find_pairs
    pairs = find_pairs(Path(transcripts_folder), Path(audio_folder or ""),
                       text_exts, audio_exts)
    items = [(p.transcript_path, p.audio_path) for p in pairs.pairs]
    items += [(t, "") for t in pairs.unmatched_text]
    words = words_audio = n_with_audio = 0
    secs = 0.0
    to_align = 0.0 if output_folder else None
    for transcript, audio in items:
        w = int(_cached(Path(transcript), _count_words))
        words += w
        if audio:
            n_with_audio += 1
            words_audio += w
            a_secs = _cached(Path(audio), _audio_seconds)
            secs += a_secs
            if to_align is not None and not _alignment_fresh(
                    output_folder, Path(transcript), Path(audio), g2p):
                to_align += a_secs
    return InputStats(len(items), n_with_audio, words, words_audio, float(secs),
                      n_audio, all_secs,
                      None if to_align is None else float(to_align))


# --- disk --------------------------------------------------------------------

def _n_categories(prefs: dict) -> int:
    sem = prefs.get("semantic", {}) or {}
    backend = sem.get("backend", "auto")
    if backend in ("liwc", "empath"):
        return len(sem.get(f"categories_{backend}") or [])
    # "auto" is resolved at run time (main.py): take the larger list.
    return max(len(sem.get("categories_liwc") or []),
               len(sem.get("categories_empath") or []))


def _csv_columns(plan: Plan, prefs: dict) -> dict[str, int]:
    """Active column count per CSV artifact (see _FIXED_COLUMNS / dynamic)."""
    from main import (_doc_columns_to_drop, _phoneme_columns_to_drop,
                      _sentence_columns_to_drop, _syllable_columns_to_drop,
                      _word_columns_to_drop)
    feats = prefs.get("features", {}) or {}
    out = prefs.get("output", {}) or {}
    per_column = {
        "features": _word_columns_to_drop(feats),
        "syllables": _syllable_columns_to_drop(feats),
        "phonemes": _phoneme_columns_to_drop(feats),
        "sentences": _sentence_columns_to_drop(feats),
        "documents_summary": _doc_columns_to_drop(out, feats),
    }
    active = _active_clusters(prefs, _active_output_csvs(prefs))
    names = {c.name for c in active}
    counts: dict[str, int] = {}
    for art, fixed in _FIXED_COLUMNS.items():
        cols = {col for c in active if c.output_csv == art for col in c.emits_columns}
        cols -= set(plan.column_drops.get(art, ()))
        cols -= per_column[art]
        counts[art] = len(cols) + fixed
    n_cat = _n_categories(prefs)
    counts["features"] += len(plan.predictability_columns)
    if "semantic indicators" in names:
        counts["features"] += n_cat                      # Sem_<Backend>_<cat>
    if plan.predictability_doc:
        pred = prefs.get("predictability", {}) or {}
        n_metrics = sum(1 for v in (pred.get("metrics") or {}).values() if v)
        counts["documents_summary"] += len(pred.get("models") or []) * n_metrics
    if "doc category proportions" in names:
        counts["documents_summary"] += 2 * n_cat         # count_<cat>, prop_<cat>
    if "doc lexical norms" in names:
        counts["documents_summary"] += sum(
            n for k, n in _DOC_NORM_COLUMNS.items() if feats.get(k, False))
    return counts


def _acoustic_channels(prefs: dict) -> int:
    feats = prefs.get("features", {}) or {}
    acfg = prefs.get("acoustic_ts", {}) or {}
    n = 1                                                # t vector
    n += int(bool(feats.get("feat_acoustic_ts_envelope", True)))
    n += int(bool(feats.get("feat_acoustic_ts_onset_env", True)))
    if feats.get("feat_acoustic_ts_logmel", True):
        n += int(acfg.get("n_mels", 16))
    if feats.get("feat_acoustic_ts_mfcc", False):
        n += min(int(acfg.get("n_mfcc", 13)), int(acfg.get("n_mels", 16)))
    return n


def estimate_disk(plan: Plan, prefs: dict, stats: InputStats) -> dict[str, int]:
    """Estimated bytes per live artifact in plan.output_csvs (all stimuli)."""
    feature_rows = stats.words * K.ROWS_PER_WORD["features"]
    sentences = stats.words * K.SENTENCES_PER_WORD
    rows = {
        "features": feature_rows,
        "syllables": stats.words_with_audio * K.ROWS_PER_WORD["syllables"],
        "phonemes": stats.words_with_audio * K.ROWS_PER_WORD["phonemes"],
        "sentences": sentences,
        "documents_summary": stats.n_stimuli,
    }
    cols = _csv_columns(plan, prefs)
    n_audio = stats.n_audio_files if stats.n_audio_files is not None else stats.n_with_audio
    all_secs = (stats.all_audio_seconds if stats.all_audio_seconds is not None
                else stats.audio_seconds)
    a_rate = float((prefs.get("acoustic_ts") or {}).get("analysis_rate", 128))
    ps_sr = float((prefs.get("phoneme_surprisal") or {}).get("analysis_sr", 128))

    def mat(payload: float, n_files: int) -> int:
        return round(payload) + MAT_HEADER_BYTES * n_files if n_files else 0

    disk: dict[str, int] = {}
    for art in plan.output_csvs:
        if art in rows:
            disk[art] = round(rows[art] * cols[art] * K.CSV_BYTES_PER_CELL[art])
        elif art == "word2vec":
            disk[art] = mat(feature_rows * _WORD2VEC_DIM * _F32, stats.n_stimuli)
        elif art == "bert_word":
            disk[art] = mat(feature_rows * _BERT_LAYERS * _BERT_DIM * _F32,
                            stats.n_stimuli)
        elif art == "bert_sentence":
            disk[art] = mat(sentences * _BERT_LAYERS * _BERT_DIM * _F32,
                            stats.n_stimuli)
        elif art == "acoustic":
            disk[art] = mat(all_secs * a_rate * _acoustic_channels(prefs) * _F64,
                            n_audio)
        elif art == "phoneme_surprisal_mat":
            # impulse trains + the t vector, float64
            disk[art] = mat(stats.audio_seconds * ps_sr
                            * (_PHONEME_SURPRISAL_TRAINS + 1) * _F64,
                            stats.n_with_audio)
    return disk


def level_bytes(disk: dict[str, int], level_key: str) -> int:
    return sum(disk.get(a, 0) for a in LEVEL_ARTIFACTS[level_key])


_UNITS = ("B", "KB", "MB", "GB", "TB", "PB")


def format_bytes(n: int) -> str:
    """'≈ 410 MB' style: 1024-based units, 2 significant digits."""
    v = float(max(n, 0))
    i = 0
    while v >= 1024 and i < len(_UNITS) - 1:
        v /= 1024
        i += 1
    if v >= 100:
        s = f"{round(v, -1):.0f}"
    elif v >= 10 or i == 0:
        s = f"{round(v):.0f}"
    else:
        s = f"{v:.1f}"
    return f"≈ {s} {_UNITS[i]}"


# --- peak memory -------------------------------------------------------------

@dataclass
class RamEstimate:
    peak_gb: float       # system RAM needed (GPU-resident models excluded)
    driver: str          # display name of the heaviest item, or "the pipeline"
    machine_gb: float
    gpu_vram_gb: float
    fits: bool
    warning: str | None


def has_cuda(system: dict) -> bool:
    """One rule for "this machine can put models on a GPU": a detected CUDA
    device with memory (utils.system_info._detect_gpu)."""
    gpu = (system or {}).get("gpu") or {}
    return bool(gpu.get("name")) and float(gpu.get("vram_gb") or 0.0) > 0


def estimate_ram(plan: Plan, prefs: dict, system: dict) -> RamEstimate:
    """Baseline + the SUM of the heavy items that will run (they stay cached
    for the whole process). A model the extractor would put on the GPU counts
    its "<slug>@cuda" value against VRAM instead of RAM."""
    from extractors.predictability_models import spec_for
    from utils.timing_history import model_device

    gpu = system.get("gpu") or {}
    machine = float(system.get("system_ram_gb") or 0.0)
    vram = float(gpu.get("vram_gb") or 0.0)
    cuda = has_cuda(system)

    # (display name, GiB, on_gpu)
    items: list[tuple[str, float, bool]] = []
    if "word2vec" in plan.output_csvs:
        items.append(("Word2Vec", K.RAM_GB["word2vec"], False))
    if {"bert_word", "bert_sentence"} & set(plan.output_csvs):
        items.append(("BERT", K.RAM_GB["bert"], False))
    if plan.predictability_word or plan.predictability_doc:
        for slug in (prefs.get("predictability", {}) or {}).get("models") or []:
            on_gpu = model_device("predictability", slug, cuda) == "cuda"
            try:
                name = spec_for(slug).label
            except KeyError:
                name = slug
            gb = K.RAM_GB.get(f"{slug}@cuda" if on_gpu else slug)
            if gb is not None:
                items.append((name, gb, on_gpu))

    ram_items = [i for i in items if not i[2]]
    gpu_items = [i for i in items if i[2]]
    peak = K.RAM_GB["baseline"] + sum(i[1] for i in ram_items)
    gpu_total = sum(i[1] for i in gpu_items)
    driver = max(items, key=lambda i: i[1])[0] if items else "the pipeline"

    warning = None
    if machine > 0 and peak > machine:
        warning = _ram_warning(_pool_item(ram_items), peak, machine)
    elif vram > 0 and gpu_total > vram:
        warning = _ram_warning(_pool_item(gpu_items), gpu_total, vram, gpu=True)
    return RamEstimate(peak, driver, machine, vram, warning is None, warning)


def _pool_item(items: list[tuple[str, float, bool]]) -> str:
    """Name for the overflowing pool: the single item, else the whole run."""
    if not items:
        return "the pipeline"
    return items[0][0] if len(items) == 1 else "This run"


def _ram_warning(item: str, need: float, have: float, gpu: bool = False) -> str:
    pool, host = ("GPU memory", "this GPU") if gpu else ("memory", "this computer")
    return (f"⚠ {item} needs ≈ {need:.1f} GB of {pool}; {host} has "
            f"{have:.1f} GB. The run will likely fail or be extremely slow.")


# --- time --------------------------------------------------------------------

@dataclass
class TimeEstimate:
    low_s: float
    high_s: float
    learned: bool        # every charged step had a rate learned on this machine


STAGE_TO_STEP: dict[str, str] = {
    "mfa_align": "mfa_align", "spacy_parse": "spacy_parse",
    "predictability": "predictability", "word2vec_embed": "word2vec",
    "bert_embed": "bert", "praat_contours": "prosody_contours",
    "syllabify_pairs": "syllabify", "phoneme_surprisal": "phoneme_surprisal",
    "sentence_prosody": "sentence_prosody", "acoustic_ts": "acoustic_ts",
}

# Steps timed inside run_word_level_batch's own loop (not main.run()), so the
# doc pass never repeats them.
_WORD_PASS_ONLY_STEPS = frozenset(
    {"syllabify", "phoneme_surprisal", "sentence_prosody", "word2vec", "bert"})
# One-time cost per process, keyed by timing_history.load_key.
_LOADED_STEPS = frozenset({"word2vec", "bert", "phoneme_surprisal"})


def _stage_step(stage: str) -> str | None:
    if stage.startswith("doc_"):
        return "doc_summary"
    return STAGE_TO_STEP.get(stage)


def _pass_steps(stages) -> set[str]:
    return {s for s in map(_stage_step, stages) if s}


def estimate_time(plan: Plan, prefs: dict, stats: InputStats, history: dict,
                  cuda_available: bool) -> TimeEstimate:
    """Range for the whole run over the passes the dispatcher will start
    (gui/run_dispatch.run, run.dispatch): word, doc, acoustic.

    Each pass charges the steps its own namespace runs (emit.level_stages, the
    set to_namespace derives that pass's no_* flags from). Reuse across
    passes, as main.py does it: MFA once per document (_prealign_batch skips
    _align_cache hits), and not at all for a stimulus whose saved alignment is
    fresh (stats.audio_seconds_to_align, when known); predictability once
    while the word pass's scores still sit in
    extractors.predictability._SCORE_CACHE (FIFO, SCORE_CACHE_MAX entries of
    one model x stimulus each); Word2Vec/BERT/sub-word/sentence steps only in
    the word pass; doc_summary only in the doc pass; acoustic_ts only in the
    acoustic pass, over every audio file.
    """
    from extractors.predictability import SCORE_CACHE_MAX
    from gui.planner.emit import WORD_BATCH_ARTIFACTS, level_stages
    from utils.timing_history import (AUDIO_STEPS, fit_overhead_slope,
                                      load_key, model_device, rate)

    stages = set(plan.stages_to_run)
    outputs = set(plan.output_csvs)
    word_pass = bool(WORD_BATCH_ARTIFACTS & outputs)
    doc_pass = "documents_summary" in outputs
    models = list((prefs.get("predictability", {}) or {}).get("models") or [])

    word = (_pass_steps(stages & level_stages(plan, prefs, "word"))
            if word_pass else set())
    doc = (_pass_steps(stages & level_stages(plan, prefs, "doc"))
           if doc_pass else set())
    word -= {"predictability", "acoustic_ts", "doc_summary"}
    doc -= {"predictability", "acoustic_ts"} | _WORD_PASS_ONLY_STEPS
    if "mfa_align" in word:
        doc.discard("mfa_align")          # cached: MFA at most once per doc

    # (step, model) charged once per pass that runs it.
    charges: list[tuple[str, str | None]] = []
    charges += [(s, None) for s in sorted(word)]
    charges += [(s, None) for s in sorted(doc)]
    pred_word = word_pass and plan.predictability_word
    pred_doc = doc_pass and plan.predictability_doc
    if pred_word:
        charges += [("predictability", m) for m in models]
    if pred_doc:
        if not pred_word or len(models) * stats.n_stimuli > SCORE_CACHE_MAX:
            charges += [("predictability", m) for m in models]
    if "acoustic" in outputs and "acoustic_ts" in stages:
        charges.append(("acoustic_ts", None))

    all_secs = (stats.all_audio_seconds if stats.all_audio_seconds is not None
                else stats.audio_seconds)
    # Audio steps run on the paired audio, except: acoustic_ts covers every
    # audio file, and MFA skips stimuli already aligned on disk. syllabify is
    # a word step over the paired transcripts only.
    units_for: dict[str, float] = {s: stats.audio_seconds for s in AUDIO_STEPS}
    units_for["acoustic_ts"] = all_secs
    if stats.audio_seconds_to_align is not None:
        units_for["mfa_align"] = stats.audio_seconds_to_align
    units_for["syllabify"] = stats.words_with_audio
    learned_loads = (history or {}).get("load_seconds", {}) or {}

    total = K.STARTUP_SECONDS if charges else 0.0
    learned = True
    loads: set[str] = set()
    for step, model in charges:
        units = float(units_for.get(step, stats.words))
        if units <= 0:
            continue
        if step == "mfa_align":
            # One MFA invocation per run that aligns anything: fixed start-up
            # + slope x audio seconds (see FIRST_MFA_OVERHEAD_SECONDS).
            fit = fit_overhead_slope(history, "mfa_align")
            if fit is None:
                learned = False
                fit = (K.FIRST_MFA_OVERHEAD_SECONDS,
                       K.FIRST_ESTIMATE_RATES["mfa_align"])
            total += fit[0] + fit[1] * units
            continue
        device = model_device(step, model, cuda_available)
        r = rate(history, step, model, device)
        if r is None:
            learned = False
            r = K.FIRST_ESTIMATE_RATES.get(
                f"{step}:{model}" if model else step,
                K.FIRST_ESTIMATE_RATES[step])
        total += units * r
        if model or step in _LOADED_STEPS:
            loads.add(load_key(step, model))
    for key in loads:
        total += float(learned_loads.get(key, K.FIRST_LOAD_SECONDS.get(key, 0.0)))

    lo, hi = K.LEARNED_SPREAD if learned else K.FIRST_ESTIMATE_SPREAD
    return TimeEstimate(total * lo, total * hi, learned)



def _fmt_hours(h: float) -> str:
    return f"{round(h):.0f}" if h >= 10 or round(h, 1) == round(h) else f"{h:.1f}"


def _in_unit(x: float, unit: str) -> str:
    """x seconds written in `unit` (rounded the way that unit is shown)."""
    if unit == "s":
        return f"{round(x)}"
    if unit == "min":
        return f"{round(x / 60)}"
    return _fmt_hours(x / (3600 if unit == "h" else 86400))


def _natural_unit(x: float) -> str:
    """Unit for x seconds, picked AFTER rounding: under 60 s -> s, up to 90
    min -> min, up to 48 h -> h, else days."""
    if round(x) < 60:
        return "s"
    if round(x / 60) <= 90:
        return "min"
    if float(_in_unit(x, "h")) <= 48:
        return "h"
    return "days"


def format_duration(low_s: float, high_s: float) -> str:
    """'≈ 10–40 min'. The unit follows the (rounded) upper bound; the lower
    bound keeps that unit unless it would round below 1 there, in which case
    it gets its own ('≈ 20 s–1 min'), so a wide range never collapses. One
    value when both ends read the same."""
    high_s = max(high_s, low_s, 0.0)
    low_s = max(low_s, 0.0)
    unit = _natural_unit(high_s)
    b = _in_unit(high_s, unit)
    a = _in_unit(low_s, unit)
    if unit != "s" and float(a) < 1:
        lo_unit = _natural_unit(low_s)
        return f"≈ {_in_unit(low_s, lo_unit)} {lo_unit}–{b} {unit}"
    return f"≈ {b} {unit}" if a == b else f"≈ {a}–{b} {unit}"
