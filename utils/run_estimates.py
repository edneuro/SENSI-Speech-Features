"""Run-tab estimates: input statistics and disk size per output artifact.

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
    words: int                # whitespace words over all transcripts
    words_with_audio: int     # whitespace words over the paired transcripts
    audio_seconds: float      # duration of the paired audio files
    # The acoustic level processes every audio file under the audio folder
    # (main.run_acoustic_level_batch, recursive), paired or not. None = same
    # as the paired figures above.
    n_audio_files: int | None = None
    all_audio_seconds: float | None = None


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


def _count_words(path: Path) -> int:
    from utils.transcript_prep import prepare_for_aligner
    with open(path, encoding="utf-8", errors="replace") as f:
        text = f.read()
    return len(prepare_for_aligner(text).split())


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
                audio_exts=("wav", "flac", "mp3")) -> InputStats | None:
    """Count stimuli, words and audio seconds in the input folders.

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
    for transcript, audio in items:
        w = int(_cached(Path(transcript), _count_words))
        words += w
        if audio:
            n_with_audio += 1
            words_audio += w
            secs += _cached(Path(audio), _audio_seconds)
    return InputStats(len(items), n_with_audio, words, words_audio, float(secs),
                      n_audio, all_secs)


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


def estimate_ram(plan: Plan, prefs: dict, system: dict) -> RamEstimate:
    """Baseline + the SUM of the heavy items that will run (they stay cached
    for the whole process). A model the extractor would put on the GPU counts
    its "<slug>@cuda" value against VRAM instead of RAM."""
    from extractors.predictability_models import spec_for
    from utils.timing_history import model_device

    gpu = system.get("gpu") or {}
    machine = float(system.get("system_ram_gb") or 0.0)
    vram = float(gpu.get("vram_gb") or 0.0)
    cuda = bool(gpu.get("name")) and vram > 0

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
        warning = _ram_warning(_pool_item(gpu_items), gpu_total, vram)
    return RamEstimate(peak, driver, machine, vram, warning is None, warning)


def _pool_item(items: list[tuple[str, float, bool]]) -> str:
    """Name for the overflowing pool: the single item, else the whole run."""
    if not items:
        return "the pipeline"
    return items[0][0] if len(items) == 1 else "This run"


def _ram_warning(item: str, need: float, have: float) -> str:
    return (f"⚠ {item} needs ≈ {need:.1f} GB of memory; this computer has "
            f"{have:.1f} GB. The run will likely fail or be extremely slow.")
