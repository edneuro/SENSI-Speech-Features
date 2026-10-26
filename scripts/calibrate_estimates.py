#!/usr/bin/env python3
"""Calibrate the Run-tab estimate constants (utils/estimate_constants.py).

Runs the headless pipeline (run.py) on validation/data with a scratch config
that turns on every CPU-feasible feature (predictability: GPT-2 only; the
gated Llama models are not run), then prints:

  * row ratios      — syllable / phoneme / sentence rows per transcript word;
  * bytes per cell  — CSV file size / (rows x columns), per CSV artifact;
  * stage rates     — seconds per unit per (step, model, device), from the
                      timing history the run wrote into the scratch config dir;
  * peak memory     — each heavy item (Word2Vec, BERT, GPT-2) measured in a
                      fresh subprocess around loading + one stimulus, minus a
                      baseline subprocess (the same pipeline, no heavy models).

Never touches the live config/: every run gets SENSI_CONFIG_DIR / --config-dir
pointing at a directory under --scratch, and the script refuses a scratch dir
inside the repo.

Usage:
    python scripts/calibrate_estimates.py [--scratch DIR] [--memory-stimulus STEM ...]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("MKL_THREADING_LAYER", "sequential")

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

VALIDATION = REPO / "validation"
DATA = VALIDATION / "data"

# Sections whose boolean toggles are all switched on for the calibration run.
_TOGGLE_SECTIONS = ("features", "output")
# Feature/output keys that pull in a heavy model, per heavy item.
_HEAVY = {
    "word2vec": {"features": ["master_word2vec", "master_semantic_similarity"],
                 "output": ["word2vec_mat"]},
    "bert": {"features": ["master_bert_word", "master_bert_sentence",
                          "master_semantic_similarity",
                          "master_sentence_similarity"],
             "output": ["bert_word_mat", "bert_sentence_mat"]},
    "gpt2": {"features": ["master_predictability"],
             "output": ["doc_agg_predictability"]},
}

# Peak working set of the process, printed by the memory child.
_PEAK_SNIPPET = r"""
import os, runpy, sys
os.environ.setdefault("MKL_THREADING_LAYER", "sequential")
sys.argv = ["run.py", "--config-dir", sys.argv[1]]
runpy.run_path(os.path.join(os.getcwd(), "run.py"), run_name="__main__")
if sys.platform == "win32":
    import psutil
    peak = psutil.Process().memory_info().peak_wset
else:
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak = peak if sys.platform == "darwin" else peak * 1024  # Linux: KiB
print("CALIB_PEAK_BYTES", int(peak))
"""


def _child_env(config_dir: Path) -> dict:
    env = dict(os.environ)
    env["SENSI_CONFIG_DIR"] = str(config_dir)
    env.pop("SENSI_PREFS_PATH", None)
    env.setdefault("MKL_THREADING_LAYER", "sequential")
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def build_config(config_dir: Path, transcripts: Path, audio: Path, output: Path,
                 *, heavy: set[str] | None = None) -> None:
    """Write features.yaml + run.yaml into `config_dir` (never the live config).

    Starts from a copy of validation/config when present (otherwise the shipped
    defaults), migrates it, switches every feature/output toggle on, restricts
    predictability to GPT-2, and points the folders at absolute paths. With
    `heavy` given, the heavy items not in it are switched off (memory runs).
    """
    config_dir = config_dir.resolve()
    if REPO in config_dir.parents or config_dir == REPO:
        raise SystemExit(f"refusing a config dir inside the repo: {config_dir}")
    config_dir.mkdir(parents=True, exist_ok=True)
    for name in ("features.yaml", "run.yaml"):
        src = VALIDATION / "config" / name
        if src.exists():
            shutil.copyfile(src, config_dir / name)
    old = os.environ.get("SENSI_CONFIG_DIR")
    os.environ["SENSI_CONFIG_DIR"] = str(config_dir)
    try:
        from utils import preferences as P
        assert P.get_config_dir().resolve() == config_dir, "config dir override failed"
        prefs = P.load_preferences()
        for section in _TOGGLE_SECTIONS:
            for k, v in list(prefs[section].items()):
                if isinstance(v, bool):
                    prefs[section][k] = True
        prefs["predictability"]["models"] = ["gpt2"]
        if heavy is not None:
            for item, keys in _HEAVY.items():
                if item in heavy:
                    continue
                for section, names in keys.items():
                    for n in names:
                        if n in prefs[section] and not (
                                # semantic-similarity master is shared by w2v + BERT
                                n == "master_semantic_similarity"
                                and ({"word2vec", "bert"} & heavy)):
                            prefs[section][n] = False
        shared = prefs["shared"]
        shared["analyzer_name"] = "calibration"
        shared["last_transcripts_folder"] = str(transcripts)
        shared["last_audio_folder"] = str(audio)
        shared["last_output_folder"] = str(output)
        P.save_preferences(prefs)
    finally:
        if old is None:
            os.environ.pop("SENSI_CONFIG_DIR", None)
        else:
            os.environ["SENSI_CONFIG_DIR"] = old


def _run(config_dir: Path, log: Path, *, peak: bool = False) -> tuple[float, int | None]:
    cmd = ([sys.executable, "-c", _PEAK_SNIPPET, str(config_dir)] if peak
           else [sys.executable, str(REPO / "run.py"), "--config-dir", str(config_dir)])
    t0 = time.perf_counter()
    with log.open("w", encoding="utf-8") as f:
        proc = subprocess.run(cmd, cwd=REPO, env=_child_env(config_dir),
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, encoding="utf-8", errors="replace")
        f.write(proc.stdout)
    secs = time.perf_counter() - t0
    if proc.returncode != 0:
        raise SystemExit(f"run failed ({proc.returncode}); see {log}")
    peak_bytes = None
    for line in proc.stdout.splitlines():
        if line.startswith("CALIB_PEAK_BYTES"):
            peak_bytes = int(line.split()[1])
    return secs, peak_bytes


def _words(path: Path) -> int:
    return len(path.read_text(encoding="utf-8").split())


def _csv_shape(path: Path) -> tuple[int, int]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    return max(len(rows) - 1, 0), (len(rows[0]) if rows else 0)


def measure_outputs(output: Path, transcripts: Path) -> dict:
    words = {p.stem: _words(p) for p in sorted(transcripts.glob("*.txt"))}
    patterns = {"features": ("word", "_features.csv"),
                "syllables": ("syllable", "_syllables.csv"),
                "phonemes": ("phoneme", "_phonemes.csv"),
                "sentences": ("sentence", "_sentences.csv")}
    out: dict = {"words_total": sum(words.values()), "stimuli": len(words),
                 "rows_per_word": {}, "bytes_per_cell": {}, "csv": {}}
    for art, (level, suffix) in patterns.items():
        size = rows = cells = 0
        w = 0
        for stem, n in words.items():
            p = output / level / f"{stem}{suffix}"
            if not p.exists():
                continue
            r, c = _csv_shape(p)
            size += p.stat().st_size
            rows += r
            cells += r * c
            w += n
        out["csv"][art] = {"bytes": size, "rows": rows, "cells": cells, "words": w}
        if cells:
            out["bytes_per_cell"][art] = size / cells
        if w and art != "features":
            out["rows_per_word"][art] = rows / w
        if art == "features" and w:
            out["word_rows_per_word"] = rows / w
    docs = sorted((output / "document").glob("documents_summary_*.csv"))
    if docs:
        p = docs[-1]
        r, c = _csv_shape(p)
        out["csv"]["documents_summary"] = {"bytes": p.stat().st_size,
                                           "rows": r, "cells": r * c}
        if r * c:
            out["bytes_per_cell"]["documents_summary"] = p.stat().st_size / (r * c)
    return out


def measure_rates(history_path: Path) -> dict:
    from utils.timing_history import load_history
    hist = load_history(history_path)
    groups: dict[tuple, list[dict]] = {}
    for d in hist.get("samples", []):
        groups.setdefault((d["step"], d["model"], d["device"]), []).append(d)
    rates = {}
    for (step, model, device), ds in sorted(groups.items(), key=str):
        per = [d["seconds"] / d["units"] for d in ds if d["units"] > 0]
        rates[f"{step}|{model}|{device}"] = {
            "n": len(ds),
            "median_s_per_unit": statistics.median(per) if per else None,
            "total_s_per_unit": (sum(d["seconds"] for d in ds)
                                 / max(sum(d["units"] for d in ds), 1e-12)),
            "units_total": sum(d["units"] for d in ds),
        }
    return {"rates": rates, "load_seconds": hist.get("load_seconds", {})}


def _subset(stem: str, root: Path) -> tuple[Path, Path]:
    t, a = root / f"in_{stem}" / "transcripts", root / f"in_{stem}" / "audio"
    t.mkdir(parents=True, exist_ok=True)
    a.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(DATA / "transcripts" / f"{stem}.txt", t / f"{stem}.txt")
    shutil.copyfile(DATA / "audio" / f"{stem}.wav", a / f"{stem}.wav")
    return t, a


MEMORY_VARIANTS = [("warmup", set()), ("baseline", set()),
                   ("word2vec", {"word2vec"}), ("bert", {"bert"}),
                   ("gpt2", {"gpt2"}),
                   # All heavy items together: they are process-cached (and
                   # preloaded by main.py), so they add up rather than overlap.
                   ("all", {"word2vec", "bert", "gpt2"})]


def measure_memory(root: Path, stems: list[str],
                   only: list[str] | None = None) -> dict:
    """Peak working set per variant, one fresh subprocess each.

    `only` restricts to the named variants (e.g. ["all"]); the warm-up is then
    skipped, so the stimulus should already be aligned under --scratch."""
    res: dict = {}
    for stem in stems:
        t, a = _subset(stem, root)
        out = root / f"mem_out_{stem}"
        res[stem] = {}
        # Warm-up: align once so every measured child reuses the alignment
        # (MFA is a separate subprocess anyway; this just saves time).
        variants = [(n, h) for n, h in MEMORY_VARIANTS
                    if only is None or n in only]
        for name, heavy in variants:
            cfg = root / f"cfg_mem_{stem}_{name}"
            build_config(cfg, t, a, out, heavy=heavy)
            secs, peak = _run(cfg, root / f"mem_{stem}_{name}.log", peak=True)
            if name != "warmup":
                res[stem][name] = peak
            print(f"  [{stem}] {name:9s} peak={peak / 2**30:.3f} GiB ({secs:.0f} s)",
                  flush=True)
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scratch", type=Path,
                    default=Path(tempfile.gettempdir()) / "sensi_calibration")
    ap.add_argument("--memory-stimulus", nargs="+", default=["1_FB_P", "LJ002-0321"],
                    help="stimuli for the peak-memory runs (longest + a short one)")
    ap.add_argument("--skip-full", action="store_true",
                    help="reuse an existing full run under --scratch")
    ap.add_argument("--skip-memory", action="store_true")
    ap.add_argument("--memory-variants", nargs="+", default=None,
                    choices=[n for n, _ in MEMORY_VARIANTS],
                    help="run only these memory variants (default: all)")
    args = ap.parse_args()
    root = args.scratch.resolve()
    if REPO in root.parents or root == REPO:
        raise SystemExit(f"--scratch must be outside the repo: {root}")
    root.mkdir(parents=True, exist_ok=True)

    import soundfile as sf
    print(f"machine: {platform.node()} | {platform.processor()} | "
          f"{platform.platform()} | python {platform.python_version()}")
    audio_s = {p.stem: sf.info(str(p)).duration
               for p in sorted((DATA / "audio").glob("*.wav"))
               if (DATA / "transcripts" / f"{p.stem}.txt").exists()}
    print(f"corpus: {len(audio_s)} paired stimuli, "
          f"{sum(audio_s.values()):.1f} s audio")

    cfg = root / "cfg_full"
    out = root / "out_full"
    if not args.skip_full:
        if out.exists():
            shutil.rmtree(out)
        if (cfg / "timing_history.json").exists():
            (cfg / "timing_history.json").unlink()
        build_config(cfg, DATA / "transcripts", DATA / "audio", out)
        print("full run ...", flush=True)
        secs, _ = _run(cfg, root / "full_run.log")
        print(f"full run wall time: {secs:.1f} s")

    outputs = measure_outputs(out, DATA / "transcripts")
    rates = measure_rates(cfg / "timing_history.json")
    result = {"outputs": outputs, **rates, "audio_seconds": audio_s}

    print("\n== rows per word ==")
    print(f"  words total: {outputs['words_total']}  "
          f"(word rows / word = {outputs.get('word_rows_per_word', 0):.4f})")
    for k, v in outputs["rows_per_word"].items():
        print(f"  {k:12s} {v:.4f}")
    print("\n== CSV bytes per cell ==")
    for k, v in outputs["bytes_per_cell"].items():
        c = outputs["csv"][k]
        print(f"  {k:18s} {v:7.3f}  ({c['bytes']} B / {c['rows']} rows / {c['cells']} cells)")
    print("\n== stage rates (s per unit; audio-second or word units) ==")
    for k, v in rates["rates"].items():
        print(f"  {k:40s} n={v['n']:3d} median={v['median_s_per_unit']:.5f} "
              f"total={v['total_s_per_unit']:.5f} units={v['units_total']:.1f}")
    print("\n== one-time loads (s) ==")
    for k, v in rates["load_seconds"].items():
        print(f"  {k:30s} {v:.2f}")

    if not args.skip_memory:
        print("\n== peak memory (fresh subprocess per item) ==", flush=True)
        mem = measure_memory(root, args.memory_stimulus, args.memory_variants)
        result["memory"] = mem
        for stem, d in mem.items():
            base = d.get("baseline")
            for k, v in d.items():
                extra = (f" (+{(v - base) / 2**30:.3f} GiB over baseline)"
                         if base is not None and k != "baseline" else "")
                print(f"  [{stem}] {k:9s} {v / 2**30:.3f} GiB{extra}")

    out_json = root / ("calibration.json" if args.memory_variants is None
                       and not args.skip_full else "calibration_partial.json")
    out_json.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"\nraw results: {out_json}")


if __name__ == "__main__":
    main()
