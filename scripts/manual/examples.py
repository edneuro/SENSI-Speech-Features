"""Real SensiExtractor output on the manual's example inputs.

Every number, table and figure in the User Manual comes from the files this
module writes under dev/manual/example_outputs/<variant>/. It runs the real
headless launcher (run.py) once per variant, in a subprocess, with a scratch
config built from the app's own defaults — never the user's config/.
"""
from __future__ import annotations

import copy
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from sensi.paths import PROJECT_ROOT

EXAMPLES_DIR = PROJECT_ROOT / "dev" / "manual" / "examples"
OUTPUTS_DIR = PROJECT_ROOT / "dev" / "manual" / "example_outputs"

_ALL_LEVELS = ("level_word", "level_document", "level_subword",
               "level_sentence", "level_acoustic")

# One setting changed per variant (spec item 16). "levels" limits the run to
# the level that setting affects; "keep" is the one file the manual shows from it.
VARIANTS: dict[str, dict] = {
    "base":        {"levels": _ALL_LEVELS, "set": {}},
    "env_hilbert": {"levels": ("level_acoustic",), "set": {"acoustic_ts": {"env_method": "hilbert"}},
                    "keep": "acoustic/LJ002-0321_acoustic.mat"},
    "mel_music":   {"levels": ("level_acoustic",), "set": {"acoustic_ts": {"mel_preset": "music"}},
                    "keep": "acoustic/LJ002-0321_acoustic.mat"},
    "pitch_hz":    {"levels": ("level_word",), "set": {"prosody": {"pitch_units": "hz"}},
                    "keep": "word/LJ002-0321_features.csv"},
}

_TS = re.compile(r"_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}")
# Embedding .mat files are large (BERT: words x 12 x 768); keep them only for
# the stems the manual shows, so the committed outputs stay small.
EMBEDDING_STEMS = {"LJ002-0321", "LJ003-0007"}
_EMBEDDING_SUFFIXES = ("_bert_word.mat", "_bert_sentence.mat", "_word2vec.mat")


def _prefs(variant: str, out_dir: Path) -> dict:
    """Every feature and output on (as scripts/calibrate_estimates.py does),
    GPT-2 for predictability (no token needed), EWFG off (no data exists)."""
    from sensi.utils.preferences import DEFAULT_PREFS
    p = copy.deepcopy(DEFAULT_PREFS)
    for section in ("features", "output"):
        for k, v in list(p[section].items()):
            if isinstance(v, bool):
                p[section][k] = True
    for k in list(p["features"]):
        if k.startswith("freq_ewfg_"):
            p["features"][k] = False
    p["output"]["doc_agg_edu_zipf"] = False
    spec = VARIANTS[variant]
    for k in _ALL_LEVELS:
        p["output"][k] = k in spec["levels"]
    p["predictability"]["models"] = ["gpt2"]
    p["predictability"]["metrics"] = {m: True for m in p["predictability"]["metrics"]}
    p["semantic"]["backend"] = "empath"
    for section, values in spec["set"].items():
        p[section].update(values)
    p["shared"].update({
        "analyzer_name": "SensiExtractor manual",
        "last_transcripts_folder": str(EXAMPLES_DIR / "transcripts"),
        "last_audio_folder": str(EXAMPLES_DIR / "audio"),
        "last_output_folder": str(out_dir),
    })
    return p


def _env() -> dict:
    env = dict(os.environ)
    e = PROJECT_ROOT / "env"
    extra = [str(e), str(e / "Scripts"), str(e / "Library" / "bin")]
    env["PATH"] = os.pathsep.join(extra + [env.get("PATH", "")])
    env.setdefault("MKL_THREADING_LAYER", "sequential")
    return env


def _write_config(prefs: dict, cfg_dir: Path) -> None:
    """features.yaml + run.yaml through the app's own writer, in a child
    process so this process's SENSI_CONFIG_DIR is untouched."""
    code = ("import json,sys; from sensi.utils.preferences import save_preferences;"
            "p=json.load(sys.stdin); save_preferences(p)")
    import json
    env = _env() | {"SENSI_CONFIG_DIR": str(cfg_dir)}
    subprocess.run([sys.executable, "-c", code], input=json.dumps(prefs), text=True,
                   cwd=PROJECT_ROOT, env=env, check=True)


def _copy_outputs(raw: Path, dest: Path, keep: str | None = None) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    for f in sorted(raw.rglob("*")):
        rel = f.relative_to(raw)
        if f.is_dir() or rel.parts[0] in ("logs", "Montreal Forced Alignment"):
            continue
        name = _TS.sub("", f.name)
        if keep and (rel.parent / name).as_posix() != keep:
            continue
        if name.endswith(_EMBEDDING_SUFFIXES) and not any(name.startswith(st + "_") for st in EMBEDDING_STEMS):
            continue
        target = dest / rel.parent / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if f.suffix == ".csv" and "documents_summary" in name:
            import pandas as pd
            df = pd.read_csv(f)
            for col in ("transcript_path", "audio_path"):
                if col in df:
                    df[col] = df[col].map(
                        lambda v: Path(v).relative_to(PROJECT_ROOT).as_posix()
                        if isinstance(v, str) and v else v)
            df.to_csv(target, index=False)
        else:
            shutil.copyfile(f, target)


def refresh_examples(out_root: Path = OUTPUTS_DIR, variants=None) -> dict[str, Path]:
    """Run run.py once per variant; copy its outputs to out_root/<variant>/."""
    done = {}
    for variant in variants or VARIANTS:
        with tempfile.TemporaryDirectory(prefix=f"manual_{variant}_") as tmp:
            tmp = Path(tmp)
            raw, cfg = tmp / "out", tmp / "cfg"
            raw.mkdir(); cfg.mkdir()
            _write_config(_prefs(variant, raw), cfg)
            print(f"[manual] running {variant} …", flush=True)
            proc = subprocess.run([sys.executable, "run.py", "--config-dir", str(cfg)],
                                  cwd=PROJECT_ROOT, env=_env(), capture_output=True, text=True)
            log = proc.stdout + proc.stderr
            if proc.returncode != 0:
                raise RuntimeError(f"run.py failed for {variant}:\n{log[-3000:]}")
            # A run that silently failed to align still exits 0 and prints "Done"
            # (memory: mfa.exe on PATH) — insist on the aligner's own report.
            if "level_word" in VARIANTS[variant]["levels"] and not re.search(r"\[mfa\] \d+ stimuli:", log):
                raise RuntimeError(f"{variant}: MFA never reported aligning — check PATH.\n{log[-2000:]}")
            _copy_outputs(raw, out_root / variant, VARIANTS[variant].get("keep"))
            done[variant] = out_root / variant
    return done
