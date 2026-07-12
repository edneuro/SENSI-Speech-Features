#!/usr/bin/env python3
"""Headless SensiExtractor runner — reads config/features.yaml + config/run.yaml.

Uses the same planner + batch runners as the GUI, so behavior matches the app
exactly. features.yaml says WHAT to extract (shareable; the GUI edits it);
run.yaml says WHERE on this machine (folders, analyzer, token-file paths).
Usage:
    python run.py [--config-dir DIR]    # DIR defaults to <repo>/config
"""
import argparse
import datetime as _dt
import os
import sys

# MKL on AMD CPUs crashes with the default Intel-thread layer (0xc06d007f);
# every entry point must set this before numpy does BLAS work (cf. main.py).
os.environ.setdefault("MKL_THREADING_LAYER", "sequential")

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def load_credentials(shared: dict) -> None:
    """Set HF_TOKEN from shared.hf_token_file (gated-Llama downloads)."""
    path = shared.get("hf_token_file")
    if not path:
        return
    resolved = os.path.expanduser(os.path.expandvars(path))
    if not os.path.isfile(resolved):
        return
    try:
        token = open(resolved, encoding="utf-8").read().strip()
    except OSError as exc:
        print(f"[warn] could not read hf_token_file {resolved}: {exc}", file=sys.stderr)
        return
    if token:
        os.environ["HF_TOKEN"] = token


def _prepare(ns):
    # Mirror run_dispatch._prepare_runtime_args: edu_bands list -> comma string.
    if isinstance(getattr(ns, "edu_bands", None), list):
        ns.edu_bands = ",".join(ns.edu_bands) if ns.edu_bands else None
    return ns


def dispatch(prefs: dict) -> None:
    """Run the enabled levels from a prefs dict (mirrors gui.run_dispatch.run)."""
    from gui import planner
    plan = planner.plan_from_prefs(prefs)
    run_ts = _dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    outputs = plan.output_csvs
    align_cache: dict = {}

    word_outputs = {"features", "syllables", "phonemes"} & outputs
    if word_outputs:
        from main import run_word_level_batch
        ns = _prepare(planner.to_namespace(plan, prefs, level="word"))
        ns.run_ts = run_ts
        ns.skip_no_match = False
        ns._align_cache = align_cache
        run_word_level_batch(ns)
    if "documents_summary" in outputs:
        from main import run_doc_level_batch
        ns = _prepare(planner.to_namespace(plan, prefs, level="doc"))
        ns.run_ts = run_ts
        ns.skip_no_match = bool(word_outputs)
        ns._align_cache = align_cache
        run_doc_level_batch(ns)
    if "acoustic" in outputs:
        from main import run_acoustic_level_batch
        ns = planner.to_acoustic_namespace(prefs)
        ns.run_ts = run_ts
        run_acoustic_level_batch(ns)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config-dir", default=None,
                    help="directory containing features.yaml + run.yaml "
                         "(default: <repo>/config, or $SENSI_CONFIG_DIR)")
    args = ap.parse_args()
    if args.config_dir:
        os.environ["SENSI_CONFIG_DIR"] = args.config_dir

    import yaml as _yaml
    from utils.preferences import (load_preferences, get_features_yaml_path,
                                   get_run_yaml_path)
    from utils.config_yaml import (read_yaml, validate_features_yaml,
                                   validate_run_yaml)

    errors: list[str] = []
    for path, validate in ((get_features_yaml_path(), validate_features_yaml),
                           (get_run_yaml_path(), validate_run_yaml)):
        try:
            data = read_yaml(path)
        except _yaml.YAMLError as exc:
            errors.append(f"{path.name}: not valid YAML ({exc})")
            continue
        errors.extend(f"{path.name}: {e}" for e in validate(data))
    if errors:
        print("Invalid config:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        sys.exit(1)

    prefs = load_preferences()
    shared = prefs["shared"]
    # Expand ~ and $VARS so HPC paths like $SCRATCH/... resolve.
    for key in ("last_transcripts_folder", "last_audio_folder",
                "last_output_folder"):
        val = shared.get(key) or ""
        shared[key] = os.path.expanduser(os.path.expandvars(val)) if val else ""

    if not (shared.get("analyzer_name") or "").strip():
        print("run.yaml: shared.analyzer_name is required (name for this run).",
              file=sys.stderr)
        sys.exit(1)
    missing = [k for k in ("last_transcripts_folder", "last_output_folder")
               if not (shared.get(k) or "").strip()]
    if missing:
        print(f"run.yaml: missing required folder(s): {', '.join(missing)}",
              file=sys.stderr)
        sys.exit(1)

    load_credentials(shared)
    dispatch(prefs)
    print(f"Done. Outputs under {shared['last_output_folder']}.")


if __name__ == "__main__":
    main()
