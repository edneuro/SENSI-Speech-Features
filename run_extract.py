#!/usr/bin/env python3
"""Run the SensiExtractor pipeline on Sherlock from a portable config.yaml.

Maps the config onto the app's prefs schema and reuses the same planner + batch
runners the GUI uses, so behavior matches the app exactly. Usage:
    python run_extract.py [config.yaml]
"""
import argparse
import datetime as _dt
import os
import sys

from utils.config_yaml import read_config, validate_config, config_to_prefs
from gui import planner


def load_credentials(run: dict) -> None:
    """Set HF_TOKEN / OPENAI_API_KEY from the key-file paths in the run block."""
    for field, env in (("hf_token_file", "HF_TOKEN"),
                       ("openai_api_key_file", "OPENAI_API_KEY")):
        path = run.get(field)
        if not path:
            continue
        resolved = os.path.expanduser(os.path.expandvars(path))
        if not os.path.isfile(resolved):
            continue
        try:
            token = open(resolved, encoding="utf-8").read().strip()
        except OSError as exc:
            print(f"[warn] could not read {field} {resolved}: {exc}", file=sys.stderr)
            continue
        if token:
            os.environ[env] = token


def _prepare(ns):
    # Mirror run_dispatch._prepare_runtime_args: edu_bands list -> comma string.
    if isinstance(getattr(ns, "edu_bands", None), list):
        ns.edu_bands = ",".join(ns.edu_bands) if ns.edu_bands else None
    return ns


def dispatch(config: dict) -> None:
    """Build prefs from config and run the enabled levels (mirrors run_dispatch.run)."""
    prefs = config_to_prefs(config)
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
    ap.add_argument("config", nargs="?", default="config.yaml",
                    help="path to the YAML config (default: config.yaml)")
    args = ap.parse_args()

    config = read_config(args.config)
    errors = validate_config(config)
    if errors:
        print("Invalid config:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        sys.exit(1)

    run = config.get("run") or {}
    if not (run.get("analyzer") or "").strip():
        print("run.analyzer is required (name for this run).", file=sys.stderr)
        sys.exit(1)

    load_credentials(run)
    dispatch(config)
    out = run.get("output_folder") or "(output_folder unset)"
    print(f"Done. Outputs under {out}.")


if __name__ == "__main__":
    main()
