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
# every entry point must set this before numpy does BLAS work (cf. sensi/pipeline.py).
os.environ.setdefault("MKL_THREADING_LAYER", "sequential")

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Keep all model/cache downloads inside the project folder.
from sensi.utils.local_caches import (  # noqa: E402
    move_legacy_model_dirs, use_project_caches)
move_legacy_model_dirs()
use_project_caches()


def _prepare(ns):
    # Mirror run_dispatch._prepare_runtime_args: edu_bands list -> comma string.
    if isinstance(getattr(ns, "edu_bands", None), list):
        ns.edu_bands = ",".join(ns.edu_bands) if ns.edu_bands else None
    return ns


def dispatch(prefs: dict) -> None:
    """Run the enabled levels from a prefs dict (mirrors sensi.gui.run_dispatch.run)."""
    from sensi.gui import planner
    plan = planner.plan_from_prefs(prefs)
    run_ts = _dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    outputs = plan.output_csvs
    align_cache: dict = {}

    word_outputs = planner.WORD_BATCH_ARTIFACTS & outputs
    if word_outputs:
        from sensi.pipeline import run_word_level_batch
        ns = _prepare(planner.to_namespace(plan, prefs, level="word"))
        ns.run_ts = run_ts
        ns.skip_no_match = False
        ns._align_cache = align_cache
        run_word_level_batch(ns)
    if "documents_summary" in outputs:
        from sensi.pipeline import run_doc_level_batch
        ns = _prepare(planner.to_namespace(plan, prefs, level="doc"))
        ns.run_ts = run_ts
        ns.skip_no_match = bool(word_outputs)
        ns._align_cache = align_cache
        run_doc_level_batch(ns)
    if "acoustic" in outputs:
        from sensi.pipeline import run_acoustic_level_batch
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
    from sensi.utils.preferences import (load_preferences, get_features_yaml_path,
                                   get_run_yaml_path)
    from sensi.utils.config_yaml import (validate_features_yaml, validate_run_yaml,
                                   retired_keys_present)

    # Parse each file's raw content directly rather than via read_yaml: that
    # helper coerces a non-mapping root (e.g. a YAML list) to {}, which would
    # let a malformed features.yaml validate clean and silently run with
    # defaults. read_yaml's tolerant coercion is left alone for
    # load_preferences, which relies on it (half-import fallback).
    #
    # This validates the RAW (pre-migration) shape on purpose — it exists to
    # catch a malformed hand-edited config before an hour of work, not to
    # duplicate load_preferences()'s migration below. A key retired by a past
    # migration (e.g. v19's feat_phoneme_label) is therefore tolerated here
    # (warned, not errored) via retired_keys_present — see sensi/utils/config_yaml.py.
    errors: list[str] = []
    features_raw: dict = {}
    features_path = get_features_yaml_path()
    for path, validate in ((features_path, validate_features_yaml),
                           (get_run_yaml_path(), validate_run_yaml)):
        if path.exists():
            try:
                with path.open("r", encoding="utf-8") as f:
                    raw = _yaml.safe_load(f)
            except _yaml.YAMLError as exc:
                errors.append(f"{path.name}: not valid YAML ({exc})")
                continue
        else:
            raw = None
        if raw is not None and not isinstance(raw, dict):
            errors.append(f"{path.name}: root must be a mapping")
            continue
        if path == features_path:
            features_raw = raw or {}
            if raw is None:
                print(f"[warn] {path.name} is empty — running with shipped "
                      "defaults (minimal feature set).", file=sys.stderr)
        errors.extend(f"{path.name}: {e}" for e in validate(raw or {}))
    if errors:
        print("Invalid config:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        sys.exit(1)
    for w in retired_keys_present(features_raw):
        print(f"[warn] {w}", file=sys.stderr)

    if not get_features_yaml_path().exists():
        print(f"[warn] features.yaml not found in {get_features_yaml_path().parent} "
              f"— running with shipped defaults (minimal feature set). To start "
              f"from the documented defaults: copy config/features_defaults.yaml "
              f"to config/features.yaml and edit.", file=sys.stderr)

    prefs = load_preferences()
    shared = prefs["shared"]
    # Expand ~ and $VARS so HPC paths like $SCRATCH/... resolve. A path that is
    # still relative is anchored to the config folder (like hf_token_file), so
    # the same run.yaml means the same folders wherever the job is launched.
    config_dir = get_run_yaml_path().parent
    for key in ("last_transcripts_folder", "last_audio_folder",
                "last_output_folder"):
        val = shared.get(key) or ""
        if val:
            val = os.path.expanduser(os.path.expandvars(val))
            if not os.path.isabs(val):
                val = os.path.normpath(os.path.join(config_dir, val))
        shared[key] = val

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

    from sensi.utils.credentials import export_hf_token
    export_hf_token(shared)
    from sensi.pipeline import InputFolderError
    try:
        dispatch(prefs)
    except InputFolderError as exc:
        # Missing/typo'd input folders or no transcripts (main._check_input_folders).
        print(f"run.yaml: {exc}", file=sys.stderr)
        sys.exit(1)
    print(f"Done. Outputs under {shared['last_output_folder']}.")


if __name__ == "__main__":
    main()
