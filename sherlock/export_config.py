#!/usr/bin/env python3
"""Export the GUI's saved settings into a portable config.yaml for Sherlock.

Reads the app's preferences (respecting SENSI_PREFS_PATH) and writes a config the
Sherlock runner (run_extract.py) consumes. Fill the `run:` section on Sherlock.
"""
import argparse
import os
import sys

# This script lives in sherlock/; add the repo root to the path so `utils.*`
# imports resolve when run directly as `python sherlock/export_config.py`.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from utils.preferences import load_preferences
from utils.config_yaml import prefs_to_config, write_config

# Default config lives alongside this script, in sherlock/.
_DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=_DEFAULT_CONFIG,
                    help="output path (default: sherlock/config.yaml)")
    args = ap.parse_args()
    config = prefs_to_config(load_preferences())
    write_config(config, args.out)
    print(f"Wrote {args.out}. Before running on Sherlock, fill the `run:` section "
          f"(transcripts/audio/output folders, analyzer, and token-file paths).")


if __name__ == "__main__":
    main()
