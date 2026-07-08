#!/usr/bin/env python3
"""Export the GUI's saved settings into a portable config.yaml for Sherlock.

Reads the app's preferences (respecting SENSI_PREFS_PATH) and writes a config the
Sherlock runner (run_extract.py) consumes. Fill the `run:` section on Sherlock.
"""
import argparse

from utils.preferences import load_preferences
from utils.config_yaml import prefs_to_config, write_config


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="config.yaml", help="output path (default: config.yaml)")
    args = ap.parse_args()
    config = prefs_to_config(load_preferences())
    write_config(config, args.out)
    print(f"Wrote {args.out}. Before running on Sherlock, fill the `run:` section "
          f"(transcripts/audio/output folders, analyzer, and token-file paths).")


if __name__ == "__main__":
    main()
