#!/usr/bin/env python3
"""Regenerate the committed config templates from the code defaults.

config/features_defaults.yaml and config/run_defaults.yaml are build
artifacts of utils.preferences.DEFAULT_PREFS — the code is the source of
truth. A drift-guard test (tests/utils/test_config_templates.py) fails
whenever the schema changes without rerunning this script:

    python scripts/export_default_config.py

Headless users copy the templates to features.yaml / run.yaml and edit.
"""
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from utils.preferences import DEFAULT_PREFS  # noqa: E402
from utils.config_yaml import (  # noqa: E402
    split_prefs, write_features_yaml, write_run_yaml,
)

TEMPLATE_BANNER_FEATURES = (
    "# ============================ TEMPLATE ============================\n"
    "# Copy this file to config/features.yaml and edit — it is the shipped\n"
    "# defaults for \"what to extract\". Regenerate after a schema change:\n"
    "#     python scripts/export_default_config.py\n"
    "# ==================================================================\n"
)
TEMPLATE_BANNER_RUN = (
    "# ============================ TEMPLATE ============================\n"
    "# Copy this file to config/run.yaml and edit — machine-specific run\n"
    "# settings (folders, analyzer name, token-file path). Regenerate:\n"
    "#     python scripts/export_default_config.py\n"
    "# ==================================================================\n"
)


def main() -> None:
    config_dir = os.path.join(_REPO_ROOT, "config")
    features_data, run_data = split_prefs(DEFAULT_PREFS)
    f_path = os.path.join(config_dir, "features_defaults.yaml")
    r_path = os.path.join(config_dir, "run_defaults.yaml")
    write_features_yaml(f_path, features_data)
    write_run_yaml(r_path, run_data)
    for path, banner in ((f_path, TEMPLATE_BANNER_FEATURES),
                         (r_path, TEMPLATE_BANNER_RUN)):
        with open(path, "r", encoding="utf-8") as fh:
            body = fh.read()
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(banner + body)
    print(f"Wrote {f_path} and {r_path} from DEFAULT_PREFS "
          f"(schema v{DEFAULT_PREFS['schema_version']}).")


if __name__ == "__main__":
    main()
