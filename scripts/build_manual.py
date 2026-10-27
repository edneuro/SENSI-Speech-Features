#!/usr/bin/env python
"""Build the self-contained User Manual into docs/user-manual/.

    python scripts/build_manual.py                       # rebuild pages from committed example outputs
    python scripts/build_manual.py --refresh-examples    # first re-run SensiExtractor on the examples (needs models)
    python scripts/build_manual.py --refresh-screenshots # first re-capture the GUI screenshots (offscreen)
    python scripts/build_manual.py --check-links         # also request every external link and report

Sources: docs/features/*.md (the same files the app's ⓘ pop-ups show),
dev/manual/ (outline manifest, chapters, templates, theme, example inputs and
the real outputs SensiExtractor produced on them). See
dev/superpowers/specs/2026-10-27-user-manual-design.md.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MKL_THREADING_LAYER", "sequential")
ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from sensi.utils.local_caches import use_project_caches  # noqa: E402
use_project_caches()

OUT_DIR = ROOT / "docs" / "user-manual"


def main(argv=None) -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--refresh-examples", action="store_true")
    ap.add_argument("--examples-only", action="store_true",
                    help="with --refresh-examples: stop after the runs")
    ap.add_argument("--variants", nargs="*", help="limit --refresh-examples to these variants")
    ap.add_argument("--refresh-screenshots", action="store_true")
    ap.add_argument("--check-links", action="store_true")
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args(argv)

    if args.refresh_examples:
        from manual.examples import refresh_examples
        for v, path in refresh_examples(variants=args.variants).items():
            print(f"[manual] {v}: {sum(1 for _ in path.rglob('*') if _.is_file())} files")
        if args.examples_only:
            return 0
    if args.refresh_screenshots:
        from manual.screenshots import capture
        capture()
    from manual.site import build_site
    report = build_site(args.out, check_links=args.check_links)
    print(report.summary())
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
