# `dev/` — project history and development material

Not needed to install or run SensiExtractor. This folder is the record of
how the tool was built and why it works the way it does.

| Path | What it is |
|---|---|
| `master_plan.md` | Current project state: what's active, completed batches, roadmap, conventions. Start here. |
| `changelog.md` | Full write-up of every completed batch (deep history). |
| `feature_dependencies.md` | Map of which features depend on which pipeline stages; the planner (`sensi/gui/planner/`) implements it. |
| `superpowers/specs/`, `superpowers/plans/` | Design docs and implementation plans, one pair per batch of work. |
| `feature-docs-audit.md`, `AoA&MRC.md` | Earlier citation audit and a feature brief. |
| `validation/verify_phoneme_batch.py` | Real-stimulus check used for the phoneme-split batch. |

Kept on the maintainer's machine only (git-ignored): the validation corpus
and its runs (`validation/data/`, `validation/output/`, …), audit working
files (`audit/`), and scratch work (`unrelated/`).

**Paths in older documents.** The repo was reorganized on 2026-10-27: code
moved into `sensi/` (`main.py` became `sensi/pipeline.py`, and its command
line was retired), dictionaries and models into `data/`, and this material
from `docs/` into `dev/`. Specs, plans and the changelog written before that
date name the old paths (`extractors/…`, `utils/…`, `gui/…`, `main.py`,
`dictionaries/…`, `docs/superpowers/…`); they are left as written.
