"""Frozen dataclasses for the planner. No logic — just types.

See docs/superpowers/specs/2026-05-14-planner-design.md §5.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class Stage:
    """One unit of pipeline execution.

    consumes/produces are artifact-name tuples used by the planner's
    DAG closure (plan.py) to determine which stages must run to satisfy
    an active cluster's `requires_stages`.
    """
    name: str
    consumes: tuple[str, ...]
    produces: tuple[str, ...]
    run: Callable


@dataclass(frozen=True)
class FeatureCluster:
    """A group of user-facing columns sharing a stage and a toggle group.

    A cluster is active iff:
    - `available_when` predicate (when present) matches the live prefs
    - `master_toggle` (when present) is True in prefs
    - `output_csv` is among the user's requested outputs
    - any key in `toggle_keys` is True in prefs
    """
    name: str
    toggle_keys: tuple[str, ...]
    output_csv: str   # "features" | "syllables" | "phonemes" | "documents_summary"
    requires_stages: tuple[str, ...]
    emits_columns: tuple[str, ...]
    master_toggle: str | None = None
    available_when: tuple[str, str] | None = None


@dataclass(frozen=True)
class Plan:
    """The output of plan_from_prefs — what to run, what to write."""
    stages_to_run: tuple[str, ...]
    output_csvs: frozenset[str]
    column_drops: dict[str, frozenset[str]]
    stage_configs: dict[str, dict[str, Any]]
    # Dynamic columns not representable by any FeatureCluster.emits_columns
    # (fixed at registry-definition time). Currently just the predictability
    # <slug>_<Metric> columns — see gui/planner/plan.py's
    # _predictability_columns / _predictability_active and the "Predictability"
    # cluster's docstring in gui/planner/registry.py.
    predictability_columns: tuple[str, ...] = ()
    # Which batch pass needs the predictability models (plan.py
    # _predictability_active): the word pass for <slug>_<Metric> word-CSV
    # columns, the doc pass for the mean_<slug>_<metric> doc means. emit.py
    # hands the models/metrics only to the namespace whose path is active.
    predictability_word: bool = False
    predictability_doc: bool = False

    def emitted_columns_for(self, name: str) -> tuple[str, ...]:
        """Look up a dynamic-column group by name. Only 'predictability' is
        registered today; unknown names return an empty tuple rather than
        raising, so callers can probe without a membership check first."""
        if name == "predictability":
            return self.predictability_columns
        return ()
