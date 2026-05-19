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
    - `output_csv` is among the user's requested outputs, AND
    - any key in `toggle_keys` is True in prefs["features"].
    """
    name: str
    toggle_keys: tuple[str, ...]
    output_csv: str   # "features" | "syllables" | "phonemes" | "documents_summary"
    requires_stages: tuple[str, ...]
    emits_columns: tuple[str, ...]


@dataclass(frozen=True)
class Plan:
    """The output of plan_from_prefs — what to run, what to write."""
    stages_to_run: tuple[str, ...]
    output_csvs: frozenset[str]
    column_drops: dict[str, frozenset[str]]
    stage_configs: dict[str, dict[str, Any]]
