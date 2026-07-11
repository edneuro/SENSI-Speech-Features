"""plan_from_prefs(prefs) -> Plan.

Pure function: deterministic, no I/O.

Algorithm (see spec §6):
  1. Determine active output CSVs from prefs["output"].
  2. For each cluster, mark active iff its output_csv is wanted AND any
     of its toggle_keys is True in prefs["features"] (or prefs["output"]
     for output-keyed clusters like doc_counts, doc_lex_diversity, etc).
  3. Union the requires_stages of active clusters.
  4. Close the set under stage.consumes ↔ stage.produces.
  5. Topologically sort.
  6. Compute per-CSV column-drop sets.
  7. Build stage_configs dict.
"""
from __future__ import annotations

from typing import Any

from gui.planner.registry import STAGES, CLUSTERS
from gui.planner.stages import Plan, FeatureCluster


# Toggle keys that live in prefs["output"] rather than prefs["features"]:
# doc-count format keys, doc-level cluster activation toggles, and the two
# doc-level cluster master gates added in v7 (master_doc_prosody,
# master_doc_articulation_rate). All other cluster toggle_keys (per-column
# feat_*, masters that gate features-section clusters) live in
# prefs["features"]. See utils/preferences.py:_OUTPUT_DEFAULT_V7 for the
# canonical set of output-keyed prefs.
OUTPUT_KEYED_TOGGLES = frozenset({
    "out_count_tokens", "out_count_content_tokens",
    "out_count_sentences", "out_count_paragraphs",
    # Individual aggregate toggles (replaced single doc_aggregates in v7)
    "doc_agg_zipf_freq", "doc_agg_edu_zipf", "doc_agg_surprisal", "doc_agg_llm",
    "doc_agg_lexical_norms",
    "doc_lex_diversity", "doc_readability",
    "doc_category_proportions",
    "master_doc_prosody", "master_doc_articulation_rate",
    "master_doc_counts",
})

_OUTPUT_CSV_MAP = {
    "word_level_csv":    "features",
    "syllables_csv":     "syllables",
    "phonemes_csv":      "phonemes",
    "sentences_csv":     "sentences",
    "word2vec_mat":      "word2vec",
    "bert_word_mat":     "bert_word",
    "bert_sentence_mat": "bert_sentence",
    "document_summary":  "documents_summary",
    "acoustic_ts_mat":   "acoustic",
    "phoneme_surprisal_mat": "phoneme_surprisal_mat",
}


def _get_dotted(prefs: dict, path: str):
    """Read a value from prefs by a dotted path like 'semantic.backend'."""
    cur = prefs
    for part in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def _active_output_csvs(prefs: dict) -> frozenset[str]:
    out = prefs.get("output", {})
    return frozenset(
        csv for prefs_key, csv in _OUTPUT_CSV_MAP.items()
        if out.get(prefs_key, False)
    )


def _toggle_value(prefs: dict, key: str) -> bool:
    """Read a toggle from the right prefs section."""
    if key in OUTPUT_KEYED_TOGGLES:
        return bool(prefs.get("output", {}).get(key, False))
    return bool(prefs.get("features", {}).get(key, False))


def _predictability_config(prefs: dict) -> tuple[list[str], list[str]]:
    """(models, metrics) from prefs["predictability"], metrics filtered to
    the ones switched on, in a stable (surprisal, cloze, topmatch) order."""
    pred = prefs.get("predictability", {}) or {}
    models = list(pred.get("models") or [])
    metrics_on = pred.get("metrics", {}) or {}
    metrics = [m for m in ("surprisal", "cloze", "topmatch") if metrics_on.get(m)]
    return models, metrics


_PREDICTABILITY_METRIC_CAP = {
    "surprisal": "Surprisal", "cloze": "Cloze", "topmatch": "TopMatch",
}


def _predictability_columns(models: list[str], metrics: list[str]) -> tuple[str, ...]:
    return tuple(
        f"{slug}_{_PREDICTABILITY_METRIC_CAP[m]}"
        for slug in models for m in metrics
    )


def _predictability_active(prefs: dict, active_csvs: frozenset[str]) -> bool:
    """The predictability stage/cluster's true activation predicate (spec §12
    fallback — see the "Predictability" FeatureCluster's docstring in
    registry.py). Not expressible via toggle_keys because it depends on two
    dynamic prefs['predictability'] fields (models list, metrics dict), not a
    fixed set of boolean feature-toggle keys."""
    if "features" not in active_csvs:
        return False
    if not _toggle_value(prefs, "master_predictability"):
        return False
    models, metrics = _predictability_config(prefs)
    return bool(models) and bool(metrics)


def _active_clusters(prefs: dict, active_csvs: frozenset[str]) -> list[FeatureCluster]:
    actives = []
    for c in CLUSTERS:
        # The "Predictability" cluster's activation is dynamic (models list x
        # metrics dict) and is computed separately in plan_from_prefs via
        # _predictability_active/_predictability_columns — see its docstring
        # in registry.py. Skip it here so it never gets falsely marked active
        # just because master_predictability is on and "features" is wanted.
        if c.name == "Predictability":
            continue
        # 1. available_when: hard env gate
        if c.available_when is not None:
            path, expected = c.available_when
            if _get_dotted(prefs, path) != expected:
                continue
        # 2. master_toggle: user gate
        if c.master_toggle is not None and not _toggle_value(prefs, c.master_toggle):
            continue
        # 3. output_csv must be wanted
        if c.output_csv not in active_csvs:
            continue
        # 4. any child toggle on (empty toggle_keys = always active when output_csv + master pass)
        if c.toggle_keys and not any(_toggle_value(prefs, k) for k in c.toggle_keys):
            continue
        actives.append(c)
    return actives


def _stage_by_name() -> dict[str, Stage]:
    return {s.name: s for s in STAGES}


def _close_under_consumes(required: set[str]) -> set[str]:
    """Walk STAGES backward: for each artifact consumed by a required stage,
    pull in any Stage that produces that artifact. Repeat until stable.

    Produces entries are matched exactly — a stage that produces 'word_df.Zipf_Freq'
    does NOT satisfy a consumer of plain 'word_df'. Only the stage that produces
    'word_df' itself does. This prevents mutation-stages (word_freq, word_prosody,
    etc.) from being pulled in just because some stage consumes the base DataFrame.

    The consumer side uses the base name only (strip everything after the first dot)
    so that 'word_df.Zipf_Freq' as a consume token still resolves to the stage that
    produces 'word_df'.
    """
    by_name = _stage_by_name()
    # Map exact produce artifact → list of stage names. Do NOT strip dots here.
    artifacts_produced_by: dict[str, list[str]] = {}
    for s in STAGES:
        for a in s.produces:
            artifacts_produced_by.setdefault(a, []).append(s.name)

    closed = set(required)
    while True:
        added = set()
        for name in list(closed):
            s = by_name[name]
            for art in s.consumes:
                # Consumer uses the base name (strip dot-suffix) to match a producer.
                base = art.split(".", 1)[0]
                for producer in artifacts_produced_by.get(base, ()):
                    if producer not in closed:
                        added.add(producer)
        if not added:
            break
        closed |= added
    return closed


def _topo_sort(stage_names: set[str]) -> tuple[str, ...]:
    """Topological sort by consumes/produces. Tiebreak by registry order
    for stable output."""
    by_name = _stage_by_name()
    registry_order = {s.name: i for i, s in enumerate(STAGES)}

    # Map exact produce artifact → stage name (first producer wins).
    produced_by: dict[str, str] = {}
    for s in STAGES:
        for a in s.produces:
            produced_by.setdefault(a, s.name)

    in_set = set(stage_names)
    deps: dict[str, set[str]] = {n: set() for n in in_set}
    for n in in_set:
        for art in by_name[n].consumes:
            # Consumer resolves using the base name (strip dot-suffix).
            base = art.split(".", 1)[0]
            producer = produced_by.get(base)
            if producer and producer in in_set and producer != n:
                deps[n].add(producer)

    ordered: list[str] = []
    remaining = set(in_set)
    while remaining:
        ready = sorted(
            [n for n in remaining if not (deps[n] - set(ordered))],
            key=lambda n: registry_order[n],
        )
        if not ready:
            raise ValueError(f"Cycle in stage dependencies among {remaining}")
        ordered.extend(ready)
        remaining -= set(ready)
    return tuple(ordered)


def _column_drops(active_clusters: list[FeatureCluster],
                  active_csvs: frozenset[str]) -> dict[str, frozenset[str]]:
    """For each active CSV, columns from inactive clusters with that CSV
    go in the drop set. Active-cluster columns survive."""
    active_cluster_set = set(id(c) for c in active_clusters)
    drops: dict[str, set[str]] = {csv: set() for csv in active_csvs}
    for c in CLUSTERS:
        if c.output_csv not in active_csvs:
            continue
        if id(c) in active_cluster_set:
            continue
        drops[c.output_csv].update(c.emits_columns)
    return {csv: frozenset(cols) for csv, cols in drops.items()}


def _stage_configs(prefs: dict, stages_to_run: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    """Per-stage parameter dict; only includes stages actually in the plan."""
    configs: dict[str, dict[str, Any]] = {}
    in_plan = set(stages_to_run)
    pros = prefs.get("prosody", {})
    pitch_units = pros.get("pitch_units", "semitones")
    if "praat_contours" in in_plan:
        configs["praat_contours"] = {
            "pitch_floor":   float(pros.get("pitch_floor_hz", 75.0)),
            "pitch_ceiling": float(pros.get("pitch_ceiling_hz", 500.0)),
        }
    if "word_prosody" in in_plan:
        configs["word_prosody"] = {"pitch_units": pitch_units}
    if "syllable_prosody" in in_plan:
        configs["syllable_prosody"] = {"pitch_units": pitch_units}
    if "sentence_prosody" in in_plan:
        configs["sentence_prosody"] = {"pitch_units": pitch_units}
    if "doc_prosody" in in_plan:
        configs["doc_prosody"] = {
            "pitch_units":     pitch_units,
            "pause_threshold": float(pros.get("pause_threshold_s", 0.25)),
        }
    if "predictability" in in_plan:
        models, metrics = _predictability_config(prefs)
        configs["predictability"] = {"models": models, "metrics": metrics}
    return configs


def plan_from_prefs(prefs: dict) -> Plan:
    """Build the Plan from a user-prefs dict. Pure function; deterministic."""
    active_csvs = _active_output_csvs(prefs)
    active = _active_clusters(prefs, active_csvs)
    # Restrict output_csvs to only those CSVs that have at least one active
    # cluster — avoids writing empty files when the master toggle is off but
    # the output key is on (e.g. acoustic_ts_mat=True with master_acoustic_ts=False).
    live_csvs = frozenset(c.output_csv for c in active)
    required: set[str] = set()
    for c in active:
        required.update(c.requires_stages)

    # Predictability: dynamic activation, handled outside the declarative
    # cluster loop above (see _predictability_active's docstring).
    pred_active = _predictability_active(prefs, active_csvs)
    pred_models, pred_metrics = (
        _predictability_config(prefs) if pred_active else ([], [])
    )
    pred_columns = _predictability_columns(pred_models, pred_metrics) if pred_active else ()
    if pred_active:
        required.update({"spacy_parse", "predictability"})
        live_csvs = live_csvs | {"features"}

    closed = _close_under_consumes(required)
    ordered = _topo_sort(closed)
    drops = _column_drops(active, live_csvs)
    configs = _stage_configs(prefs, ordered)
    return Plan(stages_to_run=ordered, output_csvs=live_csvs,
                column_drops=drops, stage_configs=configs,
                predictability_columns=pred_columns)
