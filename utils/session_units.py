"""Per-run column -> unit table writer.

Reads the Plan's stage_configs (pitch_units, pause_threshold, etc.) plus
hard-coded constants (Volume in dB, durations in seconds) to produce a
machine-readable sidecar listing units for every emittable column. Lets
downstream readers know units without parsing the run manifest JSON.
"""
from __future__ import annotations

from gui.planner.stages import Plan


# Static (csv, column) -> unit map. Pitch_* columns get their unit
# resolved dynamically from plan.stage_configs (pitch_units toggle).
_STATIC_UNITS: dict[tuple[str, str], str] = {
    # Word-level features CSV
    ("features", "Word_Duration"):    "seconds",
    ("features", "Word_Gap"):         "seconds",
    ("features", "Onset"):            "seconds",
    ("features", "End"):              "seconds",
    ("features", "Volume_Mean"):      "dB",
    ("features", "Volume_Min"):       "dB",
    ("features", "Volume_Max"):       "dB",
    ("features", "Volume_Std"):       "dB",
    ("features", "Volume_Delta"):     "dB",
    # Frequency: Zipf is log10(per-billion-words) with a +3 offset (cf. wordfreq).
    ("features", "Zipf_Freq"):        "Zipf (log10 per-billion + 3)",
    ("features", "Edu_Zipf_K2"):      "Zipf (log10 per-billion + 3)",
    ("features", "Edu_Zipf_3to5"):    "Zipf (log10 per-billion + 3)",
    ("features", "Edu_Zipf_6to8"):    "Zipf (log10 per-billion + 3)",
    ("features", "Edu_Zipf_9to12"):   "Zipf (log10 per-billion + 3)",
    ("features", "Edu_Zipf_Overall"): "Zipf (log10 per-billion + 3)",
    # Predictability: dynamic, roster-driven '<slug>_<Metric>' columns — see
    # _predictability_unit_rows below, resolved from plan.predictability_columns
    # rather than hard-coded here (the old fixed Surprisal/LLM_* column names
    # no longer exist post-unification; extractors/predictability.py).
    # Syllable sidecar
    ("syllables", "Onset"):           "seconds",
    ("syllables", "End"):             "seconds",
    ("syllables", "Duration"):        "seconds",
    ("syllables", "Volume_Mean"):     "dB",
    ("syllables", "Volume_Delta"):    "dB",
    # Phoneme sidecar
    ("phonemes",  "Onset"):           "seconds",
    ("phonemes",  "End"):             "seconds",
    ("phonemes",  "Duration"):        "seconds",
    # Doc-level summary
    ("documents_summary", "doc_volume_mean"):           "dB",
    ("documents_summary", "doc_volume_min"):            "dB",
    ("documents_summary", "doc_volume_max"):            "dB",
    ("documents_summary", "doc_volume_std"):            "dB",
    ("documents_summary", "doc_volume_delta_mean"):     "dB",
    ("documents_summary", "doc_hnr"):                   "dB",
    ("documents_summary", "doc_articulation_rate"):     "syllables/second",
    ("documents_summary", "doc_pause_count"):           "count",
    ("documents_summary", "doc_pause_mean_duration"):   "seconds",
    ("documents_summary", "doc_pause_total_duration"):  "seconds",
    ("documents_summary", "doc_pause_pct_silence"):     "percent",
    # Doc-level composite psych (Pennebaker LIWC-style scores, ~0-100)
    ("documents_summary", "psych_authentic"):           "Pennebaker score (~0-100)",
    ("documents_summary", "psych_analytic"):            "Pennebaker score (~0-100)",
    ("documents_summary", "psych_tone"):                "Pennebaker score (~0-100)",
    ("documents_summary", "psych_clout"):               "Pennebaker score (~0-100)",
}

# Pitch-bearing columns whose unit comes from plan.stage_configs.
_PITCH_COLUMNS: dict[str, tuple[str, ...]] = {
    "features":          ("Pitch_Mean", "Pitch_Min", "Pitch_Max",
                          "Pitch_Std", "Pitch_Delta"),
    "syllables":         ("Pitch_Mean", "Pitch_Delta"),
    "documents_summary": ("doc_pitch_mean", "doc_pitch_min", "doc_pitch_max",
                          "doc_pitch_std", "doc_pitch_delta_mean"),
}


# Predictability metric suffix -> unit. Roster columns are named
# '<slug>_<Metric>' (extractors/predictability.py's _METRIC_COLS caps); the
# slug prefix varies per selected model so only the suffix is matched.
_PREDICTABILITY_METRIC_UNITS: dict[str, str] = {
    "Surprisal": "bits",
    "Cloze":     "probability (0-1)",
    "TopMatch":  "boolean",
}


def _predictability_unit_rows(plan: Plan) -> list[dict]:
    """One row per active '<slug>_<Metric>' predictability column.

    Resolved dynamically from plan.predictability_columns (rather than a
    static table) because the column set depends on the user's selected
    model roster — see gui/planner/plan.py:_predictability_columns.
    """
    if "features" not in plan.output_csvs:
        return []
    rows: list[dict] = []
    for col in plan.predictability_columns:
        for suffix, unit in _PREDICTABILITY_METRIC_UNITS.items():
            if col.endswith(f"_{suffix}"):
                rows.append({"output_csv": "features", "column": col, "unit": unit})
                break
    return rows


def _resolve_pitch_units(plan: Plan) -> str:
    """Return the active pitch_units string ("Hz" or "semitones").

    Reads from praat_contours stage_config first (the upstream stage), then
    falls back to any prosody-emitter stage. Defaults to "semitones".
    """
    sc = plan.stage_configs
    for stage in ("praat_contours", "word_prosody", "syllable_prosody", "doc_prosody"):
        cfg = sc.get(stage, {})
        unit = cfg.get("pitch_units")
        if unit:
            return unit
    return "semitones"


def build_session_units(plan: Plan) -> list[dict]:
    """Return one row per (active CSV x emitted column) with its unit string."""
    rows: list[dict] = []
    pitch_units = _resolve_pitch_units(plan)

    for (csv, col), unit in _STATIC_UNITS.items():
        if csv in plan.output_csvs:
            rows.append({"output_csv": csv, "column": col, "unit": unit})

    for csv, cols in _PITCH_COLUMNS.items():
        if csv not in plan.output_csvs:
            continue
        for col in cols:
            rows.append({"output_csv": csv, "column": col, "unit": pitch_units})

    rows.extend(_predictability_unit_rows(plan))

    return rows
