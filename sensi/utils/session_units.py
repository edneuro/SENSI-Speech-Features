"""Per-run column -> unit table writer.

Reads the Plan's stage_configs (pitch_units, pause_threshold, etc.) plus
hard-coded constants (Volume in dB, durations in seconds) to produce a
machine-readable sidecar listing units for every emittable column. Lets
downstream readers know units without parsing the run manifest JSON.
"""
from __future__ import annotations

from sensi.gui.planner.stages import Plan


_ZIPF = "Zipf (log10 frequency per billion words)"

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
    # Frequency: Zipf = log10(frequency per billion words)
    #              = log10(per-million) + 3 (cf. wordfreq).
    ("features", "Zipf_Freq"):        _ZIPF,
    ("features", "Edu_Zipf_K2"):      _ZIPF,
    ("features", "Edu_Zipf_3to5"):    _ZIPF,
    ("features", "Edu_Zipf_6to8"):    _ZIPF,
    ("features", "Edu_Zipf_9to12"):   _ZIPF,
    ("features", "Edu_Zipf_Overall"): _ZIPF,
    # Lexical norms (extractors/lexical_norms.py): rating-scale units.
    ("features", "AoA_Mean"):         "years (rated age of acquisition)",
    ("features", "AoA_SD"):           "years",
    ("features", "Concreteness"):     "rating (1-5)",
    ("features", "Imageability"):     "rating (100-700; MRC 1-7 x 100)",
    ("features", "Familiarity"):      "rating (100-700; MRC 1-7 x 100)",
    ("features", "Valence"):          "rating (1-9)",
    ("features", "Arousal"):          "rating (1-9)",
    ("features", "Dominance"):        "rating (1-9)",
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
    ("syllables", "P_Center"):        "seconds",
    # Phoneme sidecar
    ("phonemes",  "Onset"):           "seconds",
    ("phonemes",  "End"):             "seconds",
    ("phonemes",  "Duration"):        "seconds",
    ("phonemes",  "phoneme_surprisal"): "bits",
    ("phonemes",  "cohort_entropy"):  "bits",
    # Sentence sidecar (extractors/sentence_prosody.py)
    ("sentences", "Onset"):           "seconds",
    ("sentences", "End"):             "seconds",
    ("sentences", "Volume_Mean"):     "dB",
    ("sentences", "Volume_Min"):      "dB",
    ("sentences", "Volume_Max"):      "dB",
    ("sentences", "Volume_Std"):      "dB",
    ("sentences", "Volume_Delta"):    "dB",
    ("sentences", "HNR"):             "dB",
    # Doc-level summary
    ("documents_summary", "n_tokens"):                  "count",
    ("documents_summary", "n_content_tokens"):          "count",
    ("documents_summary", "n_sentences"):               "count",
    ("documents_summary", "n_paragraphs"):              "count",
    ("documents_summary", "mean_zipf_freq"):            _ZIPF,
    ("documents_summary", "mean_edu_zipf_k2"):          _ZIPF,
    ("documents_summary", "mean_edu_zipf_3to5"):        _ZIPF,
    ("documents_summary", "mean_edu_zipf_6to8"):        _ZIPF,
    ("documents_summary", "mean_edu_zipf_9to12"):       _ZIPF,
    ("documents_summary", "mean_edu_zipf_overall"):     _ZIPF,
    ("documents_summary", "ttr"):                       "ratio (0-1)",
    ("documents_summary", "mtld"):                      "tokens (mean factor length)",
    ("documents_summary", "flesch_reading_ease"):       "Flesch score (~0-100)",
    ("documents_summary", "flesch_kincaid_grade"):      "US grade level",
    # Doc-level lexical-norm aggregates (over content words; utils/doc_summary.py)
    ("documents_summary", "mean_aoa"):                  "years",
    ("documents_summary", "median_aoa"):                "years",
    ("documents_summary", "max_aoa"):                   "years",
    ("documents_summary", "sd_aoa"):                    "years",
    ("documents_summary", "prop_late_acquired"):        "proportion (0-1)",
    ("documents_summary", "mean_concreteness"):         "rating (1-5)",
    ("documents_summary", "mean_imageability"):         "rating (100-700)",
    ("documents_summary", "mean_familiarity"):          "rating (100-700)",
    ("documents_summary", "mean_valence"):              "rating (1-9)",
    ("documents_summary", "mean_arousal"):              "rating (1-9)",
    ("documents_summary", "mean_dominance"):            "rating (1-9)",
    ("documents_summary", "aoa_coverage"):              "proportion (0-1)",
    ("documents_summary", "concreteness_coverage"):     "proportion (0-1)",
    ("documents_summary", "imageability_coverage"):     "proportion (0-1)",
    ("documents_summary", "familiarity_coverage"):      "proportion (0-1)",
    ("documents_summary", "valence_coverage"):          "proportion (0-1)",
    ("documents_summary", "arousal_coverage"):          "proportion (0-1)",
    ("documents_summary", "dominance_coverage"):        "proportion (0-1)",
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
    "sentences":         ("Pitch_Mean", "Pitch_Min", "Pitch_Max",
                          "Pitch_Std", "Pitch_Delta"),
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

# Doc-level means of the same columns: 'mean_<slug>_<metric lower>'
# (utils/doc_summary.py). A mean of a boolean is a proportion.
_DOC_PREDICTABILITY_METRIC_UNITS: dict[str, str] = {
    "Surprisal": "bits",
    "Cloze":     "probability (0-1)",
    "TopMatch":  "proportion (0-1)",
}


def _predictability_unit_rows(plan: Plan) -> list[dict]:
    """One row per active '<slug>_<Metric>' predictability column, plus its
    doc-level 'mean_<slug>_<metric>' aggregate when the document CSV is on.

    Resolved dynamically from plan.predictability_columns (rather than a
    static table) because the column set depends on the user's selected
    model roster — see gui/planner/plan.py:_predictability_columns.
    """
    rows: list[dict] = []
    for col in plan.predictability_columns:
        for suffix, unit in _PREDICTABILITY_METRIC_UNITS.items():
            if not col.endswith(f"_{suffix}"):
                continue
            if "features" in plan.output_csvs:
                rows.append({"output_csv": "features", "column": col, "unit": unit})
            if "documents_summary" in plan.output_csvs:
                slug = col[: -len(suffix) - 1]
                rows.append({"output_csv": "documents_summary",
                             "column": f"mean_{slug}_{suffix.lower()}",
                             "unit": _DOC_PREDICTABILITY_METRIC_UNITS[suffix]})
            break
    return rows


def _resolve_pitch_units(plan: Plan) -> str:
    """Return the active pitch_units string ("Hz" or "semitones").

    Reads from praat_contours stage_config first (the upstream stage), then
    falls back to any prosody-emitter stage. Defaults to "semitones".
    """
    sc = plan.stage_configs
    for stage in ("praat_contours", "word_prosody", "syllable_prosody",
                  "sentence_prosody", "doc_prosody"):
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
