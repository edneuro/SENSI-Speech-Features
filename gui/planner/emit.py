"""Adapter: Plan → argparse.Namespace that main.py consumes.

Single source of truth for the no_* flag mappings. Refactor in v2 when
main.py learns to consume Plans directly.

See spec §7.
"""
from __future__ import annotations

import argparse

from gui.planner.stages import Plan

# Word-level output CSVs — used to filter which stages are relevant for the
# word-level namespace (so doc-level stage requirements don't bleed into
# word-level no_* flags like no_align).
_WORD_LEVEL_CSVS = frozenset({"features", "syllables", "phonemes"})

_WORD_STAGE_FLAGS = {
    # stage name -> attr name on the namespace (no_X is True iff stage NOT in plan)
    "spacy_parse":    "no_parse",
    "mfa_align":      "no_align",
    "praat_contours": None,    # implicit; no flag in main.py
    "word_freq":      "no_freq",
    "ewfg_freq":      "no_edu_freq",
    "gpt2_surprisal": "no_surprisal",
    "llm_predict":    "no_llm_cloze",
    "sem_domain":     "no_sem_domain",
    "word_prosody":   "no_prosody",
}


def _word_only_stages(plan: Plan, prefs: dict) -> frozenset[str]:
    """Return the stages required only by word-level output CSVs.

    This prevents doc-level stage requirements (e.g. mfa_align pulled in by
    doc_prosody) from affecting word-level no_* flags like no_align.

    A word-level cluster is "active" iff its output_csv is in the plan's
    active output CSVs AND at least one of its toggle_keys is True in prefs.
    Only active clusters' required stages contribute to the word-level view.
    """
    from gui.planner.registry import CLUSTERS
    from gui.planner.plan import OUTPUT_KEYED_TOGGLES

    def _toggle_value(key: str) -> bool:
        if key in OUTPUT_KEYED_TOGGLES:
            return bool(prefs.get("output", {}).get(key, False))
        return bool(prefs.get("features", {}).get(key, False))

    required: set[str] = set()
    for c in CLUSTERS:
        if c.output_csv not in _WORD_LEVEL_CSVS:
            continue
        if c.output_csv not in plan.output_csvs:
            continue
        if not any(_toggle_value(k) for k in c.toggle_keys):
            continue
        required.update(c.requires_stages)
    return frozenset(required)


def _enabled_ewfg_bands(features: dict) -> str | None:
    bands = [
        suf for suf, key in [
            ("k2", "freq_ewfg_k2"), ("3to5", "freq_ewfg_3to5"),
            ("6to8", "freq_ewfg_6to8"), ("9to12", "freq_ewfg_9to12"),
            ("overall", "freq_ewfg_overall"),
        ] if features.get(key, False)
    ]
    return ",".join(bands) if bands else None


def _categories_for_level(prefs: dict, level: str) -> str | None:
    """Return the comma-joined active categories for `level`.

    v7 unified: word and doc levels share `semantic.categories_<backend>`.
    `level` is unused inside; kept in the signature so callers can pass it
    transparently if a future divergence is ever needed.
    """
    backend = prefs.get("semantic", {}).get("backend", "auto")
    if backend == "auto":
        # Defer to runtime detection; emit None so main.run() falls back.
        return None
    cats = prefs.get("semantic", {}).get(f"categories_{backend}", [])
    return ",".join(cats) if cats else None


def to_namespace(plan: Plan, prefs: dict, level: str) -> argparse.Namespace:
    """Build the argparse.Namespace for main.run_word_level_batch (level='word')
    or main.run_doc_level_batch (level='doc')."""
    features = prefs.get("features", {})
    output = prefs.get("output", {})
    shared = prefs.get("shared", {})

    ns = argparse.Namespace()

    # I/O folders (passthrough from prefs)
    ns.transcripts_folder = shared.get("last_transcripts_folder", "")
    ns.audio_folder       = shared.get("last_audio_folder", "")
    ns.output_folder      = shared.get("last_output_folder", "")
    ns.analyzer           = shared.get("analyzer_name", "")
    ns.text_extensions    = "txt,tsv,md"
    ns.audio_extensions   = "wav,flac,mp3"
    ns.save_prefs         = False

    # Word-stage no_* flags
    # Use the full plan set for doc-level (doc stages can pull in mfa_align etc);
    # use a word-only subset for word-level so doc-level requirements don't bleed
    # into word-level no_* flags (e.g. doc_prosody → mfa_align → no_align=False
    # should not affect the word batch when user has no acoustic/prosody features on).
    in_plan = set(plan.stages_to_run)
    word_stages = _word_only_stages(plan, prefs) if level == "word" else in_plan
    for stage_name, attr in _WORD_STAGE_FLAGS.items():
        if attr is None:
            continue
        setattr(ns, attr, stage_name not in word_stages)

    # If syllable_prosody is active, praat contours are already being extracted.
    # no_prosody must be False so main.py also runs word_prosody (it shares the
    # contour data that syllable_prosody needs anyway).
    if "syllable_prosody" in word_stages:
        ns.no_prosody = False

    # Sidecar / output flags
    ns.no_word_csv  = "features" not in plan.output_csvs
    ns.no_syllables = "syllables" not in plan.output_csvs
    ns.no_phonemes  = "phonemes" not in plan.output_csvs

    # Pass-throughs that main.py reads directly from args
    ns.feature_toggles = dict(features)
    ns.output_toggles  = dict(output)
    ns.edu_bands       = _enabled_ewfg_bands(features)
    ns.sem_dict        = prefs.get("semantic", {}).get("backend", "auto")
    ns.sem_categories  = _categories_for_level(prefs, level)
    ns.llm_model       = prefs.get("predictability", {}).get("llm_model", "") or None
    ns.edu_freq_lookup = None

    # Prosody runtime params
    pros = prefs.get("prosody", {})
    ns.pitch_units     = pros.get("pitch_units", "semitones")
    ns.pitch_floor     = float(pros.get("pitch_floor_hz", 75.0))
    ns.pitch_ceiling   = float(pros.get("pitch_ceiling_hz", 500.0))
    ns.pause_threshold = float(pros.get("pause_threshold_s", 0.25))

    # Doc-level extra flags
    if level == "doc":
        # Doc prosody requires word-level prosody to also run (it consumes
        # Pitch_Delta / Volume_Delta that word_prosody populates).  Force
        # no_prosody=False whenever doc_prosody is in the plan even when none
        # of the 10 feat_prosody_* word-level toggles are on.
        if "doc_prosody" in in_plan:
            ns.no_prosody = False

        ns.no_doc_counts          = "doc_counts" not in in_plan
        ns.no_doc_mean_zipf       = ("doc_aggregates" not in in_plan
                                      or not features.get("freq_subtlex", False))
        ns.no_doc_mean_edu_zipf   = ("doc_aggregates" not in in_plan
                                      or ns.edu_bands is None)
        ns.no_doc_mean_surprisal  = ("doc_aggregates" not in in_plan
                                      or not features.get("pred_surprisal_gpt2", False))
        ns.no_doc_mean_llm_cloze  = ("doc_aggregates" not in in_plan or not (
            features.get("pred_surprisal_llm", False)
            or features.get("pred_cloze", False)
            or features.get("pred_topmatch", False)))
        ns.no_doc_ttr             = "doc_lex_diversity" not in in_plan
        ns.no_doc_mtld            = "doc_lex_diversity" not in in_plan
        ns.no_doc_flesch_ease     = "doc_readability"   not in in_plan
        ns.no_doc_fk_grade        = "doc_readability"   not in in_plan
        ns.no_doc_liwc_authentic  = "doc_liwc"          not in in_plan
        ns.no_doc_liwc_analytic   = "doc_liwc"          not in in_plan
        ns.no_doc_liwc_tone       = "doc_liwc"          not in in_plan
        ns.no_doc_liwc_clout      = "doc_liwc"          not in in_plan
        ns.no_doc_category_proportions = "doc_category_proportions" not in in_plan
        ns.no_doc_prosody         = "doc_prosody"       not in in_plan

    return ns
