"""Adapter: Plan → argparse.Namespace that main.py consumes.

Single source of truth for the no_* flag mappings. Refactor in v2 when
main.py learns to consume Plans directly.

See spec §7.
"""
from __future__ import annotations

import argparse

from gui.planner.stages import Plan
from extractors.acoustic_ts import MTRF_COMP

# Word-level output CSVs — used to filter which stages are relevant for the
# word-level namespace (so doc-level stage requirements don't bleed into
# word-level no_* flags like no_align).
_WORD_LEVEL_CSVS = frozenset({
    "features", "syllables", "phonemes", "sentences",
    "word2vec", "bert_word", "bert_sentence",
})

# Doc-level output CSVs — used to scope the doc namespace's word-stage flags
# to only what *doc-level* clusters require, so word-level stage requirements
# (e.g. mfa_align pulled in by word prosody / sub-word) don't bleed into the
# doc batch's no_* flags.
_DOC_LEVEL_CSVS = frozenset({"documents_summary"})

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
    "lexical_norms":  "no_lexical_norms",
    "word_prosody":   "no_prosody",
}


def _word_only_stages(plan: Plan, prefs: dict) -> frozenset[str]:
    """Return the stages required only by *active* word-level output CSVs.

    This prevents doc-level stage requirements (e.g. mfa_align pulled in by
    doc_prosody) from affecting word-level no_* flags like no_align.

    Cluster activation is delegated to plan._active_clusters — the single
    source of truth — so the same gates apply here as when the plan was
    built: available_when (env), master_toggle (user master switch),
    output_csv wanted, and any child toggle on. Re-deriving the predicate
    locally is what caused the 2026-06-24 bug: a cluster disabled via its
    master toggle (children left on) still flipped its no_* flag, so main.py
    ran an extractor the user had switched off. `plan` is accepted for call
    compatibility; activation reads straight from prefs.
    """
    from gui.planner.plan import _active_clusters, _active_output_csvs

    active_csvs = _active_output_csvs(prefs)
    required: set[str] = set()
    for c in _active_clusters(prefs, active_csvs):
        if c.output_csv not in _WORD_LEVEL_CSVS:
            continue
        required.update(c.requires_stages)
    return frozenset(required)


def _doc_only_stages(prefs: dict) -> frozenset[str]:
    """Return the stages required only by *active* doc-level clusters.

    Symmetric to _word_only_stages. Without this, the doc namespace derived
    its no_* flags from the whole plan, so mfa_align pulled in by a *word*
    cluster (word prosody, sub-word, sentence prosody, acoustic) made the doc
    batch run MFA even when no doc-level feature needed alignment.

    The set is closed under stage.consumes (like plan_from_prefs) so that a
    doc stage which consumes the base word_df pulls in spacy_parse (keeping
    no_parse correct), and doc_prosody — which consumes word_alignment /
    contours — pulls in mfa_align / praat_contours *only when it is active*.
    That closure is exactly what makes no_align False for the doc batch iff a
    doc-level feature (doc prosody / articulation rate) actually needs it.
    Because doc aggregate stages consume the base word_df (not its dotted
    columns like word_df.Surprisal), the underlying word extractors
    (freq / surprisal / sem / norms) are NOT pulled in here; they are
    (re)enabled independently by main.run_doc_level_batch from the doc output
    toggles.
    """
    from gui.planner.plan import (
        _active_clusters, _active_output_csvs, _close_under_consumes,
    )

    active_csvs = _active_output_csvs(prefs)
    required: set[str] = set()
    for c in _active_clusters(prefs, active_csvs):
        if c.output_csv not in _DOC_LEVEL_CSVS:
            continue
        required.update(c.requires_stages)
    return frozenset(_close_under_consumes(required))


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


def to_acoustic_namespace(prefs: dict) -> argparse.Namespace:
    """Namespace for main.run_acoustic_level_batch. Audio-only; no transcript
    or word/doc stage flags."""
    shared = prefs.get("shared", {})
    feats = prefs.get("features", {})
    acfg = prefs.get("acoustic_ts", {})
    ns = argparse.Namespace()
    ns.audio_folder     = shared.get("last_audio_folder", "")
    ns.output_folder    = shared.get("last_output_folder", "")
    ns.analyzer         = shared.get("analyzer_name", "")
    ns.audio_extensions = "wav,flac,mp3"
    ns.acoustic_ts = {
        "envelope":  bool(feats.get("feat_acoustic_ts_envelope", True)),
        "onset_env": bool(feats.get("feat_acoustic_ts_onset_env", True)),
        "logmel":    bool(feats.get("feat_acoustic_ts_logmel", True)),
        "mfcc":      bool(feats.get("feat_acoustic_ts_mfcc", False)),
        "env_method":   acfg.get("env_method", "rms"),
        "env_comp":     float(acfg.get("env_comp", MTRF_COMP)),
        "env_window":   float(acfg.get("env_window", 1.0)),
        "n_mels":       int(acfg.get("n_mels", 16)),
        "n_mfcc":       int(acfg.get("n_mfcc", 13)),
        "mel_preset":   acfg.get("mel_preset", "speech"),
        "analysis_rate": float(acfg.get("analysis_rate", 128)),
    }
    return ns


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

    # Word-stage no_* flags.
    # Each level's flags are scoped to the stages required by *that level's*
    # active clusters, so requirements don't bleed across levels:
    #   - word level: doc_prosody → mfa_align must not flip the word batch's
    #     no_align (that was the 2026-06-24 bug fixed by _word_only_stages);
    #   - doc level: a word-level prosody/sub-word feature → mfa_align must not
    #     flip the doc batch's no_align, or the doc batch runs MFA with no
    #     doc-level feature that needs it.
    # The underlying word extractors that feed doc aggregates are re-enabled
    # by main.run_doc_level_batch from the doc output toggles, so scoping the
    # doc flags here is safe.
    in_plan = set(plan.stages_to_run)
    if level == "word":
        word_stages = _word_only_stages(plan, prefs)
    elif level == "doc":
        word_stages = _doc_only_stages(prefs)
    else:
        word_stages = in_plan
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
    ns.no_sentences     = "sentences"    not in plan.output_csvs
    ns.no_word2vec      = "word2vec"     not in plan.output_csvs
    ns.no_bert_word     = "bert_word"    not in plan.output_csvs
    ns.no_bert_sentence = "bert_sentence" not in plan.output_csvs

    # Pass-throughs that main.py reads directly from args
    ns.feature_toggles = dict(features)
    ns.output_toggles  = dict(output)
    ns.edu_bands       = _enabled_ewfg_bands(features)
    ns.sem_dict        = prefs.get("semantic", {}).get("backend", "auto")
    ns.sem_categories  = _categories_for_level(prefs, level)
    ns.llm_model       = prefs.get("predictability", {}).get("llm_model", "") or None
    ns.edu_freq_lookup = None
    _norms = prefs.get("lexical_norms", {})
    ns.lexical_norms_paths = _norms.get("paths") or None
    ns.late_acquired_threshold = float(_norms.get("late_acquired_threshold", 10.0))

    # Prosody runtime params
    pros = prefs.get("prosody", {})
    ns.pitch_units     = pros.get("pitch_units", "semitones")
    ns.pitch_floor     = float(pros.get("pitch_floor_hz", 75.0))
    ns.pitch_ceiling   = float(pros.get("pitch_ceiling_hz", 500.0))
    ns.pause_threshold = float(pros.get("pause_threshold_s", 0.25))
    ns.pcenter_k       = float(pros.get("pcenter_k", 0.3))

    # Doc-level extra flags
    if level == "doc":
        # Doc prosody requires word-level prosody to also run (it consumes
        # Pitch_Delta / Volume_Delta that word_prosody populates).  Force
        # no_prosody=False whenever doc_prosody is in the plan even when none
        # of the 10 feat_prosody_* word-level toggles are on.
        if "doc_prosody" in in_plan:
            ns.no_prosody = False

        ns.no_doc_counts          = "doc_counts" not in in_plan
        # Each aggregate family has its own output toggle; the extractor
        # does not re-run — it aggregates whatever columns word_df already has.
        ns.no_doc_mean_zipf       = not output.get("doc_agg_zipf_freq", False)
        ns.no_doc_mean_edu_zipf   = not output.get("doc_agg_edu_zipf", False) or ns.edu_bands is None
        ns.no_doc_mean_surprisal  = not output.get("doc_agg_surprisal", False)
        ns.no_doc_mean_llm_cloze  = not output.get("doc_agg_llm", False)
        ns.no_doc_ttr             = "doc_lex_diversity" not in in_plan
        ns.no_doc_mtld            = "doc_lex_diversity" not in in_plan
        ns.no_doc_flesch_ease     = "doc_readability"   not in in_plan
        ns.no_doc_fk_grade        = "doc_readability"   not in in_plan
        ns.no_doc_lexical_norms   = not output.get("doc_agg_lexical_norms", False)
        ns.no_doc_psych_authentic = "doc_psych"         not in in_plan
        ns.no_doc_psych_analytic  = "doc_psych"         not in in_plan
        ns.no_doc_psych_tone      = "doc_psych"         not in in_plan
        ns.no_doc_psych_clout     = "doc_psych"         not in in_plan
        ns.no_doc_category_proportions = "doc_category_proportions" not in in_plan
        ns.no_doc_prosody         = "doc_prosody"       not in in_plan

    return ns
