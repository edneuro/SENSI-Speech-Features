"""emit.to_namespace adapter tests. Given a Plan + prefs, verify the
resulting Namespace has the expected no_* flags and pass-through fields."""
import argparse

from gui.planner.plan import plan_from_prefs
from gui.planner.emit import to_namespace


# v7 master toggle defaults (all True) — included in every test prefs dict
# so individual tests that don't care about master gates still see clusters
# as active (matching real v7 default behaviour).
_MASTER_DEFAULTS_FEATURES = {
    "master_word_prosody":       True,
    "master_psych_composite":    True,
    "master_syllable_intervals": True,
    "master_syllable_labels":    True,
    "master_syllable_prosody":   True,
    "master_phoneme_intervals":  True,
    "master_structural":         True,
    "master_lexling":            True,
    "master_frequency":          True,
    "master_predictability":     True,
    "master_acoustic":           True,
}
_MASTER_DEFAULTS_OUTPUT = {
    "master_doc_prosody":           True,
    "master_doc_articulation_rate": True,
    "master_doc_counts":            True,
    "doc_agg_zipf_freq":            True,
    "doc_agg_edu_zipf":             True,
    "doc_agg_surprisal":            True,
    "doc_agg_llm":                  True,
}


def _prefs(output: dict, features: dict, **kw) -> dict:
    merged_features = {**_MASTER_DEFAULTS_FEATURES, **features}
    merged_output   = {**_MASTER_DEFAULTS_OUTPUT,   **output}
    return {
        "output":   merged_output,
        "features": merged_features,
        "semantic": kw.get("semantic", {"backend": "empath",
                                         "categories_empath": []}),
        "prosody":  kw.get("prosody", {"pitch_units": "semitones",
                                        "pitch_floor_hz": 75.0,
                                        "pitch_ceiling_hz": 500.0,
                                        "pause_threshold_s": 0.25}),
        "predictability": kw.get("predictability", {"llm_model": ""}),
        "shared":   kw.get("shared", {"last_transcripts_folder": "",
                                       "last_audio_folder": "",
                                       "last_output_folder": "",
                                       "analyzer_name": ""}),
    }


def test_word_level_namespace_basic():
    prefs = _prefs(
        output={"word_level_csv": True, "syllables_csv": False,
                "phonemes_csv": False, "document_summary": False},
        features={"freq_subtlex": True},
    )
    plan = plan_from_prefs(prefs)
    ns = to_namespace(plan, prefs, level="word")
    assert ns.no_parse is False
    assert ns.no_freq is False
    assert ns.no_align is True
    assert ns.no_surprisal is True
    assert ns.no_llm_cloze is True
    assert ns.no_sem_domain is True
    assert ns.no_word_csv is False  # word CSV wanted
    assert ns.no_syllables is True
    assert ns.no_phonemes is True


def test_sidecars_only_namespace():
    """User config from the recent bug report: word CSV off, sidecars on."""
    prefs = _prefs(
        output={"word_level_csv": False, "syllables_csv": True,
                "phonemes_csv": True, "document_summary": False},
        features={"feat_syllable_onset": True, "feat_syllable_label": True,
                  "feat_phoneme_label": True},
    )
    plan = plan_from_prefs(prefs)
    ns = to_namespace(plan, prefs, level="word")
    assert ns.no_parse is False, "subword needs parse"
    assert ns.no_align is False, "subword needs MFA"
    assert ns.no_word_csv is True, "user did NOT request features CSV"
    assert ns.no_syllables is False
    assert ns.no_phonemes is False


def test_syllable_prosody_forces_prosody_flag_off():
    """no_prosody must be False so main.py runs praat extract + word prosody
    (contours are then shared with syllable_prosody)."""
    prefs = _prefs(
        output={"word_level_csv": False, "syllables_csv": True,
                "phonemes_csv": False, "document_summary": False},
        features={"feat_syllable_pitch_mean": True},
    )
    plan = plan_from_prefs(prefs)
    ns = to_namespace(plan, prefs, level="word")
    assert ns.no_prosody is False
    assert ns.no_align is False


def test_doc_namespace_separate_no_flags():
    prefs = _prefs(
        output={"word_level_csv": False, "syllables_csv": False,
                "phonemes_csv": False, "document_summary": True,
                "doc_agg_zipf_freq": True, "doc_agg_edu_zipf": False,
                "doc_agg_surprisal": False, "doc_agg_llm": False,
                "doc_lex_diversity": True, "doc_readability": False,
                "doc_category_proportions": False,
                "out_count_tokens": True, "out_count_content_tokens": False,
                "out_count_sentences": False, "out_count_paragraphs": False},
        features={"freq_subtlex": True, "pred_surprisal_gpt2": False},
    )
    plan = plan_from_prefs(prefs)
    ns = to_namespace(plan, prefs, level="doc")
    assert ns.no_doc_counts is False
    assert ns.no_doc_mean_zipf is False     # doc_agg_zipf_freq True
    assert ns.no_doc_mean_surprisal is True  # doc_agg_surprisal False
    assert ns.no_doc_ttr is False
    assert ns.no_doc_flesch_ease is True
    assert ns.no_doc_category_proportions is True


def test_feature_toggles_pass_through_verbatim():
    """main.py's _word_columns_to_drop reads feature_toggles directly,
    so the planner must pass it through unchanged.

    The full prefs features dict (including v7 master defaults) is passed
    through; we assert that the caller-supplied keys are present with the
    correct values and that the namespace dict is a superset of features_in.
    """
    features_in = {"freq_subtlex": True, "feat_struct_doc_id": False,
                   "feat_acoustic_onset": False}
    prefs = _prefs(
        output={"word_level_csv": True, "syllables_csv": False,
                "phonemes_csv": False, "document_summary": False},
        features=features_in,
    )
    plan = plan_from_prefs(prefs)
    ns = to_namespace(plan, prefs, level="word")
    # feature_toggles is the full prefs["features"] dict (a superset of
    # features_in because _prefs merges v7 master defaults).
    for key, val in features_in.items():
        assert ns.feature_toggles.get(key) == val, \
            f"feature_toggles[{key!r}] = {ns.feature_toggles.get(key)!r}, expected {val!r}"


def test_prosody_config_propagates_to_namespace():
    prefs = _prefs(
        output={"word_level_csv": True, "syllables_csv": False,
                "phonemes_csv": False, "document_summary": False},
        features={"feat_prosody_pitch_mean": True},
        prosody={"pitch_units": "Hz", "pitch_floor_hz": 60.0,
                 "pitch_ceiling_hz": 700.0, "pause_threshold_s": 0.4},
    )
    plan = plan_from_prefs(prefs)
    ns = to_namespace(plan, prefs, level="word")
    assert ns.pitch_units == "Hz"
    assert ns.pitch_floor == 60.0
    assert ns.pitch_ceiling == 700.0
    assert ns.pause_threshold == 0.4


def test_edu_bands_string_for_runtime():
    prefs = _prefs(
        output={"word_level_csv": True, "syllables_csv": False,
                "phonemes_csv": False, "document_summary": False},
        features={"freq_ewfg_k2": True, "freq_ewfg_overall": True},
    )
    plan = plan_from_prefs(prefs)
    ns = to_namespace(plan, prefs, level="word")
    # main.py.run() does args.edu_bands.split(',') — must be a comma-joined string
    assert isinstance(ns.edu_bands, str) or ns.edu_bands is None
    if ns.edu_bands:
        assert "k2" in ns.edu_bands
        assert "overall" in ns.edu_bands


def test_individual_agg_toggles_control_no_doc_mean_flags():
    """Each doc_agg_* output toggle independently controls the corresponding
    no_doc_mean_* flag — they are no longer linked to word-level feature toggles."""
    prefs = _prefs(
        output={"word_level_csv": False, "document_summary": True,
                "syllables_csv": False, "phonemes_csv": False,
                "doc_agg_zipf_freq": True,
                "doc_agg_edu_zipf": False,
                "doc_agg_surprisal": False,
                "doc_agg_llm": True},
        features={"freq_subtlex": True, "pred_surprisal_gpt2": True},
    )
    plan = plan_from_prefs(prefs)
    ns = to_namespace(plan, prefs, level="doc")
    assert ns.no_doc_mean_zipf is False       # doc_agg_zipf_freq is True
    assert ns.no_doc_mean_surprisal is True   # doc_agg_surprisal is False
    assert ns.no_doc_mean_llm_cloze is False  # doc_agg_llm is True


def test_sem_categories_unified_key_used_for_both_levels():
    """v7 unified: word and doc levels both read semantic.categories_<backend>.
    The old split word_/doc_categories_* keys no longer exist in prefs; both
    levels must return the same value from the shared key."""
    prefs = _prefs(
        output={"word_level_csv": True, "syllables_csv": False,
                "phonemes_csv": False, "document_summary": True},
        features={"sem_indicators": True},
        semantic={"backend": "empath",
                  "categories_empath": ["positive_emotion", "work", "family"]},
    )
    plan = plan_from_prefs(prefs)
    ns_word = to_namespace(plan, prefs, level="word")
    ns_doc  = to_namespace(plan, prefs, level="doc")
    # Both levels must return the same unified selection.
    assert set(ns_word.sem_categories.split(",")) == {"positive_emotion", "work", "family"}
    assert ns_doc.sem_categories == ns_word.sem_categories


def test_doc_level_stages_dont_bleed_into_word_namespace():
    """Doc-prosody pulls mfa_align/praat_contours into plan.stages_to_run.
    The word namespace must NOT then set no_align=False unless a word-level
    cluster also wants alignment."""
    prefs = _prefs(
        output={"word_level_csv": True, "syllables_csv": False,
                "phonemes_csv": False, "document_summary": True},
        features={"freq_subtlex": True,                  # word: just SUBTLEX
                  "feat_doc_prosody_pitch_mean": True},  # doc: prosody
    )
    plan = plan_from_prefs(prefs)
    assert "mfa_align" in plan.stages_to_run, "doc_prosody must pull MFA"
    ns_word = to_namespace(plan, prefs, level="word")
    ns_doc  = to_namespace(plan, prefs, level="doc")
    assert ns_word.no_align is True, \
        "word namespace must not see mfa_align as active — no word-level toggle needs it"
    assert ns_word.no_prosody is True
    assert ns_doc.no_align is False
    assert ns_doc.no_prosody is False  # doc_prosody forces word_prosody to run too
