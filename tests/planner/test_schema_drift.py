"""Schema-drift guard. Every prefs key that controls feature behavior
must be referenced by some FeatureCluster's toggle_keys — otherwise
the planner silently ignores it."""
from gui.planner.registry import CLUSTERS
from utils.preferences import DEFAULT_PREFS

# Track the *active* prefs schema (not a pinned version) so new batches that
# add keys are validated against the registry automatically.
_FEATURES_DEFAULT_V10 = DEFAULT_PREFS["features"]
_OUTPUT_DEFAULT_V10 = DEFAULT_PREFS["output"]


# Keys that are intentionally NOT in any cluster's toggle_keys (config
# knobs routed via the master_toggle field instead).
_INTENTIONALLY_UNROUTED_FEATURES: set[str] = {
    # Cluster master switches — routed via FeatureCluster.master_toggle,
    # NOT via toggle_keys. Checked separately by
    # test_every_master_toggle_key_exists_in_v10_prefs_schema.
    "master_word_prosody",
    "master_psych_composite",
    "master_syllable_intervals",
    "master_syllable_labels",
    "master_syllable_prosody",
    "master_phoneme_intervals",
    # master_syllable_pcenter kept in prefs schema for migration compat but is
    # no longer wired to a ClusterCard master toggle (P-center has only one
    # toggle so the master is redundant; see gui/section_subword_features.py).
    "master_syllable_pcenter",
    # Word-level card-group masters
    "master_structural",
    "master_lexling",
    "master_frequency",
    "master_predictability",
    "master_acoustic",
    "master_lexical_norms",     # routes via master_toggle field
    "master_sentence_timing",   # routes via master_toggle field
    "master_sentence_prosody",  # routes via master_toggle field
    "master_word2vec",          # routes via master_toggle field
    "master_bert_word",         # routes via master_toggle field
    "master_bert_sentence",     # routes via master_toggle field
    # v13 acoustic level: master_acoustic_ts is vestigial — the acoustic level is
    # gated solely by the acoustic_ts_mat output key (Run tab), not a cluster
    # master. Kept in the schema for migration compat (cf. master_syllable_pcenter).
    "master_acoustic_ts",
    # v16 phoneme surprisal — routes via FeatureCluster.master_toggle
    "master_phoneme_surprisal",
}

# Output keys NOT routed via clusters (the master output toggles that
# select which CSV gets written are handled by _OUTPUT_CSV_MAP in
# plan.py, not by clusters).
_OUTPUT_KEYS_VIA_CSV_MAP: set[str] = {
    "word_level_csv", "syllables_csv", "phonemes_csv", "document_summary",
    "sentences_csv",       # routed via _OUTPUT_CSV_MAP
    "word2vec_mat",        # routed via _OUTPUT_CSV_MAP
    "bert_word_mat",       # routed via _OUTPUT_CSV_MAP
    "bert_sentence_mat",   # routed via _OUTPUT_CSV_MAP
    # Cluster master switches in the output section — routed via
    # FeatureCluster.master_toggle, not toggle_keys.
    "master_doc_prosody",
    # Vestigial: articulation rate is now gated by master_doc_prosody (it is a
    # child of the Document-prosody card). Kept in the schema for migration
    # compat only (cf. master_acoustic_ts).
    "master_doc_articulation_rate",
    "master_doc_counts",
    "acoustic_ts_mat",       # routed via _OUTPUT_CSV_MAP
    "phoneme_surprisal_mat",  # routed via _OUTPUT_CSV_MAP
}


def _all_cluster_toggle_keys() -> set[str]:
    return {k for c in CLUSTERS for k in c.toggle_keys}


def test_every_feature_pref_key_is_in_some_cluster():
    routed = _all_cluster_toggle_keys()
    feature_keys = set(_FEATURES_DEFAULT_V10.keys())
    unrouted = feature_keys - routed - _INTENTIONALLY_UNROUTED_FEATURES
    assert not unrouted, (
        f"Feature pref keys not referenced by any cluster (the planner will "
        f"silently ignore them when deciding what runs): {sorted(unrouted)}"
    )


def test_every_output_pref_key_is_either_master_csv_or_in_some_cluster():
    routed = _all_cluster_toggle_keys()
    output_keys = set(_OUTPUT_DEFAULT_V10.keys())
    unrouted = output_keys - routed - _OUTPUT_KEYS_VIA_CSV_MAP
    assert not unrouted, (
        f"Output pref keys not referenced by any cluster or master CSV map "
        f"(planner ignores them): {sorted(unrouted)}"
    )


def test_no_cluster_references_nonexistent_pref_key():
    """The reverse direction: cluster toggle_keys must exist in the prefs schema."""
    all_pref_keys = set(_FEATURES_DEFAULT_V10.keys()) | set(_OUTPUT_DEFAULT_V10.keys())
    routed = _all_cluster_toggle_keys()
    phantom = routed - all_pref_keys
    assert not phantom, (
        f"Clusters reference pref keys not in the v10 schema "
        f"(typo or stale registry): {sorted(phantom)}"
    )


def test_every_master_toggle_key_exists_in_v10_prefs_schema():
    """Each cluster's master_toggle (when present) must be a real key in
    the v9 prefs schema (either features or output). Catches typos and
    forgotten schema additions."""
    all_pref_keys = set(_FEATURES_DEFAULT_V10) | set(_OUTPUT_DEFAULT_V10)
    masters = {c.master_toggle for c in CLUSTERS if c.master_toggle}
    missing = masters - all_pref_keys
    assert not missing, (
        f"Cluster master_toggle(s) not in v10 schema "
        f"(typo or missing v9 default): {sorted(missing)}"
    )


def test_acoustic_level_routed_when_output_on():
    # The acoustic level is gated by the acoustic_ts_mat output key (Run-tab
    # toggle) plus a feature toggle. Under opt-in defaults the features are off,
    # so enable envelope explicitly.
    from gui.planner.plan import plan_from_prefs
    from utils.preferences import DEFAULT_PREFS
    import copy
    prefs = copy.deepcopy(DEFAULT_PREFS)
    prefs["output"]["acoustic_ts_mat"] = True
    prefs["features"]["feat_acoustic_ts_envelope"] = True
    plan = plan_from_prefs(prefs)
    assert "acoustic" in plan.output_csvs
    assert "acoustic_ts" in plan.stages_to_run


def test_acoustic_level_off_by_default():
    from gui.planner.plan import plan_from_prefs
    from utils.preferences import DEFAULT_PREFS
    plan = plan_from_prefs(DEFAULT_PREFS)
    assert "acoustic" not in plan.output_csvs   # acoustic_ts_mat defaults OFF


def test_acoustic_level_off_when_all_features_off():
    # Output key on but every acoustic feature off -> cluster inactive -> no file.
    from gui.planner.plan import plan_from_prefs
    from utils.preferences import DEFAULT_PREFS
    import copy
    prefs = copy.deepcopy(DEFAULT_PREFS)
    prefs["output"]["acoustic_ts_mat"] = True
    for k in ("feat_acoustic_ts_envelope", "feat_acoustic_ts_onset_env",
              "feat_acoustic_ts_logmel", "feat_acoustic_ts_mfcc"):
        prefs["features"][k] = False
    plan = plan_from_prefs(prefs)
    assert "acoustic" not in plan.output_csvs
