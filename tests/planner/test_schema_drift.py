"""Schema-drift guard. Every prefs key that controls feature behavior
must be referenced by some FeatureCluster's toggle_keys — otherwise
the planner silently ignores it."""
from gui.planner.registry import CLUSTERS
from utils.preferences import _FEATURES_DEFAULT_V7, _OUTPUT_DEFAULT_V7


# Keys that are intentionally NOT in any cluster's toggle_keys (config
# knobs routed via the master_toggle field instead).
_INTENTIONALLY_UNROUTED_FEATURES: set[str] = {
    # Cluster master switches — routed via FeatureCluster.master_toggle,
    # NOT via toggle_keys. Checked separately by
    # test_every_master_toggle_key_exists_in_v7_prefs_schema.
    "master_word_prosody",
    "master_psych_composite",
    "master_syllable_intervals",
    "master_syllable_labels",
    "master_syllable_prosody",
    "master_phoneme_intervals",
}

# Output keys NOT routed via clusters (the master output toggles that
# select which CSV gets written are handled by _OUTPUT_CSV_MAP in
# plan.py, not by clusters).
_OUTPUT_KEYS_VIA_CSV_MAP: set[str] = {
    "word_level_csv", "syllables_csv", "phonemes_csv", "document_summary",
    # Cluster master switches in the output section — routed via
    # FeatureCluster.master_toggle, not toggle_keys.
    "master_doc_prosody",
    "master_doc_articulation_rate",
}


def _all_cluster_toggle_keys() -> set[str]:
    return {k for c in CLUSTERS for k in c.toggle_keys}


def test_every_feature_pref_key_is_in_some_cluster():
    routed = _all_cluster_toggle_keys()
    feature_keys = set(_FEATURES_DEFAULT_V7.keys())
    unrouted = feature_keys - routed - _INTENTIONALLY_UNROUTED_FEATURES
    assert not unrouted, (
        f"Feature pref keys not referenced by any cluster (the planner will "
        f"silently ignore them when deciding what runs): {sorted(unrouted)}"
    )


def test_every_output_pref_key_is_either_master_csv_or_in_some_cluster():
    routed = _all_cluster_toggle_keys()
    output_keys = set(_OUTPUT_DEFAULT_V7.keys())
    unrouted = output_keys - routed - _OUTPUT_KEYS_VIA_CSV_MAP
    assert not unrouted, (
        f"Output pref keys not referenced by any cluster or master CSV map "
        f"(planner ignores them): {sorted(unrouted)}"
    )


def test_no_cluster_references_nonexistent_pref_key():
    """The reverse direction: cluster toggle_keys must exist in the prefs schema."""
    all_pref_keys = set(_FEATURES_DEFAULT_V7.keys()) | set(_OUTPUT_DEFAULT_V7.keys())
    routed = _all_cluster_toggle_keys()
    phantom = routed - all_pref_keys
    assert not phantom, (
        f"Clusters reference pref keys not in the v7 schema "
        f"(typo or stale registry): {sorted(phantom)}"
    )


def test_every_master_toggle_key_exists_in_v7_prefs_schema():
    """Each cluster's master_toggle (when present) must be a real key in
    the v7 prefs schema (either features or output). Catches typos and
    forgotten schema additions."""
    all_pref_keys = set(_FEATURES_DEFAULT_V7) | set(_OUTPUT_DEFAULT_V7)
    masters = {c.master_toggle for c in CLUSTERS if c.master_toggle}
    missing = masters - all_pref_keys
    assert not missing, (
        f"Cluster master_toggle(s) not in v7 schema "
        f"(typo or missing v7 default): {sorted(missing)}"
    )
