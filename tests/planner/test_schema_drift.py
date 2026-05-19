"""Schema-drift guard. Every prefs key that controls feature behavior
must be referenced by some FeatureCluster's toggle_keys — otherwise
the planner silently ignores it."""
from gui.planner.registry import CLUSTERS
from utils.preferences import _FEATURES_DEFAULT_V6, _OUTPUT_DEFAULT_V6


# Keys that are intentionally NOT in any cluster (config knobs, not
# feature toggles).
_INTENTIONALLY_UNROUTED_FEATURES: set[str] = set()  # currently empty

# Output keys NOT routed via clusters (the master output toggles that
# select which CSV gets written are handled by _OUTPUT_CSV_MAP in
# plan.py, not by clusters).
_OUTPUT_KEYS_VIA_CSV_MAP: set[str] = {
    "word_level_csv", "syllables_csv", "phonemes_csv", "document_summary",
}


def _all_cluster_toggle_keys() -> set[str]:
    return {k for c in CLUSTERS for k in c.toggle_keys}


def test_every_feature_pref_key_is_in_some_cluster():
    routed = _all_cluster_toggle_keys()
    feature_keys = set(_FEATURES_DEFAULT_V6.keys())
    unrouted = feature_keys - routed - _INTENTIONALLY_UNROUTED_FEATURES
    assert not unrouted, (
        f"Feature pref keys not referenced by any cluster (the planner will "
        f"silently ignore them when deciding what runs): {sorted(unrouted)}"
    )


def test_every_output_pref_key_is_either_master_csv_or_in_some_cluster():
    routed = _all_cluster_toggle_keys()
    output_keys = set(_OUTPUT_DEFAULT_V6.keys())
    unrouted = output_keys - routed - _OUTPUT_KEYS_VIA_CSV_MAP
    assert not unrouted, (
        f"Output pref keys not referenced by any cluster or master CSV map "
        f"(planner ignores them): {sorted(unrouted)}"
    )


def test_no_cluster_references_nonexistent_pref_key():
    """The reverse direction: cluster toggle_keys must exist in the prefs schema."""
    all_pref_keys = set(_FEATURES_DEFAULT_V6.keys()) | set(_OUTPUT_DEFAULT_V6.keys())
    routed = _all_cluster_toggle_keys()
    phantom = routed - all_pref_keys
    assert not phantom, (
        f"Clusters reference pref keys not in the v6 schema "
        f"(typo or stale registry): {sorted(phantom)}"
    )
