"""Sanity / drift tests over STAGES. CLUSTERS-specific tests in test_cluster_registry
(Task 3); full algorithm tests in test_plan.py (Task 4)."""
from gui.planner.registry import STAGES, CLUSTERS


def test_stage_names_are_unique():
    names = [s.name for s in STAGES]
    assert len(names) == len(set(names)), \
        f"Duplicate stage names: {[n for n in names if names.count(n) > 1]}"


def test_every_consumed_artifact_is_produced_by_some_stage():
    """The DAG must close: every artifact consumed by some stage must
    be produced by some other stage OR be a root input ('text', 'audio',
    'prepared_text' — the latter is produced by transcript_prep, which
    we treat as implicit since it always runs before MFA)."""
    ROOTS = {"text", "audio", "prepared_text"}
    produced = set()
    for s in STAGES:
        produced.update(s.produces)
    consumed = set()
    for s in STAGES:
        consumed.update(s.consumes)
    orphan = consumed - produced - ROOTS
    # Strip namespaced sub-artifacts like word_df.Zipf_Freq — those are produced
    # by a stage modifying the word_df. The base name "word_df" is what matters
    # for consumes-resolution.
    orphan = {a for a in orphan if "." not in a}
    assert not orphan, f"Stage consumes unproduced artifact(s): {orphan}"


def test_each_stage_produces_at_least_one_artifact():
    for s in STAGES:
        assert s.produces, f"Stage {s.name!r} produces nothing"


def test_shared_stages_present():
    names = {s.name for s in STAGES}
    assert "spacy_parse" in names
    assert "mfa_align" in names
    assert "praat_contours" in names


def test_subword_stages_present():
    names = {s.name for s in STAGES}
    assert "syllabify_pairs" in names
    assert "syllable_prosody" in names


def test_doc_stages_present():
    names = {s.name for s in STAGES}
    for n in ("doc_counts", "doc_aggregates", "doc_lex_diversity",
              "doc_readability", "doc_psych", "doc_category_proportions",
              "doc_prosody", "doc_articulation_rate"):
        assert n in names, f"missing doc stage {n}"


def test_cluster_names_are_unique():
    names = [c.name for c in CLUSTERS]
    assert len(names) == len(set(names))


def test_every_cluster_required_stage_exists():
    stage_names = {s.name for s in STAGES}
    for c in CLUSTERS:
        for rs in c.requires_stages:
            assert rs in stage_names, \
                f"Cluster {c.name!r} requires unknown stage {rs!r}"


def test_every_cluster_output_csv_is_valid():
    VALID = {
        "features", "syllables", "phonemes", "sentences", "documents_summary",
        "word2vec", "bert_word", "bert_sentence", "acoustic",
        "phoneme_surprisal_mat",
    }
    for c in CLUSTERS:
        assert c.output_csv in VALID, \
            f"Cluster {c.name!r} has invalid output_csv {c.output_csv!r}"


# Clusters that legitimately have NO child toggle and NO master (activated
# purely by their Run-tab level via LEVEL_ARTIFACTS). Since v21 every cluster
# has a toggle or a section switch, so the allowlist is empty; keep it explicit
# so the invariant below still catches a cluster left permanently inactive.
_OUTPUT_KEY_ONLY_GATED_CSVS: set[str] = set()


def test_every_cluster_has_at_least_one_toggle_key_or_master():
    for c in CLUSTERS:
        assert (c.toggle_keys or c.master_toggle
                or c.output_csv in _OUTPUT_KEY_ONLY_GATED_CSVS), (
            f"Cluster {c.name!r} has no toggle_keys, no master_toggle, and is "
            f"not an allowlisted output-key-only-gated cluster "
            f"(would be permanently inactive)"
        )


def _cluster(name):
    return next(c for c in CLUSTERS if c.name == name)


def test_phoneme_cluster_uses_the_split_toggles():
    c = _cluster("phoneme intervals")
    assert "feat_phoneme_label" not in c.toggle_keys
    assert {"feat_phoneme_arpabet", "feat_phoneme_stress",
            "feat_phoneme_ipa", "feat_phoneme_pron_source"} <= set(c.toggle_keys)
    assert {"ARPABET", "Stress", "IPA", "Pron_Source"} <= set(c.emits_columns)
    assert "Label" not in c.emits_columns


def test_syllable_labels_cluster_gains_ipa():
    c = _cluster("syllable labels")
    assert "feat_syllable_ipa" in c.toggle_keys
    assert "IPA" in c.emits_columns
    assert "ARPABET" in c.emits_columns   # unchanged
