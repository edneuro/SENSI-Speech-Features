"""Declarative feature/stage dependency graph.

STAGES is the canonical list of pipeline stages. Each stage declares
what artifacts it consumes and produces, plus the extractor function
to call.

CLUSTERS (added in Task 3) maps user-facing toggle groups to the
stages they require.

Adding a new feature is a registry edit here plus three small touch-ups
elsewhere (see spec §9): (1) the new extractor module, (2) one Stage +
one FeatureCluster in this file, (3) the new prefs key in
utils/preferences.py, (4) the toggle widget in gui/section_*.py. No
edits to main.py, gui/run_dispatch.py, or any existing extractor.

See docs/feature_dependencies.md for the inventory this operationalizes.
"""
from __future__ import annotations

from gui.planner.stages import Stage, FeatureCluster

# Extractor callables are bound lazily: planning reads only Stage metadata
# (name/consumes/produces), so importing this module must NOT drag in the
# heavy ML stack (torch/transformers/spaCy). Each name below imports its
# extractor on first *call* (i.e. when a stage actually runs), caching it.
from importlib import import_module


def _lazy(module_path: str, attr: str):
    """Return a callable that imports `module_path`.`attr` on first call and
    caches it, then delegates. Keeps GUI startup free of torch/spaCy/etc.

    A bad module path/attr fails only at call time (when the stage runs), not
    at import — the deliberate trade-off that buys the fast startup."""
    cache: dict = {}

    def call(*args, **kwargs):
        fn = cache.get("fn")
        if fn is None:
            fn = getattr(import_module(module_path), attr)
            cache["fn"] = fn
        return fn(*args, **kwargs)

    # Carry the target's identity so diagnostics that log stage.run.__name__
    # show the extractor, not "call".
    call.__name__ = attr
    call.__qualname__ = f"{module_path}.{attr}"
    return call


_spacy_parse           = _lazy("extractors.text_parser", "parse")
_mfa_align             = _lazy("extractors.aligner", "align_with_phones")
_praat_extract         = _lazy("extractors.audio_features", "extract")
add_word_freq          = _lazy("extractors.word_freq", "add_word_freq")
add_edu_freq           = _lazy("extractors.educator_freq", "add_edu_freq")
add_predictability     = _lazy("extractors.predictability", "add_predictability")
add_semantic_domain    = _lazy("extractors.semantic_domain", "add_semantic_domain")
_add_lexical_norms     = _lazy("extractors.lexical_norms", "add_lexical_norms")
add_word_prosody       = _lazy("extractors.prosody", "add_word_prosody")
add_timing_columns     = _lazy("extractors.timing", "add_timing_columns")
syllabify_pairs        = _lazy("extractors.syllabify", "syllabify_pairs")
add_syllable_prosody   = _lazy("extractors.syllable_prosody", "add_syllable_prosody")
_add_pcenter           = _lazy("extractors.pcenter", "add_pcenter")
_add_sentence_prosody  = _lazy("extractors.sentence_prosody", "add_sentence_prosody")
_embed_word2vec        = _lazy("extractors.word2vec_embed", "embed_word2vec")
_embed_bert            = _lazy("extractors.bert_embed", "embed_bert_words_and_sentences")
_extract_acoustic      = _lazy("extractors.acoustic_ts", "extract_acoustic")
_add_phoneme_surprisal = _lazy("extractors.phoneme_surprisal", "add_phoneme_surprisal")

STAGES: list[Stage] = [
    # --- §0 Shared stages ---
    Stage(name="spacy_parse",
          consumes=("text",),
          produces=("word_df",),
          run=_spacy_parse),

    Stage(name="mfa_align",
          consumes=("audio", "prepared_text"),
          produces=("word_alignment", "phones_df"),
          run=_mfa_align),

    Stage(name="praat_contours",
          consumes=("audio",),
          produces=("contours",),
          run=_praat_extract),

    # --- §1 Word-level extractors ---
    Stage(name="word_freq",
          consumes=("word_df",),
          produces=("word_df.Zipf_Freq",),
          run=add_word_freq),

    Stage(name="ewfg_freq",
          consumes=("word_df",),
          produces=("word_df.Edu_Zipf",),
          run=add_edu_freq),

    # v18: unified dynamic predictability stage — replaces the old
    # gpt2_surprisal/llm_predict pair. One call to add_predictability scores
    # every selected roster model and writes only the requested metric
    # columns, namespaced <slug>_<Metric>. See extractors/predictability.py.
    Stage(name="predictability",
          consumes=("word_df",),
          produces=("word_df.predictability",),
          run=add_predictability),

    Stage(name="sem_domain",
          consumes=("word_df",),
          produces=("word_df.Sem",),
          run=add_semantic_domain),

    Stage(name="lexical_norms",
          consumes=("word_df",),
          produces=("word_df.Norms",),
          run=_add_lexical_norms),

    Stage(name="word_timing",
          consumes=("word_df", "word_alignment"),
          produces=("word_df.Word_Duration", "word_df.Word_Gap"),
          run=add_timing_columns),

    Stage(name="word_prosody",
          consumes=("word_df", "word_alignment", "contours"),
          produces=("word_df.Prosody",),
          run=add_word_prosody),

    # --- §2 Sub-word ---
    Stage(name="syllabify_pairs",
          consumes=("word_df", "phones_df"),
          produces=("syllables_df", "phonemes_df"),
          run=syllabify_pairs),

    Stage(name="syllable_prosody",
          consumes=("syllables_df", "contours"),
          produces=("syllables_df.Prosody",),
          run=add_syllable_prosody),

    Stage(name="pcenter",
          consumes=("syllables_df", "phonemes_df"),
          produces=("syllables_df.P_Center",),
          run=_add_pcenter),

    Stage(name="phoneme_surprisal",
          consumes=("phonemes_df",),
          produces=("phonemes_df.phoneme_surprisal", "phonemes_df.cohort_entropy"),
          run=_add_phoneme_surprisal),

    # --- §3 Sentence level ---
    Stage(name="sentence_prosody",
          consumes=("word_df", "contours"),
          produces=("sentence_df",),
          run=_add_sentence_prosody),

    # --- §5 Embeddings ---
    Stage(name="word2vec_embed",
          consumes=("word_df",),
          produces=("word2vec_mat",),
          run=_embed_word2vec),

    Stage(name="bert_embed",
          consumes=("word_df",),
          produces=("bert_word_mat", "bert_sentence_mat"),
          run=_embed_bert),

    # --- §6 Acoustic level (audio-only continuous TRF features) ---
    Stage(name="acoustic_ts",
          consumes=("audio",),
          produces=("acoustic_mat",),
          run=_extract_acoustic),

    # --- §4 Document level ---
    # Document stages are intentionally NOT extracted into separate functions
    # in v1 — utils/doc_summary.py:build_doc_summary computes them as one pass.
    # We model them as logical stages so the planner knows what's needed; the
    # adapter (Task 5) collapses them into the existing no_doc_* flags.
    Stage(name="doc_counts",
          consumes=("word_df",),
          produces=("doc_row.counts",),
          run=lambda *a, **kw: None),  # marker stage; logic stays in doc_summary

    Stage(name="doc_aggregates",
          consumes=("word_df",),  # actual deps vary by which means are requested
          produces=("doc_row.aggregates",),
          run=lambda *a, **kw: None),

    Stage(name="doc_lex_diversity",
          consumes=("word_df",),
          produces=("doc_row.ttr", "doc_row.mtld"),
          run=lambda *a, **kw: None),

    Stage(name="doc_readability",
          consumes=("text",),
          produces=("doc_row.flesch", "doc_row.fk_grade"),
          run=lambda *a, **kw: None),

    Stage(name="doc_psych",
          consumes=("word_df",),
          produces=("doc_row.psych",),
          run=lambda *a, **kw: None),

    Stage(name="doc_category_proportions",
          consumes=("word_df",),  # needs Sem_* on word_df
          produces=("doc_row.categories",),
          run=lambda *a, **kw: None),

    Stage(name="doc_prosody",
          consumes=("word_alignment", "contours"),
          produces=("doc_row.prosody",),
          run=lambda *a, **kw: None),

    Stage(name="doc_articulation_rate",
          consumes=("word_df", "word_alignment"),  # also needs pyphen but that's a lib not an artifact
          produces=("doc_row.articulation_rate",),
          run=lambda *a, **kw: None),
]


# The predictability cluster is registered (so test_registry/test_schema_drift
# can validate its shape) but its activation + dynamic columns are computed
# separately in gui/planner/plan.py. Both the cluster definition below and
# plan.py's skip-and-compute logic reference this single constant, so the
# coupling is import-visible rather than a bare duplicated string literal.
PREDICTABILITY_CLUSTER_NAME = "Predictability"


CLUSTERS: list[FeatureCluster] = [
    # --- Word level (§1 of feature_dependencies.md) ---

    # §1.1 Structural identifiers — no extractor runs; columns are passthrough
    # from spacy_parse. The cluster's requires_stages is just spacy_parse;
    # per-column toggles drive the CSV-write drop step (handled by
    # feature_toggles passthrough in emit.py).
    FeatureCluster(
        name="word structural identifiers",
        toggle_keys=("feat_struct_doc_id", "feat_struct_paragraph_id",
                     "feat_struct_sentence_id", "feat_struct_sentence_text",
                     "feat_struct_char_start", "feat_struct_char_end",
                     "feat_struct_token_id", "feat_struct_token_in_sentence_id"),
        output_csv="features",
        requires_stages=("spacy_parse",),
        emits_columns=("doc_id", "paragraph_id", "sentence_id", "sentence",
                       "start", "end", "token_id", "token_in_sentence_id"),
        master_toggle="master_structural",
    ),

    # §1.2 Lex-ling — also passthrough from spacy_parse
    FeatureCluster(
        name="lex-ling",
        toggle_keys=("lexling_tokens", "lexling_pos", "lexling_morph", "lexling_dep"),
        output_csv="features",
        requires_stages=("spacy_parse",),
        emits_columns=("token", "lemma", "upos", "xpos", "feats",
                       "dep_rel", "head_token_id"),
        master_toggle="master_lexling",
    ),

    # §1.3 SUBTLEX frequency
    FeatureCluster(
        name="SUBTLEX frequency",
        toggle_keys=("freq_subtlex",),
        output_csv="features",
        requires_stages=("spacy_parse", "word_freq"),
        emits_columns=("Zipf_Freq",),
        master_toggle="master_frequency",
    ),

    # §1.4 EWFG — one cluster covering all 5 bands; the band-subset choice
    # is plumbed via prefs/edu_bands and applied in the extractor itself.
    FeatureCluster(
        name="EWFG grade-band frequency",
        toggle_keys=("freq_ewfg_k2", "freq_ewfg_3to5", "freq_ewfg_6to8",
                     "freq_ewfg_9to12", "freq_ewfg_overall"),
        output_csv="features",
        requires_stages=("spacy_parse", "ewfg_freq"),
        emits_columns=("Edu_Zipf_K2", "Edu_Zipf_3to5", "Edu_Zipf_6to8",
                       "Edu_Zipf_9to12", "Edu_Zipf_Overall"),
        master_toggle="master_frequency",
    ),

    # §1.5/1.6 Predictability (v18) — unified dynamic stage/cluster replacing
    # the old "GPT-2 surprisal" + "LLM predictability" clusters. Columns are
    # <slug>_<Metric> (e.g. "gpt2_Surprisal", "llama3_8b_Cloze") for every
    # roster model in prefs["predictability"]["models"] crossed with every
    # metric switched on in prefs["predictability"]["metrics"] — a set not
    # knowable at registry-definition time, so toggle_keys/emits_columns
    # can't carry it declaratively (there is no fixed list of feature-toggle
    # keys, one per column, the way every other cluster works).
    #
    # This cluster is therefore "static but considered" (spec §12 fallback):
    # it exists so the stage/CSV/master wiring is documented here and
    # test_registry.py can validate its shape, but gui/planner/plan.py's
    # _active_clusters skips it by name and computes its true activation
    # (master_predictability AND models non-empty AND >=1 metric on) plus its
    # dynamic column list directly from prefs["predictability"]
    # (_predictability_active / _predictability_columns). Do not add
    # toggle_keys here expecting them to gate anything — they won't be read.
    FeatureCluster(
        name=PREDICTABILITY_CLUSTER_NAME,
        toggle_keys=(),
        output_csv="features",
        requires_stages=("spacy_parse", "predictability"),
        emits_columns=(),
        master_toggle="master_predictability",
    ),

    # §1.7 Semantic indicators — single master toggle; categories selection
    # plumbed via prefs and applied in the extractor.
    FeatureCluster(
        name="semantic indicators",
        toggle_keys=("sem_indicators",),
        output_csv="features",
        requires_stages=("spacy_parse", "sem_domain"),
        emits_columns=("Sem_Categories",),  # plus dynamic Sem_<Backend>_<cat> columns
        master_toggle="master_semantic_indicators",
    ),

    # §1.7b Lexical norms — user-supplied AoA / concreteness / imageability /
    # familiarity / valence-arousal-dominance. Opt-in (master defaults off);
    # no data ships.
    FeatureCluster(
        name="lexical norms",
        toggle_keys=("feat_norm_aoa_mean", "feat_norm_aoa_sd",
                     "feat_norm_concreteness", "feat_norm_imageability",
                     "feat_norm_familiarity", "feat_norm_valence",
                     "feat_norm_arousal", "feat_norm_dominance"),
        output_csv="features",
        requires_stages=("spacy_parse", "lexical_norms"),
        emits_columns=("AoA_Mean", "AoA_SD", "Concreteness", "Imageability",
                       "Familiarity", "Valence", "Arousal", "Dominance"),
        master_toggle="master_lexical_norms",
    ),

    # §1.8 Word acoustic alignment — Onset, End columns
    FeatureCluster(
        name="word acoustic alignment",
        toggle_keys=("feat_acoustic_onset", "feat_acoustic_end"),
        output_csv="features",
        requires_stages=("mfa_align",),
        emits_columns=("Onset", "End"),
        master_toggle="master_acoustic",
    ),

    # §1.9 Derived timing
    FeatureCluster(
        name="word derived timing",
        toggle_keys=("feat_acoustic_word_duration", "feat_acoustic_word_gap"),
        output_csv="features",
        requires_stages=("mfa_align", "word_timing"),
        emits_columns=("Word_Duration", "Word_Gap"),
        master_toggle="master_acoustic",
    ),

    # §1.10 Word prosody — 10 columns, 10 toggles, one stage
    FeatureCluster(
        name="word prosody",
        toggle_keys=("feat_prosody_pitch_mean", "feat_prosody_pitch_min",
                     "feat_prosody_pitch_max", "feat_prosody_pitch_std",
                     "feat_prosody_pitch_delta",
                     "feat_prosody_volume_mean", "feat_prosody_volume_min",
                     "feat_prosody_volume_max", "feat_prosody_volume_std",
                     "feat_prosody_volume_delta"),
        output_csv="features",
        requires_stages=("spacy_parse", "mfa_align", "praat_contours", "word_prosody"),
        emits_columns=("Pitch_Mean", "Pitch_Min", "Pitch_Max", "Pitch_Std", "Pitch_Delta",
                       "Volume_Mean", "Volume_Min", "Volume_Max", "Volume_Std", "Volume_Delta"),
        master_toggle="master_word_prosody",
    ),

    # --- Sub-word level (§2 of feature_dependencies.md) ---

    # §2.1 Phoneme intervals — phoneme half of syllabify_pairs
    FeatureCluster(
        name="phoneme intervals",
        toggle_keys=("feat_phoneme_arpabet", "feat_phoneme_stress",
                     "feat_phoneme_ipa", "feat_phoneme_pron_source",
                     "feat_phoneme_onset", "feat_phoneme_end",
                     "feat_phoneme_duration"),
        output_csv="phonemes",
        requires_stages=("spacy_parse", "mfa_align", "syllabify_pairs"),
        emits_columns=("ARPABET", "Stress", "IPA", "Pron_Source",
                       "Onset", "End", "Duration"),
        master_toggle="master_phoneme_intervals",
    ),

    # §2.2a Syllabification intervals (timing columns)
    FeatureCluster(
        name="syllabification intervals",
        toggle_keys=("feat_syllable_onset", "feat_syllable_end",
                     "feat_syllable_duration"),
        output_csv="syllables",
        requires_stages=("spacy_parse", "mfa_align", "syllabify_pairs"),
        emits_columns=("Onset", "End", "Duration"),
        master_toggle="master_syllable_intervals",
    ),

    # §2.2b Syllable labels (Label + ARPABET + IPA)
    FeatureCluster(
        name="syllable labels",
        toggle_keys=("feat_syllable_label", "feat_syllable_arpabet",
                     "feat_syllable_ipa"),
        output_csv="syllables",
        requires_stages=("spacy_parse", "mfa_align", "syllabify_pairs"),
        emits_columns=("Label", "ARPABET", "IPA"),
        master_toggle="master_syllable_labels",
    ),

    # §2.3 Syllable prosody
    FeatureCluster(
        name="syllable prosody",
        toggle_keys=("feat_syllable_pitch_mean", "feat_syllable_volume_mean",
                     "feat_syllable_pitch_delta", "feat_syllable_volume_delta"),
        output_csv="syllables",
        requires_stages=("spacy_parse", "mfa_align", "praat_contours",
                         "syllabify_pairs", "syllable_prosody"),
        emits_columns=("Pitch_Mean", "Volume_Mean", "Pitch_Delta", "Volume_Delta"),
        master_toggle="master_syllable_prosody",
    ),

    # §2.4 P-center
    FeatureCluster(
        name="p-center",
        toggle_keys=("feat_syllable_pcenter",),
        output_csv="syllables",
        requires_stages=("spacy_parse", "mfa_align", "syllabify_pairs", "pcenter"),
        emits_columns=("P_Center",),
        master_toggle="master_syllable_pcenter",
    ),

    # §2.5 Phoneme surprisal (CSV columns) — Brodbeck/Gillis cohort model
    FeatureCluster(
        name="phoneme surprisal",
        toggle_keys=("feat_phoneme_surprisal", "feat_phoneme_cohort_entropy"),
        output_csv="phonemes",
        requires_stages=("spacy_parse", "mfa_align", "syllabify_pairs", "phoneme_surprisal"),
        emits_columns=("phoneme_surprisal", "cohort_entropy"),
        master_toggle="master_phoneme_surprisal",
    ),

    # §2.5b Phoneme surprisal (.mat impulse trains) — a Sub-word-level artifact
    # gated by the Phoneme-surprisal section switch + its .mat toggle (v21).
    FeatureCluster(
        name="phoneme surprisal mat",
        toggle_keys=("feat_phoneme_surprisal_mat",),
        output_csv="phoneme_surprisal_mat",
        requires_stages=("spacy_parse", "mfa_align", "syllabify_pairs", "phoneme_surprisal"),
        emits_columns=(),
        master_toggle="master_phoneme_surprisal",
    ),

    # --- Sentence level (§3) ---

    FeatureCluster(
        name="sentence timing",
        toggle_keys=("feat_sentence_text", "feat_sentence_onset", "feat_sentence_end"),
        output_csv="sentences",
        requires_stages=("spacy_parse", "mfa_align"),
        emits_columns=("sentence", "Onset", "End"),
        master_toggle="master_sentence_timing",
    ),

    FeatureCluster(
        name="sentence prosody",
        toggle_keys=(
            "feat_sentence_pitch_mean", "feat_sentence_pitch_min",
            "feat_sentence_pitch_max",  "feat_sentence_pitch_std",
            "feat_sentence_pitch_delta",
            "feat_sentence_volume_mean", "feat_sentence_volume_min",
            "feat_sentence_volume_max",  "feat_sentence_volume_std",
            "feat_sentence_volume_delta",
            "feat_sentence_hnr",
        ),
        output_csv="sentences",
        requires_stages=("spacy_parse", "mfa_align", "praat_contours", "sentence_prosody"),
        emits_columns=(
            "Pitch_Mean", "Pitch_Min", "Pitch_Max", "Pitch_Std", "Pitch_Delta",
            "Volume_Mean", "Volume_Min", "Volume_Max", "Volume_Std", "Volume_Delta",
            "HNR",
        ),
        master_toggle="master_sentence_prosody",
    ),

    # --- Embeddings (§5) ---

    FeatureCluster(
        name="word2vec",
        toggle_keys=("master_word2vec",),
        output_csv="word2vec",
        requires_stages=("spacy_parse", "word2vec_embed"),
        emits_columns=(),
        master_toggle="master_semantic_similarity",
    ),

    FeatureCluster(
        name="bert word embeddings",
        toggle_keys=("master_bert_word",),
        output_csv="bert_word",
        requires_stages=("spacy_parse", "bert_embed"),
        emits_columns=(),
        master_toggle="master_semantic_similarity",
    ),

    FeatureCluster(
        name="bert sentence embeddings",
        toggle_keys=("master_bert_sentence",),
        output_csv="bert_sentence",
        requires_stages=("spacy_parse", "bert_embed"),
        emits_columns=(),
        master_toggle="master_sentence_similarity",
    ),

    # --- Acoustic level (§6) ---
    # Gated by the Acoustic level switch (output.level_acoustic, Run tab) + at
    # least one feature toggle. The level key doubles as the master: the
    # acoustic level is one cluster, so a separate master would duplicate it.
    FeatureCluster(
        name="acoustic time-series",
        toggle_keys=("feat_acoustic_ts_envelope", "feat_acoustic_ts_onset_env",
                     "feat_acoustic_ts_logmel", "feat_acoustic_ts_mfcc"),
        output_csv="acoustic",
        requires_stages=("acoustic_ts",),
        emits_columns=(),
        master_toggle="level_acoustic",
    ),

    # --- Document level (§4 of feature_dependencies.md) ---

    # §4.1 Token counts (output-keyed toggles, not feature-keyed)
    FeatureCluster(
        name="doc counts",
        toggle_keys=("out_count_tokens", "out_count_content_tokens",
                     "out_count_sentences", "out_count_paragraphs"),
        output_csv="documents_summary",
        requires_stages=("spacy_parse", "doc_counts"),
        emits_columns=("n_tokens", "n_content_tokens", "n_sentences", "n_paragraphs"),
        master_toggle="master_doc_counts",
    ),

    # §4.2 Word-feature aggregates — four independent toggles, one shared stage.
    # Each toggle controls one family of aggregate means in the doc summary.
    FeatureCluster(
        name="doc agg zipf freq",
        toggle_keys=("doc_agg_zipf_freq",),
        output_csv="documents_summary",
        # The source extractor is pulled in so the mean is computed even when
        # the word-level Frequency section / Word level is off (the doc pass
        # forces it on; its columns never reach the word CSV).
        requires_stages=("spacy_parse", "word_freq", "doc_aggregates"),
        emits_columns=("mean_zipf_freq",),
        master_toggle="master_doc_aggregates",
    ),
    FeatureCluster(
        name="doc agg edu zipf",
        toggle_keys=("doc_agg_edu_zipf",),
        output_csv="documents_summary",
        # Source extractor pulled in like word_freq above; the grade bands
        # themselves are the shared freq_ewfg_* ticks (no band = no means).
        requires_stages=("spacy_parse", "ewfg_freq", "doc_aggregates"),
        emits_columns=("mean_edu_zipf_k2", "mean_edu_zipf_3to5",
                       "mean_edu_zipf_6to8", "mean_edu_zipf_9to12",
                       "mean_edu_zipf_overall"),
        master_toggle="master_doc_aggregates",
    ),
    # Doc-level predictability aggregates (v18) — single toggle mirroring the
    # word-level Predictability cluster's roster-driven shape. Unlike that
    # cluster, this one's *activation* is a plain boolean toggle
    # (doc_agg_predictability), so it does not need the _active_clusters
    # skip-and-compute treatment — only its emitted columns are dynamic
    # (mean_<slug>_<metric>, one per active roster model x metric, computed
    # by utils.doc_summary.build_doc_summary from whatever word-level
    # <slug>_<Metric> columns the run produced). Registered here with
    # emits_columns=() (static-but-considered, spec §12 fallback) so
    # test_schema_drift can still validate the toggle's shape; there is no
    # fixed column list to declare.
    FeatureCluster(
        name="doc agg predictability",
        toggle_keys=("doc_agg_predictability",),
        output_csv="documents_summary",
        requires_stages=("spacy_parse", "doc_aggregates"),
        emits_columns=(),
        master_toggle="master_doc_aggregates",
    ),

    # §4.3 Lex diversity
    FeatureCluster(
        name="doc lex diversity",
        toggle_keys=("doc_lex_diversity",),
        output_csv="documents_summary",
        requires_stages=("spacy_parse", "doc_lex_diversity"),
        emits_columns=("ttr", "mtld"),
        master_toggle="master_doc_lex_diversity",
    ),

    # §4.4 Readability
    FeatureCluster(
        name="doc readability",
        toggle_keys=("doc_readability",),
        output_csv="documents_summary",
        requires_stages=("doc_readability",),
        emits_columns=("flesch_reading_ease", "flesch_kincaid_grade"),
        master_toggle="master_doc_readability",
    ),

    # §4.5 Composite psychological scores (license-clean, spaCy + VADER).
    # No backend gate: the scores no longer depend on a LIWC dictionary.
    FeatureCluster(
        name="doc composite psych",
        toggle_keys=("feat_psych_authentic", "feat_psych_analytic",
                     "feat_psych_tone", "feat_psych_clout"),
        output_csv="documents_summary",
        requires_stages=("doc_psych",),
        emits_columns=("psych_authentic", "psych_analytic", "psych_tone", "psych_clout"),
        master_toggle="master_psych_composite",
    ),

    # §4.5b Document-level lexical-norm aggregates (means, AoA spread,
    # prop_late_acquired, coverage). Single output toggle; computed in
    # doc_summary from the word-level norm columns.
    FeatureCluster(
        name="doc lexical norms",
        toggle_keys=("doc_agg_lexical_norms",),
        output_csv="documents_summary",
        requires_stages=("spacy_parse", "lexical_norms", "doc_aggregates"),
        emits_columns=(),  # dynamic — means + coverage columns
        master_toggle="master_doc_aggregates",
    ),

    # §4.6 Category proportions
    FeatureCluster(
        name="doc category proportions",
        toggle_keys=("doc_category_proportions",),
        output_csv="documents_summary",
        requires_stages=("spacy_parse", "sem_domain", "doc_category_proportions"),
        emits_columns=(),  # dynamic — count_<cat>, prop_<cat>
        master_toggle="master_doc_category_proportions",
    ),

    # §4.7 Doc prosody
    FeatureCluster(
        name="doc prosody",
        toggle_keys=("feat_doc_prosody_pitch_mean", "feat_doc_prosody_pitch_min",
                     "feat_doc_prosody_pitch_max", "feat_doc_prosody_pitch_std",
                     "feat_doc_prosody_volume_mean", "feat_doc_prosody_volume_min",
                     "feat_doc_prosody_volume_max", "feat_doc_prosody_volume_std",
                     "feat_doc_prosody_pitch_delta_mean",
                     "feat_doc_prosody_volume_delta_mean",
                     "feat_doc_prosody_hnr",
                     "feat_doc_prosody_pause_count",
                     "feat_doc_prosody_pause_mean_duration",
                     "feat_doc_prosody_pause_total_duration",
                     "feat_doc_prosody_pause_pct_silence"),
        output_csv="documents_summary",
        requires_stages=("mfa_align", "praat_contours", "doc_prosody"),
        emits_columns=("doc_pitch_mean", "doc_pitch_min", "doc_pitch_max", "doc_pitch_std",
                       "doc_volume_mean", "doc_volume_min", "doc_volume_max", "doc_volume_std",
                       "doc_pitch_delta_mean", "doc_volume_delta_mean",
                       "doc_hnr",
                       "doc_pause_count", "doc_pause_mean_duration",
                       "doc_pause_total_duration", "doc_pause_pct_silence"),
        master_toggle="master_doc_prosody",
    ),

    # §4.8 Articulation rate (split from doc prosody because of pyphen dep).
    # Gated by master_doc_prosody: in the GUI it is a child row of the
    # Document-prosody card, so turning that card off must deactivate it too
    # (otherwise it stays active and pulls MFA). master_doc_articulation_rate
    # is vestigial — kept in the prefs schema for migration compat but never
    # wired to a UI master (cf. master_acoustic_ts).
    FeatureCluster(
        name="doc articulation rate",
        toggle_keys=("feat_doc_prosody_articulation_rate",),
        output_csv="documents_summary",
        requires_stages=("spacy_parse", "mfa_align", "doc_articulation_rate"),
        emits_columns=("doc_articulation_rate",),
        master_toggle="master_doc_prosody",
    ),
]
