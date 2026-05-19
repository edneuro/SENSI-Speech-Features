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

# Extractor function imports — kept local at module top so the registry
# entries below can reference them by identity.
from extractors.text_parser    import parse           as _spacy_parse
from extractors.aligner        import align_with_phones as _mfa_align
from extractors.audio_features import extract         as _praat_extract
from extractors.word_freq      import add_word_freq
from extractors.educator_freq  import add_edu_freq
from extractors.surprisal      import add_surprisal
from extractors.llm_cloze      import add_llm_cloze
from extractors.semantic_domain import add_semantic_domain
from extractors.prosody        import add_word_prosody
from extractors.timing         import add_timing_columns
from extractors.syllabify      import syllabify_pairs
from extractors.syllable_prosody import add_syllable_prosody

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

    Stage(name="gpt2_surprisal",
          consumes=("word_df",),
          produces=("word_df.Surprisal",),
          run=add_surprisal),

    Stage(name="llm_predict",
          consumes=("word_df",),
          produces=("word_df.LLM",),
          run=add_llm_cloze),

    Stage(name="sem_domain",
          consumes=("word_df",),
          produces=("word_df.Sem",),
          run=add_semantic_domain),

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

    Stage(name="doc_liwc",
          consumes=("text",),
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
                     "feat_struct_term_id", "feat_struct_token_id"),
        output_csv="features",
        requires_stages=("spacy_parse",),
        emits_columns=("doc_id", "paragraph_id", "sentence_id", "sentence",
                       "start", "end", "term_id", "token_id"),
    ),

    # §1.2 Lex-ling — also passthrough from spacy_parse
    FeatureCluster(
        name="lex-ling",
        toggle_keys=("lexling_tokens", "lexling_pos", "lexling_morph", "lexling_dep"),
        output_csv="features",
        requires_stages=("spacy_parse",),
        emits_columns=("token", "lemma", "upos", "xpos", "feats",
                       "dep_rel", "head_token_id", "deps", "misc"),
    ),

    # §1.3 SUBTLEX frequency
    FeatureCluster(
        name="SUBTLEX frequency",
        toggle_keys=("freq_subtlex",),
        output_csv="features",
        requires_stages=("spacy_parse", "word_freq"),
        emits_columns=("Zipf_Freq",),
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
    ),

    # §1.5 GPT-2 surprisal
    FeatureCluster(
        name="GPT-2 surprisal",
        toggle_keys=("pred_surprisal_gpt2",),
        output_csv="features",
        requires_stages=("spacy_parse", "gpt2_surprisal"),
        emits_columns=("Surprisal",),
    ),

    # §1.6 LLM predictability — three columns, three toggles, one stage
    FeatureCluster(
        name="LLM predictability",
        toggle_keys=("pred_surprisal_llm", "pred_cloze", "pred_topmatch"),
        output_csv="features",
        requires_stages=("spacy_parse", "llm_predict"),
        emits_columns=("LLM_Surprisal", "LLM_Cloze", "LLM_TopMatch"),
    ),

    # §1.7 Semantic indicators — single master toggle; categories selection
    # plumbed via prefs and applied in the extractor.
    FeatureCluster(
        name="semantic indicators",
        toggle_keys=("sem_indicators",),
        output_csv="features",
        requires_stages=("spacy_parse", "sem_domain"),
        emits_columns=("Sem_Categories",),  # plus dynamic Sem_<Backend>_<cat> columns
    ),

    # §1.8 Word acoustic alignment — Onset, End columns
    FeatureCluster(
        name="word acoustic alignment",
        toggle_keys=("feat_acoustic_onset", "feat_acoustic_end"),
        output_csv="features",
        requires_stages=("mfa_align",),
        emits_columns=("Onset", "End"),
    ),

    # §1.9 Derived timing
    FeatureCluster(
        name="word derived timing",
        toggle_keys=("feat_acoustic_word_duration", "feat_acoustic_word_gap"),
        output_csv="features",
        requires_stages=("mfa_align", "word_timing"),
        emits_columns=("Word_Duration", "Word_Gap"),
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
        toggle_keys=("feat_phoneme_label", "feat_phoneme_onset",
                     "feat_phoneme_end", "feat_phoneme_duration"),
        output_csv="phonemes",
        requires_stages=("spacy_parse", "mfa_align", "syllabify_pairs"),
        emits_columns=("Label", "Onset", "End", "Duration"),
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

    # §2.2b Syllable labels (Label + ARPABET)
    FeatureCluster(
        name="syllable labels",
        toggle_keys=("feat_syllable_label", "feat_syllable_arpabet"),
        output_csv="syllables",
        requires_stages=("spacy_parse", "mfa_align", "syllabify_pairs"),
        emits_columns=("Label", "ARPABET"),
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

    # --- Document level (§4 of feature_dependencies.md) ---

    # §4.1 Token counts (output-keyed toggles, not feature-keyed)
    FeatureCluster(
        name="doc counts",
        toggle_keys=("out_count_tokens", "out_count_content_tokens",
                     "out_count_sentences", "out_count_paragraphs"),
        output_csv="documents_summary",
        requires_stages=("spacy_parse", "doc_counts"),
        emits_columns=("n_tokens", "n_content_tokens", "n_sentences", "n_paragraphs"),
    ),

    # §4.2 Word-feature aggregates (means)
    FeatureCluster(
        name="doc word-feature aggregates",
        toggle_keys=("doc_aggregates",),
        output_csv="documents_summary",
        requires_stages=("spacy_parse", "doc_aggregates"),
        emits_columns=("mean_zipf_freq", "mean_edu_zipf_k2", "mean_edu_zipf_3to5",
                       "mean_edu_zipf_6to8", "mean_edu_zipf_9to12", "mean_edu_zipf_overall",
                       "mean_surprisal", "mean_llm_surprisal",
                       "mean_llm_cloze", "mean_llm_top_match_prop"),
    ),

    # §4.3 Lex diversity
    FeatureCluster(
        name="doc lex diversity",
        toggle_keys=("doc_lex_diversity",),
        output_csv="documents_summary",
        requires_stages=("spacy_parse", "doc_lex_diversity"),
        emits_columns=("ttr", "mtld"),
    ),

    # §4.4 Readability
    FeatureCluster(
        name="doc readability",
        toggle_keys=("doc_readability",),
        output_csv="documents_summary",
        requires_stages=("doc_readability",),
        emits_columns=("flesch_reading_ease", "flesch_kincaid_grade"),
    ),

    # §4.5 LIWC composite
    FeatureCluster(
        name="doc LIWC composite",
        toggle_keys=("feat_psych_authentic", "feat_psych_analytic",
                     "feat_psych_tone", "feat_psych_clout"),
        output_csv="documents_summary",
        requires_stages=("doc_liwc",),
        emits_columns=("psych_authentic", "psych_analytic", "psych_tone", "psych_clout"),
        master_toggle="master_psych_composite",
        available_when=("semantic.backend", "liwc"),
    ),

    # §4.6 Category proportions
    FeatureCluster(
        name="doc category proportions",
        toggle_keys=("doc_category_proportions",),
        output_csv="documents_summary",
        requires_stages=("spacy_parse", "sem_domain", "doc_category_proportions"),
        emits_columns=(),  # dynamic — count_<cat>, prop_<cat>
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

    # §4.8 Articulation rate (split from doc prosody because of pyphen dep)
    FeatureCluster(
        name="doc articulation rate",
        toggle_keys=("feat_doc_prosody_articulation_rate",),
        output_csv="documents_summary",
        requires_stages=("spacy_parse", "mfa_align", "doc_articulation_rate"),
        emits_columns=("doc_articulation_rate",),
        master_toggle="master_doc_articulation_rate",
    ),
]
