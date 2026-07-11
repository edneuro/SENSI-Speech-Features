"""Load and save user preferences.

Schema (v13 — active):
    {
      "schema_version": 13,
      "shared":  {analyzer_name, last_*_folder, theme},
      "features": {feat_struct_*, lexling_*, freq_*, pred_*, sem_*,
                   feat_acoustic_*, feat_psych_*,
                   feat_prosody_* (10 word-level), feat_doc_prosody_* (16 doc-level),
                   feat_syllable_* (9 syllable), feat_syllable_pcenter,
                   feat_phoneme_* (4 phoneme),
                   feat_sentence_* (14 sentence-level),
                   master_* (6 cluster master toggles), master_syllable_pcenter,
                   master_sentence_timing, master_sentence_prosody,
                   master_word2vec, master_bert_word, master_bert_sentence,
                   master_acoustic_ts, feat_acoustic_ts_* (4 acoustic-TS toggles)},
      "predictability": {llm_model},
      "semantic": {backend, categories_liwc, categories_empath},
      "output":   {word_level_csv, document_summary, out_count_*, doc_*,
                   syllables_csv, phonemes_csv, sentences_csv,
                   word2vec_mat, bert_word_mat, bert_sentence_mat,
                   master_doc_prosody, master_doc_articulation_rate,
                   acoustic_ts_mat},
      "prosody":  {pitch_units, pitch_floor_hz, pitch_ceiling_hz, pause_threshold_s, pcenter_k},
      "acoustic_ts": {analysis_rate, env_method, env_comp, n_mels, n_mfcc, mel_preset}
    }

Migration chain (all paths terminate at the current version):
  v1 → v2 → v3 → v4 → v5 → v6 → v7 → v8 → v9 → v10
  v10..current files are deep-merged against the current defaults (additive).
  Finalize steps applied to any older source: v11 coerces liwc→empath,
    v14 recovers the struct-id rename, v17 upgrades the old rounded
    acoustic_ts.env_comp default (0.3) to the exact mTRF constant log10(2)
    and coerces stored OpenAI (gpt-*) llm_model to "" (auto).
  Later additive versions (v12 lexical-norms, v13 acoustic_ts, v15 opt-in
    defaults, v16 phoneme surprisal, v17 above) only add keys / correct
    defaults; missing keys are filled from the current defaults.

Default location: <project_root>/config/preferences.json
Override via SENSI_PREFS_PATH environment variable.
"""

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# v2 constants (preserved so v1→v2 intermediate step still works)
# ---------------------------------------------------------------------------

SCHEMA_VERSION_V2 = 2

_SOP_CATEGORIES = [
    'social', 'cognition', 'emotion_positive', 'emotion_negative',
    'body', 'motion', 'perception', 'time', 'space',
    'math', 'science', 'work', 'family', 'communication', 'money',
]

_WORD_TOGGLES_DEFAULT = {
    'spacy_parse': True,
    'zipf_freq': True,
    'edu_freq': True,
    'sem_domain': True,
    'gpt2_surprisal': True,
    'llm_cloze': True,
    'alignment': True,
}

_DOC_TOGGLES_DEFAULT = {
    'counts': True,
    'mean_zipf_freq': True,
    'mean_edu_zipf': True,
    'mean_surprisal': True,
    'mean_llm_cloze': True,
    'ttr': True,
    'mtld': True,
    'flesch_reading_ease': True,
    'flesch_kincaid_grade': True,
    'category_proportions': True,
    'liwc_authentic': True,
    'liwc_analytic': True,
    'liwc_tone': True,
    'liwc_clout': True,
}

_DEFAULT_PREFS_V2: dict[str, Any] = {
    'schema_version': SCHEMA_VERSION_V2,
    'shared': {
        'analyzer_name': '',
        'last_transcripts_folder': '',
        'last_audio_folder': '',
        'last_output_folder': '',
        'dictionary_backend': 'auto',
    },
    'word_level': {
        'feature_toggles': dict(_WORD_TOGGLES_DEFAULT),
        'semantic_categories': list(_SOP_CATEGORIES),
        'llm_model': '',
    },
    'document_level': {
        'feature_toggles': dict(_DOC_TOGGLES_DEFAULT),
        'semantic_categories': list(_SOP_CATEGORIES),
        'llm_model': '',
    },
}

# ---------------------------------------------------------------------------
# v3 constants (active schema)
# ---------------------------------------------------------------------------

SCHEMA_VERSION_V3 = 3

# Empath native category names (mapped to the closest available Empath categories).
# These are the actual category keys in the Empath lexicon (~194 total).
_SOP_CATEGORIES_EMPATH = [
    'friends', 'philosophy', 'positive_emotion', 'negative_emotion',
    'body', 'movement', 'hearing', 'anticipation',
    'science', 'work', 'family', 'communication', 'money',
]

# LIWC-22 native category names (placeholder — to be confirmed against the .dic file).
# Names below are best-effort matches to the Empath SOP list.
_SOP_CATEGORIES_LIWC = [
    "affect", "cogproc", "emo_pos", "emo_neg",
    "body", "motion", "percept", "time", "space",
    "work", "family", "money", "relig", "focuspresent", "social",
]

_FEATURES_DEFAULT = {
    "structural": True,
    "lexling_tokens": True,
    "lexling_pos": True,
    "lexling_morph": True,
    "lexling_dep": True,
    "freq_subtlex": True,
    "freq_ewfg_k2": True,
    "freq_ewfg_3to5": True,
    "freq_ewfg_6to8": True,
    "freq_ewfg_9to12": True,
    "freq_ewfg_overall": True,
    "pred_surprisal_gpt2": True,
    "pred_surprisal_llm": True,
    "pred_cloze": True,
    "pred_topmatch": True,
    "sem_indicators": True,
    "sem_composite_psych": True,
    "acoustic_boundaries": True,
}

_OUTPUT_DEFAULT = {
    "word_level_csv": True,
    "document_summary": True,
    "doc_counts": True,
    "doc_aggregates": True,
    "doc_lex_diversity": True,
    "doc_readability": True,
    "doc_category_proportions": True,
}

DEFAULT_PREFS_V3: dict[str, Any] = {
    "schema_version": SCHEMA_VERSION_V3,
    "shared": {
        "analyzer_name": "",
        "last_transcripts_folder": "",
        "last_audio_folder": "",
        "last_output_folder": "",
        "theme": "system",
    },
    "features": dict(_FEATURES_DEFAULT),
    "predictability": {
        "llm_model": "",
    },
    "semantic": {
        "backend": "auto",
        "categories_liwc": list(_SOP_CATEGORIES_LIWC),
        "categories_empath": list(_SOP_CATEGORIES_EMPATH),
    },
    "output": dict(_OUTPUT_DEFAULT),
}

# ---------------------------------------------------------------------------
# v4 constants (active schema)
#
# Batch 3 changes vs v3:
#   - structural toggle expanded into per-column feat_struct_* (8 toggles)
#   - acoustic_boundaries expanded into per-column feat_acoustic_* (4 toggles)
#   - sem_composite_psych expanded into per-column feat_psych_* (4 toggles)
#   - output.doc_counts expanded into per-column out_count_* (4 toggles)
#   - semantic.categories_{liwc,empath} split per level into
#     {word,doc}_categories_{liwc,empath} (4 lists total)
# ---------------------------------------------------------------------------

SCHEMA_VERSION_V4 = 4

_FEATURES_DEFAULT_V4 = {
    # Lexical-linguistic — same keys as v3
    "lexling_tokens": True,
    "lexling_pos": True,
    "lexling_morph": True,
    "lexling_dep": True,
    # Frequency — same keys as v3
    "freq_subtlex": True,
    "freq_ewfg_k2": True,
    "freq_ewfg_3to5": True,
    "freq_ewfg_6to8": True,
    "freq_ewfg_9to12": True,
    "freq_ewfg_overall": True,
    # Predictability — same keys as v3
    "pred_surprisal_gpt2": True,
    "pred_surprisal_llm": True,
    "pred_cloze": True,
    "pred_topmatch": True,
    # Semantic indicators — same key
    "sem_indicators": True,
    # Structural — per-column (was single `structural` toggle in v3)
    "feat_struct_doc_id": True,
    "feat_struct_paragraph_id": True,
    "feat_struct_sentence_id": True,
    "feat_struct_sentence_text": True,
    "feat_struct_char_start": True,
    "feat_struct_char_end": True,
    "feat_struct_term_id": True,
    "feat_struct_token_id": True,
    # Acoustic — per-column (was single `acoustic_boundaries` in v3)
    "feat_acoustic_onset": True,
    "feat_acoustic_end": True,
    "feat_acoustic_word_duration": True,
    "feat_acoustic_word_gap": True,
    # Composite psychological scores — per-column (was `sem_composite_psych` in v3)
    "feat_psych_authentic": True,
    "feat_psych_analytic": True,
    "feat_psych_tone": True,
    "feat_psych_clout": True,
}

_OUTPUT_DEFAULT_V4 = {
    "word_level_csv": True,
    "document_summary": True,
    # Per-column doc counts (was single `doc_counts` in v3)
    "out_count_tokens": True,
    "out_count_content_tokens": True,
    "out_count_sentences": True,
    "out_count_paragraphs": True,
    # Other doc-level groups unchanged
    "doc_aggregates": True,
    "doc_lex_diversity": True,
    "doc_readability": True,
    "doc_category_proportions": True,
}

DEFAULT_PREFS_V4: dict[str, Any] = {
    "schema_version": SCHEMA_VERSION_V4,
    "shared": {
        "analyzer_name": "",
        "last_transcripts_folder": "",
        "last_audio_folder": "",
        "last_output_folder": "",
        "theme": "system",
    },
    "features": dict(_FEATURES_DEFAULT_V4),
    "predictability": {
        "llm_model": "",
    },
    "semantic": {
        "backend": "auto",
        "word_categories_liwc": list(_SOP_CATEGORIES_LIWC),
        "word_categories_empath": list(_SOP_CATEGORIES_EMPATH),
        "doc_categories_liwc": list(_SOP_CATEGORIES_LIWC),
        "doc_categories_empath": list(_SOP_CATEGORIES_EMPATH),
    },
    "output": dict(_OUTPUT_DEFAULT_V4),
}

# ---------------------------------------------------------------------------
# v5 constants (active schema)
#
# Batch 4 changes vs v4:
#   - Word-level prosody feature toggles (10 keys: feat_prosody_*)
#   - Doc-level prosody feature toggles (16 keys: feat_doc_prosody_*)
#   - New top-level "prosody" config section with pitch/pause parameters
# ---------------------------------------------------------------------------

SCHEMA_VERSION_V5 = 5

_FEATURES_DEFAULT_V5 = {
    **_FEATURES_DEFAULT_V4,
    # Word-level prosody
    "feat_prosody_pitch_mean": True,
    "feat_prosody_pitch_min": True,
    "feat_prosody_pitch_max": True,
    "feat_prosody_pitch_std": True,
    "feat_prosody_pitch_delta": True,
    "feat_prosody_volume_mean": True,
    "feat_prosody_volume_min": True,
    "feat_prosody_volume_max": True,
    "feat_prosody_volume_std": True,
    "feat_prosody_volume_delta": True,
    # Doc-level prosody
    "feat_doc_prosody_pitch_mean": True,
    "feat_doc_prosody_pitch_min": True,
    "feat_doc_prosody_pitch_max": True,
    "feat_doc_prosody_pitch_std": True,
    "feat_doc_prosody_volume_mean": True,
    "feat_doc_prosody_volume_min": True,
    "feat_doc_prosody_volume_max": True,
    "feat_doc_prosody_volume_std": True,
    "feat_doc_prosody_pitch_delta_mean": True,
    "feat_doc_prosody_volume_delta_mean": True,
    "feat_doc_prosody_hnr": True,
    "feat_doc_prosody_articulation_rate": True,
    "feat_doc_prosody_pause_count": True,
    "feat_doc_prosody_pause_mean_duration": True,
    "feat_doc_prosody_pause_total_duration": True,
    "feat_doc_prosody_pause_pct_silence": True,
}

_PROSODY_DEFAULT_V5 = {
    "pitch_units": "semitones",
    "pitch_floor_hz": 75,
    "pitch_ceiling_hz": 500,
    "pause_threshold_s": 0.25,
}

DEFAULT_PREFS_V5: dict[str, Any] = {
    **DEFAULT_PREFS_V4,
    "schema_version": SCHEMA_VERSION_V5,
    "features": dict(_FEATURES_DEFAULT_V5),
    "prosody": dict(_PROSODY_DEFAULT_V5),
}

# ---------------------------------------------------------------------------
# v6 constants (active schema)
#
# Phoneme/syllable batch (2026-05-13) changes vs v5:
#   - 9 word-level syllable per-column toggles (feat_syllable_*)
#   - 4 word-level phoneme per-column toggles (feat_phoneme_*)
#   - 2 sidecar output toggles (output.syllables_csv, output.phonemes_csv)
# ---------------------------------------------------------------------------

SCHEMA_VERSION_V6 = 6

_FEATURES_DEFAULT_V6 = {
    **_FEATURES_DEFAULT_V5,
    # Syllable per-column (9 keys)
    "feat_syllable_onset":        True,
    "feat_syllable_end":          True,
    "feat_syllable_duration":     True,
    "feat_syllable_label":        True,
    "feat_syllable_arpabet":      False,  # opt-in; verbose column
    "feat_syllable_pitch_mean":   True,
    "feat_syllable_volume_mean":  True,
    "feat_syllable_pitch_delta":  True,
    "feat_syllable_volume_delta": True,
    # Phoneme per-column (4 keys)
    "feat_phoneme_label":         True,
    "feat_phoneme_onset":         True,
    "feat_phoneme_end":           True,
    "feat_phoneme_duration":      True,
}

_OUTPUT_DEFAULT_V6 = {
    **DEFAULT_PREFS_V5["output"],
    "syllables_csv": True,
    "phonemes_csv":  True,
}

DEFAULT_PREFS_V6: dict[str, Any] = {
    **DEFAULT_PREFS_V5,
    "schema_version": SCHEMA_VERSION_V6,
    "features": dict(_FEATURES_DEFAULT_V6),
    "output":   dict(_OUTPUT_DEFAULT_V6),
}

# ---------------------------------------------------------------------------
# v7 constants (active schema)
#
# GUI validation unification batch (2026-05-19) changes vs v6:
#   - 8 master_* keys (6 features + 2 output) defaulting to True
#   - semantic.{word,doc}_categories_<backend> → semantic.categories_<backend>
#     (lossless union migration)
# ---------------------------------------------------------------------------

SCHEMA_VERSION_V7 = 7

# Cluster master toggles introduced in v7. All default True so existing
# user workflows behave identically until a master is explicitly toggled.
_MASTERS_V7_FEATURES = {
    "master_word_prosody":         True,
    "master_psych_composite":      True,
    "master_syllable_intervals":   True,
    "master_syllable_labels":      True,
    "master_syllable_prosody":     True,
    "master_phoneme_intervals":    True,
    # Card-level masters for word-level feature groups
    "master_structural":           True,
    "master_lexling":              True,
    "master_frequency":            True,
    "master_predictability":       True,
    "master_acoustic":             True,
}

_MASTERS_V7_OUTPUT = {
    "master_doc_prosody":           True,
    "master_doc_articulation_rate": True,
    "master_doc_counts":            True,
}

# Individual doc-level aggregate toggles (replaced single doc_aggregates key).
# Existing prefs that have doc_aggregates:true will carry it forward as a
# stale unknown key; the new keys default True so behaviour is preserved.
_AGG_KEYS_V7 = {
    "doc_agg_zipf_freq": True,
    "doc_agg_edu_zipf":  True,
    "doc_agg_surprisal": True,
    "doc_agg_llm":       True,
}

_FEATURES_DEFAULT_V7 = {**_FEATURES_DEFAULT_V6, **_MASTERS_V7_FEATURES}
_OUTPUT_DEFAULT_V7   = {
    **{k: v for k, v in _OUTPUT_DEFAULT_V6.items() if k != "doc_aggregates"},
    **_AGG_KEYS_V7,
    **_MASTERS_V7_OUTPUT,
}

DEFAULT_PREFS_V7: dict[str, Any] = {
    **DEFAULT_PREFS_V6,
    "schema_version": SCHEMA_VERSION_V7,
    "features": dict(_FEATURES_DEFAULT_V7),
    "output":   dict(_OUTPUT_DEFAULT_V7),
    "semantic": {
        **{k: v for k, v in DEFAULT_PREFS_V6["semantic"].items()
           if not (k.startswith("word_categories_") or k.startswith("doc_categories_"))},
        "categories_liwc":   list(_SOP_CATEGORIES_LIWC),
        "categories_empath": list(_SOP_CATEGORIES_EMPATH),
    },
}

# ---------------------------------------------------------------------------
# v8 constants (active schema)
#
# P-center batch (2026-05-20) changes vs v7:
#   - 2 new feature toggles: feat_syllable_pcenter, master_syllable_pcenter
#   - 1 new prosody param: pcenter_k (default 0.3)
# ---------------------------------------------------------------------------

SCHEMA_VERSION_V8 = 8

_FEATURES_DEFAULT_V8 = {
    **_FEATURES_DEFAULT_V7,
    "feat_syllable_pcenter":   True,
    "master_syllable_pcenter": True,
}

_OUTPUT_DEFAULT_V8 = dict(_OUTPUT_DEFAULT_V7)  # No output schema changes in v8

DEFAULT_PREFS_V8: dict[str, Any] = {
    **DEFAULT_PREFS_V7,
    "schema_version": SCHEMA_VERSION_V8,
    "features": dict(_FEATURES_DEFAULT_V8),
    "prosody": {**DEFAULT_PREFS_V7["prosody"], "pcenter_k": 0.3},
}

# ---------------------------------------------------------------------------
# v9 constants
#
# Sentence-level tier (2026-05-22) adds vs v8:
#   - 14 new feature toggles: feat_sentence_* (11 prosody + onset + end + text)
#     plus 2 master toggles: master_sentence_timing, master_sentence_prosody
#   - 1 new output toggle: sentences_csv
# ---------------------------------------------------------------------------

SCHEMA_VERSION_V9 = 9

_FEATURES_DEFAULT_V9 = {
    **_FEATURES_DEFAULT_V8,
    "feat_sentence_text":           True,
    "feat_sentence_onset":          True,
    "feat_sentence_end":            True,
    "feat_sentence_pitch_mean":     True,
    "feat_sentence_pitch_min":      True,
    "feat_sentence_pitch_max":      True,
    "feat_sentence_pitch_std":      True,
    "feat_sentence_pitch_delta":    True,
    "feat_sentence_volume_mean":    True,
    "feat_sentence_volume_min":     True,
    "feat_sentence_volume_max":     True,
    "feat_sentence_volume_std":     True,
    "feat_sentence_volume_delta":   True,
    "feat_sentence_hnr":            True,
    "master_sentence_timing":       True,
    "master_sentence_prosody":      True,
}

_OUTPUT_DEFAULT_V9 = {
    **_OUTPUT_DEFAULT_V8,
    "sentences_csv": True,
}

DEFAULT_PREFS_V9: dict[str, Any] = {
    **DEFAULT_PREFS_V8,
    "schema_version": SCHEMA_VERSION_V9,
    "features": dict(_FEATURES_DEFAULT_V9),
    "output":   dict(_OUTPUT_DEFAULT_V9),
}

# ---------------------------------------------------------------------------
# v10 constants
#
# Embeddings batch (2026-05-22) adds vs v9:
#   - 3 new feature master toggles: master_word2vec, master_bert_word,
#     master_bert_sentence
#   - 3 new output toggles: word2vec_mat, bert_word_mat, bert_sentence_mat
# ---------------------------------------------------------------------------

SCHEMA_VERSION_V10 = 10

_FEATURES_DEFAULT_V10 = {
    **_FEATURES_DEFAULT_V9,
    "master_word2vec":      True,
    "master_bert_word":     True,
    "master_bert_sentence": True,
}

_OUTPUT_DEFAULT_V10 = {
    **_OUTPUT_DEFAULT_V9,
    "word2vec_mat":      True,
    "bert_word_mat":     True,
    "bert_sentence_mat": True,
}

DEFAULT_PREFS_V10: dict[str, Any] = {
    **DEFAULT_PREFS_V9,
    "schema_version": SCHEMA_VERSION_V10,
    "features": dict(_FEATURES_DEFAULT_V10),
    "output":   dict(_OUTPUT_DEFAULT_V10),
}

# ---------------------------------------------------------------------------
# v11 constants
#
# LIWC removal (2026-06-10): the LIWC-22 EULA forbids using LIWC dictionary
# material outside LIWC's software, so LIWC is disabled as a shipped
# backend. No new keys — only the default semantic backend changes from
# "auto" to "empath", and the migration coerces any stored "liwc" backend
# to "empath" so no run silently reaches the disabled LIWC path. (The
# composite-psych scores are reimplemented license-clean and no longer
# depend on the backend at all.)
# ---------------------------------------------------------------------------

SCHEMA_VERSION_V11 = 11

DEFAULT_PREFS_V11: dict[str, Any] = {
    **DEFAULT_PREFS_V10,
    "schema_version": SCHEMA_VERSION_V11,
    "semantic": {
        **DEFAULT_PREFS_V10["semantic"],
        "backend": "empath",
    },
}

# ---------------------------------------------------------------------------
# v12 constants
#
# Lexical-norms batch (2026-06-10): per-word AoA / concreteness / imageability
# / familiarity / valence-arousal-dominance from user-supplied parquets.
# Opt-in (no data ships): master defaults OFF; per-column children default ON
# so flipping the master on yields all columns. Additive only.
# ---------------------------------------------------------------------------

SCHEMA_VERSION_V12 = 12

_NORM_FEATURE_KEYS = {
    "master_lexical_norms":   False,   # opt-in: no norm data ships
    "feat_norm_aoa_mean":     True,
    "feat_norm_aoa_sd":       True,
    "feat_norm_concreteness": True,
    "feat_norm_imageability": True,
    "feat_norm_familiarity":  True,
    "feat_norm_valence":      True,
    "feat_norm_arousal":      True,
    "feat_norm_dominance":    True,
}

# EWFG is disabled (its source is print-only / not digitized — see
# docs/features/edu_zipf.md). Default the grade-band toggles + the doc
# aggregate OFF so a fresh install / planner run does not emit all-NaN
# Edu_Zipf_* columns. (Existing user prefs keep their values; the GUI
# additionally greys these out.)
_EWFG_DISABLED_DEFAULTS = {
    "freq_ewfg_k2": False, "freq_ewfg_3to5": False, "freq_ewfg_6to8": False,
    "freq_ewfg_9to12": False, "freq_ewfg_overall": False,
}

_FEATURES_DEFAULT_V12 = {
    **_FEATURES_DEFAULT_V10, **_NORM_FEATURE_KEYS, **_EWFG_DISABLED_DEFAULTS,
}
_OUTPUT_DEFAULT_V12 = {
    **_OUTPUT_DEFAULT_V10,
    "doc_agg_lexical_norms": False,
    "doc_agg_edu_zipf": False,
}

DEFAULT_PREFS_V12: dict[str, Any] = {
    **DEFAULT_PREFS_V11,
    "schema_version": SCHEMA_VERSION_V12,
    "features": dict(_FEATURES_DEFAULT_V12),
    "output":   dict(_OUTPUT_DEFAULT_V12),
    "lexical_norms": {"paths": {}, "late_acquired_threshold": 10},
}

# ---------------------------------------------------------------------------
# v13 constants
#
# Acoustic level (2026-06-13): audio-only continuous TRF features emitted as a
# combined .mat per stimulus. Opt-in: the run/save gate is the `acoustic_ts_mat`
# OUTPUT key (toggled in the Run tab, like every other level), default OFF.
# Additive only.
# ---------------------------------------------------------------------------

SCHEMA_VERSION_V13 = 13

_ACOUSTIC_TS_FEATURE_KEYS = {
    # master_acoustic_ts is vestigial: the acoustic level is gated solely by the
    # acoustic_ts_mat output key (Run tab). Kept in the schema for v13 migration
    # compatibility and exempted in the schema-drift guard (cf. master_syllable_pcenter).
    "master_acoustic_ts":         False,
    "feat_acoustic_ts_envelope":  True,
    "feat_acoustic_ts_onset_env": True,
    "feat_acoustic_ts_logmel":    True,
    "feat_acoustic_ts_mfcc":      False,   # off by default
}

_FEATURES_DEFAULT_V13 = {**_FEATURES_DEFAULT_V12, **_ACOUSTIC_TS_FEATURE_KEYS}
# acoustic_ts_mat defaults OFF — acoustic is an opt-in audio-only level selected
# from the Run tab. (Other levels' output keys default ON.)
_OUTPUT_DEFAULT_V13 = {**_OUTPUT_DEFAULT_V12, "acoustic_ts_mat": False}

DEFAULT_PREFS_V13: dict[str, Any] = {
    **DEFAULT_PREFS_V12,
    "schema_version": SCHEMA_VERSION_V13,
    "features": dict(_FEATURES_DEFAULT_V13),
    "output":   dict(_OUTPUT_DEFAULT_V13),
    "acoustic_ts": {
        "analysis_rate": 128,
        "env_method": "rms",
        "env_comp": 0.3,
        "env_window": 1.0,
        "n_mels": 16,
        "n_mfcc": 13,
        "mel_preset": "speech",
    },
}

# ---------------------------------------------------------------------------
# v14 constants
#
# Structural-id rename (2026-07-06) for clarity — "term" and "token" were
# near-synonyms and easy to confuse:
#   feat_struct_term_id  (global whole-document index)  -> feat_struct_token_id
#   feat_struct_token_id (position within the sentence)  -> feat_struct_token_in_sentence_id
# The emitted CSV columns rename in lockstep (term_id -> token_id,
# token_id -> token_in_sentence_id); head_token_id now references the global
# token_id. Not additive — the migration below renames the keys in saved prefs.
# ---------------------------------------------------------------------------

SCHEMA_VERSION_V14 = 14

_FEATURES_DEFAULT_V14 = dict(_FEATURES_DEFAULT_V13)
# Pop both old keys before re-adding so the swap can't collide.
_v14_global_default = _FEATURES_DEFAULT_V14.pop("feat_struct_term_id", True)
_v14_insent_default = _FEATURES_DEFAULT_V14.pop("feat_struct_token_id", True)
_FEATURES_DEFAULT_V14["feat_struct_token_id"] = _v14_global_default
_FEATURES_DEFAULT_V14["feat_struct_token_in_sentence_id"] = _v14_insent_default

DEFAULT_PREFS_V14: dict[str, Any] = {
    **DEFAULT_PREFS_V13,
    "schema_version": SCHEMA_VERSION_V14,
    "features": dict(_FEATURES_DEFAULT_V14),
}

# ---------------------------------------------------------------------------
# v15 constants
#
# Opt-in defaults (2026-07-06): a fresh install / "Reset settings" produces the
# minimum — only the word-level features.csv. Every cluster *master* toggle
# stays ON (so the level cards are enabled and children can be switched on),
# every non-structural feature is OFF, and only the word_level_csv output is on.
# Users opt into the expensive levels (MFA-based acoustic/prosody, sub-word,
# doc, sentence, embeddings, GPT-2/LLM predictability) explicitly.
#
# Existing prefs are NOT force-flipped: the migration is additive, so a user's
# saved toggles are preserved; the new lean baseline reaches them only on a
# fresh install or via the Run-tab "Reset settings" button.
#
# Exception: master_acoustic_ts stays OFF — it is vestigial (no UI master; the
# acoustic level is gated solely by the acoustic_ts_mat output toggle), so
# flipping it on would be a no-op that contradicts its documented default.
# ---------------------------------------------------------------------------

SCHEMA_VERSION_V15 = 15

# `master_` keys that are NOT card-masters-with-children — the master IS the
# feature's on/off (its cluster has no child toggles), so treat it like any
# other feature and default it OFF. Leaving them on would show a checked GUI
# toggle whose output is off (nothing runs), which is exactly the confusing
# "BERT looks on but doesn't run" state.
_V15_FEATURE_MASTERS_OFF = frozenset({
    "master_acoustic_ts",   # vestigial: acoustic gated by the acoustic_ts_mat output
    "master_word2vec",      # embedding on/off (no child toggles)
    "master_bert_word",     # embedding on/off
    "master_bert_sentence", # embedding on/off
})


def _v15_feature_default(key: str) -> bool:
    if key in _V15_FEATURE_MASTERS_OFF:
        return False
    return key.startswith("master_") or key.startswith("feat_struct_")


def _v15_output_default(key: str) -> bool:
    return key.startswith("master_") or key == "word_level_csv"


_FEATURES_DEFAULT_V15 = {k: _v15_feature_default(k) for k in _FEATURES_DEFAULT_V14}
_OUTPUT_DEFAULT_V15 = {k: _v15_output_default(k) for k in DEFAULT_PREFS_V14["output"]}

DEFAULT_PREFS_V15: dict[str, Any] = {
    **DEFAULT_PREFS_V14,
    "schema_version": SCHEMA_VERSION_V15,
    "features": dict(_FEATURES_DEFAULT_V15),
    "output":   dict(_OUTPUT_DEFAULT_V15),
}

# ---------------------------------------------------------------------------
# v16 constants
#
# Phoneme surprisal + cohort entropy (2026-07-06). Additive. Card master
# defaults ON (v15 policy: card enabled, children off); feature toggles + the
# .mat output key default OFF (opt-in, like acoustic_ts_mat). New
# phoneme_surprisal config section holds analysis_sr + the functional flags.
# ---------------------------------------------------------------------------

SCHEMA_VERSION_V16 = 16

_FEATURES_DEFAULT_V16 = {
    **_FEATURES_DEFAULT_V15,
    "master_phoneme_surprisal": True,
    "feat_phoneme_surprisal": False,
    "feat_phoneme_cohort_entropy": False,
}
_OUTPUT_DEFAULT_V16 = {**DEFAULT_PREFS_V15["output"], "phoneme_surprisal_mat": False}

DEFAULT_PREFS_V16: dict[str, Any] = {
    **DEFAULT_PREFS_V15,
    "schema_version": SCHEMA_VERSION_V16,
    "features": dict(_FEATURES_DEFAULT_V16),
    "output":   dict(_OUTPUT_DEFAULT_V16),
    "phoneme_surprisal": {
        "analysis_sr": 128,
        "strip_stress": True,
        "primary_only": True,
    },
}

# ---------------------------------------------------------------------------
# v17 constants
#
# Release-hardening batch (2026-07-10). Two default corrections, no new keys:
#   - acoustic_ts.env_comp: 0.3 → log10(2) ≈ 0.30103. The envelope extractor
#     replicates mTRFenvelope.m bit-for-bit ONLY at exactly log10(2)
#     (see extractors/acoustic_ts.MTRF_COMP); the rounded 0.3 default made
#     every GUI/CLI run deviate from the documented replication claim.
#     Migration coerces a stored value of exactly 0.3 (the old default) from
#     any pre-v17 file; any other stored value is the user's and is kept.
#   - predictability.llm_model: stored OpenAI ids (gpt-*) are coerced to ""
#     (auto) — the OpenAI scoring endpoint no longer exists, so a saved gpt-*
#     selection could only fail at run time (see extractors/llm_cloze.py).
# ---------------------------------------------------------------------------

SCHEMA_VERSION_V17 = 17

# Exact mTRF power-law compression exponent. Kept as a literal here so
# utils.preferences never imports extractors; cross-module equality with
# extractors.acoustic_ts.MTRF_COMP is pinned by a migration test.
import math as _math
MTRF_ENV_COMP = float(_math.log10(2))

DEFAULT_PREFS_V17: dict[str, Any] = {
    **DEFAULT_PREFS_V16,
    "schema_version": SCHEMA_VERSION_V17,
    "acoustic_ts": {**DEFAULT_PREFS_V16["acoustic_ts"],
                    "env_comp": MTRF_ENV_COMP},
}

# ---------------------------------------------------------------------------
# Active schema aliases
# ---------------------------------------------------------------------------

SCHEMA_VERSION = SCHEMA_VERSION_V17  # active version
DEFAULT_PREFS: dict[str, Any] = DEFAULT_PREFS_V17  # active defaults


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def get_preferences_path() -> Path:
    override = os.environ.get('SENSI_PREFS_PATH')
    if override:
        return Path(override)
    project_root = Path(__file__).resolve().parent.parent
    return project_root / 'config' / 'preferences.json'


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_preferences() -> dict[str, Any]:
    """Load preferences; auto-migrate from any earlier schema version if needed."""
    path = get_preferences_path()
    merged: dict[str, Any] = deepcopy(DEFAULT_PREFS)

    if not path.exists():
        return merged
    try:
        with path.open('r', encoding='utf-8') as f:
            user = json.load(f)
    except (json.JSONDecodeError, OSError):
        return merged
    if not isinstance(user, dict):
        return merged

    version = user.get('schema_version', 1)
    # Repair leftover `document_categories_*` keys written by the
    # CategoryPicker before the doc/document mismatch was fixed. Move any
    # such entry into the canonical `doc_categories_*` slot so existing
    # selections aren't lost when the picker switches to the right key.
    user_sem = user.get('semantic') if isinstance(user.get('semantic'), dict) else None
    if user_sem is not None:
        for backend in ('liwc', 'empath'):
            wrong_key = f'document_categories_{backend}'
            right_key = f'doc_categories_{backend}'
            if wrong_key in user_sem:
                user_sem[right_key] = user_sem.pop(wrong_key)
    if version in (SCHEMA_VERSION_V10, SCHEMA_VERSION_V11,
                   SCHEMA_VERSION_V12, SCHEMA_VERSION_V13, SCHEMA_VERSION_V14,
                   SCHEMA_VERSION_V15, SCHEMA_VERSION_V16, SCHEMA_VERSION_V17):
        # v10–v15 are additive: missing keys are filled from the current
        # defaults (merged starts as DEFAULT_PREFS). v11's backend coercion and
        # the version stamp are applied in the finalize step below, which
        # rewrites the file for any source older than the current version.
        _deep_merge_current(merged, user)
    elif version == SCHEMA_VERSION_V9:
        _migrate_v9_into_v10(merged, user)
        _write(path, merged)
    elif version == SCHEMA_VERSION_V8:
        v9_intermediate = deepcopy(DEFAULT_PREFS_V9)
        _migrate_v8_into_v9(v9_intermediate, user)
        _migrate_v9_into_v10(merged, v9_intermediate)
        _write(path, merged)
    elif version == SCHEMA_VERSION_V7:
        v8_intermediate = deepcopy(DEFAULT_PREFS_V8)
        _migrate_v7_into_v8(v8_intermediate, user)
        v9_intermediate = deepcopy(DEFAULT_PREFS_V9)
        _migrate_v8_into_v9(v9_intermediate, v8_intermediate)
        _migrate_v9_into_v10(merged, v9_intermediate)
        _write(path, merged)
    elif version == SCHEMA_VERSION_V6:
        v7_intermediate = deepcopy(DEFAULT_PREFS_V7)
        _migrate_v6_into_v7(v7_intermediate, user)
        v8_intermediate = deepcopy(DEFAULT_PREFS_V8)
        _migrate_v7_into_v8(v8_intermediate, v7_intermediate)
        v9_intermediate = deepcopy(DEFAULT_PREFS_V9)
        _migrate_v8_into_v9(v9_intermediate, v8_intermediate)
        _migrate_v9_into_v10(merged, v9_intermediate)
        _write(path, merged)
    elif version == SCHEMA_VERSION_V5:
        v6_intermediate = deepcopy(DEFAULT_PREFS_V6)
        _migrate_v5_into_v6(v6_intermediate, user)
        v7_intermediate = deepcopy(DEFAULT_PREFS_V7)
        _migrate_v6_into_v7(v7_intermediate, v6_intermediate)
        v8_intermediate = deepcopy(DEFAULT_PREFS_V8)
        _migrate_v7_into_v8(v8_intermediate, v7_intermediate)
        v9_intermediate = deepcopy(DEFAULT_PREFS_V9)
        _migrate_v8_into_v9(v9_intermediate, v8_intermediate)
        _migrate_v9_into_v10(merged, v9_intermediate)
        _write(path, merged)
    elif version == SCHEMA_VERSION_V4:
        v5_intermediate = deepcopy(DEFAULT_PREFS_V5)
        _migrate_v4_into_v5(v5_intermediate, user)
        v6_intermediate = deepcopy(DEFAULT_PREFS_V6)
        _migrate_v5_into_v6(v6_intermediate, v5_intermediate)
        v7_intermediate = deepcopy(DEFAULT_PREFS_V7)
        _migrate_v6_into_v7(v7_intermediate, v6_intermediate)
        v8_intermediate = deepcopy(DEFAULT_PREFS_V8)
        _migrate_v7_into_v8(v8_intermediate, v7_intermediate)
        v9_intermediate = deepcopy(DEFAULT_PREFS_V9)
        _migrate_v8_into_v9(v9_intermediate, v8_intermediate)
        _migrate_v9_into_v10(merged, v9_intermediate)
        _write(path, merged)
    elif version == 3:
        v4_intermediate = deepcopy(DEFAULT_PREFS_V4)
        _migrate_v3_into_v4(v4_intermediate, user)
        v5_intermediate = deepcopy(DEFAULT_PREFS_V5)
        _migrate_v4_into_v5(v5_intermediate, v4_intermediate)
        v6_intermediate = deepcopy(DEFAULT_PREFS_V6)
        _migrate_v5_into_v6(v6_intermediate, v5_intermediate)
        v7_intermediate = deepcopy(DEFAULT_PREFS_V7)
        _migrate_v6_into_v7(v7_intermediate, v6_intermediate)
        v8_intermediate = deepcopy(DEFAULT_PREFS_V8)
        _migrate_v7_into_v8(v8_intermediate, v7_intermediate)
        v9_intermediate = deepcopy(DEFAULT_PREFS_V9)
        _migrate_v8_into_v9(v9_intermediate, v8_intermediate)
        _migrate_v9_into_v10(merged, v9_intermediate)
        _write(path, merged)
    elif version == 2:
        v3_intermediate = deepcopy(DEFAULT_PREFS_V3)
        _migrate_v2_into_v3(v3_intermediate, user)
        v4_intermediate = deepcopy(DEFAULT_PREFS_V4)
        _migrate_v3_into_v4(v4_intermediate, v3_intermediate)
        v5_intermediate = deepcopy(DEFAULT_PREFS_V5)
        _migrate_v4_into_v5(v5_intermediate, v4_intermediate)
        v6_intermediate = deepcopy(DEFAULT_PREFS_V6)
        _migrate_v5_into_v6(v6_intermediate, v5_intermediate)
        v7_intermediate = deepcopy(DEFAULT_PREFS_V7)
        _migrate_v6_into_v7(v7_intermediate, v6_intermediate)
        v8_intermediate = deepcopy(DEFAULT_PREFS_V8)
        _migrate_v7_into_v8(v8_intermediate, v7_intermediate)
        v9_intermediate = deepcopy(DEFAULT_PREFS_V9)
        _migrate_v8_into_v9(v9_intermediate, v8_intermediate)
        _migrate_v9_into_v10(merged, v9_intermediate)
        _write(path, merged)
    else:
        # v1 flat → v2 → v3 → v4 → v5 → v6 → v7 → v8 → v9 → v10
        v2_intermediate = deepcopy(_DEFAULT_PREFS_V2)
        _migrate_v1_into(v2_intermediate, user)
        v3_intermediate = deepcopy(DEFAULT_PREFS_V3)
        _migrate_v2_into_v3(v3_intermediate, v2_intermediate)
        v4_intermediate = deepcopy(DEFAULT_PREFS_V4)
        _migrate_v3_into_v4(v4_intermediate, v3_intermediate)
        v5_intermediate = deepcopy(DEFAULT_PREFS_V5)
        _migrate_v4_into_v5(v5_intermediate, v4_intermediate)
        v6_intermediate = deepcopy(DEFAULT_PREFS_V6)
        _migrate_v5_into_v6(v6_intermediate, v5_intermediate)
        v7_intermediate = deepcopy(DEFAULT_PREFS_V7)
        _migrate_v6_into_v7(v7_intermediate, v6_intermediate)
        v8_intermediate = deepcopy(DEFAULT_PREFS_V8)
        _migrate_v7_into_v8(v8_intermediate, v7_intermediate)
        v9_intermediate = deepcopy(DEFAULT_PREFS_V9)
        _migrate_v8_into_v9(v9_intermediate, v8_intermediate)
        _migrate_v9_into_v10(merged, v9_intermediate)
        _write(path, merged)

    # v10 → v11 finalize (applies to every source version): LIWC is disabled
    # under its EULA, so coerce any stored "liwc" backend to "empath" and
    # stamp the current schema version. Rewrite only when something changed
    # (i.e. the source was not already v11).
    sem = merged.get("semantic")
    if isinstance(sem, dict) and sem.get("backend") == "liwc":
        sem["backend"] = "empath"
    # v14 struct-id rename finalize (applies to every pre-v14 source). Any
    # merge above copies a pre-v14 file's keys wholesale, so at this point the
    # old within-sentence value sits in `feat_struct_token_id` and the old
    # global value in `feat_struct_term_id`. Recover the swap into the new keys.
    f = merged.get("features")
    if isinstance(f, dict) and "feat_struct_term_id" in f:
        f["feat_struct_token_in_sentence_id"] = f.get("feat_struct_token_id", True)
        f["feat_struct_token_id"] = f.pop("feat_struct_term_id")
    # v17 finalize (pre-v17 sources only — post-v17 values are the user's):
    #  (1) upgrade the old rounded env_comp default to the exact mTRF constant;
    #  (2) clear a stored OpenAI (gpt-*) llm_model — OpenAI cloze scoring no
    #      longer works, so an auto ("") selection picks a working local model
    #      instead of failing at run time.
    if isinstance(version, int) and version < SCHEMA_VERSION_V17:
        ac = merged.get("acoustic_ts")
        if isinstance(ac, dict) and ac.get("env_comp") == 0.3:
            ac["env_comp"] = MTRF_ENV_COMP
        pred = merged.get("predictability")
        if isinstance(pred, dict) and str(pred.get("llm_model", "")).startswith("gpt-"):
            pred["llm_model"] = ""
    if merged.get("schema_version") != SCHEMA_VERSION:
        merged["schema_version"] = SCHEMA_VERSION
        _write(path, merged)

    return merged


def save_preferences(updates: dict[str, Any]) -> None:
    """Merge `updates` into existing preferences and write to disk."""
    path = get_preferences_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    current = load_preferences()
    _deep_merge_current(current, updates)
    current['schema_version'] = SCHEMA_VERSION
    _write(path, current)


def reset_to_defaults(*, keep_shared: bool = True) -> dict[str, Any]:
    """Overwrite the preferences file with the shipped defaults.

    Every feature/output/config selection returns to DEFAULT_PREFS. By default
    the `shared` section (I/O folder paths, analyzer name, theme) is preserved
    so the user doesn't have to re-pick folders after a reset; pass
    keep_shared=False for a full wipe.

    The preferences file is local (gitignored), so this only affects this
    machine. Returns the fresh prefs dict.
    """
    path = get_preferences_path()
    fresh = deepcopy(DEFAULT_PREFS)
    if keep_shared:
        current = load_preferences()
        if isinstance(current.get("shared"), dict):
            fresh["shared"] = deepcopy(current["shared"])
    _write(path, fresh)
    return fresh


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _write(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


def _deep_merge_v3(into: dict, src: dict) -> None:
    """Merge `src` into `into` using v3 top-level shape.

    For known top-level dict sections (shared, features, predictability,
    semantic, output), merge their inner contents so unspecified keys keep
    defaults. For other keys, replace wholesale. Unknown keys from `src` are
    preserved (forward-compat).
    """
    _V3_SECTIONS = ('shared', 'features', 'predictability', 'semantic', 'output')
    for key, value in src.items():
        if key in _V3_SECTIONS and isinstance(value, dict):
            existing = into.get(key, {})
            if not isinstance(existing, dict):
                existing = {}
            existing.update(value)
            into[key] = existing
        else:
            into[key] = value


_CURRENT_SECTIONS = ('shared', 'features', 'predictability',
                     'semantic', 'output', 'prosody', 'lexical_norms',
                     'acoustic_ts', 'phoneme_surprisal')


def _deep_merge_current(into: dict, src: dict) -> None:
    """Merge a current-schema (v6) src dict into into.

    Top-level dict sections in _CURRENT_SECTIONS merge by inner key so
    unspecified keys keep defaults. Unknown keys from src are preserved
    wholesale (forward-compat).
    """
    for key, value in src.items():
        if key in _CURRENT_SECTIONS and isinstance(value, dict):
            existing = into.get(key, {})
            if not isinstance(existing, dict):
                existing = {}
            existing.update(value)
            into[key] = existing
        else:
            into[key] = value


def _deep_merge_sections(into: dict, src: dict) -> None:
    """Merge `src` into `into` using v2 top-level shape (shared, word_level,
    document_level). Kept for internal use during v1→v2 intermediate step.
    For known top-level dict keys, merge their inner contents so unspecified
    keys keep defaults. For other keys, replace wholesale.
    """
    for key, value in src.items():
        if key in ('shared', 'word_level', 'document_level') and isinstance(value, dict):
            existing = into.get(key, {})
            if not isinstance(existing, dict):
                existing = {}
            for sub_key, sub_value in value.items():
                if sub_key == 'feature_toggles' and isinstance(sub_value, dict):
                    base = dict(existing.get('feature_toggles', {}))
                    base.update(sub_value)
                    existing['feature_toggles'] = base
                else:
                    existing[sub_key] = sub_value
            into[key] = existing
        else:
            into[key] = value


def _migrate_v1_into(merged: dict, v1: dict) -> None:
    """Move v1 flat fields into the v2 nested layout in `merged`."""
    shared = merged['shared']
    word = merged['word_level']

    for key in ('analyzer_name', 'last_transcripts_folder',
                'last_audio_folder', 'last_output_folder',
                'dictionary_backend'):
        if key in v1:
            shared[key] = v1[key]

    if 'semantic_categories' in v1:
        word['semantic_categories'] = list(v1['semantic_categories'])
        merged['document_level']['semantic_categories'] = list(v1['semantic_categories'])
    if 'llm_model' in v1:
        word['llm_model'] = v1['llm_model']
    if isinstance(v1.get('feature_toggles'), dict):
        base = dict(_WORD_TOGGLES_DEFAULT)
        for k, v in v1['feature_toggles'].items():
            if k in base:
                base[k] = v
        word['feature_toggles'] = base


def _migrate_v2_into_v3(merged: dict, v2: dict) -> None:
    """Convert v2-shape `v2` into the v3-shape `merged` (mutates merged)."""
    v2_shared = v2.get("shared", {})
    v2_word = v2.get("word_level", {})
    v2_doc = v2.get("document_level", {})
    v2_word_toggles = v2_word.get("feature_toggles", {})
    v2_doc_toggles = v2_doc.get("feature_toggles", {})

    # shared
    for key in ("analyzer_name", "last_transcripts_folder",
                "last_audio_folder", "last_output_folder"):
        if key in v2_shared:
            merged["shared"][key] = v2_shared[key]

    # backend
    if "dictionary_backend" in v2_shared:
        merged["semantic"]["backend"] = v2_shared["dictionary_backend"]

    # features
    f = merged["features"]
    spacy_on = bool(v2_word_toggles.get("spacy_parse", True))
    for sub in ("structural", "lexling_tokens", "lexling_pos",
                "lexling_morph", "lexling_dep"):
        f[sub] = spacy_on
    f["freq_subtlex"] = bool(v2_word_toggles.get("zipf_freq", True))
    edu_on = bool(v2_word_toggles.get("edu_freq", True))
    for band in ("freq_ewfg_k2", "freq_ewfg_3to5", "freq_ewfg_6to8",
                 "freq_ewfg_9to12", "freq_ewfg_overall"):
        f[band] = edu_on
    f["pred_surprisal_gpt2"] = bool(v2_word_toggles.get("gpt2_surprisal", True))
    llm_on = bool(v2_word_toggles.get("llm_cloze", True))
    f["pred_surprisal_llm"] = llm_on
    f["pred_cloze"] = llm_on
    f["pred_topmatch"] = llm_on
    f["sem_indicators"] = bool(v2_word_toggles.get("sem_domain", True))
    f["acoustic_boundaries"] = bool(v2_word_toggles.get("alignment", True))
    # composite psych = any of the four liwc_* was on in v2
    f["sem_composite_psych"] = any(
        bool(v2_doc_toggles.get(k, False))
        for k in ("liwc_authentic", "liwc_analytic", "liwc_tone", "liwc_clout")
    )

    # predictability
    if "llm_model" in v2_word:
        merged["predictability"]["llm_model"] = v2_word["llm_model"]

    # semantic categories — route to whichever backend was active
    backend = merged["semantic"]["backend"]
    cats = v2_word.get("semantic_categories")
    if cats is not None:
        if backend == "liwc":
            merged["semantic"]["categories_liwc"] = list(cats)
        elif backend == "empath":
            merged["semantic"]["categories_empath"] = list(cats)
        else:  # auto — write to both, user can change later
            merged["semantic"]["categories_liwc"] = list(cats)
            merged["semantic"]["categories_empath"] = list(cats)

    # output
    o = merged["output"]
    o["word_level_csv"] = True
    o["document_summary"] = True
    o["doc_counts"] = bool(v2_doc_toggles.get("counts", True))
    o["doc_aggregates"] = any(
        bool(v2_doc_toggles.get(k, False))
        for k in ("mean_zipf_freq", "mean_edu_zipf",
                  "mean_surprisal", "mean_llm_cloze")
    )
    o["doc_lex_diversity"] = (
        bool(v2_doc_toggles.get("ttr", False))
        or bool(v2_doc_toggles.get("mtld", False))
    )
    o["doc_readability"] = (
        bool(v2_doc_toggles.get("flesch_reading_ease", False))
        or bool(v2_doc_toggles.get("flesch_kincaid_grade", False))
    )
    o["doc_category_proportions"] = bool(
        v2_doc_toggles.get("category_proportions", True)
    )


def _migrate_v3_into_v4(merged: dict, v3: dict) -> None:
    """Convert v3-shape `v3` into v4-shape `merged` (mutates merged).

    Changes:
      - structural (single bool) -> 8 feat_struct_* (all same value)
      - acoustic_boundaries (single bool) -> 4 feat_acoustic_* (all same value)
      - sem_composite_psych (single bool) -> 4 feat_psych_* (all same value)
      - output.doc_counts (single bool) -> 4 out_count_* (all same value)
      - semantic.categories_{liwc,empath} -> duplicated into both
        word_categories_* and doc_categories_* (so existing user
        selections start identical at word and doc level; they can
        diverge afterward).
    """
    # shared (unchanged keys)
    for k, v in v3.get("shared", {}).items():
        merged["shared"][k] = v

    # predictability (unchanged)
    for k, v in v3.get("predictability", {}).items():
        merged["predictability"][k] = v

    # semantic — backend stays; categories duplicated to both levels
    v3_sem = v3.get("semantic", {})
    merged["semantic"]["backend"] = v3_sem.get("backend", "auto")
    cats_liwc = v3_sem.get("categories_liwc", list(_SOP_CATEGORIES_LIWC))
    cats_empath = v3_sem.get("categories_empath", list(_SOP_CATEGORIES_EMPATH))
    merged["semantic"]["word_categories_liwc"] = list(cats_liwc)
    merged["semantic"]["doc_categories_liwc"] = list(cats_liwc)
    merged["semantic"]["word_categories_empath"] = list(cats_empath)
    merged["semantic"]["doc_categories_empath"] = list(cats_empath)

    # features — most keys unchanged; expand 3 collapsed parents
    v3_feat = v3.get("features", {})
    f = merged["features"]
    for k in (
        "lexling_tokens", "lexling_pos", "lexling_morph", "lexling_dep",
        "freq_subtlex",
        "freq_ewfg_k2", "freq_ewfg_3to5", "freq_ewfg_6to8",
        "freq_ewfg_9to12", "freq_ewfg_overall",
        "pred_surprisal_gpt2", "pred_surprisal_llm",
        "pred_cloze", "pred_topmatch",
        "sem_indicators",
    ):
        if k in v3_feat:
            f[k] = v3_feat[k]
    struct_on = bool(v3_feat.get("structural", True))
    for k in ("feat_struct_doc_id", "feat_struct_paragraph_id",
              "feat_struct_sentence_id", "feat_struct_sentence_text",
              "feat_struct_char_start", "feat_struct_char_end",
              "feat_struct_term_id", "feat_struct_token_id"):
        f[k] = struct_on
    acoustic_on = bool(v3_feat.get("acoustic_boundaries", True))
    for k in ("feat_acoustic_onset", "feat_acoustic_end",
              "feat_acoustic_word_duration", "feat_acoustic_word_gap"):
        f[k] = acoustic_on
    psych_on = bool(v3_feat.get("sem_composite_psych", True))
    for k in ("feat_psych_authentic", "feat_psych_analytic",
              "feat_psych_tone", "feat_psych_clout"):
        f[k] = psych_on

    # output — most keys unchanged; expand doc_counts
    v3_out = v3.get("output", {})
    o = merged["output"]
    for k in ("word_level_csv", "document_summary",
              "doc_aggregates", "doc_lex_diversity",
              "doc_readability", "doc_category_proportions"):
        if k in v3_out:
            o[k] = v3_out[k]
    counts_on = bool(v3_out.get("doc_counts", True))
    for k in ("out_count_tokens", "out_count_content_tokens",
              "out_count_sentences", "out_count_paragraphs"):
        o[k] = counts_on


def _migrate_v4_into_v5(merged: dict, v4: dict) -> None:
    """Convert v4-shape `v4` into v5-shape `merged` (mutates merged).
    Additive only — v5 defaults already include the new prosody keys.

    Note: any `prosody` section in the v4 source is intentionally not copied —
    v5 defaults are authoritative for newly-introduced sections.
    """
    for section in ("shared", "predictability", "semantic", "output"):
        for k, v in v4.get(section, {}).items():
            merged[section][k] = v
    for k, v in v4.get("features", {}).items():
        merged["features"][k] = v


def _migrate_v5_into_v6(merged: dict, v5: dict) -> None:
    """Convert v5-shape `v5` into v6-shape `merged` (mutates merged).
    Additive only — v6 defaults already include the new sub-word toggles
    and the new sidecar output keys.
    """
    for section in ("shared", "predictability", "semantic", "output", "prosody"):
        for k, v in v5.get(section, {}).items():
            merged[section][k] = v
    for k, v in v5.get("features", {}).items():
        merged["features"][k] = v


def _migrate_v6_into_v7(target: dict[str, Any], user_v6: dict[str, Any]) -> None:
    """v6 → v7: copy user's v6 state into the v7-shaped target, then
    union word/doc categories into the unified semantic.categories_<backend>
    keys and drop the split keys.

    Master toggle defaults (all True) come from the v7 defaults already
    populated into target — we only need to copy over user-set values
    where the user has them, then they will be preserved by _deep_merge_current.
    """
    _deep_merge_current(target, user_v6)
    sem = target.setdefault("semantic", {})
    # Read the user's pre-existing unified value (if any) directly from the
    # v6 input, distinct from any SOP default that target may carry. A v6
    # file with both the split keys AND a forward-compat unified key should
    # not silently lose the unified value.
    user_sem_in = user_v6.get("semantic", {}) if isinstance(user_v6.get("semantic"), dict) else {}
    for backend in ("liwc", "empath"):
        word_key = f"word_categories_{backend}"
        doc_key  = f"doc_categories_{backend}"
        unified  = f"categories_{backend}"
        word_cats = sem.get(word_key, []) or []
        doc_cats  = sem.get(doc_key,  []) or []
        if word_cats or doc_cats:
            # Lossless union (preserve order of first appearance). Include
            # any unified value the user actually set in the v6 file (not
            # the SOP default that target may carry).
            user_unified = user_sem_in.get(unified, []) or []
            sem[unified] = list(dict.fromkeys([*user_unified, *word_cats, *doc_cats]))
        # If neither was set explicitly, leave the v7 default categories_<backend>
        # already in target (DEFAULT_PREFS_V7 populated SOP defaults).
        sem.pop(word_key, None)
        sem.pop(doc_key, None)
    target["schema_version"] = SCHEMA_VERSION_V7


def _migrate_v7_into_v8(merged: dict, v7: dict) -> None:
    """v7 → v8: additive only.

    v8 defaults (already in merged) supply pcenter_k=0.3 and the two new
    feature toggles. Any forward-compat value the user set in the v7 file
    is preserved by deep-merging their data over the defaults.
    """
    _deep_merge_current(merged, v7)
    merged["schema_version"] = SCHEMA_VERSION_V8


def _migrate_v8_into_v9(merged: dict, v8: dict) -> None:
    """v8 → v9: additive only.

    v9 defaults (already in merged) supply all sentence-level keys.
    """
    _deep_merge_current(merged, v8)
    merged["schema_version"] = SCHEMA_VERSION_V9


def _migrate_v9_into_v10(merged: dict, v9: dict) -> None:
    """v9 → v10: additive only.

    v10 defaults (already in merged) supply all embedding keys.
    """
    _deep_merge_current(merged, v9)
    merged["schema_version"] = SCHEMA_VERSION_V10
