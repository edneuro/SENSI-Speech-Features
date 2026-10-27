"""Load and save user preferences.

Schema (v21 — active; DEFAULT_PREFS_V21 is the authoritative key set):
    {
      "schema_version": 21,
      "shared":  {analyzer_name, last_*_folder, theme, hf_token_file},
      "features": {per-column toggles (feat_*, lexling_*, freq_*, sem_*),
                   master_* section switches (one per feature card, incl.
                   master_semantic_indicators / _similarity,
                   master_sentence_similarity), master_word2vec /
                   master_bert_word / master_bert_sentence (embedding ticks),
                   feat_phoneme_surprisal_mat, vestigial master_acoustic_ts},
      "predictability": {models, metrics, context},
      "semantic": {backend, categories_liwc, categories_empath},
      "output":   {level_word, level_document, level_subword, level_sentence,
                   level_acoustic (Run-tab level switches), out_count_*,
                   doc_* toggles, master_doc_* section switches},
      "prosody":  {pitch_units, pitch_floor_hz, pitch_ceiling_hz, pause_threshold_s, pcenter_k},
      "lexical_norms": {paths, late_acquired_threshold},
      "acoustic_ts": {analysis_rate, env_method, env_comp, env_window, n_mels, n_mfcc, mel_preset},
      "phoneme_surprisal": {analysis_sr, strip_stress, primary_only},
      "alignment": {g2p_oov}
    }

Migration chain (all paths terminate at the current version):
  v1 → v2 → v3 → v4 → v5 → v6 → v7 → v8 → v9 → v10
  v10..current files are deep-merged against the current defaults (additive).
  Finalize steps applied to any older source: v11 coerces liwc→empath,
    v14 recovers the struct-id rename, v17 upgrades the old rounded
    acoustic_ts.env_comp default (0.3) to the exact mTRF constant log10(2)
    and coerces stored OpenAI (gpt-*) llm_model to "" (auto).
    Later removal-migrations: v18 predictability unification, v20 phoneme
    label split, v21 per-file output toggles → output.level_* switches
    (_migrate_outputs_to_levels).
  Later additive versions (v12 lexical-norms, v13 acoustic_ts, v15 opt-in
    defaults, v16 phoneme surprisal, v17 above) only add keys / correct
    defaults; missing keys are filled from the current defaults.

Default location: <project_root>/config/{features.yaml,run.yaml} — a two-file
    YAML store (Project C). `features.yaml` holds schema_version/features/
    output/predictability/semantic/prosody/... (global, reset-able);
    `run.yaml` holds `shared` (machine-specific folders, analyzer name,
    theme, token-file paths). A legacy `preferences.json` is imported once
    if neither YAML file exists yet (see `_load_legacy_json`).
Override the directory holding both files via SENSI_CONFIG_DIR, or (for
    hermetic tests that only know the old single-file var) via
    SENSI_PREFS_PATH's parent directory — SENSI_CONFIG_DIR wins if both are
    set. SENSI_PREFS_PATH also independently overrides the legacy json path
    read by `_load_legacy_json`.
"""

import json
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

from sensi.paths import CONFIG_DIR_DEFAULT
from sensi.utils.config_yaml import (FEATURES_YAML, RUN_YAML, read_yaml,
                               split_prefs, join_yaml_pair,
                               write_features_yaml, write_run_yaml)

# Config files already reported as unparseable this session (warn once each).
_WARNED_UNPARSEABLE: set[str] = set()

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
    "doc_agg_predictability": True,
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
    # master_acoustic_ts is vestigial: the acoustic level is gated solely by its
    # Run-tab level switch (output.level_acoustic since v21; acoustic_ts_mat before). Kept in the schema for v13 migration
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
# acoustic level is gated solely by its Run-tab level switch,
# output.level_acoustic since v21 / acoustic_ts_mat before), so
# flipping it on would be a no-op that contradicts its documented default.
# ---------------------------------------------------------------------------

SCHEMA_VERSION_V15 = 15

# `master_` keys that are NOT card-masters-with-children — the master IS the
# feature's on/off (its cluster has no child toggles), so treat it like any
# other feature and default it OFF. Leaving them on would show a checked GUI
# toggle whose output is off (nothing runs), which is exactly the confusing
# "BERT looks on but doesn't run" state.
_V15_FEATURE_MASTERS_OFF = frozenset({
    "master_acoustic_ts",   # vestigial: acoustic gated by output.level_acoustic
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
#     (see sensi/extractors/acoustic_ts.MTRF_COMP); the rounded 0.3 default made
#     every GUI/CLI run deviate from the documented replication claim.
#     Migration coerces a stored value of exactly 0.3 (the old default) from
#     any pre-v17 file; any other stored value is the user's and is kept.
#   - predictability.llm_model: stored OpenAI ids (gpt-*) are coerced to ""
#     (auto) — the OpenAI scoring endpoint no longer exists, so a saved gpt-*
#     selection could only fail at run time (see sensi/extractors/llm_cloze.py).
# ---------------------------------------------------------------------------

SCHEMA_VERSION_V17 = 17

# Exact mTRF power-law compression exponent. Kept as a literal here so
# sensi.utils.preferences never imports extractors; cross-module equality with
# sensi.extractors.acoustic_ts.MTRF_COMP is pinned by a migration test.
import math as _math
MTRF_ENV_COMP = float(_math.log10(2))

DEFAULT_PREFS_V17: dict[str, Any] = {
    **DEFAULT_PREFS_V16,
    "schema_version": SCHEMA_VERSION_V17,
    "acoustic_ts": {**DEFAULT_PREFS_V16["acoustic_ts"],
                    "env_comp": MTRF_ENV_COMP},
}

# ---------------------------------------------------------------------------
# v18 constants
#
# Predictability model/metric unification (2026-07-11): model selection moves
# out of the four pred_* feature toggles (pred_surprisal_gpt2, pred_surprisal_llm,
# pred_cloze, pred_topmatch) and predictability.llm_model into a single
# `predictability` settings section: {"models": [<slug>, ...], "metrics":
# {"surprisal": bool, "cloze": bool, "topmatch": bool}}. Model slugs come from
# sensi.extractors.predictability_models (the roster module — pure, no torch).
#
# Not additive — the migration (in the finalize block below) removes the four
# old feature keys and predictability.llm_model, mapping their old on/off
# state forward onto model slugs + metrics (precedent: the v14 struct-id
# rename, also a removal-migration handled in a finalize block rather than a
# per-version step).
#
# Fresh-install default (v15 opt-in policy): predictability.models == [] AND
# all metrics off — predictability is fully off until the user opts in. A
# pre-checked metric with no model is contradictory (can never produce a
# column) and would fire a day-one "metric on, no model" preflight warning on
# an untouched install; the "models selected but no metric" warning guides the
# natural pick-a-model-then-a-metric workflow instead.
# ---------------------------------------------------------------------------

SCHEMA_VERSION_V18 = 18

_FEATURES_DEFAULT_V18 = {
    k: v for k, v in DEFAULT_PREFS_V17["features"].items()
    if k not in ("pred_surprisal_gpt2", "pred_surprisal_llm",
                 "pred_cloze", "pred_topmatch")
}

DEFAULT_PREFS_V18: dict[str, Any] = {
    **DEFAULT_PREFS_V17,
    "schema_version": SCHEMA_VERSION_V18,
    "features": dict(_FEATURES_DEFAULT_V18),
    "predictability": {
        "models": [],
        "metrics": {"surprisal": False, "cloze": False, "topmatch": False},
    },
}

# ---------------------------------------------------------------------------
# v19 constants
#
# Per-model predictability context size (Project B). Additive: a `context`
# map (slug → max context tokens) is added to the predictability section. An
# absent slug means "use the model's native ceiling", so an empty map = every
# model at its native window = identical to v18 behavior. No value transforms.
# ---------------------------------------------------------------------------

SCHEMA_VERSION_V19 = 19

DEFAULT_PREFS_V19: dict[str, Any] = {
    **DEFAULT_PREFS_V18,
    "schema_version": SCHEMA_VERSION_V19,
    # hf_token_file added 2026-07-12 (Project C, run.yaml): path to a file
    # holding a HuggingFace token for headless runs. Additive — no bump.
    # Defaults to the conventional gitignored config/hf_token.txt (relative
    # values resolve against get_config_dir(); see sensi/utils/credentials.py).
    "shared": {**DEFAULT_PREFS_V18["shared"], "hf_token_file": "hf_token.txt"},
    "predictability": {
        **DEFAULT_PREFS_V18["predictability"],
        "context": {},
    },
}

# ---------------------------------------------------------------------------
# v20 constants
#
# Phoneme label split + IPA + OOV coverage. REMOVAL-migration (precedent: v14's
# struct-id rename, v18's predictability unification): the fused
# `feat_phoneme_label` toggle is replaced by four per-column toggles, because
# the phoneme CSV's `Label` column (ARPABET + stress digit fused, e.g. 'EY2')
# becomes ARPABET / Stress / IPA / Pron_Source.
#
# Additive alongside it: `feat_syllable_ipa` (opt-in verbose column, matching
# feat_syllable_arpabet) and a new `alignment` section holding `g2p_oov`
# (opt-in MFA G2P for out-of-dictionary words — off by default so no run
# silently gains modeled pronunciations).
# ---------------------------------------------------------------------------

SCHEMA_VERSION_V20 = 20

_FEATURES_DEFAULT_V20 = {
    k: v for k, v in DEFAULT_PREFS_V19["features"].items()
    if k != "feat_phoneme_label"
}
_FEATURES_DEFAULT_V20.update({
    "feat_phoneme_arpabet":     False,
    "feat_phoneme_stress":      False,
    "feat_phoneme_ipa":         False,
    "feat_phoneme_pron_source": False,
    "feat_syllable_ipa":        False,
})

DEFAULT_PREFS_V20: dict[str, Any] = {
    **DEFAULT_PREFS_V19,
    "schema_version": SCHEMA_VERSION_V20,
    "features": dict(_FEATURES_DEFAULT_V20),
    "alignment": {"g2p_oov": False},
}

# ---------------------------------------------------------------------------
# v21 constants
#
# Output switches. REMOVAL-migration (precedent: v14, v18, v20): the ten
# per-file output toggles (word_level_csv, document_summary, syllables_csv,
# phonemes_csv, sentences_csv, acoustic_ts_mat, phoneme_surprisal_mat,
# word2vec_mat, bert_word_mat, bert_sentence_mat) are replaced by five
# per-level switches (output.level_*). What used to be a file toggle becomes
# a section switch on the level's page: new section masters for the doc page
# (output.master_doc_*), the word page's semantic cards, the sentence page's
# BERT card, and a per-file `feat_phoneme_surprisal_mat` toggle (the .mat is
# a separate file within the sub-word level). The migration
# (_migrate_outputs_to_levels) maps a v20 file onto these so the files a run
# writes stay identical.
# ---------------------------------------------------------------------------

SCHEMA_VERSION_V21 = 21

# The v20 output keys retired by v21 (also listed in
# sensi.utils.config_yaml.RETIRED_FEATURE_KEYS so a stale features.yaml validates).
RETIRED_OUTPUT_KEYS_V21 = (
    "word_level_csv", "document_summary", "syllables_csv", "phonemes_csv",
    "sentences_csv", "acoustic_ts_mat", "phoneme_surprisal_mat",
    "word2vec_mat", "bert_word_mat", "bert_sentence_mat",
)

_OUTPUT_DEFAULT_V21 = {
    # v15 opt-in policy: only the word level is on for a fresh install.
    "level_word":     True,
    "level_document": False,
    "level_subword":  False,
    "level_sentence": False,
    "level_acoustic": False,
    **{k: v for k, v in DEFAULT_PREFS_V20["output"].items()
       if k not in RETIRED_OUTPUT_KEYS_V21},
    # Doc-page section switches (card masters: ON, children off).
    "master_doc_aggregates":           True,
    "master_doc_lex_diversity":        True,
    "master_doc_readability":          True,
    "master_doc_category_proportions": True,
}

_FEATURES_DEFAULT_V21 = {
    **_FEATURES_DEFAULT_V20,
    "master_semantic_indicators": True,
    "master_semantic_similarity": True,
    "master_sentence_similarity": True,
    "feat_phoneme_surprisal_mat": False,
}

DEFAULT_PREFS_V21: dict[str, Any] = {
    **DEFAULT_PREFS_V20,
    "schema_version": SCHEMA_VERSION_V21,
    "features": dict(_FEATURES_DEFAULT_V21),
    "output":   dict(_OUTPUT_DEFAULT_V21),
}


def _migrate_outputs_to_levels(merged: dict, old_output: dict) -> None:
    """v20 → v21: map the ten retired per-file output toggles onto the five
    output.level_* switches + section switches, so a migrated config writes
    exactly the files it wrote before. Mutates `merged`; a key missing from
    `old_output` takes its v20 default. Removes the retired keys.
    """
    v20_out = DEFAULT_PREFS_V20["output"]

    def old(key: str) -> bool:
        return bool(old_output.get(key, v20_out[key]))

    wl, ds = old("word_level_csv"), old("document_summary")
    syl, ph = old("syllables_csv"), old("phonemes_csv")
    psrm, sent = old("phoneme_surprisal_mat"), old("sentences_csv")
    ac = old("acoustic_ts_mat")
    w2v, bw, bs = old("word2vec_mat"), old("bert_word_mat"), old("bert_sentence_mat")

    o = merged.setdefault("output", {})
    f = merged.setdefault("features", {})
    o["level_word"] = wl or w2v or bw
    o["level_document"] = ds
    o["level_subword"] = syl or ph or psrm
    o["level_sentence"] = sent or bs
    o["level_acoustic"] = ac

    f["master_word2vec"] = bool(f.get("master_word2vec", False)) and w2v
    f["master_bert_word"] = bool(f.get("master_bert_word", False)) and bw
    f["master_bert_sentence"] = bool(f.get("master_bert_sentence", False)) and bs
    f["feat_phoneme_surprisal_mat"] = psrm
    # v20's P-center and phoneme-surprisal .mat clusters had no section
    # master (P_Center rode on the syllables CSV, the .mat on its output key
    # alone); v21 gates both on a section switch, so force it on where the
    # v20 config wrote the artifact.
    if syl:
        f["master_syllable_pcenter"] = True
    psr_master_was_off = not bool(f.get("master_phoneme_surprisal", True))
    if psrm and psr_master_was_off:
        # The .mat was written although the CSV-column section was off:
        # turn the section on for the .mat, untick its CSV columns so the
        # phonemes CSV still omits them.
        f["master_phoneme_surprisal"] = True
        f["feat_phoneme_surprisal"] = False
        f["feat_phoneme_cohort_entropy"] = False

    off: list[str] = []
    if o["level_word"] and not wl:
        off += ["master_structural", "master_lexling", "master_frequency",
                "master_predictability", "master_lexical_norms",
                "master_acoustic", "master_word_prosody",
                "master_semantic_indicators"]
    if o["level_subword"] and not syl:
        off += ["master_syllable_intervals", "master_syllable_labels",
                "master_syllable_prosody", "master_syllable_pcenter"]
    if o["level_subword"] and not ph:
        off.append("master_phoneme_intervals")
        if psrm:
            # The section stays on for the .mat; its CSV-column ticks would
            # otherwise make the (unwanted) phonemes CSV live.
            off += ["feat_phoneme_surprisal", "feat_phoneme_cohort_entropy"]
        else:
            off.append("master_phoneme_surprisal")  # keep the user's ticks
    if o["level_sentence"] and not sent:
        off += ["master_sentence_timing", "master_sentence_prosody"]
    for key in off:
        f[key] = False

    for key in RETIRED_OUTPUT_KEYS_V21:
        o.pop(key, None)

# ---------------------------------------------------------------------------
# Active schema aliases
# ---------------------------------------------------------------------------

SCHEMA_VERSION = SCHEMA_VERSION_V21  # active version
DEFAULT_PREFS: dict[str, Any] = DEFAULT_PREFS_V21  # active defaults


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def get_preferences_path() -> Path:
    """LEGACY json path — only read for the one-time import to the YAML pair."""
    override = os.environ.get('SENSI_PREFS_PATH')
    if override:
        return Path(override)
    return get_config_dir() / 'preferences.json'


def get_config_dir() -> Path:
    """Directory holding features.yaml + run.yaml.

    Precedence: SENSI_CONFIG_DIR > parent of SENSI_PREFS_PATH (so the many
    existing test isolation sites that only set the legacy var stay hermetic)
    > <project_root>/config.
    """
    override = os.environ.get('SENSI_CONFIG_DIR')
    if override:
        return Path(override)
    legacy = os.environ.get('SENSI_PREFS_PATH')
    if legacy:
        return Path(legacy).parent
    return CONFIG_DIR_DEFAULT


def get_features_yaml_path() -> Path:
    return get_config_dir() / FEATURES_YAML


def get_run_yaml_path() -> Path:
    return get_config_dir() / RUN_YAML


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _merged_from_user(user: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Run the full v1->v21 migration/merge/finalize chain on a parsed prefs dict.

    Pure: performs no disk I/O. Returns (merged, changed) where `changed` is
    True iff the source `schema_version` differed from SCHEMA_VERSION (i.e.
    the caller should persist the result).
    """
    merged: dict[str, Any] = deepcopy(DEFAULT_PREFS)

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
                   SCHEMA_VERSION_V15, SCHEMA_VERSION_V16, SCHEMA_VERSION_V17,
                   SCHEMA_VERSION_V18, SCHEMA_VERSION_V19, SCHEMA_VERSION_V20,
                   SCHEMA_VERSION_V21):
        # v10–v15 are additive: missing keys are filled from the current
        # defaults (merged starts as DEFAULT_PREFS). v11's backend coercion and
        # the version stamp are applied in the finalize step below, which
        # rewrites the file for any source older than the current version.
        _deep_merge_current(merged, user)
    elif version == SCHEMA_VERSION_V9:
        _migrate_v9_into_v10(merged, user)
    elif version == SCHEMA_VERSION_V8:
        v9_intermediate = deepcopy(DEFAULT_PREFS_V9)
        _migrate_v8_into_v9(v9_intermediate, user)
        _migrate_v9_into_v10(merged, v9_intermediate)
    elif version == SCHEMA_VERSION_V7:
        v8_intermediate = deepcopy(DEFAULT_PREFS_V8)
        _migrate_v7_into_v8(v8_intermediate, user)
        v9_intermediate = deepcopy(DEFAULT_PREFS_V9)
        _migrate_v8_into_v9(v9_intermediate, v8_intermediate)
        _migrate_v9_into_v10(merged, v9_intermediate)
    elif version == SCHEMA_VERSION_V6:
        v7_intermediate = deepcopy(DEFAULT_PREFS_V7)
        _migrate_v6_into_v7(v7_intermediate, user)
        v8_intermediate = deepcopy(DEFAULT_PREFS_V8)
        _migrate_v7_into_v8(v8_intermediate, v7_intermediate)
        v9_intermediate = deepcopy(DEFAULT_PREFS_V9)
        _migrate_v8_into_v9(v9_intermediate, v8_intermediate)
        _migrate_v9_into_v10(merged, v9_intermediate)
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
    # v18 finalize (pre-v18 sources only — post-v18 values are the user's):
    # predictability model/metric unification. The four pred_* feature
    # toggles and predictability.llm_model are replaced by
    # predictability.models (a list of model slugs) and predictability.metrics
    # ({surprisal, cloze, topmatch}). Map the old on/off state forward:
    #   - pred_surprisal_gpt2 on -> add "gpt2" to models, metrics.surprisal=True
    #   - any of pred_surprisal_llm/pred_cloze/pred_topmatch on -> add the
    #     stored llm_model's slug (default "llama32_3b" if unset/unknown) to
    #     models, and set the matching metric(s).
    # Not additive — the four feature keys and llm_model are removed, not
    # merely filled from defaults (precedent: the v14 struct-id rename).
    #
    # Same version gate also folds in the doc-aggregate residue: the v7-era
    # doc_agg_surprisal / doc_agg_llm output toggles are unified into a
    # single doc_agg_predictability toggle (see sensi/utils/doc_summary.py's
    # predictability_means / sensi/gui/planner/registry.py's "doc agg
    # predictability" cluster). Only acts when a pre-v18 source file literally
    # has one of the old keys (sentinel None below) — pre-v10 chains never
    # carry these names forward as real user data (only as v7-template
    # defaults already reconciled by the additive merge), so this must not
    # clobber that already-correct value with a spurious False.
    if isinstance(version, int) and version < SCHEMA_VERSION_V18:
        out = merged.get("output", {})
        surp_on = out.pop("doc_agg_surprisal", None)
        llm_on = out.pop("doc_agg_llm", None)
        if surp_on is not None or llm_on is not None:
            out["doc_agg_predictability"] = bool(surp_on) or bool(llm_on)
    if isinstance(version, int) and version < SCHEMA_VERSION_V18:
        from sensi.extractors.predictability_models import slug_for_hf_id
        f = merged.get("features", {})
        pred = merged.setdefault("predictability", {})
        models = list(pred.get("models") or [])
        metrics = pred.setdefault("metrics", {"surprisal": False, "cloze": False, "topmatch": False})
        gpt2_on = f.pop("pred_surprisal_gpt2", False)
        llm_on = f.pop("pred_surprisal_llm", False)
        cloze_on = f.pop("pred_cloze", False)
        top_on = f.pop("pred_topmatch", False)
        if gpt2_on and "gpt2" not in models:
            models.append("gpt2")
        if llm_on or cloze_on or top_on:
            slug = slug_for_hf_id(pred.get("llm_model", "")) or "llama32_3b"
            if slug not in models:
                models.append(slug)
        # Metrics reflect ONLY the old toggles — hard-assigned over the v18
        # defaults (whose surprisal defaults True), so e.g. a cloze-only user
        # is not silently given surprisal. surprisal = gpt2 OR llm surprisal.
        metrics["surprisal"] = gpt2_on or llm_on
        metrics["cloze"] = cloze_on
        metrics["topmatch"] = top_on
        pred["models"] = models
        pred.pop("llm_model", None)

    # v20 finalize (pre-v20 sources only — post-v20 values are the user's):
    # the fused phoneme `Label` column is split, so its single toggle becomes
    # four. Carry the old on/off state to all four so a user who had the
    # column off keeps every replacement off. Not additive — the key is
    # removed, not merely defaulted (precedent: v14, v18).
    if isinstance(version, int) and version < SCHEMA_VERSION_V20:
        f = merged.get("features", {})
        label_on = f.pop("feat_phoneme_label", None)
        if label_on is not None:
            for key in ("feat_phoneme_arpabet", "feat_phoneme_stress",
                        "feat_phoneme_ipa", "feat_phoneme_pron_source"):
                f[key] = bool(label_on)

    # v21 finalize (pre-v21 sources only — post-v21 values are the user's):
    # the ten per-file output toggles become five output.level_* switches +
    # section switches. Every merge above copies the source's output keys
    # wholesale, so the old toggles sit in merged["output"] here (a key the
    # source never had takes its v20 default inside the helper).
    if isinstance(version, int) and version < SCHEMA_VERSION_V21:
        _migrate_outputs_to_levels(merged, dict(merged.get("output", {})))

    changed = merged.get("schema_version") != SCHEMA_VERSION
    merged["schema_version"] = SCHEMA_VERSION
    return merged, changed


def _read_yaml_tolerant(path: Path) -> tuple[dict[str, Any], bool]:
    """Best-effort read of one half of the YAML pair.

    Returns (data, parsed_ok). `parsed_ok` is False only when `path` exists
    but could not be parsed (malformed YAML or an OS-level read error) — the
    caller uses this to know it must not overwrite that file. A missing or
    legitimately empty file is not a parse failure: it returns ({}, True).
    """
    import yaml as _yaml
    try:
        return read_yaml(path), True
    except (_yaml.YAMLError, OSError):
        return {}, False


def _write_yaml_pair(merged: dict[str, Any], *,
                      write_features: bool = True,
                      write_run: bool = True) -> None:
    """Write the halves of `merged` back to disk.

    write_features/write_run let the caller refuse to rewrite a half whose
    on-disk file could not be parsed in the first place (see load_preferences)
    — overwriting a file we never understood would destroy the only evidence
    a user has for diagnosing their own corrupted config.
    """
    features_data, run_data = split_prefs(merged)
    if write_features:
        write_features_yaml(get_features_yaml_path(), features_data)
    if write_run:
        write_run_yaml(get_run_yaml_path(), run_data)


def load_preferences() -> dict[str, Any]:
    """Load preferences from the YAML pair; one-time import from legacy json."""
    f_path, r_path = get_features_yaml_path(), get_run_yaml_path()
    if f_path.exists() or r_path.exists():
        f_data, f_ok = _read_yaml_tolerant(f_path)
        r_data, r_ok = _read_yaml_tolerant(r_path)
        user = join_yaml_pair(f_data, r_data)
        if get_preferences_path().exists() and (
                not user or not (f_path.exists() and r_path.exists())):
            # Half-import window: a crash (or torn write) between writing
            # features.yaml and run.yaml can leave the pair incomplete — one
            # file missing, empty, or unparseable — while a legacy
            # preferences.json still sits on disk. A COMPLETED import always
            # renames the json to .bak, so "json present + incomplete pair"
            # can only mean an interrupted import (or hand-tampering): rather
            # than silently loading defaults for the missing half and
            # orphaning the json forever, fall through and (re)run the import
            # so the pair gets completed.
            return _load_legacy_json()
        # Warn once per file per session: load_preferences runs on every
        # GUI refresh, which printed the same warning dozens of times.
        for path_, ok_ in ((f_path, f_ok), (r_path, r_ok)):
            if not ok_ and str(path_) not in _WARNED_UNPARSEABLE:
                _WARNED_UNPARSEABLE.add(str(path_))
                print(f"SensiExtractor: warning: {path_} could not be parsed "
                      "as YAML — it will be left unchanged on disk and this "
                      "session falls back to defaults for its settings.",
                      file=sys.stderr)
        # The YAML store began at v19; a missing/hand-stripped schema_version
        # must not fall into the v1 flat-json migration, so a NON-EMPTY
        # features half (real section content, just missing/miscast the
        # version key) is assumed to be v19 (sectioned-shaped), NOT the
        # current SCHEMA_VERSION. Assuming "current" would skip every pending
        # finalize block below (v20's feat_phoneme_label removal in
        # particular), letting a retired key survive forever with `changed`
        # False so the file is never rewritten to self-heal. A quoted version
        # (`schema_version: '19'`) is coerced to int the same way.
        # This is deliberately based on f_data (the parsed features.yaml
        # payload alone), NOT the joined `user` dict: run.yaml contributing a
        # non-empty `shared` section must never make an emptied/corrupted
        # features.yaml look like it has content to migrate — see
        # test_corrupted_features_yaml_with_valid_run_yaml_is_not_overwritten.
        has_content = bool(f_data.keys() - {'schema_version'})
        assumed_version = SCHEMA_VERSION_V19 if has_content else SCHEMA_VERSION
        raw_version = user.get('schema_version')
        if raw_version is None:
            user['schema_version'] = assumed_version
        else:
            try:
                user['schema_version'] = int(raw_version)
            except (TypeError, ValueError):
                user['schema_version'] = assumed_version
        merged, changed = _merged_from_user(user)
        if changed:
            try:
                # Never rewrite a half we could not parse in the first
                # place — that would silently destroy a corrupted file the
                # user needs to diagnose their own settings.
                _write_yaml_pair(merged, write_features=f_ok, write_run=r_ok)
            except OSError:
                pass  # read-only config dir: settings still load for the
                      # session, persistence just doesn't happen
        return merged
    return _load_legacy_json()


def _load_legacy_json() -> dict[str, Any]:
    """One-time import: legacy preferences.json → migration chain → YAML pair.

    The original json is renamed to preferences.json.bak (pristine — the
    migration runs in memory), so the user can inspect what was imported.
    A corrupted json is left in place and defaults are returned, matching the
    old loader's tolerance.
    """
    path = get_preferences_path()
    if not path.exists():
        return deepcopy(DEFAULT_PREFS)
    try:
        with path.open('r', encoding='utf-8') as f:
            user = json.load(f)
    except (json.JSONDecodeError, OSError):
        return deepcopy(DEFAULT_PREFS)
    if not isinstance(user, dict):
        return deepcopy(DEFAULT_PREFS)
    merged, _ = _merged_from_user(user)
    try:
        _write_yaml_pair(merged)
    except OSError:
        # Config dir is read-only (or otherwise unwritable): the app must
        # still start up and use the imported settings for this session. Do
        # NOT rename the json to .bak here — the YAML pair write failed, so
        # the json is the only surviving store; destroying it would lose the
        # user's settings permanently.
        return merged
    try:
        os.replace(path, path.with_name(path.name + '.bak'))
    except OSError:
        pass  # YAML pair now exists and takes priority; a stuck json is inert
    return merged


def save_preferences(updates: dict[str, Any]) -> None:
    """Merge `updates` into current prefs; write only the touched YAML file(s)."""
    current = load_preferences()
    _deep_merge_current(current, updates)
    current['schema_version'] = SCHEMA_VERSION
    features_data, run_data = split_prefs(current)
    f_path, r_path = get_features_yaml_path(), get_run_yaml_path()
    touched_run = 'shared' in updates
    touched_features = any(k != 'shared' for k in updates)
    try:
        if touched_features or not f_path.exists():
            write_features_yaml(f_path, features_data)
        if touched_run or not r_path.exists():
            write_run_yaml(r_path, run_data)
    except OSError as exc:
        # The write already retried a transient lock (config_yaml._write_text).
        # The GUI registers a handler so the failure is shown, not just printed.
        if _save_error_handler is None:
            raise
        _save_error_handler(f"Couldn't save your settings ({exc}). The last "
                            "change will be lost when the app closes.")


_save_error_handler = None


def set_save_error_handler(handler) -> None:
    """Register a callable(message) that reports a failed settings save (the
    GUI shows it); without one, save_preferences raises the OSError."""
    global _save_error_handler
    _save_error_handler = handler


def reset_to_defaults(*, keep_shared: bool = True) -> dict[str, Any]:
    """Rewrite features.yaml with the shipped defaults.

    run.yaml (folders, analyzer, theme, token paths) is never touched by a
    default reset — pass keep_shared=False to reset it too. Files are local
    (gitignored), so this only affects this machine. Returns the fresh prefs.
    """
    current = load_preferences()          # also triggers the legacy-json import
    fresh = deepcopy(DEFAULT_PREFS)
    if keep_shared and isinstance(current.get('shared'), dict):
        fresh['shared'] = deepcopy(current['shared'])
    features_data, run_data = split_prefs(fresh)
    write_features_yaml(get_features_yaml_path(), features_data)
    if not keep_shared:
        write_run_yaml(get_run_yaml_path(), run_data)
    return fresh


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

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
                     'acoustic_ts', 'phoneme_surprisal', 'alignment')


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
