"""Load and save user preferences.

Schema (v6 — active):
    {
      "schema_version": 6,
      "shared":  {analyzer_name, last_*_folder, theme},
      "features": {feat_struct_*, lexling_*, freq_*, pred_*, sem_*,
                   feat_acoustic_*, feat_psych_*,
                   feat_prosody_* (10 word-level), feat_doc_prosody_* (16 doc-level),
                   feat_syllable_* (9 syllable), feat_phoneme_* (4 phoneme)},
      "predictability": {llm_model},
      "semantic": {backend, word_categories_*, doc_categories_*},
      "output":   {word_level_csv, document_summary, out_count_*, doc_*,
                   syllables_csv, phonemes_csv},
      "prosody":  {pitch_units, pitch_floor_hz, pitch_ceiling_hz, pause_threshold_s}
    }

Migration chain (all paths terminate at v6):
  v1 → v2 → v3 → v4 → v5 → v6
  v2 → v3 → v4 → v5 → v6
  v3 → v4 → v5 → v6
  v4 → v5 → v6
  v5 → v6
  v6 files are deep-merged against v6 defaults

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
# Active schema aliases
# ---------------------------------------------------------------------------

SCHEMA_VERSION = SCHEMA_VERSION_V6  # active version
DEFAULT_PREFS: dict[str, Any] = DEFAULT_PREFS_V6  # active defaults


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
    """Load preferences; auto-migrate from v1 or v2 schema if needed."""
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
    if version == SCHEMA_VERSION_V6:
        _deep_merge_current(merged, user)
    elif version == SCHEMA_VERSION_V5:
        _migrate_v5_into_v6(merged, user)
        _write(path, merged)
    elif version == SCHEMA_VERSION_V4:
        v5_intermediate = deepcopy(DEFAULT_PREFS_V5)
        _migrate_v4_into_v5(v5_intermediate, user)
        _migrate_v5_into_v6(merged, v5_intermediate)
        _write(path, merged)
    elif version == 3:
        v4_intermediate = deepcopy(DEFAULT_PREFS_V4)
        _migrate_v3_into_v4(v4_intermediate, user)
        v5_intermediate = deepcopy(DEFAULT_PREFS_V5)
        _migrate_v4_into_v5(v5_intermediate, v4_intermediate)
        _migrate_v5_into_v6(merged, v5_intermediate)
        _write(path, merged)
    elif version == 2:
        v3_intermediate = deepcopy(DEFAULT_PREFS_V3)
        _migrate_v2_into_v3(v3_intermediate, user)
        v4_intermediate = deepcopy(DEFAULT_PREFS_V4)
        _migrate_v3_into_v4(v4_intermediate, v3_intermediate)
        v5_intermediate = deepcopy(DEFAULT_PREFS_V5)
        _migrate_v4_into_v5(v5_intermediate, v4_intermediate)
        _migrate_v5_into_v6(merged, v5_intermediate)
        _write(path, merged)
    else:
        # v1 flat → v2 → v3 → v4 → v5 → v6
        v2_intermediate = deepcopy(_DEFAULT_PREFS_V2)
        _migrate_v1_into(v2_intermediate, user)
        v3_intermediate = deepcopy(DEFAULT_PREFS_V3)
        _migrate_v2_into_v3(v3_intermediate, v2_intermediate)
        v4_intermediate = deepcopy(DEFAULT_PREFS_V4)
        _migrate_v3_into_v4(v4_intermediate, v3_intermediate)
        v5_intermediate = deepcopy(DEFAULT_PREFS_V5)
        _migrate_v4_into_v5(v5_intermediate, v4_intermediate)
        _migrate_v5_into_v6(merged, v5_intermediate)
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
                     'semantic', 'output', 'prosody')


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
