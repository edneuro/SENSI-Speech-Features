"""Loader for per-feature documentation markdown files.

Each GUI toggle key resolves to a `docs/features/<key>.md` file (or to a
shared file via _SHARED_DOC_MAP when multiple toggles describe one concept,
e.g. all five pitch_* columns mapped to prosody_pitch.md).

Files use YAML frontmatter for popup-summary fields (name, libraries,
status, levels…) and Markdown body for the long-form content.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml


DOCS_ROOT = Path(__file__).resolve().parent.parent / "docs" / "features"

# Toggle keys that share a single .md file. When several toggles describe
# facets of one concept (the five `_pitch_*` derived stats; the
# bookkeeping ID columns), they all resolve to the same markdown body.
_SHARED_DOC_MAP: dict[str, str] = {
    # Structural identifier columns — one shared explainer table
    "feat_struct_doc_id":       "structural_ids.md",
    "feat_struct_paragraph_id": "structural_ids.md",
    "feat_struct_sentence_id":  "structural_ids.md",
    "feat_struct_term_id":      "structural_ids.md",
    "feat_struct_token_id":     "structural_ids.md",
    # Structural character offsets — one shared explainer
    "feat_struct_char_start":   "structural_offsets.md",
    "feat_struct_char_end":     "structural_offsets.md",
    # EWFG grade-band frequency — all five bands share one explainer
    "freq_ewfg_k2":             "edu_zipf.md",
    "freq_ewfg_3to5":           "edu_zipf.md",
    "freq_ewfg_6to8":           "edu_zipf.md",
    "freq_ewfg_9to12":          "edu_zipf.md",
    "freq_ewfg_overall":        "edu_zipf.md",
    # Acoustic word boundaries (Onset/End from MFA) — one shared explainer
    "feat_acoustic_onset":         "acoustic_boundaries.md",
    "feat_acoustic_end":           "acoustic_boundaries.md",
    # Word duration / gap — derived from boundaries, one shared explainer
    "feat_acoustic_word_duration": "acoustic_durations.md",
    "feat_acoustic_word_gap":      "acoustic_durations.md",
    # Word prosody — pitch family
    "feat_prosody_pitch_mean":   "prosody_pitch.md",
    "feat_prosody_pitch_min":    "prosody_pitch.md",
    "feat_prosody_pitch_max":    "prosody_pitch.md",
    "feat_prosody_pitch_std":    "prosody_pitch.md",
    "feat_prosody_pitch_delta":  "prosody_pitch.md",
    # Word prosody — volume family
    "feat_prosody_volume_mean":  "prosody_volume.md",
    "feat_prosody_volume_min":   "prosody_volume.md",
    "feat_prosody_volume_max":   "prosody_volume.md",
    "feat_prosody_volume_std":   "prosody_volume.md",
    "feat_prosody_volume_delta": "prosody_volume.md",
    # Document-level count columns
    "output_out_count_tokens":         "doc_counts.md",
    "output_out_count_content_tokens": "doc_counts.md",
    "output_out_count_sentences":      "doc_counts.md",
    "output_out_count_paragraphs":     "doc_counts.md",
    # Document-level word-feature aggregates
    "output_doc_agg_zipf_freq": "doc_aggregates.md",
    "output_doc_agg_edu_zipf":  "doc_aggregates.md",
    "output_doc_agg_surprisal": "doc_aggregates.md",
    "output_doc_agg_llm":       "doc_aggregates.md",
    # Composite psych scores (Pennebaker) — all four share one explainer
    "feat_psych_authentic": "doc_psych_composite.md",
    "feat_psych_analytic":  "doc_psych_composite.md",
    "feat_psych_tone":      "doc_psych_composite.md",
    "feat_psych_clout":     "doc_psych_composite.md",
    # Document-level prosody — pitch family
    "feat_doc_prosody_pitch_mean":       "doc_prosody_pitch.md",
    "feat_doc_prosody_pitch_min":        "doc_prosody_pitch.md",
    "feat_doc_prosody_pitch_max":        "doc_prosody_pitch.md",
    "feat_doc_prosody_pitch_std":        "doc_prosody_pitch.md",
    "feat_doc_prosody_pitch_delta_mean": "doc_prosody_pitch.md",
    # Document-level prosody — volume family
    "feat_doc_prosody_volume_mean":       "doc_prosody_volume.md",
    "feat_doc_prosody_volume_min":        "doc_prosody_volume.md",
    "feat_doc_prosody_volume_max":        "doc_prosody_volume.md",
    "feat_doc_prosody_volume_std":        "doc_prosody_volume.md",
    "feat_doc_prosody_volume_delta_mean": "doc_prosody_volume.md",
    # Document-level prosody — pauses (4 toggles share one explainer)
    "feat_doc_prosody_pause_count":          "doc_prosody_pauses.md",
    "feat_doc_prosody_pause_mean_duration":  "doc_prosody_pauses.md",
    "feat_doc_prosody_pause_total_duration": "doc_prosody_pauses.md",
    "feat_doc_prosody_pause_pct_silence":    "doc_prosody_pauses.md",
    # Syllable timing (3 toggles share one explainer)
    "feat_syllable_onset":    "syllable_timing.md",
    "feat_syllable_end":      "syllable_timing.md",
    "feat_syllable_duration": "syllable_timing.md",
    # Syllable labels (orthographic + ARPABET)
    "feat_syllable_label":   "syllable_labels.md",
    "feat_syllable_arpabet": "syllable_labels.md",
    # Syllable prosody (4 toggles share one explainer)
    "feat_syllable_pitch_mean":   "syllable_prosody.md",
    "feat_syllable_volume_mean":  "syllable_prosody.md",
    "feat_syllable_pitch_delta":  "syllable_prosody.md",
    "feat_syllable_volume_delta": "syllable_prosody.md",
    # Phoneme-level columns (4 toggles share one explainer)
    "feat_phoneme_label":    "phoneme.md",
    "feat_phoneme_onset":    "phoneme.md",
    "feat_phoneme_end":      "phoneme.md",
    "feat_phoneme_duration": "phoneme.md",
    # Sentence timing (3 toggles share one explainer)
    "feat_sentence_onset": "sentence_timing.md",
    "feat_sentence_end":   "sentence_timing.md",
    "feat_sentence_text":  "sentence_timing.md",
    # Sentence prosody — pitch family (5 toggles)
    "feat_sentence_pitch_mean":  "sentence_prosody_pitch.md",
    "feat_sentence_pitch_min":   "sentence_prosody_pitch.md",
    "feat_sentence_pitch_max":   "sentence_prosody_pitch.md",
    "feat_sentence_pitch_std":   "sentence_prosody_pitch.md",
    "feat_sentence_pitch_delta": "sentence_prosody_pitch.md",
    # Sentence prosody — volume family (5 toggles)
    "feat_sentence_volume_mean":  "sentence_prosody_volume.md",
    "feat_sentence_volume_min":   "sentence_prosody_volume.md",
    "feat_sentence_volume_max":   "sentence_prosody_volume.md",
    "feat_sentence_volume_std":   "sentence_prosody_volume.md",
    "feat_sentence_volume_delta": "sentence_prosody_volume.md",
}


@dataclass
class FeatureDoc:
    key: str
    frontmatter: dict
    body: str

    @property
    def name(self) -> str:
        return self.frontmatter.get("name", self.key)


_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?(.*)\Z", re.DOTALL)


def _doc_path(toggle_key: str) -> Path:
    filename = _SHARED_DOC_MAP.get(toggle_key, f"{toggle_key}.md")
    return DOCS_ROOT / filename


def has_feature_doc(toggle_key: str) -> bool:
    """Cheap existence check — drives whether the GUI renders an info icon."""
    return _doc_path(toggle_key).is_file()


def load_feature_doc(toggle_key: str) -> FeatureDoc | None:
    """Return the parsed FeatureDoc for `toggle_key`, or None if no .md exists."""
    path = _doc_path(toggle_key)
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8")
    m = _FRONTMATTER_RE.match(text)
    if m is None:
        return FeatureDoc(key=toggle_key, frontmatter={}, body=text.strip())
    frontmatter = yaml.safe_load(m.group(1)) or {}
    body = m.group(2).strip()
    return FeatureDoc(key=toggle_key, frontmatter=frontmatter, body=body)
