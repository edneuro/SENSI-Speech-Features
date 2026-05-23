# Sentence-Level Tier — Design Spec

**Date:** 2026-05-22  
**Status:** Design finalized; plan at `docs/superpowers/plans/2026-05-22-sentence-level-tier.md`

---

## 1. Output

New sidecar CSV per stimulus: `sentence/<stim>_sentences.csv`

| Column | Type | Notes |
|--------|------|-------|
| `doc_id` | str | always present (FK to word CSV) |
| `sentence_id` | int | always present (FK to word CSV) |
| `sentence` | str | full sentence text |
| `Onset` | float | min Onset over sentence's aligned words (NaN if no alignment) |
| `End` | float | max End over sentence's aligned words |
| `Pitch_Mean` | float | pitch stats over [Onset, End] window |
| `Pitch_Min` | float | 5th percentile of voiced frames |
| `Pitch_Max` | float | 95th percentile of voiced frames |
| `Pitch_Std` | float | std dev of voiced frames |
| `Pitch_Delta` | float | Pitch_Mean[i] − Pitch_Mean[i−1] (NaN for first sentence) |
| `Volume_Mean` | float | intensity stats over window |
| `Volume_Min` | float | |
| `Volume_Max` | float | |
| `Volume_Std` | float | |
| `Volume_Delta` | float | Volume_Mean[i] − Volume_Mean[i−1] |
| `HNR` | float | mean HNR over voiced frames in window |

11 prosody columns — mirrors doc-level but **no pause statistics**.

## 2. HNR at sentence level

`audio_features.Contours` gains two new optional fields:
- `hnr_frame_db: np.ndarray | None` — per-frame HNR values (sentinel −200 → NaN)
- `hnr_frame_times: np.ndarray | None` — timestamps for `hnr_frame_db` (harmonicity grid)

Sentence-level HNR = mean of non-NaN frames in [Onset, End]. If fields absent → NaN.

## 3. Prefs: v8 → v9 (additive)

New `features` keys: `feat_sentence_text`, `feat_sentence_onset`, `feat_sentence_end`,  
`feat_sentence_pitch_mean/min/max/std/delta`, `feat_sentence_volume_mean/min/max/std/delta`, `feat_sentence_hnr`,  
`master_sentence_timing`, `master_sentence_prosody`

New `output` key: `sentences_csv` (default True)

## 4. Planner

New stage: `sentence_prosody` (consumes `word_df`, `contours`; produces `sentence_df`)

New clusters: `sentence timing` (onset/end/text), `sentence prosody` (11 prosody columns)

`_OUTPUT_CSV_MAP` gains `"sentences_csv" → "sentences"`. `_WORD_LEVEL_CSVS` gains `"sentences"`.

## 5. GUI

New sidebar page "Sentence features" between Sub-word and Run. Two ClusterCard sections:
timing (Onset, End, sentence text) and prosody (11 columns).
