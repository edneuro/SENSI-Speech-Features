"""Per-stimulus aggregation — one-row summary for each processed pair.

Reads the per-word DataFrame produced by the extractor pipeline and emits
a flat dict of doc-level fields. Each feature group is controlled by a
toggle key (see DOC_FEATURE_KEYS); toggled-off groups are absent from
the output.

When 'category_proportions' is on, scans Sem_* columns and emits
`count_<cat>` / `prop_<cat>` over content tokens.

When `psych_*` toggles are requested and `dictionary_used == 'LIWC'`,
adds the corresponding composite psychological scores using
`extractors.liwc_summary` (Pennebaker et al. 2014 formula approximations).
"""

from pathlib import Path

import pandas as pd

from extractors.lexical_diversity import compute_ttr, compute_mtld
from extractors.readability import compute_readability

CONTENT_POS = {'NOUN', 'VERB', 'ADJ', 'ADV', 'PROPN'}

DOC_FEATURE_KEYS = (
    'counts',
    'mean_zipf_freq',
    'mean_edu_zipf',
    'mean_surprisal',
    'mean_llm_cloze',
    'ttr',
    'mtld',
    'flesch_reading_ease',
    'flesch_kincaid_grade',
    'category_proportions',
    'psych_authentic',
    'psych_analytic',
    'psych_tone',
    'psych_clout',
)


def build_doc_summary(
    token_df: pd.DataFrame,
    stimulus_id: str,
    dictionary_used: str,
    text: str,
    toggles: dict[str, bool],
    liwc_dictionary_path: Path | str | None = None,
    *,
    contours=None,                       # extractors.audio_features.Contours | None
    pause_threshold_s: float = 0.25,
    pitch_units: str = 'semitones',
) -> dict:
    """Aggregate a per-word DataFrame into a one-row summary dict."""
    df = token_df
    summary: dict = {
        'stimulus_id': stimulus_id,
        'dictionary_used': dictionary_used,
    }

    if toggles.get('counts'):
        summary['n_tokens'] = int(len(df))
        summary['n_content_tokens'] = int(df['upos'].isin(CONTENT_POS).sum()) if 'upos' in df else 0
        summary['n_sentences'] = int(df['sentence_id'].nunique()) if 'sentence_id' in df else 0
        summary['n_paragraphs'] = int(df['paragraph_id'].nunique()) if 'paragraph_id' in df else 0

    if toggles.get('mean_zipf_freq') and 'Zipf_Freq' in df.columns:
        summary['mean_zipf_freq'] = _mean(df['Zipf_Freq'])

    if toggles.get('mean_edu_zipf'):
        for src, dst in [
            ('Edu_Zipf_K2', 'mean_edu_zipf_k2'),
            ('Edu_Zipf_3to5', 'mean_edu_zipf_3to5'),
            ('Edu_Zipf_6to8', 'mean_edu_zipf_6to8'),
            ('Edu_Zipf_9to12', 'mean_edu_zipf_9to12'),
            ('Edu_Zipf_Overall', 'mean_edu_zipf_overall'),
        ]:
            if src in df.columns:
                summary[dst] = _mean(df[src])

    if toggles.get('mean_surprisal') and 'Surprisal' in df.columns:
        summary['mean_surprisal'] = _mean(df['Surprisal'])

    if toggles.get('mean_llm_cloze'):
        for src, dst in [
            ('LLM_Cloze', 'mean_llm_cloze'),
            ('LLM_Surprisal', 'mean_llm_surprisal'),
            ('LLM_TopMatch', 'mean_llm_top_match_prop'),
        ]:
            if src in df.columns:
                summary[dst] = _mean(df[src])

    content_tokens = []
    if 'upos' in df.columns and 'token' in df.columns:
        content_tokens = [
            str(t) for t, p in zip(df['token'], df['upos']) if p in CONTENT_POS
        ]

    if toggles.get('ttr'):
        summary['ttr'] = compute_ttr(content_tokens)
    if toggles.get('mtld'):
        summary['mtld'] = compute_mtld(content_tokens)

    if toggles.get('flesch_reading_ease') or toggles.get('flesch_kincaid_grade'):
        read = compute_readability(text)
        if toggles.get('flesch_reading_ease'):
            summary['flesch_reading_ease'] = read['flesch_reading_ease']
        if toggles.get('flesch_kincaid_grade'):
            summary['flesch_kincaid_grade'] = read['flesch_kincaid_grade']

    if toggles.get('category_proportions'):
        sem_cols = [c for c in df.columns
                    if c.startswith('Sem_') and c != 'Sem_Categories']
        n_content = summary.get('n_content_tokens')
        if n_content is None:
            n_content = int(df['upos'].isin(CONTENT_POS).sum()) if 'upos' in df else 0
        for col in sem_cols:
            parts = col.split('_', 2)
            if len(parts) < 3:
                continue
            category = parts[2]
            count = int(df[col].fillna(0).astype(int).sum())
            summary[f'count_{category}'] = count
            summary[f'prop_{category}'] = (count / n_content) if n_content > 0 else 0.0

    psych_wanted = any(toggles.get(k) for k in
                       ('psych_authentic', 'psych_analytic', 'psych_tone', 'psych_clout'))
    if psych_wanted and dictionary_used == 'LIWC' and liwc_dictionary_path:
        from extractors.liwc_summary import compute_liwc_summary
        scores = compute_liwc_summary(text, liwc_dictionary_path)
        for short_key in ('authentic', 'analytic', 'tone', 'clout'):
            full_key = f'psych_{short_key}'
            if toggles.get(full_key):
                summary[full_key] = scores[short_key]

    # Prosody — single master toggle. Emits all 16 cols; per-column dropping
    # happens later in main.py's CSV writer via _doc_columns_to_drop.
    if contours is not None and toggles.get('doc_prosody'):
        from extractors.prosody import compute_doc_prosody
        prosody = compute_doc_prosody(
            token_df, contours,
            text=text, pause_threshold_s=pause_threshold_s,
            pitch_units=pitch_units,
        )
        summary.update(prosody)

    return summary


def _mean(series: pd.Series) -> float | None:
    """Mean that skips NaN; returns None if all values are NaN."""
    if series.notna().any():
        return float(series.mean())
    return None
