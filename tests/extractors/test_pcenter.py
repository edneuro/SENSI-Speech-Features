"""Unit tests for extractors.pcenter.add_pcenter."""
import math
import pandas as pd

from sensi.extractors.pcenter import add_pcenter


def _split_label(label):
    """'AE1' -> ('AE', 1); 'B' -> ('B', None). Mirrors syllabify._split_label."""
    text = str(label)
    if text and text[-1].isdigit():
        return text[:-1], int(text[-1])
    return text, None


def _phon_df(token_in_sentence_id, syl_id, phonemes):
    """phonemes: list of (label, onset, end). `label` is the fused ARPABET
    form (e.g. 'AE1'); split here into ARPABET + Stress to match the
    post-Task-3 phoneme frame schema."""
    rows = []
    for i, (label, onset, end) in enumerate(phonemes):
        code, stress = _split_label(label)
        rows.append({
            'doc_id': 'd', 'sentence_id': 1, 'token_in_sentence_id': token_in_sentence_id,
            'token': 'w', 'syllable_id': syl_id, 'phoneme_id': i + 1,
            'ARPABET': code, 'Stress': stress, 'Onset': onset, 'End': end,
            'Duration': round(end - onset, 4),
        })
    df = pd.DataFrame(rows)
    df['Stress'] = df['Stress'].astype('Int64')
    return df


def _syl_df(token_in_sentence_id, syl_id, onset, end):
    return pd.DataFrame([{
        'doc_id': 'd', 'sentence_id': 1, 'token_in_sentence_id': token_in_sentence_id,
        'token': 'w', 'syllable_id': syl_id,
        'Label': 'x', 'ARPABET': 'x',
        'Onset': onset, 'End': end,
        'Duration': round(end - onset, 4) if not math.isnan(onset) else float('nan'),
    }])


def test_standard_onset_cluster():
    """BAT [B, AE1, T] with k=0.3 -> nucleus_onset=0.05, onset_dur=0.05."""
    phon = _phon_df(1, 1, [('B', 0.0, 0.05), ('AE1', 0.05, 0.15), ('T', 0.15, 0.20)])
    syl = _syl_df(1, 1, 0.0, 0.20)
    result = add_pcenter(syl, phon, k=0.3)
    expected = round(0.05 - 0.3 * 0.05, 4)
    assert abs(result['P_Center'].iloc[0] - expected) < 1e-6


def test_vowel_initial_syllable():
    """[AE1, T] — onset_consonant_duration=0 -> P_Center == nucleus_onset."""
    phon = _phon_df(1, 1, [('AE1', 0.10, 0.20), ('T', 0.20, 0.25)])
    syl = _syl_df(1, 1, 0.10, 0.25)
    result = add_pcenter(syl, phon, k=0.3)
    assert abs(result['P_Center'].iloc[0] - 0.10) < 1e-6


def test_no_vowel_syllable_falls_back_to_acoustic_onset():
    """[HH, M] — no nucleus -> P_Center == syllable.Onset."""
    phon = _phon_df(1, 1, [('HH', 0.05, 0.10), ('M', 0.10, 0.20)])
    syl = _syl_df(1, 1, 0.05, 0.20)
    result = add_pcenter(syl, phon, k=0.3)
    assert abs(result['P_Center'].iloc[0] - 0.05) < 1e-6


def test_multi_consonant_onset():
    """[S, T, R, AY1, NG] — correction = sum(S+T+R) * k."""
    phon = _phon_df(1, 1, [
        ('S', 0.0, 0.05), ('T', 0.05, 0.10), ('R', 0.10, 0.15),
        ('AY1', 0.15, 0.30), ('NG', 0.30, 0.40),
    ])
    syl = _syl_df(1, 1, 0.0, 0.40)
    result = add_pcenter(syl, phon, k=0.3)
    expected = round(0.15 - 0.3 * 0.15, 4)
    assert abs(result['P_Center'].iloc[0] - expected) < 1e-6


def test_nan_when_syllable_onset_is_nan():
    """MFA alignment missing (Onset=NaN) -> P_Center=NaN."""
    phon = _phon_df(1, 1, [('B', 0.0, 0.05), ('AE1', 0.05, 0.15)])
    syl = _syl_df(1, 1, float('nan'), float('nan'))
    result = add_pcenter(syl, phon, k=0.3)
    assert math.isnan(result['P_Center'].iloc[0])


def test_k_zero_yields_nucleus_onset():
    """k=0 means no correction: P_Center == nucleus_onset exactly."""
    phon = _phon_df(1, 1, [('B', 0.0, 0.05), ('AE1', 0.05, 0.15), ('T', 0.15, 0.20)])
    syl = _syl_df(1, 1, 0.0, 0.20)
    result = add_pcenter(syl, phon, k=0.0)
    assert abs(result['P_Center'].iloc[0] - 0.05) < 1e-6


def test_syllable_with_no_phoneme_rows_falls_back_to_acoustic_onset():
    """Syllable FK absent from phonemes_df -> falls back to syllable.Onset."""
    phon = _phon_df(1, 1, [('B', 0.0, 0.05), ('AE1', 0.05, 0.15)])
    syl = pd.concat([
        _syl_df(1, 1, 0.0, 0.15),
        _syl_df(1, 2, 0.15, 0.25),  # syllable_id=2 has no phonemes
    ], ignore_index=True)
    result = add_pcenter(syl, phon, k=0.3)
    assert abs(result['P_Center'].iloc[1] - 0.15) < 1e-6


def test_two_words_no_cross_contamination():
    """Two tokens × two syllables each — verifies both token_in_sentence_id and syllable_id in FK grouping."""
    # Token 1, Syllable 1: [B, AE1, T] (onset 0.0–0.20)
    phon_t1s1 = _phon_df(1, 1, [('B', 0.0, 0.05), ('AE1', 0.05, 0.15), ('T', 0.15, 0.20)])
    # Token 1, Syllable 2: [K, AH1, T] (onset 0.22–0.45)
    phon_t1s2 = _phon_df(1, 2, [('K', 0.22, 0.27), ('AH1', 0.27, 0.40), ('T', 0.40, 0.45)])
    # Token 2, Syllable 1: [AY1, NG] (onset 0.50–0.70) — vowel-initial
    phon_t2s1 = _phon_df(2, 1, [('AY1', 0.50, 0.65), ('NG', 0.65, 0.70)])
    # Token 2, Syllable 2: [G, IH1, N] (onset 0.72–0.90)
    phon_t2s2 = _phon_df(2, 2, [('G', 0.72, 0.77), ('IH1', 0.77, 0.85), ('N', 0.85, 0.90)])
    phon = pd.concat([phon_t1s1, phon_t1s2, phon_t2s1, phon_t2s2], ignore_index=True)

    syl_t1s1 = _syl_df(1, 1, 0.0, 0.20)
    syl_t1s2 = _syl_df(1, 2, 0.22, 0.45)
    syl_t2s1 = _syl_df(2, 1, 0.50, 0.70)
    syl_t2s2 = _syl_df(2, 2, 0.72, 0.90)
    syl = pd.concat([syl_t1s1, syl_t1s2, syl_t2s1, syl_t2s2], ignore_index=True)

    result = add_pcenter(syl, phon, k=0.3)

    # Token 1, Syllable 1: nucleus_onset=0.05, onset_dur=0.05, P_Center=0.05-0.3*0.05=0.035
    assert abs(result['P_Center'].iloc[0] - round(0.05 - 0.3 * 0.05, 4)) < 1e-6
    # Token 1, Syllable 2: nucleus_onset=0.27, onset_dur=0.05, P_Center=0.27-0.3*0.05=0.255
    assert abs(result['P_Center'].iloc[1] - round(0.27 - 0.3 * 0.05, 4)) < 1e-6
    # Token 2, Syllable 1: vowel-initial, onset_dur=0, P_Center=nucleus_onset=0.50
    assert abs(result['P_Center'].iloc[2] - 0.50) < 1e-6
    # Token 2, Syllable 2: nucleus_onset=0.77, onset_dur=0.05, P_Center=0.77-0.3*0.05=0.755
    assert abs(result['P_Center'].iloc[3] - round(0.77 - 0.3 * 0.05, 4)) < 1e-6


# ---------------------------------------------------------------------------
# Task 4: vowel detection via Stress, not the fused Label
# ---------------------------------------------------------------------------

def _phon(arpabet, stress, onset, end, phoneme_id):
    return {"doc_id": 1, "sentence_id": 1, "token_in_sentence_id": 1,
            "syllable_id": 1, "phoneme_id": phoneme_id,
            "ARPABET": arpabet, "Stress": stress,
            "Onset": onset, "End": end, "Duration": round(end - onset, 4)}


def test_nucleus_is_found_via_stress_column():
    """Regression: vowel detection must not depend on a fused Label string."""
    syll = pd.DataFrame([{
        "doc_id": 1, "sentence_id": 1, "token_in_sentence_id": 1,
        "syllable_id": 1, "Onset": 0.0, "End": 0.30,
    }])
    phon = pd.DataFrame([
        _phon("K",  None, 0.00, 0.10, 1),
        _phon("AH", 1,    0.10, 0.20, 2),
        _phon("P",  None, 0.20, 0.30, 3),
    ]).astype({"Stress": "Int64"})

    out = add_pcenter(syll, phon, k=0.3)
    # nucleus_onset 0.10 - 0.3 * onset_dur 0.10 = 0.07
    assert out["P_Center"].iloc[0] == 0.07
    assert out["P_Center"].iloc[0] != 0.0, "must not degrade to syllable onset"


def test_unstressed_vowel_still_counts_as_a_nucleus():
    """Stress == 0 is a vowel. Only <NA> means 'not a vowel'."""
    syll = pd.DataFrame([{
        "doc_id": 1, "sentence_id": 1, "token_in_sentence_id": 1,
        "syllable_id": 1, "Onset": 0.0, "End": 0.20,
    }])
    phon = pd.DataFrame([
        _phon("B",  None, 0.00, 0.10, 1),
        _phon("AH", 0,    0.10, 0.20, 2),
    ]).astype({"Stress": "Int64"})

    out = add_pcenter(syll, phon, k=0.3)
    assert out["P_Center"].iloc[0] == 0.07


def test_vowelless_syllable_falls_back_to_syllable_onset():
    syll = pd.DataFrame([{
        "doc_id": 1, "sentence_id": 1, "token_in_sentence_id": 1,
        "syllable_id": 1, "Onset": 1.5, "End": 1.70,
    }])
    phon = pd.DataFrame([
        _phon("HH", None, 1.50, 1.60, 1),
        _phon("M",  None, 1.60, 1.70, 2),
    ]).astype({"Stress": "Int64"})

    out = add_pcenter(syll, phon, k=0.3)
    assert out["P_Center"].iloc[0] == 1.5
