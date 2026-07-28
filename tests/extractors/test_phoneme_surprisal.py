import math
import pandas as pd
import pytest
from extractors.phoneme_surprisal import (
    add_phoneme_surprisal, build_cohort_trie, phoneme_scores,
)

# Lexicon {cat:3, cab:1, dog:2} with stress-stripped ARPABET pronunciations.
_ENTRIES = [
    (("K", "AE", "T"), 3.0),
    (("K", "AE", "B"), 1.0),
    (("D", "AO", "G"), 2.0),
]


def test_cat_surprisal_matches_hand_computation():
    trie = build_cohort_trie(_ENTRIES)
    surp, _ = phoneme_scores(trie, ["K", "AE", "T"])
    assert surp == pytest.approx([0.585, 0.0, 0.415], abs=1e-3)


def test_cat_entropy_matches_hand_computation():
    trie = build_cohort_trie(_ENTRIES)
    _, ent = phoneme_scores(trie, ["K", "AE", "T"])
    # after-node entropy: {cat3,cab1}=0.811, {cat3,cab1}=0.811, {cat3}=0.0
    assert ent == pytest.approx([0.811, 0.811, 0.0], abs=1e-3)


def test_first_phoneme_uses_empty_context_root():
    # surprisal(K) = log2(6) - log2(4): Z_root=6 (all words), after-K Z=4.
    trie = build_cohort_trie(_ENTRIES)
    surp, _ = phoneme_scores(trie, ["K", "AE", "B"])
    assert surp[0] == pytest.approx(math.log2(6) - math.log2(4), abs=1e-6)


def test_oov_path_caps_surprisal_and_zeros_entropy():
    trie = build_cohort_trie(_ENTRIES)
    # 'K AE Z' leaves the trie at Z (only T/B follow AE).
    surp, ent = phoneme_scores(trie, ["K", "AE", "Z"])
    assert surp[2] == pytest.approx(trie.cap, abs=1e-9)
    assert ent[2] == 0.0
    # cap == -log2(f_min / z_root) == -log2(1/6)
    assert trie.cap == pytest.approx(-math.log2(1.0 / 6.0), abs=1e-9)


def test_guard_flags_raise_on_non_default():
    trie = build_cohort_trie(_ENTRIES)
    with pytest.raises(NotImplementedError):
        phoneme_scores(trie, ["K"], context_independent=False)
    with pytest.raises(NotImplementedError):
        phoneme_scores(trie, ["K"], first_phoneme_empty_context=False)


def test_load_mfa_lexicon_parses_variable_columns(tmp_path):
    from extractors.phoneme_surprisal import load_mfa_lexicon
    d = tmp_path / "mini.dict"
    d.write_text(
        "cat\t0.99\t0.62\t0.58\t1.07\tK AE1 T\n"   # 4 prob cols then phones
        "dog\t0.09\t0.48\t1.01\t1.0\tD AO1 G\n"      # first-listed dog variant
        "dog\t0.99\t0.41\t0.49\t1.06\tD AA1 G\n"     # second variant (dropped when primary_only)
        "'ve\tV\n"                                     # no prob columns
        "[bracketed]\t0.99\t0.18\t1.0\t1.0\tspn\n",  # special token -> skipped
        encoding="utf-8",
    )
    entries = dict((p, f) for p, f in
                   load_mfa_lexicon(str(d), strip_stress=True, primary_only=True))
    assert ("K", "AE", "T") in entries          # stress stripped
    assert ("D", "AO", "G") in entries           # first dog variant kept
    assert ("D", "AA", "G") not in entries       # second variant dropped
    assert ("V",) in entries                     # no-prob-column line parsed
    assert not any("spn" in p for p in entries)  # special token skipped
    # frequencies came from wordfreq (all > 0 for these common words)
    assert entries[("K", "AE", "T")] > 0


def _cat_phon_df():
    # One word "cat", one syllable, three phonemes at 0.10/0.15/0.20 s.
    df = pd.DataFrame([
        dict(doc_id=1, sentence_id=1, token_in_sentence_id=1, token="cat",
             syllable_id=1, phoneme_id=1, ARPABET="K", Stress=None, Onset=0.10, End=0.15, Duration=0.05),
        dict(doc_id=1, sentence_id=1, token_in_sentence_id=1, token="cat",
             syllable_id=1, phoneme_id=2, ARPABET="AE", Stress=1, Onset=0.15, End=0.18, Duration=0.03),
        dict(doc_id=1, sentence_id=1, token_in_sentence_id=1, token="cat",
             syllable_id=1, phoneme_id=3, ARPABET="T", Stress=None, Onset=0.18, End=0.20, Duration=0.02),
    ])
    df["Stress"] = df["Stress"].astype("Int64")
    return df


def test_add_phoneme_surprisal_appends_two_columns(monkeypatch):
    from extractors import phoneme_surprisal as ps
    trie = ps.build_cohort_trie([
        (("K", "AE", "T"), 3.0), (("K", "AE", "B"), 1.0), (("D", "AO", "G"), 2.0)])
    monkeypatch.setattr(ps, "get_cohort_trie", lambda *a, **k: trie)
    out = ps.add_phoneme_surprisal(_cat_phon_df())
    assert list(out["phoneme_surprisal"]) == pytest.approx([0.585, 0.0, 0.415], abs=1e-3)
    assert list(out["cohort_entropy"]) == pytest.approx([0.811, 0.811, 0.0], abs=1e-3)


def test_phoneme_surprisal_mat_vars_impulses_and_controls():
    from extractors import phoneme_surprisal as ps
    df = _cat_phon_df()
    df["phoneme_surprisal"] = [0.585, 0.0, 0.415]
    df["cohort_entropy"] = [0.811, 0.811, 0.0]
    out = ps.phoneme_surprisal_mat_vars(df, duration_s=1.0, analysis_sr=100,
                                        config={"analysis_sr": 100})
    N = 100  # round(1.0 * 100)
    for k in ("phoneme_surprisal", "cohort_entropy", "phoneme_onset", "word_onset", "t"):
        assert out[k].shape[0] == N
    assert out["fs"] == 100
    # impulses land at round(onset*sr): 0.10->10, 0.15->15, 0.18->18
    assert out["phoneme_surprisal"][10, 0] == pytest.approx(0.585, abs=1e-3)
    assert out["phoneme_surprisal"][18, 0] == pytest.approx(0.415, abs=1e-3)
    assert out["phoneme_onset"][10, 0] == 1 and out["phoneme_onset"][18, 0] == 1
    # only the first phoneme of the word is a word onset
    assert out["word_onset"][10, 0] == 1
    assert out["word_onset"][15, 0] == 0 and out["word_onset"][18, 0] == 0


# ---------------------------------------------------------------------------
# Task 5: read ARPABET/Stress, not the fused Label; spn -> NaN
# ---------------------------------------------------------------------------

def _row(arpabet, stress, phoneme_id, tis_id=1):
    return {"doc_id": 1, "sentence_id": 1, "token_in_sentence_id": tis_id,
            "token": "cup", "syllable_id": 1, "phoneme_id": phoneme_id,
            "ARPABET": arpabet, "Stress": stress,
            "Onset": 0.1 * phoneme_id, "End": 0.1 * phoneme_id + 0.1,
            "Duration": 0.1}


def test_reads_arpabet_column():
    df = pd.DataFrame([
        _row("K", None, 1), _row("AH", 1, 2), _row("P", None, 3),
    ]).astype({"Stress": "Int64"})
    out = add_phoneme_surprisal(df)
    assert "phoneme_surprisal" in out.columns
    assert out["phoneme_surprisal"].iloc[0] > 0
    assert out["phoneme_surprisal"].notna().all()


def test_spn_row_scores_nan_not_the_oov_cap():
    df = pd.DataFrame([
        {"doc_id": 1, "sentence_id": 1, "token_in_sentence_id": 1,
         "token": "zorblax", "syllable_id": 1, "phoneme_id": 1,
         "ARPABET": "spn", "Stress": None, "Onset": 0.0, "End": 0.33,
         "Duration": 0.33},
    ]).astype({"Stress": "Int64"})
    out = add_phoneme_surprisal(df)
    assert pd.isna(out["phoneme_surprisal"].iloc[0])
    assert pd.isna(out["cohort_entropy"].iloc[0])


def test_strip_stress_false_recombines_arpabet_and_stress():
    """The stress-sensitive trie needs the fused form back."""
    df = pd.DataFrame([
        _row("K", None, 1), _row("AH", 1, 2), _row("P", None, 3),
    ]).astype({"Stress": "Int64"})
    out = add_phoneme_surprisal(df, strip_stress=False, primary_only=True)
    # A stress-sensitive trie still finds 'K AH1 P', so the first phoneme is
    # scored from the real cohort rather than pinned at the OOV cap.
    assert out["phoneme_surprisal"].iloc[0] > 0
    assert out["phoneme_surprisal"].notna().all()
