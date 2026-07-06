import math
import pytest
from extractors.phoneme_surprisal import build_cohort_trie, phoneme_scores

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
