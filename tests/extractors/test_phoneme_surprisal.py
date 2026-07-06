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
