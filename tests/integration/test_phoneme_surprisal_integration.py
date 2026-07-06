import numpy as np
import pandas as pd
import scipy.io
from extractors import phoneme_surprisal as ps


def test_mat_roundtrip_has_four_trains(tmp_path, monkeypatch):
    trie = ps.build_cohort_trie([
        (("K", "AE", "T"), 3.0), (("K", "AE", "B"), 1.0), (("D", "AO", "G"), 2.0)])
    monkeypatch.setattr(ps, "get_cohort_trie", lambda *a, **k: trie)
    df = pd.DataFrame([
        dict(doc_id=1, sentence_id=1, token_in_sentence_id=1, token="cat",
             syllable_id=1, phoneme_id=i + 1, Label=lbl, Onset=on, End=on + 0.03, Duration=0.03)
        for i, (lbl, on) in enumerate([("K", 0.10), ("AE1", 0.15), ("T", 0.18)])
    ])
    df = ps.add_phoneme_surprisal(df)
    out = ps.phoneme_surprisal_mat_vars(df, 1.0, 100, {"analysis_sr": 100})
    p = tmp_path / "s_phoneme_surprisal.mat"
    scipy.io.savemat(str(p), out)
    loaded = scipy.io.loadmat(str(p))
    for k in ("phoneme_surprisal", "cohort_entropy", "phoneme_onset", "word_onset"):
        assert loaded[k].shape == (100, 1)
    assert loaded["word_onset"][10, 0] == 1
