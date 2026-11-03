import numpy as np
import pandas as pd
import pytest
import scipy.io

from sensi.gui.planner import registry
from sensi.utils.event_mat import (
    merge_rule, merge_spoken_words, syllable_event_vars, word_event_vars,
)

NAN = np.nan
RELOOK = lambda w: {"Zipf_Freq": 4.1}  # noqa: E731


def _frame():
    return pd.DataFrame({
        "token": ["The", "debtor", "'s", "side", ".", "ghost"],
        "lemma": ["the", "debtor", "'s", "side", ".", "ghost"],
        "upos": ["DET", "NOUN", "PART", "NOUN", "PUNCT", "NOUN"],
        "Transcript": ["the", "debtor's", "debtor's", "side", NAN, NAN],
        "Onset": [0.0, 0.3, 0.3, 0.9, NAN, NAN],
        "End": [0.2, 0.8, 0.8, 1.2, NAN, NAN],
        "Word_Duration": [0.2, 0.5, 0.5, 0.3, NAN, NAN],
        "token_id": [0, 1, 2, 3, 4, 5],
        "sentence_id": [0, 0, 0, 0, 0, 0],
        "gpt2_Surprisal": [1.0, 3.0, 2.0, 1.5, NAN, NAN],
        "gpt2_Cloze": [0.9, 0.5, 0.2, 0.4, NAN, NAN],
        "gpt2_TopMatch": [True, True, False, True, NAN, NAN],
        "Sem_Empath_money": [0, 0, 1, 0, NAN, NAN],
        "Zipf_Freq": [7.0, 3.0, 5.0, 5.5, NAN, NAN],
        "AoA_Mean": [3.0, 8.0, 2.0, 4.0, NAN, NAN],
        "Pitch_Mean": [100.0, 110.0, 120.0, 130.0, NAN, NAN],
    })


def test_merge_contraction():
    df = _frame()
    df = df[df.upos != "PUNCT"]
    df = df[df.Onset.notna()].reset_index(drop=True)
    m = merge_spoken_words(df, RELOOK)
    assert len(m) == 3
    assert list(m.token) == ["The", "debtor's", "side"]
    assert m.gpt2_Surprisal[1] == 5.0
    assert m.gpt2_Cloze[1] == pytest.approx(0.1)
    assert m.gpt2_TopMatch[1] == 0
    assert m.Sem_Empath_money[1] == 1
    assert m.Zipf_Freq[1] == 4.1
    assert np.isnan(m.AoA_Mean[1])
    assert m.Pitch_Mean[1] == 110.0
    assert m.End[1] == 0.8 and m.Word_Duration[1] == pytest.approx(0.5)


def test_nan_part_makes_sum_nan():
    df = _frame().iloc[:4].copy()
    df.loc[2, "gpt2_Surprisal"] = NAN
    m = merge_spoken_words(df, RELOOK)
    assert np.isnan(m.gpt2_Surprisal[1])


def _keep(df):
    return list(df.columns)


def test_filters_punct_and_unaligned():
    df = _frame()
    v = word_event_vars(df, _keep(df), "txt", {}, RELOOK)
    assert list(v["token"]) == ["The", "debtor's", "side"]
    assert not np.isnan(v["onset"]).any()
    n = len(v["onset"])
    for k in ("offset", "duration", "token", "sentence_id", "token_id",
              "gpt2_Surprisal", "Zipf_Freq"):
        assert len(v[k]) == n
    assert v["text"] == "txt"


def test_keep_columns_respected():
    df = _frame()
    keep = [c for c in df.columns if c != "Zipf_Freq"]
    v = word_event_vars(df, keep, "t", {}, RELOOK)
    assert "Zipf_Freq" not in v
    assert "Zipf_Freq" not in list(v["columns"])
    assert "lemma" not in v and "Transcript" not in v and "Onset" not in v


def test_units_aligned_with_columns():
    df = _frame()
    v = word_event_vars(df, _keep(df), "t", {"gpt2_Surprisal": "bits"}, RELOOK)
    cols = list(v["columns"])
    assert len(v["units"]) == len(cols)
    assert v["units"][cols.index("gpt2_Surprisal")] == "bits"
    assert v["units"][cols.index("AoA_Mean")] == ""


def test_invalid_name_raises():
    df = _frame()
    df["bad-name"] = 1.0
    with pytest.raises(ValueError):
        word_event_vars(df, _keep(df), "t", {}, RELOOK)


def test_none_when_nothing_aligned():
    df = _frame()
    df["Onset"] = NAN
    assert word_event_vars(df, _keep(df), "t", {}, RELOOK) is None
    assert word_event_vars(_frame().drop(columns=["Onset"]), ["token"], "t",
                           {}, RELOOK) is None


def test_every_registry_word_column_has_rule():
    for c in registry.CLUSTERS:
        if c.output_csv == "features":
            for col in c.emits_columns:
                assert merge_rule(col) is not None, col
    assert merge_rule("llama3_8b_Surprisal") == "sum"
    assert merge_rule("Sem_Categories") == "first"


def _syll():
    return pd.DataFrame({
        "token": ["hello", "hello", "."],
        "Label": ["HH AH", "L OW", NAN],
        "ARPABET": ["a", "b", NAN],
        "Onset": [0.0, 0.2, NAN],
        "End": [0.2, 0.5, NAN],
        "Duration": [0.2, 0.3, NAN],
        "P_Center": [0.1, 0.3, NAN],
        "sentence_id": [0, 0, 0],
        "syllable_id": [0, 1, 2],
        "Pitch_Mean": [100.0, 90.0, NAN],
    })


def test_syllable_vars():
    s = _syll()
    v = syllable_event_vars(s, list(s.columns), "t", {})
    assert list(v["label"]) == ["HH AH", "L OW"]
    assert list(v["token"]) == ["hello", "hello"]
    assert list(v["pcenter"]) == [0.1, 0.3]
    assert list(v["columns"]) == ["Pitch_Mean"]
    keep = [c for c in s.columns if c != "P_Center"]
    assert "pcenter" not in syllable_event_vars(s, keep, "t", {})


def test_savemat_roundtrip(tmp_path):
    df = _frame()
    v = word_event_vars(df, _keep(df), "t", {}, RELOOK)
    p = tmp_path / "w.mat"
    scipy.io.savemat(p, v, oned_as="column")
    m = scipy.io.loadmat(p)
    assert m["token"].shape == (3, 1)
    assert m["token"].dtype == object
    assert m["onset"].shape == (3, 1)
