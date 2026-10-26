"""v21 prefs schema: output.level_* switches replace the ten per-file output
toggles; v20 configs migrate onto them (_migrate_outputs_to_levels)."""
import json
from copy import deepcopy
from pathlib import Path

import pytest

from utils.config_yaml import (
    retired_keys_present, split_prefs, validate_features_yaml,
    write_features_yaml,
)
from utils.preferences import (
    DEFAULT_PREFS, DEFAULT_PREFS_V20, RETIRED_OUTPUT_KEYS_V21,
    _merged_from_user, get_features_yaml_path, load_preferences,
)

_CASES = json.loads(
    (Path(__file__).resolve().parents[1] / "fixtures" / "output_switch_snapshots"
     / "v20_cases.json").read_text(encoding="utf-8"))
_CASE_BY_NAME = {c["name"]: c for c in _CASES}

_LEVELS = ("level_word", "level_document", "level_subword",
           "level_sentence", "level_acoustic")


def _deep_merge(base: dict, over: dict) -> dict:
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = deepcopy(v)
    return base


def _v20_prefs(case: dict) -> dict:
    return _deep_merge(deepcopy(DEFAULT_PREFS_V20), case["prefs_overrides"])


def _migrate(name: str) -> dict:
    merged, changed = _merged_from_user(_v20_prefs(_CASE_BY_NAME[name]))
    assert changed
    return merged


def test_defaults_v21_levels():
    out = DEFAULT_PREFS["output"]
    assert out["level_word"] is True
    for lvl in _LEVELS[1:]:
        assert out[lvl] is False
    assert not set(RETIRED_OUTPUT_KEYS_V21) & set(out)
    for k in ("master_doc_aggregates", "master_doc_lex_diversity",
              "master_doc_readability", "master_doc_category_proportions"):
        assert out[k] is True
    f = DEFAULT_PREFS["features"]
    for k in ("master_semantic_indicators", "master_semantic_similarity",
              "master_sentence_similarity"):
        assert f[k] is True
    assert f["feat_phoneme_surprisal_mat"] is False


# Expected (word, document, sub-word, sentence, acoustic) level values per
# snapshot case: a level is on iff the v20 config wrote any of its files.
_EXPECTED_LEVELS = {
    "defaults":                    (True, False, False, False, False),
    "all_on":                      (True, True, True, True, True),
    "embeddings_without_word_csv": (True, False, False, False, False),
    "syllables_not_phonemes":      (True, False, True, False, False),
    "psr_mat_without_phonemes":    (True, False, True, False, False),
    "bert_sentence_without_csv":   (True, False, False, True, False),
    "acoustic_only":               (False, False, False, False, True),
    "doc_only_with_pred_means":    (False, True, False, False, False),
}


def test_expected_levels_cover_every_case():
    assert set(_EXPECTED_LEVELS) == set(_CASE_BY_NAME)


@pytest.mark.parametrize("case", _CASES, ids=[c["name"] for c in _CASES])
def test_v20_file_migrates_to_levels(case):
    features_data, _ = split_prefs(_v20_prefs(case))
    write_features_yaml(get_features_yaml_path(), features_data)
    prefs = load_preferences()
    assert prefs["schema_version"] == 21
    assert not set(RETIRED_OUTPUT_KEYS_V21) & set(prefs["output"])
    assert tuple(prefs["output"][lvl] for lvl in _LEVELS) ==         _EXPECTED_LEVELS[case["name"]]
    # Persisted: the rewritten file is already v21 and has no retired key.
    reread = load_preferences()
    assert reread == prefs


def test_v20_yaml_validates_with_warning():
    v20_payload, _ = split_prefs(deepcopy(DEFAULT_PREFS_V20))
    assert validate_features_yaml(v20_payload) == []
    assert any("word_level_csv" in w for w in retired_keys_present(v20_payload))


def test_migration_defaults_word_only():
    m = _migrate("defaults")
    assert [m["output"][lvl] for lvl in _LEVELS] == [True, False, False, False, False]
    assert m["features"]["master_structural"] is True  # word CSV was on


def test_migration_embeddings_without_word_csv():
    m = _migrate("embeddings_without_word_csv")
    f = m["features"]
    assert m["output"]["level_word"] is True
    assert f["master_word2vec"] is True and f["master_bert_word"] is True
    # The word CSV was off: every word-CSV section is switched off.
    for k in ("master_structural", "master_lexling", "master_frequency",
              "master_predictability", "master_lexical_norms",
              "master_acoustic", "master_word_prosody",
              "master_semantic_indicators"):
        assert f[k] is False, k


def test_migration_psr_mat_without_phonemes():
    m = _migrate("psr_mat_without_phonemes")
    f = m["features"]
    assert m["output"]["level_subword"] is True
    assert f["feat_phoneme_surprisal_mat"] is True
    assert f["master_phoneme_intervals"] is False
    assert f["feat_phoneme_surprisal"] is False
    assert f["feat_phoneme_cohort_entropy"] is False
    for k in ("master_syllable_intervals", "master_syllable_labels",
              "master_syllable_prosody", "master_syllable_pcenter"):
        assert f[k] is False, k


def test_migration_bert_sentence_without_csv():
    m = _migrate("bert_sentence_without_csv")
    f = m["features"]
    assert m["output"]["level_sentence"] is True
    assert f["master_bert_sentence"] is True
    assert f["master_sentence_timing"] is False
    assert f["master_sentence_prosody"] is False


def test_migration_masters_cleared_when_mat_off():
    # A master on but its .mat output off produced nothing in v20 — it must
    # not start producing a file after migration.
    v20 = deepcopy(DEFAULT_PREFS_V20)
    v20["features"].update(master_word2vec=True, master_bert_word=True,
                           master_bert_sentence=True)
    m, _ = _merged_from_user(v20)
    for k in ("master_word2vec", "master_bert_word", "master_bert_sentence"):
        assert m["features"][k] is False, k


def test_v21_source_not_remigrated():
    v21 = deepcopy(DEFAULT_PREFS)
    v21["output"]["level_word"] = False
    v21["features"]["master_structural"] = True
    m, changed = _merged_from_user(v21)
    assert not changed
    assert m["output"]["level_word"] is False
    assert m["features"]["master_structural"] is True
