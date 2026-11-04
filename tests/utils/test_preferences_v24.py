"""v24 prefs schema: predictability.precision (per-model weight precision)."""
from copy import deepcopy

from sensi.utils.preferences import (
    DEFAULT_PREFS, DEFAULT_PREFS_V23, SCHEMA_VERSION, _merged_from_user,
)


def test_active_version_is_24():
    assert SCHEMA_VERSION == 24
    assert DEFAULT_PREFS["schema_version"] == 24
    assert DEFAULT_PREFS["predictability"]["precision"] == {}


def test_v23_file_gets_empty_precision():
    merged, changed = _merged_from_user(deepcopy(DEFAULT_PREFS_V23))
    assert changed is True and merged["schema_version"] == 24
    assert merged["predictability"]["precision"] == {}


def test_v24_precision_choice_is_kept():
    prefs = deepcopy(DEFAULT_PREFS)
    prefs["predictability"]["precision"] = {"llama3_8b": "4bit"}
    merged, changed = _merged_from_user(prefs)
    assert changed is False
    assert merged["predictability"]["precision"] == {"llama3_8b": "4bit"}
