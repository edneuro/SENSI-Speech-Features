"""v22 prefs schema: additive event .mat toggles."""
from copy import deepcopy

from sensi.utils.preferences import (
    DEFAULT_PREFS, DEFAULT_PREFS_V21, SCHEMA_VERSION, _merged_from_user,
)

_NEW = {
    "master_word_events": True,
    "feat_word_events_mat": False,
    "feat_syllable_events_mat": False,
}


def test_active_version_is_22():
    assert SCHEMA_VERSION == 22
    assert DEFAULT_PREFS["schema_version"] == 22


def test_v21_file_gets_new_keys_at_defaults():
    merged, changed = _merged_from_user(deepcopy(DEFAULT_PREFS_V21))
    assert changed is True
    assert merged["schema_version"] == 22
    for k, v in _NEW.items():
        assert merged["features"][k] is v


def test_v22_file_round_trips_enabled_toggle():
    prefs = deepcopy(DEFAULT_PREFS)
    prefs["features"]["feat_word_events_mat"] = True
    merged, changed = _merged_from_user(prefs)
    assert changed is False
    assert merged["features"]["feat_word_events_mat"] is True
