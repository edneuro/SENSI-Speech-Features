"""v23 prefs schema: feat_acoustic_word_gap_after, seeded from the gap-before toggle."""
from copy import deepcopy

from sensi.utils.preferences import (
    DEFAULT_PREFS, DEFAULT_PREFS_V22, SCHEMA_VERSION, _merged_from_user,
)


def test_active_version_is_at_least_23():
    assert SCHEMA_VERSION >= 23
    assert DEFAULT_PREFS["schema_version"] >= 23
    assert DEFAULT_PREFS["features"]["feat_acoustic_word_gap_after"] is False


def test_v22_file_with_gap_on_gets_gap_after_on():
    old = deepcopy(DEFAULT_PREFS_V22)
    old["features"]["feat_acoustic_word_gap"] = True
    merged, changed = _merged_from_user(old)
    assert changed is True and merged["schema_version"] >= 23
    assert merged["features"]["feat_acoustic_word_gap_after"] is True


def test_v22_file_with_gap_off_keeps_gap_after_off():
    merged, _ = _merged_from_user(deepcopy(DEFAULT_PREFS_V22))
    assert merged["features"]["feat_acoustic_word_gap_after"] is False


def test_v23_choice_is_kept():
    prefs = deepcopy(DEFAULT_PREFS)
    prefs["features"]["feat_acoustic_word_gap"] = True   # gap-after stays the user's own
    merged, changed = _merged_from_user(prefs)
    assert changed is False
    assert merged["features"]["feat_acoustic_word_gap_after"] is False
