"""Widget tests for ClusterCard."""
import pytest
from utils.preferences import save_preferences
from gui.widgets import ClusterCard, ToggleRow


def test_cluster_card_disables_children_when_master_off(qtbot, prefs_tmp):
    save_preferences({"features": {"master_word_prosody": False}})
    card = ClusterCard("Prosody", "Word-level prosody",
                       master_key="master_word_prosody")
    qtbot.addWidget(card)
    child = ToggleRow(label="Pitch_Mean")
    card.add_child(child)
    assert child.isEnabled() is False


def test_cluster_card_enables_children_when_master_on(qtbot, prefs_tmp):
    save_preferences({"features": {"master_word_prosody": True}})
    card = ClusterCard("Prosody", "Word-level prosody",
                       master_key="master_word_prosody")
    qtbot.addWidget(card)
    child = ToggleRow(label="Pitch_Mean")
    card.add_child(child)
    assert child.isEnabled() is True


def test_cluster_card_master_toggle_persists(qtbot, prefs_tmp):
    save_preferences({"features": {"master_word_prosody": True}})
    card = ClusterCard("Prosody", "Word-level prosody",
                       master_key="master_word_prosody")
    qtbot.addWidget(card)
    card.master_toggle().set_checked(False)
    # set_checked emits toggled; the slot calls save_preferences
    from utils.preferences import load_preferences
    assert load_preferences()["features"]["master_word_prosody"] is False


def test_cluster_card_unavailable_disables_master(qtbot, prefs_tmp):
    save_preferences({"features": {"master_psych_composite": True},
                      "semantic": {"backend": "empath"}})
    card = ClusterCard("Psych", "LIWC psych composite",
                       master_key="master_psych_composite",
                       available=False,
                       disabled_subtitle="Requires LIWC backend")
    qtbot.addWidget(card)
    # When available=False, the master checkbox itself is disabled (greyed
    # out at the Qt level) regardless of the prefs value.
    assert card.master_toggle().check.isEnabled() is False


def test_cluster_card_output_section_persists_correctly(qtbot, prefs_tmp):
    save_preferences({"output": {"master_doc_prosody": True}})
    card = ClusterCard("Doc prosody", "Document prosody",
                       master_key="master_doc_prosody",
                       master_section="output")
    qtbot.addWidget(card)
    card.master_toggle().set_checked(False)
    from utils.preferences import load_preferences
    assert load_preferences()["output"]["master_doc_prosody"] is False
