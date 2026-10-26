"""Section-switch card (ClusterCard): switch, tag, groups, shared settings,
full-depth greying."""
from PySide6.QtWidgets import (
    QCheckBox, QGridLayout, QHBoxLayout, QPushButton, QSpinBox, QWidget,
)

from gui.widgets import ClusterCard, SectionSwitch, ToggleRow
from utils.preferences import load_preferences, save_preferences


def _isolate(tmp_path, monkeypatch):
    monkeypatch.setenv("SENSI_PREFS_PATH", str(tmp_path / "prefs.json"))


def _nested_card(qtbot):
    card = ClusterCard("Prosody", "Word-level prosody",
                       master_key="master_word_prosody")
    qtbot.addWidget(card)
    grid = QGridLayout()
    row = QHBoxLayout()
    cb = QCheckBox("Pitch_Mean")
    row.addWidget(cb)
    grid.addLayout(row, 0, 0)
    card.add_child_layout(grid)
    return card, cb


def test_switch_off_greys_nested_rows(qtbot, tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    save_preferences({"features": {"master_word_prosody": True}})
    card, cb = _nested_card(qtbot)
    assert isinstance(card.master_toggle(), SectionSwitch)
    assert cb.isEnabled() is True
    card.master_toggle().set_checked(False)
    assert cb.isEnabled() is False
    card.master_toggle().set_checked(True)
    assert cb.isEnabled() is True

    # First paint with the key saved False.
    save_preferences({"features": {"master_word_prosody": False}})
    card2, cb2 = _nested_card(qtbot)
    assert card2.master_toggle().is_checked() is False
    assert cb2.isEnabled() is False


def test_switch_does_not_change_ticks(qtbot, tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    save_preferences({"features": {"master_word_prosody": True}})
    card = ClusterCard("Prosody", "", master_key="master_word_prosody")
    qtbot.addWidget(card)
    rows = [ToggleRow(label=f"c{i}") for i in range(3)]
    rows[0].set_checked(True)
    rows[2].set_checked(True)
    for r in rows:
        card.add_child(r)
    before = [r.is_checked() for r in rows]
    card.master_toggle().set_checked(False)
    assert [r.is_checked() for r in rows] == before
    card.master_toggle().set_checked(True)
    assert [r.is_checked() for r in rows] == before


def test_all_none_links_set_group_only(qtbot, tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    save_preferences({"features": {"master_predictability": True}})
    card = ClusterCard("Predictability", "", master_key="master_predictability")
    qtbot.addWidget(card)
    content = QWidget()
    lay = QHBoxLayout(content)
    boxes = [QCheckBox(n) for n in ("GPT-2", "Llama-3.2-3B")]
    boxes[0].setChecked(True)
    for b in boxes:
        lay.addWidget(b)
    card.add_group("Models", content)
    links = {b.text(): b for b in card.findChildren(QPushButton)}

    links["None"].click()
    assert [b.isChecked() for b in boxes] == [False, False]
    assert card.master_toggle().is_checked() is True
    assert load_preferences()["features"]["master_predictability"] is True

    links["All"].click()
    assert [b.isChecked() for b in boxes] == [True, True]
    assert card.master_toggle().is_checked() is True

    card.master_toggle().set_checked(False)
    assert links["All"].isEnabled() is False
    assert links["None"].isEnabled() is False


def test_shared_widget_never_greyed(qtbot, tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    save_preferences({"features": {"master_word_prosody": True}})
    card = ClusterCard("Prosody", "", master_key="master_word_prosody")
    qtbot.addWidget(card)
    floor = QSpinBox()
    card.add_shared(floor, "Pitch settings (used by all prosody sections)")
    card.master_toggle().set_checked(False)
    assert floor.isEnabled() is True


def test_perma_disabled_row_stays_disabled(qtbot, tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    save_preferences({"features": {"master_frequency": False}})
    card = ClusterCard("Frequency", "", master_key="master_frequency")
    qtbot.addWidget(card)
    row = ToggleRow(label="EWFG")
    row.set_permanently_disabled("not digitized")
    card.add_child(row)
    card.master_toggle().set_checked(True)
    assert row.check.isEnabled() is False


def test_tag_states(qtbot, tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    save_preferences({"features": {"master_word_prosody": True}})
    card = ClusterCard("Prosody", "", master_key="master_word_prosody")
    qtbot.addWidget(card)
    row = ToggleRow(label="Pitch_Mean")
    card.add_child(row)
    assert card.tag_text() == "Included — nothing selected"
    row.set_checked(True)
    assert card.tag_text() == "Included"
    assert card.tag_label.text() == "Included"
    card.master_toggle().set_checked(False)
    assert card.tag_text() == "Left out of outputs"
    assert card.tag_label.text() == "Left out of outputs"


def test_set_master_checked_does_not_save(qtbot, tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    save_preferences({"features": {"master_word_prosody": True}})
    card, cb = _nested_card(qtbot)
    seen = []
    card.settings_changed.connect(lambda: seen.append(1))
    card.set_master_checked(False)
    assert cb.isEnabled() is False
    assert load_preferences()["features"]["master_word_prosody"] is True
    assert seen == []
