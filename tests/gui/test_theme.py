def test_accent_color_is_hex():
    from gui.theme import ACCENT_COLOR
    assert ACCENT_COLOR.startswith("#")
    assert len(ACCENT_COLOR) == 7


def test_grid_unit_is_8():
    from gui.theme import GRID_UNIT
    assert GRID_UNIT == 8


def test_qss_for_appearance_returns_valid_css(qtbot):
    from gui.theme import qss_for_appearance
    # Light-only theme (dark removed in batch 3 polish).
    qss = qss_for_appearance("light")
    assert "background" in qss
    assert "QCheckBox" in qss  # checkbox indicator must be themed
