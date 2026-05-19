"""Tests for the planner-gated cost panel in RunSection.

The cost panel must appear only when both conditions hold:
  1. 'llm_predict' is in plan.stages_to_run (i.e. at least one LLM toggle is on
     AND the features CSV output is requested).
  2. The configured LLM model starts with 'gpt-'.
"""

import pytest
from utils.preferences import save_preferences


def test_cost_panel_hidden_when_llm_toggles_off(qtbot, prefs_tmp):
    """GPT model set but all LLM feature toggles are off — cost panel must be hidden."""
    save_preferences({
        "output": {"word_level_csv": True},
        "predictability": {"llm_model": "gpt-4o-mini"},
        "features": {
            "pred_surprisal_llm": False,
            "pred_cloze": False,
            "pred_topmatch": False,
        },
    })
    from gui.section_run import RunSection
    section = RunSection(get_session_api_key=lambda: "", status_strip=None)
    qtbot.addWidget(section)
    section.refresh()
    # setVisible(False) sets the explicit-hide flag; isHidden() reflects it
    # even for unshown top-level widgets.
    assert section.cost_panel.isHidden()


def test_cost_panel_visible_when_llm_toggle_on(qtbot, prefs_tmp):
    """GPT model set AND at least one LLM toggle is on — cost panel must not be hidden."""
    save_preferences({
        "output": {"word_level_csv": True},
        "predictability": {"llm_model": "gpt-4o-mini"},
        "features": {
            "pred_surprisal_llm": True,
            "pred_cloze": False,
            "pred_topmatch": False,
        },
    })
    from gui.section_run import RunSection
    section = RunSection(get_session_api_key=lambda: "", status_strip=None)
    qtbot.addWidget(section)
    section.refresh()
    # isVisible() is False for unshown top-level widgets regardless of setVisible().
    # isHidden() reflects the widget's own explicit-hide flag — False means
    # setVisible(True) was called (the cost panel should be shown when rendered).
    assert not section.cost_panel.isHidden()
