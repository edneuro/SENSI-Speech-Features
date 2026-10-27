"""Tests for the (now permanently dead) cost panel in RunSection.

Predictability offers local HuggingFace models only (see
sensi/extractors/predictability.py, sensi/extractors/predictability_models.py) — the
OpenAI backend was removed, so there is no paid usage left to estimate. The
panel is kept in the widget tree (hidden) rather than deleted so a future
paid backend could be wired back in without rebuilding the UI; these tests
lock in that it stays hidden regardless of predictability configuration.
"""

from sensi.utils.preferences import save_preferences


def test_cost_panel_hidden_by_default(qtbot, prefs_tmp):
    from sensi.gui.section_run import RunSection
    section = RunSection(get_session_api_key=lambda: "", status_strip=None)
    qtbot.addWidget(section)
    section.refresh()
    assert section.cost_panel.isHidden()


def test_cost_panel_stays_hidden_with_models_and_metrics_selected(qtbot, prefs_tmp):
    """Even a fully-configured local-model predictability setup (which does
    real, free, local computation) must not surface the cost panel — there
    is no paid backend left for it to estimate."""
    save_preferences({
        "output": {"level_word": True},
        "predictability": {
            "models": ["gpt2", "llama3_8b"],
            "metrics": {"surprisal": True, "cloze": True, "topmatch": True},
        },
    })
    from sensi.gui.section_run import RunSection
    section = RunSection(get_session_api_key=lambda: "", status_strip=None)
    qtbot.addWidget(section)
    section.refresh()
    assert section.cost_panel.isHidden()
