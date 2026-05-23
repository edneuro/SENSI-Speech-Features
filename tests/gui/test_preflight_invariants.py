"""Preflight invariants — assertions that hold across all feature configs.

The planner owns the toggle→stage dependency graph. Preflight must
never instruct the user to flip a toggle to satisfy a dependency
the planner would resolve automatically.
"""
import pytest

from utils.preferences import save_preferences
from gui.preflight import compute_warnings


def _no_outputs(features=None):
    return {"output": {"word_level_csv": False, "document_summary": False,
                       "syllables_csv": False, "phonemes_csv": False,
                       "sentences_csv": False,
                       "word2vec_mat": False, "bert_word_mat": False,
                       "bert_sentence_mat": False},
            "features": features or {}}


_FEATURE_MATRIX = [
    # Each entry is a (prefs_update, label) pair representing a representative
    # toggle configuration to sweep over.
    ({"output": {"word_level_csv": True, "syllables_csv": False,
                  "phonemes_csv": False, "document_summary": False},
       "features": {"freq_subtlex": True}}, "word_subtlex_only"),
    ({"output": {"word_level_csv": False, "syllables_csv": True,
                  "phonemes_csv": True, "document_summary": False},
       "features": {"feat_syllable_onset": True, "feat_phoneme_label": True}},
     "sidecars_only"),
    ({"output": {"word_level_csv": True, "syllables_csv": False,
                  "phonemes_csv": False, "document_summary": False},
       "features": {"feat_prosody_pitch_mean": True}}, "word_prosody"),
    ({"output": {"word_level_csv": False, "syllables_csv": False,
                  "phonemes_csv": False, "document_summary": True,
                  "doc_agg_zipf_freq": True},
       "features": {"freq_subtlex": True}}, "doc_agg_zipf"),
    ({"output": {"word_level_csv": True, "syllables_csv": False,
                  "phonemes_csv": False, "document_summary": False},
       "features": {"sem_indicators": True},
       "semantic": {"backend": "liwc"}}, "liwc_sem"),
]


@pytest.mark.parametrize("prefs_update,label", _FEATURE_MATRIX,
                          ids=[lbl for _, lbl in _FEATURE_MATRIX])
def test_no_preflight_warning_instructs_user_to_enable(prefs_tmp, prefs_update, label):
    save_preferences(prefs_update)
    for w in compute_warnings(session_api_key=""):
        assert "enable" not in w.message.lower(), (
            f"Preflight told user to enable something ({label}): "
            f"{w.message!r}"
        )


def test_empty_plan_yields_single_info_hint(prefs_tmp):
    save_preferences(_no_outputs())
    warnings = compute_warnings(session_api_key="")
    assert len(warnings) == 1
    assert warnings[0].severity == "info"
    assert warnings[0].toggle_key is None
    assert "No outputs selected" in warnings[0].message


def test_mfa_warning_attaches_to_no_toggle(prefs_tmp, monkeypatch):
    """The MFA binary-missing warning is now a session/env warning,
    not a toggle warning — toggle_key should be None."""
    import gui.preflight as pf
    monkeypatch.setattr(pf, "_mfa_available", False)
    save_preferences({"output": {"syllables_csv": True},
                      "features": {"feat_syllable_onset": True}})
    warnings = compute_warnings(session_api_key="")
    mfa_warnings = [w for w in warnings if "MFA" in w.message]
    assert mfa_warnings, "Expected an MFA-missing warning given monkeypatched _mfa_available=False"
    assert all(w.toggle_key is None for w in mfa_warnings)
