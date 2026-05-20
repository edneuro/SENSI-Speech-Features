"""Continuous pre-flight check engine.

Planner consumer: reads current preferences + session-only state (API
key), runs the planner to learn which stages will fire, then emits
warnings ONLY for environment/asset prerequisites of those stages.

Toggle-dependency warnings (e.g. 'enable Onset/End in the Acoustic
card') are no longer needed — the planner pulls dependent stages
automatically. This module enforces an invariant (see
tests/gui/test_preflight_invariants.py) that no warning ever
instructs the user to enable a toggle.
"""

from dataclasses import dataclass
from pathlib import Path

from utils.preferences import load_preferences
from gui.planner import plan_from_prefs
from extractors.llm_cloze import LLM_OPTIONS


@dataclass(frozen=True)
class Warning:
    toggle_key: str | None  # None for non-toggle-attached info hints
    severity: str            # "info" | "amber" | "red"
    message: str


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_EWFG_PARQUET = _PROJECT_ROOT / "dictionaries" / "edu_zipf.parquet"
_LIWC_DIC_GLOB = "*.dic"

# MFA detection is expensive (shutil.which is cheap, but the subprocess
# verification we used historically wasn't). Cache the result for the
# process lifetime — MFA install state doesn't change while the GUI is
# running.
_mfa_available: bool | None = None


def _mfa_on_path() -> bool:
    global _mfa_available
    if _mfa_available is None:
        import shutil
        _mfa_available = shutil.which("mfa") is not None
    return _mfa_available


def _parselmouth_importable() -> bool:
    try:
        import parselmouth  # type: ignore  # noqa: F401
        return True
    except ImportError:
        return False


def _pyphen_importable() -> bool:
    try:
        import pyphen  # type: ignore  # noqa: F401
        return True
    except ImportError:
        return False


_PROSODY_STAGES = frozenset({
    "word_prosody", "syllable_prosody", "doc_prosody", "praat_contours",
})


def compute_warnings(session_api_key: str = "") -> list[Warning]:
    prefs = load_preferences()
    plan = plan_from_prefs(prefs)
    in_plan = plan.stages_to_run
    out: list[Warning] = []

    # Empty plan — give a single info hint and bail. Users with zero outputs
    # selected don't need to see env warnings for stages that won't fire.
    if not plan.output_csvs:
        out.append(Warning(
            toggle_key=None, severity="info",
            message="No outputs selected — pick at least one CSV in the Outputs section."))
        return out

    # EWFG dictionary
    if "ewfg_freq" in in_plan and not _EWFG_PARQUET.exists():
        for band in ("freq_ewfg_k2", "freq_ewfg_3to5", "freq_ewfg_6to8",
                     "freq_ewfg_9to12", "freq_ewfg_overall"):
            if prefs["features"].get(band, False):
                out.append(Warning(
                    toggle_key=band, severity="amber",
                    message="EWFG parquet not found at dictionaries/edu_zipf.parquet "
                            "— columns will be NaN. See README for acquisition."))

    # OpenAI without API key
    if "llm_predict" in in_plan:
        llm_model = prefs["predictability"].get("llm_model", "") or LLM_OPTIONS[0][0]
        if llm_model.startswith("gpt-") and not session_api_key:
            for k in ("pred_surprisal_llm", "pred_cloze", "pred_topmatch"):
                if prefs["features"].get(k, False):
                    out.append(Warning(
                        toggle_key=k, severity="amber",
                        message=f"OpenAI model '{llm_model}' selected but no API key "
                                "pasted — paste your key in Predictability."))

    # LIWC backend selected but no .dic
    if "sem_domain" in in_plan and prefs["semantic"].get("backend") == "liwc":
        dict_dir = _PROJECT_ROOT / "dictionaries"
        if not any(dict_dir.glob(_LIWC_DIC_GLOB)):
            out.append(Warning(
                toggle_key="sem_indicators", severity="amber",
                message="LIWC backend selected but no .dic file found in "
                        "dictionaries/. Will fall back to Empath at run time."))

    # MFA binary
    if "mfa_align" in in_plan and not _mfa_on_path():
        out.append(Warning(
            toggle_key=None, severity="red",
            message="MFA binary not found on PATH. Activate the ./env conda env "
                    "before running the GUI."))

    # parselmouth (any prosody stage in plan)
    if set(in_plan) & _PROSODY_STAGES:
        if not _parselmouth_importable():
            out.append(Warning(
                toggle_key=None, severity="amber",
                message="praat-parselmouth not installed — prosody columns "
                        "will be NaN. pip install praat-parselmouth."))

    # pyphen — doc articulation rate
    if "doc_articulation_rate" in in_plan and not _pyphen_importable():
        out.append(Warning(
            toggle_key="feat_doc_prosody_articulation_rate", severity="amber",
            message="pyphen not installed — doc articulation rate will be "
                    "NaN. pip install pyphen."))

    # pyphen — syllable label fallback
    if ("syllabify_pairs" in in_plan
            and prefs["features"].get("feat_syllable_label", False)
            and not _pyphen_importable()):
        out.append(Warning(
            toggle_key="feat_syllable_label", severity="amber",
            message="pyphen not installed — syllable Label will fall back to "
                    "ARPABET-without-stress. pip install pyphen for orthographic labels."))

    return out
