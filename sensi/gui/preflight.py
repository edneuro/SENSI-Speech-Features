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

from sensi.utils.preferences import load_preferences
from sensi.gui.planner import plan_from_prefs
from sensi.gui.planner.plan import (
    DOC_AGG_PREDICTABILITY_CLUSTER_NAME, _active_clusters, _active_output_csvs,
)
# Imported as a module (not `from ... import has_hf_token`) so that tests can
# monkeypatch `extractors.predictability.has_hf_token` and have it take
# effect here — a direct-name import would bind a stale reference at
# module-load time.
from sensi.extractors import predictability
from sensi.extractors.predictability_models import is_gated, all_specs
# Imported as a module (see the `predictability` note above) so tests can
# monkeypatch `extractors.aligner.g2p_model_available` and have it take
# effect here.
from sensi.extractors import aligner


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


def compute_warnings(session_api_key: str = "", session_hf_token: str = "") -> list[Warning]:
    prefs = load_preferences()
    plan = plan_from_prefs(prefs)
    in_plan = plan.stages_to_run
    out: list[Warning] = []

    # Empty plan — give a single info hint and bail. Users with zero outputs
    # selected don't need to see env warnings for stages that won't fire.
    if not plan.output_csvs:
        out.append(Warning(
            toggle_key=None, severity="info",
            message="Nothing will be written — turn on at least one level on the Run "
                    "page, and tick a feature on that level's page."))
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

    # Predictability (v18: models/metrics, not a single llm_model). Local
    # HuggingFace models only — OpenAI is no longer offered, so there is no
    # API-key warning here. Relevant when either predictability path is in
    # play: the word-level Predictability section (Word level on) or the
    # document predictability means (Document level, Aggregates section).
    _active_csvs = _active_output_csvs(prefs)
    pred_section_on = (
        ("features" in _active_csvs
         and bool(prefs["features"].get("master_predictability", False)))
        or any(c.name == DOC_AGG_PREDICTABILITY_CLUSTER_NAME
               for c in _active_clusters(prefs, _active_csvs)))
    pred = prefs.get("predictability", {}) or {}
    models = list(pred.get("models") or [])
    metrics_on = pred.get("metrics", {}) or {}
    metrics = [m for m in ("surprisal", "cloze", "topmatch") if metrics_on.get(m)]
    if plan.predictability_word or plan.predictability_doc:
        _valid_slugs = {spec.slug for spec in all_specs()}
        for slug in models:
            # Skip unrecognized slugs (e.g. a stale/hand-edited prefs file) —
            # is_gated() would KeyError on an unknown slug; preflight must
            # never crash on a malformed-but-loadable prefs file.
            if slug not in _valid_slugs:
                continue
            if (is_gated(slug) and not session_hf_token
                    and not predictability.has_hf_token()):
                out.append(Warning(
                    toggle_key=None, severity="amber",
                    message=f"Model '{slug}' is gated and needs a HuggingFace "
                            "token + accepted license — paste a token in "
                            "Predictability or run `hf auth login`."))
    if pred_section_on:
        if models and not metrics:
            out.append(Warning(
                toggle_key=None, severity="amber",
                message="Predictability model(s) selected but no metric is "
                        "turned on — nothing will be produced. Turn on "
                        "Surprisal, Cloze, or Top-match in Predictability."))
        if metrics and not models:
            out.append(Warning(
                toggle_key=None, severity="amber",
                message="Predictability metric(s) turned on but no model is "
                        "selected — nothing will be produced. Pick at least "
                        "one model in Predictability."))

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

    # G2P for OOV words requested but the vendored model isn't present —
    # today this silently degrades every OOV word to a single `spn` phone
    # (see extractors/aligner._run_mfa), with only a run-time stderr line.
    if ("mfa_align" in in_plan and prefs.get("alignment", {}).get("g2p_oov", False)
            and not aligner.g2p_model_available()):
        out.append(Warning(
            toggle_key=None, severity="amber",
            message="G2P for out-of-dictionary words is on, but the vendored "
                    "MFA G2P model is missing — OOV words will align to a "
                    "single spn phone instead. Run: python utils/download_models.py"))

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
