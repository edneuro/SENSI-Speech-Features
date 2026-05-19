"""Continuous pre-flight check engine.

Reads current preferences + session-only state (API key) and returns a
list of inline warnings keyed by the toggle they should appear next to.
"""

from dataclasses import dataclass
from pathlib import Path

from utils.preferences import load_preferences


@dataclass(frozen=True)
class Warning:
    toggle_key: str   # which ToggleRow it attaches to
    severity: str     # "amber" or "red"
    message: str


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_EWFG_PARQUET = _PROJECT_ROOT / "dictionaries" / "edu_zipf.parquet"
_LIWC_DIC_GLOB = "*.dic"

# MFA detection is expensive (subprocess + Kaldi init can be 1-3s on Windows).
# Cache the result for the lifetime of the process — MFA install state doesn't
# change while the GUI is running.
_mfa_available: bool | None = None


def _mfa_on_path() -> bool:
    global _mfa_available
    if _mfa_available is None:
        import shutil
        _mfa_available = shutil.which("mfa") is not None
    return _mfa_available


def _resolve_categories_backend(prefs: dict) -> str:
    """Concrete 'liwc' or 'empath' for level-keyed category lookups."""
    backend = prefs["semantic"].get("backend", "auto")
    if backend == "auto":
        from extractors.semantic_domain import detect_backend
        backend, _ = detect_backend()
    return backend


def compute_warnings(session_api_key: str = "") -> list[Warning]:
    prefs = load_preferences()
    f = prefs["features"]
    out: list[Warning] = []

    # EWFG missing
    if not _EWFG_PARQUET.exists():
        for band in ("freq_ewfg_k2", "freq_ewfg_3to5", "freq_ewfg_6to8",
                     "freq_ewfg_9to12", "freq_ewfg_overall"):
            if f.get(band, False):
                out.append(Warning(
                    toggle_key=band, severity="amber",
                    message="EWFG parquet not found at dictionaries/edu_zipf.parquet "
                            "— columns will be NaN. See README for acquisition."))

    # OpenAI without API key
    llm_model = prefs["predictability"].get("llm_model", "")
    if llm_model.startswith("gpt-") and not session_api_key:
        for k in ("pred_surprisal_llm", "pred_cloze", "pred_topmatch"):
            if f.get(k, False):
                out.append(Warning(
                    toggle_key=k, severity="amber",
                    message=f"OpenAI model '{llm_model}' selected but no API key "
                            "pasted — paste your key in Predictability."))

    # LIWC backend selected but no .dic
    if prefs["semantic"]["backend"] == "liwc":
        dict_dir = _PROJECT_ROOT / "dictionaries"
        if not any(dict_dir.glob(_LIWC_DIC_GLOB)):
            if f.get("sem_indicators", False):
                out.append(Warning(
                    toggle_key="sem_indicators", severity="amber",
                    message="LIWC backend selected but no .dic file found in "
                            "dictionaries/. Will fall back to Empath at run time."))

    # MFA detection — uses shutil.which (filesystem check, no subprocess) and
    # caches the result per process. Was previously subprocess.run() with a 3s
    # timeout, which froze the GUI on every toggle.
    if f.get("acoustic_boundaries", False) and not _mfa_on_path():
        out.append(Warning(
            toggle_key="acoustic_boundaries", severity="red",
            message="MFA binary not found on PATH. Activate the sensi conda env "
                    "or disable this feature."))

    # Prosody requires MFA alignment, which run_dispatch enables whenever
    # ANY of the 4 feat_acoustic_* toggles is on. Mirror that here so we
    # don't false-positive when the user has e.g. only feat_acoustic_end on.
    _any_acoustic_on = any(f.get(k, True) for k in (
        "feat_acoustic_onset", "feat_acoustic_end",
        "feat_acoustic_word_duration", "feat_acoustic_word_gap"))
    if not _any_acoustic_on:
        for key in ("feat_prosody_pitch_mean", "feat_prosody_pitch_min",
                    "feat_prosody_pitch_max", "feat_prosody_pitch_std",
                    "feat_prosody_pitch_delta",
                    "feat_prosody_volume_mean", "feat_prosody_volume_min",
                    "feat_prosody_volume_max", "feat_prosody_volume_std",
                    "feat_prosody_volume_delta"):
            if f.get(key, True):
                out.append(Warning(
                    toggle_key=key, severity="amber",
                    message="Prosody requires MFA alignment — enable Onset/End in the Acoustic card.",
                ))

    # Prosody requires praat-parselmouth.
    try:
        import parselmouth  # type: ignore  # noqa: F401
        _parselmouth_ok = True
    except ImportError:
        _parselmouth_ok = False
    if not _parselmouth_ok and f.get("feat_prosody_pitch_mean", True):
        out.append(Warning(
            toggle_key="feat_prosody_pitch_mean", severity="amber",
            message="Install praat-parselmouth to enable prosody features.",
        ))

    # Doc articulation rate requires pyphen.
    try:
        import pyphen  # type: ignore  # noqa: F401
        _pyphen_ok = True
    except ImportError:
        _pyphen_ok = False
    if not _pyphen_ok and f.get("feat_doc_prosody_articulation_rate", True):
        out.append(Warning(
            toggle_key="feat_doc_prosody_articulation_rate", severity="amber",
            message="Install pyphen to enable articulation rate.",
        ))

    # Categories selected but the parent toggle that consumes them is off.
    # Without this warning the user's selection is silently ignored — they
    # see the count in the picker dialog but no Sem_* / count_* columns
    # appear in the output. Compute the backend once so a missing dict /
    # detect_backend failure doesn't poison both checks.
    try:
        backend = _resolve_categories_backend(prefs)
    except Exception:
        backend = None

    if backend is not None:
        if not f.get("sem_indicators", False):
            word_cats = prefs["semantic"].get(f"word_categories_{backend}", [])
            if word_cats:
                n = len(word_cats)
                out.append(Warning(
                    toggle_key="sem_indicators", severity="amber",
                    message=f"{n} word-level categor{'y' if n == 1 else 'ies'} "
                            "selected but Semantic indicators is off — enable it "
                            "or the selection won't produce any Sem_* columns."))

        if not prefs["output"].get("doc_category_proportions", False):
            doc_cats = prefs["semantic"].get(f"doc_categories_{backend}", [])
            if doc_cats:
                n = len(doc_cats)
                out.append(Warning(
                    toggle_key="output_doc_category_proportions", severity="amber",
                    message=f"{n} doc-level categor{'y' if n == 1 else 'ies'} "
                            "selected but Category proportions is off — enable it "
                            "or no count_<cat> / prop_<cat> columns will appear."))

    # Sub-word features require MFA alignment. Defaults mirror the prosody
    # check above: schema defaults are True for the 12 syllable/phoneme keys
    # (feat_syllable_arpabet is the lone False, but load_preferences merges
    # all keys into the dict so the default rarely fires in practice).
    _any_subword = any(f.get(k, True) for k in (
        "feat_syllable_onset", "feat_syllable_end", "feat_syllable_duration",
        "feat_syllable_label", "feat_syllable_arpabet",
        "feat_syllable_pitch_mean", "feat_syllable_volume_mean",
        "feat_syllable_pitch_delta", "feat_syllable_volume_delta",
        "feat_phoneme_label", "feat_phoneme_onset",
        "feat_phoneme_end", "feat_phoneme_duration",
    ))
    _any_acoustic = any(f.get(k, True) for k in (
        "feat_acoustic_onset", "feat_acoustic_end",
        "feat_acoustic_word_duration", "feat_acoustic_word_gap"))
    if _any_subword and not _any_acoustic:
        out.append(Warning(
            toggle_key="feat_syllable_onset", severity="amber",
            message="Sub-word features require MFA alignment — enable Onset/End "
                    "in the Acoustic card on the Word-level features page.",
        ))

    # Syllable Label degrades to ARPABET-without-stress when pyphen is
    # unavailable. _pyphen_ok was already computed above for the articulation
    # rate check; reuse it here rather than re-importing.
    if not _pyphen_ok and f.get("feat_syllable_label", True):
        out.append(Warning(
            toggle_key="feat_syllable_label", severity="amber",
            message="pyphen not installed — syllable Label will fall back to "
                    "ARPABET-without-stress. Install pyphen for orthographic labels.",
        ))

    return out
