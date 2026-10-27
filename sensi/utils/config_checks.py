"""Value checks for the feature config: ranges, allowed values, cross-field rules.

sensi.utils.config_yaml.validate_features_yaml checks a hand-edited features.yaml for
unknown keys and wrong types; this module checks that the values themselves
make sense (a pitch floor below the ceiling, a positive analysis rate, a known
model slug, ...). Both the headless run.py (via validate_features_yaml) and the
GUI Run tab (on the loaded prefs) call value_errors, so a setting that would
silently produce empty or wrong output is refused the same way on every path.

`data` may be a partial features.yaml payload: a value missing from a section
falls back to DEFAULT_PREFS, and a value of the wrong type is skipped here
(the type check reports it).
"""
from __future__ import annotations

_PITCH_UNITS = ("semitones", "hz")
_ENV_METHODS = ("rms", "hilbert")
_MEL_PRESETS = ("speech", "music", "general")
_SEM_BACKENDS = ("auto", "empath", "liwc")
_METRICS = ("surprisal", "cloze", "topmatch")


_empath_cats: set | None = None


def _empath_categories() -> set | None:
    """Empath's category names, loaded once (the Run tab re-checks often)."""
    global _empath_cats
    if _empath_cats is None:
        try:
            from empath import Empath
            _empath_cats = set(Empath().cats)
        except Exception:  # empath missing/broken: skip this check
            return None
    return _empath_cats


def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def value_errors(data: dict) -> list[str]:
    """Plain-language errors for out-of-range or inconsistent settings; [] = ok."""
    from sensi.utils.preferences import DEFAULT_PREFS as defaults
    if not isinstance(data, dict):
        return []
    errors: list[str] = []

    def section(name: str) -> dict:
        raw = data.get(name)
        merged = dict(defaults.get(name) or {})
        if isinstance(raw, dict):
            merged.update(raw)
        return merged

    def positive(sec: str, key: str, label: str) -> None:
        v = section(sec).get(key)
        if _num(v) and not v > 0:
            errors.append(f"{label} ({sec}.{key}) must be greater than 0, got {v}")

    def one_of(sec: str, key: str, allowed) -> None:
        v = section(sec).get(key)
        if isinstance(v, str) and v not in allowed:
            errors.append(f"{sec}.{key} must be one of {', '.join(allowed)}; got {v!r}")

    # --- prosody --------------------------------------------------------
    pros = section("prosody")
    one_of("prosody", "pitch_units", _PITCH_UNITS)
    positive("prosody", "pitch_floor_hz", "Pitch floor")
    positive("prosody", "pitch_ceiling_hz", "Pitch ceiling")
    floor, ceiling = pros.get("pitch_floor_hz"), pros.get("pitch_ceiling_hz")
    if _num(floor) and _num(ceiling) and floor >= ceiling:
        errors.append(
            f"Pitch floor ({floor:g} Hz) must be below pitch ceiling "
            f"({ceiling:g} Hz); otherwise no pitch is found and every pitch "
            "column is blank (Word features > Prosody)")
    positive("prosody", "pause_threshold_s", "Pause threshold")
    k = pros.get("pcenter_k")
    if _num(k) and not 0 <= k <= 1:
        errors.append(f"P-center k (prosody.pcenter_k) must be between 0 and 1, got {k}")

    # --- acoustic time series -------------------------------------------
    ac = section("acoustic_ts")
    positive("acoustic_ts", "analysis_rate", "Analysis rate")
    one_of("acoustic_ts", "env_method", _ENV_METHODS)
    positive("acoustic_ts", "env_comp", "Envelope compression")
    positive("acoustic_ts", "env_window", "Envelope window")
    one_of("acoustic_ts", "mel_preset", _MEL_PRESETS)
    n_mels, n_mfcc = ac.get("n_mels"), ac.get("n_mfcc")
    for key, v in (("n_mels", n_mels), ("n_mfcc", n_mfcc)):
        if _num(v) and (v < 1 or v != int(v)):
            errors.append(f"acoustic_ts.{key} must be a whole number of at least 1, got {v}")
    if _num(n_mels) and _num(n_mfcc) and n_mfcc > n_mels:
        errors.append(
            f"MFCC count ({n_mfcc:g}) can't exceed the number of mel bands "
            f"({n_mels:g}); MFCCs are computed from those bands (Acoustic features)")

    # --- phoneme surprisal / lexical norms ------------------------------
    positive("phoneme_surprisal", "analysis_sr", "Phoneme analysis rate")
    lat = section("lexical_norms").get("late_acquired_threshold")
    if _num(lat) and lat < 0:
        errors.append(f"lexical_norms.late_acquired_threshold must be 0 or more, got {lat}")
    paths = section("lexical_norms").get("paths")
    if isinstance(paths, dict):
        for name, p in paths.items():
            if not isinstance(p, str):
                errors.append(f"lexical_norms.paths.{name} must be a file path (text)")

    # --- predictability -------------------------------------------------
    from sensi.extractors.predictability_models import all_specs
    slugs = {s.slug for s in all_specs()}
    pred = section("predictability")
    models = pred.get("models")
    if isinstance(models, list):
        for m in models:
            if m not in slugs:
                errors.append(f"unknown predictability model {m!r}; "
                              f"choose from {', '.join(sorted(slugs))}")
    metrics = pred.get("metrics")
    if isinstance(metrics, dict):
        for m, on in metrics.items():
            if m not in _METRICS:
                errors.append(f"unknown predictability metric {m!r}; "
                              f"choose from {', '.join(_METRICS)}")
            elif not isinstance(on, bool):
                errors.append(f"predictability.metrics.{m} must be true/false")
    context = pred.get("context")
    if isinstance(context, dict):
        for m, n in context.items():
            if m not in slugs:
                errors.append(f"predictability.context has unknown model {m!r}")
            elif not (_num(n) and n == int(n) and n > 0):
                errors.append(f"predictability.context.{m} must be a positive whole "
                              f"number of tokens, got {n!r}")

    # --- semantic -------------------------------------------------------
    one_of("semantic", "backend", _SEM_BACKENDS)
    cats = section("semantic").get("categories_empath")
    if isinstance(cats, list):
        bad = [c for c in cats if not isinstance(c, str)]
        if bad:
            errors.append("semantic.categories_empath must be a list of names")
        else:
            known = _empath_categories()
            if known is not None:
                unknown = [c for c in cats if c not in known]
                if unknown:
                    errors.append("unknown Empath categor"
                                  f"{'y' if len(unknown) == 1 else 'ies'}: "
                                  f"{', '.join(unknown)}")
    return errors
