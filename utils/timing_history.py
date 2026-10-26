"""Machine-local per-step timing history for the Run-tab time estimate.

The batch runners in main.py time each expensive stage per stimulus with a
`StageTimer` and append the samples of every *successful* stimulus to
`<config dir>/timing_history.json` (git-ignored, next to run.yaml; honours
SENSI_CONFIG_DIR via utils.preferences.get_config_dir). The estimator reads
the median seconds-per-unit for each (step, model, device) key with `rate()`.

Units: audio seconds for the audio steps (AUDIO_STEPS), whitespace-separated
transcript words for every other step. One-time model loads are kept apart in
`load_seconds` (keyed by `load_key`: "<step>:<model>") so a first run does not inflate the
per-stimulus rates.

The file is disposable learning data: a missing or unreadable file is an empty
history, never an error; deleting it just resets estimates to first guesses.
"""
from __future__ import annotations

import json
import statistics
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path

STEP_NAMES: tuple[str, ...] = (
    "mfa_align", "spacy_parse", "predictability", "word2vec", "bert",
    "prosody_contours", "syllabify", "phoneme_surprisal", "sentence_prosody",
    "acoustic_ts", "doc_summary",
)

# Steps whose `units` are audio seconds; all others count transcript words.
AUDIO_STEPS: frozenset[str] = frozenset(
    {"mfa_align", "prosody_contours", "acoustic_ts", "phoneme_surprisal"})

HISTORY_FILENAME = "timing_history.json"
MAX_SAMPLES_PER_KEY = 200
MIN_SAMPLES_FOR_RATE = 3
_FORMAT_VERSION = 1


@dataclass
class Sample:
    step: str
    model: str | None
    device: str          # "cpu" | "cuda"
    units: float         # audio seconds (AUDIO_STEPS) or words
    seconds: float


@dataclass
class StageTimer:
    """Collects per-stage wall-time samples for one batch run."""
    samples: list[Sample] = field(default_factory=list)
    load_seconds: dict[str, float] = field(default_factory=dict)

    @contextmanager
    def stage(self, step: str, units: float, model: str | None = None,
              device: str = "cpu"):
        """Time the body; append a Sample only if it completes without raising."""
        t0 = time.perf_counter()
        yield
        self.samples.append(Sample(step=step, model=model, device=device,
                                   units=float(units),
                                   seconds=time.perf_counter() - t0))


def load_key(step: str, model: str | None = None) -> str:
    """Key into `load_seconds`: "<step>:<model>", or "<step>" for a step
    with a single fixed model (word2vec, bert)."""
    return f"{step}:{model}" if model else step


def model_device(step: str, model: str | None, cuda_available: bool) -> str:
    """"cuda" or "cpu": where the extractor itself puts this step's model.

    Predictability follows extractors/predictability.py's _HuggingFaceBackend:
    GPU tiers are tried only when CUDA is available and the model is not one
    pinned to full-precision CPU (_is_full_precision_cpu, i.e. GPT-2).
    BERT (extractors/bert_embed.py) never moves its model off the CPU, and
    every other step is CPU-only.
    """
    if step != "predictability" or not cuda_available or not model:
        return "cpu"
    from extractors.predictability import _is_full_precision_cpu
    from extractors.predictability_models import spec_for
    try:
        hf_id = spec_for(model).hf_id
    except KeyError:
        return "cpu"
    return "cpu" if _is_full_precision_cpu(hf_id) else "cuda"


def _default_path() -> Path:
    from utils.preferences import get_config_dir
    return get_config_dir() / HISTORY_FILENAME


def _valid_sample(d) -> bool:
    try:
        return (isinstance(d, dict) and isinstance(d["step"], str)
                and isinstance(d["device"], str)
                and (d["model"] is None or isinstance(d["model"], str))
                and float(d["units"]) >= 0 and float(d["seconds"]) >= 0)
    except (KeyError, TypeError, ValueError):
        return False


def load_history(path: Path | None = None) -> dict:
    """Parsed history ({} when the file is missing, unreadable or malformed).

    Shape: {"version": 1, "samples": [<Sample as dict>, ...],
            "load_seconds": {"<step>:<model>": seconds}}.
    """
    p = Path(path) if path is not None else _default_path()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return {}
    if not isinstance(data, dict) or not isinstance(data.get("samples", []), list):
        return {}
    samples = [s for s in data.get("samples", []) if _valid_sample(s)]
    loads = data.get("load_seconds", {})
    if not isinstance(loads, dict):
        loads = {}
    loads = {str(k): float(v) for k, v in loads.items()
             if isinstance(v, (int, float))}
    return {"version": _FORMAT_VERSION, "samples": samples, "load_seconds": loads}


def _key(d: dict) -> tuple:
    return (d["step"], d["model"], d["device"])


def append_samples(samples: list[Sample], path: Path | None = None, *,
                   load_seconds: dict[str, float] | None = None) -> None:
    """Append samples (and latest model-load times), keeping the most recent
    MAX_SAMPLES_PER_KEY per (step, model, device); atomic write."""
    from utils.config_yaml import _write_text
    p = Path(path) if path is not None else _default_path()
    hist = load_history(p) or {"version": _FORMAT_VERSION, "samples": [],
                               "load_seconds": {}}
    merged = hist["samples"] + [asdict(s) for s in samples]
    # Keep the newest N per key: walk newest-first, then restore order.
    counts: dict[tuple, int] = {}
    kept_rev = []
    for d in reversed(merged):
        k = _key(d)
        if counts.get(k, 0) < MAX_SAMPLES_PER_KEY:
            counts[k] = counts.get(k, 0) + 1
            kept_rev.append(d)
    hist["samples"] = list(reversed(kept_rev))
    if load_seconds:
        hist["load_seconds"].update({k: float(v) for k, v in load_seconds.items()})
    _write_text(p, json.dumps(hist, indent=1) + "\n")


def rate(history: dict, step: str, model: str | None, device: str) -> float | None:
    """Median seconds per unit for (step, model, device); None with < 3 samples."""
    per_unit = [float(d["seconds"]) / float(d["units"])
                for d in (history or {}).get("samples", [])
                if _key(d) == (step, model, device) and float(d["units"]) > 0]
    if len(per_unit) < MIN_SAMPLES_FOR_RATE:
        return None
    return statistics.median(per_unit)
