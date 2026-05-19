"""OpenAI cost estimation using tiktoken + a versioned rate file."""

import json
from dataclasses import dataclass
from pathlib import Path


_PRICING_PATH = Path(__file__).resolve().parent.parent / "config" / "openai_pricing.json"


@dataclass(frozen=True)
class CostEstimate:
    model_id: str
    n_input_tokens: int
    n_output_tokens: int
    usd_input: float
    usd_output: float
    usd_total: float
    rates_last_updated: str
    is_local: bool


def _load_rates() -> dict:
    if not _PRICING_PATH.exists():
        return {"last_updated": "unknown", "rates_per_million_tokens": {}}
    return json.loads(_PRICING_PATH.read_text(encoding="utf-8"))


def estimate(model_id: str, n_input_tokens: int,
             n_output_tokens: int) -> CostEstimate:
    if not model_id.startswith("gpt-"):
        return CostEstimate(
            model_id=model_id, n_input_tokens=n_input_tokens,
            n_output_tokens=n_output_tokens, usd_input=0.0,
            usd_output=0.0, usd_total=0.0,
            rates_last_updated="n/a", is_local=True,
        )
    rates = _load_rates()
    table = rates.get("rates_per_million_tokens", {}).get(model_id, {})
    in_rate = table.get("input", 0.0)
    out_rate = table.get("output", 0.0)
    usd_in = n_input_tokens / 1_000_000 * in_rate
    usd_out = n_output_tokens / 1_000_000 * out_rate
    return CostEstimate(
        model_id=model_id, n_input_tokens=n_input_tokens,
        n_output_tokens=n_output_tokens, usd_input=usd_in,
        usd_output=usd_out, usd_total=usd_in + usd_out,
        rates_last_updated=rates.get("last_updated", "unknown"),
        is_local=False,
    )


def count_input_tokens(transcript_paths: list[Path], model_id: str) -> int:
    import tiktoken
    try:
        enc = tiktoken.encoding_for_model(model_id)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    n = 0
    for p in transcript_paths:
        try:
            n += len(enc.encode(p.read_text(encoding="utf-8")))
        except (OSError, UnicodeDecodeError):
            continue
    return n
