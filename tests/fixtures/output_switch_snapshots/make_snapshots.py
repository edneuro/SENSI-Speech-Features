"""Snapshot the v20 planner's behaviour for representative configs.

Run on the unmodified v20 code to (re)generate v20_cases.json. Later tasks
rebuild each case from ``prefs_overrides`` and compare.
"""
import json
import os
import sys
import tempfile
from copy import deepcopy
from pathlib import Path

os.environ["SENSI_CONFIG_DIR"] = tempfile.mkdtemp(prefix="sensi_snap_")
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from utils.preferences import DEFAULT_PREFS  # noqa: E402
from gui.planner import plan_from_prefs  # noqa: E402


def deep_merge(base: dict, over: dict) -> dict:
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_merge(base[k], v)
        else:
            base[k] = deepcopy(v)
    return base


def _bools(section: str, prefix: str = "") -> dict:
    return {k: True for k, v in DEFAULT_PREFS[section].items()
            if isinstance(v, bool) and k.startswith(prefix)}


def _feat(*prefixes: str) -> dict:
    return {k: True for k, v in DEFAULT_PREFS["features"].items()
            if isinstance(v, bool) and k.startswith(prefixes)}


PRED = {"models": ["gpt2"], "metrics": {"surprisal": True, "cloze": True}}

CASES = [
    ("defaults", {}),
    ("all_on", {
        "features": _bools("features"),
        "output": _bools("output"),
        "predictability": PRED,
    }),
    ("embeddings_without_word_csv", {
        "output": {"word_level_csv": False, "word2vec_mat": True, "bert_word_mat": True},
        "features": {"master_word2vec": True, "master_bert_word": True},
    }),
    ("syllables_not_phonemes", {
        "output": {"syllables_csv": True, "phonemes_csv": False},
        "features": _feat("feat_syllable_", "feat_phoneme_"),
    }),
    ("psr_mat_without_phonemes", {
        "output": {"phonemes_csv": False, "phoneme_surprisal_mat": True},
        "features": {"feat_phoneme_surprisal": True},
    }),
    ("bert_sentence_without_csv", {
        "output": {"sentences_csv": False, "bert_sentence_mat": True},
        "features": {"master_bert_sentence": True, **_feat("feat_sentence_")},
    }),
    ("acoustic_only", {
        "output": {"word_level_csv": False, "acoustic_ts_mat": True},
        "features": {"feat_acoustic_ts_envelope": True},
    }),
    ("doc_only_with_pred_means", {
        "output": {"word_level_csv": False, "document_summary": True,
                   "doc_agg_predictability": True},
        "predictability": {"models": ["gpt2"], "metrics": {"surprisal": True}},
    }),
]


def main() -> None:
    out = []
    for name, over in CASES:
        merged = deep_merge(deepcopy(DEFAULT_PREFS), over)
        plan = plan_from_prefs(merged)
        out.append({
            "name": name,
            "prefs_overrides": over,
            "output_csvs": sorted(plan.output_csvs),
            "stages_to_run": list(plan.stages_to_run),
        })
    path = Path(__file__).with_name("v20_cases.json")
    path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    for c in out:
        print(c["name"], c["output_csvs"])


if __name__ == "__main__":
    main()
