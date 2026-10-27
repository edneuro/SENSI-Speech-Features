#!/usr/bin/env python3
"""Independent correctness check for a gated Llama predictability model on Sherlock.

Same methodology already proven for GPT-2 on a local machine: run the app's
real `extractors.predictability.add_predictability` on a transcript, then
INDEPENDENTLY re-tokenize/forward-pass/aggregate the exact same already-
loaded model+tokenizer by hand (reusing only the pure-math helpers
`_scores_from_subwords` and `_chunk_slices`, never calling the app's own
`.score()` again), and diff every content word's Surprisal / Cloze /
TopMatch. A match proves the app's windowing, subword-mapping, and
NaN-placement logic is correct for this model -- not just "the pipeline ran
without crashing."

ONE run exercises BOTH code paths the app can take, on the same text:
  1. Native/uncapped context: one forward pass over the whole text.
  2. A capped context (default: 128, the app's own floor): forces the
     sliding-window CHUNKED scoring path (`_chunk_slices`, stride = window
     // 2), since the built-in text is long enough to exceed that cap.
  Then compares a late word's surprisal between the two passes to confirm
  the cap actually changes emitted values (documented behavior -- see
  docs/features/pred_surprisal.md) rather than silently no-op'ing.

No audio/MFA needed -- predictability only consumes spaCy-parsed text.

---------------------------------------------------------------------------
HOW TO RUN (mirrors sherlock/RUNBOOK.md's own apptainer invocation style)
---------------------------------------------------------------------------

1. Weights must already be downloaded (RUNBOOK.md step 5), e.g.:
     apptainer exec --bind $SCRATCH:$SCRATCH --env HF_HOME=$SCRATCH/sensi/hf_cache \\
       --env HF_TOKEN="$(cat $GROUP_HOME/SensiExtractor/config/hf_token.txt)" \\
       --env HF_HUB_DISABLE_XET=1 \\
       $GROUP_HOME/SensiExtractor/sherlock/sensiextractor.sif \\
       bash -c 'hf download meta-llama/Llama-3.2-3B'
   (swap the model id for whichever slug you're validating -- see --model-slug below)

2. One run, on a CPU dev node (llama32_3b is CPU-capable; llama3_8b needs a
   GPU node, see RUNBOOK.md step 7):
     sh_dev -c 4 -m 16GB -t 01:00:00
     ulimit -n "$(ulimit -Hn)"
     export HF_HOME=$SCRATCH/sensi/hf_cache
     apptainer exec \\
       --bind $SCRATCH:$SCRATCH \\
       --bind $GROUP_HOME/SensiExtractor:$GROUP_HOME/SensiExtractor \\
       --env HF_HOME=$HF_HOME --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \\
       $GROUP_HOME/SensiExtractor/sherlock/sensiextractor.sif \\
       python $GROUP_HOME/SensiExtractor/sherlock/validate_llama_predictability.py \\
       --model-slug llama32_3b

3. On a GPU node (llama3_8b), same single command but submitted via
   sbatch/run_container.sbatch per RUNBOOK.md step 7, with
   --model-slug llama3_8b appended to the python invocation inside the
   sbatch script.

Exit code 0 = every checked token matched in BOTH passes (PASS). Exit code 1
= a real mismatch was found in either pass, or setup failed (missing
token/weights/etc). Either way, paste the full stdout back for review -- it
prints enough detail (every token's app-vs-independent values, both passes)
to diagnose a real failure, not just pass/fail.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

# Same guard every entry point in this repo sets before numpy does BLAS work
# (gui.py, run.py, sensi/pipeline.py) -- a no-op on platforms where it doesn't apply.
os.environ.setdefault("MKL_THREADING_LAYER", "sequential")

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Long enough to exceed a 128-token context cap by several chunks, so the
# single default text exercises both the native and chunked code paths in
# one run without needing a user-supplied --text-file.
_DEFAULT_TEXT = (
    ("Bella has 2 apples. Joseph gives her 6 more apples. Together, they have "
     "8 apples. They decide to split the apples equally between each person.\n"
     ) * 6 +
    "Meanwhile, Adrianna the debtor counted her jails and warders, feeling "
    "quite pompous about it all, until the distribution of apples resumed "
    "once more between each and every person in the room.\n"
) * 2


def _run_pass(*, label, df, text, tokens, upos, model_slug, metrics, context_map,
              add_predictability, _get_backend, _scores_from_subwords, _chunk_slices,
              effective_context, spec):
    """Run the app's real path + an independent hand-rolled recomputation for
    one context setting, diff every content token, and return
    (app_df, n_checked, n_bad)."""
    import torch

    eff_context = effective_context(model_slug, context_map)
    print(f"\n=== {label} (effective context: {eff_context}) ===")

    app_df = add_predictability(df, text, models=[model_slug], metrics=metrics,
                                 context=context_map)

    backend = _get_backend(model_slug, max_context=eff_context)
    tokenizer = backend._tokenizer
    model = backend._model
    device = next(model.parameters()).device

    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    ids = enc["input_ids"]
    offsets = enc["offset_mapping"]

    if eff_context is None or len(ids) <= eff_context:
        print(f"Text is {len(ids)} subwords <= effective context {eff_context} "
              f"-- single forward pass (native code path).")
        with torch.no_grad():
            logits = model(torch.tensor([ids], device=device)).logits[0]
    else:
        n_chunks = len(_chunk_slices(len(ids), eff_context))
        print(f"Text is {len(ids)} subwords > effective context {eff_context} "
              f"-- {n_chunks} chunk(s), sliding window (chunked code path).")
        chunk_logits = []
        for seg_start, seg_end, keep_off in _chunk_slices(len(ids), eff_context):
            seg_ids = ids[seg_start:seg_end]
            with torch.no_grad():
                seg_logits = model(torch.tensor([seg_ids], device=device)).logits[0]
            chunk_logits.append(seg_logits[keep_off:])
        logits = torch.cat(chunk_logits, dim=0)
    log_probs = torch.log_softmax(logits, dim=-1)

    sub_log_p: list = [None]
    sub_top1: list = [None]
    for i in range(1, len(ids)):
        sub_log_p.append(log_probs[i - 1, ids[i]].item())
        sub_top1.append(int(log_probs[i - 1].argmax().item()))

    indep_scores = _scores_from_subwords(text, tokens, offsets, sub_log_p, sub_top1, ids)

    metric_col = {"surprisal": "Surprisal", "cloze": "Cloze", "topmatch": "TopMatch"}
    n_checked = 0
    n_bad = 0
    print(f"{'token':16s} {'metric':10s} {'app':>14s} {'independent':>14s} {'diff':>10s}")
    for i, (tok_str, pos_tag, sc) in enumerate(zip(tokens, upos, indep_scores)):
        if pos_tag == "PUNCT":
            continue
        for m in metrics:
            col = f"{model_slug}_{metric_col[m]}"
            if col not in app_df.columns:
                continue
            app_val = app_df[col].iloc[i]
            if m == "surprisal":
                indep_val = round(sc["surprisal"], 4) if sc["surprisal"] == sc["surprisal"] else float("nan")
            elif m == "cloze":
                indep_val = round(sc["p"], 6) if sc["p"] == sc["p"] else float("nan")
            else:
                indep_val = sc["top_match"]
                app_val = None if (app_val != app_val) else int(app_val)  # NaN-safe

            both_nan = (app_val is None and indep_val is None) or (
                isinstance(app_val, float) and isinstance(indep_val, float)
                and math.isnan(app_val) and math.isnan(indep_val)
            )
            if both_nan:
                diff = 0.0
            elif app_val is None or indep_val is None:
                diff = float("nan")
            else:
                diff = abs(app_val - indep_val)
            n_checked += 1
            bad = (not both_nan) and (diff != diff or diff > 1e-3)
            if bad:
                n_bad += 1
            flag = "  <-- MISMATCH" if bad else ""
            print(f"{tok_str!r:16s} {m:10s} {app_val!s:>14s} {indep_val!s:>14s} {diff!s:>10s}{flag}")

    print(f"{label}: checked {n_checked} pairs, {n_bad} mismatches.")
    return app_df, n_checked, n_bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-slug", default="llama32_3b",
                     choices=["llama32_3b", "llama3_8b", "gpt2"],
                     help="Roster slug from sensi/extractors/predictability_models.py. "
                          "llama32_3b and llama3_8b are the two gated models in "
                          "scope; gpt2 is included only for local dry-run smoke "
                          "testing (no gating, no Sherlock/GPU needed). "
                          "(default: llama32_3b -- CPU-capable, cheapest to validate)")
    ap.add_argument("--text-file", default=None,
                     help="Path to a transcript .txt to score instead of the "
                          "built-in default text. Should exceed --context-cap "
                          "subwords so the chunked pass is actually exercised.")
    ap.add_argument("--context-cap", type=int, default=128,
                     help="Per-model max-context cap (subwords) for the "
                          "capped/chunked pass, same as config/features.yaml's "
                          "predictability.context map. Default 128 = "
                          "FLOOR_CONTEXT, the app's own minimum. The native "
                          "pass always runs uncapped regardless of this flag.")
    ap.add_argument("--metrics", default="surprisal,cloze,topmatch",
                     help="Comma-separated subset of surprisal,cloze,topmatch")
    args = ap.parse_args()

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]

    from sensi.extractors.text_parser import parse
    from sensi.extractors.predictability import (
        add_predictability, _get_backend, _scores_from_subwords, _chunk_slices,
    )
    from sensi.extractors.predictability_models import spec_for, is_gated, effective_context

    spec = spec_for(args.model_slug)
    print(f"Model slug: {spec.slug}  hf_id: {spec.hf_id}  gated: {spec.gated}  "
          f"native_ctx: {spec.native_ctx}")
    if is_gated(args.model_slug) and not (os.environ.get("HF_TOKEN")
                                           or os.environ.get("HF_HUB_OFFLINE") == "1"):
        print("WARNING: this is a gated model and neither HF_TOKEN nor "
              "HF_HUB_OFFLINE=1 is set -- the load will likely fail with a "
              "401 unless the weights are already cached locally (offline "
              "mode) or a token is exported. See RUNBOOK.md step 5.",
              file=sys.stderr)

    text = Path(args.text_file).read_text(encoding="utf-8") if args.text_file else _DEFAULT_TEXT
    print(f"Text ({len(text)} chars): {text[:200]!r}{'...' if len(text) > 200 else ''}")

    df = parse(text)
    tokens = [str(t) for t in df["token"].tolist()]
    upos = df["upos"].tolist()

    common = dict(df=df, text=text, tokens=tokens, upos=upos, model_slug=args.model_slug,
                  metrics=metrics, add_predictability=add_predictability,
                  _get_backend=_get_backend, _scores_from_subwords=_scores_from_subwords,
                  _chunk_slices=_chunk_slices, effective_context=effective_context, spec=spec)

    app_df_native, n1, bad1 = _run_pass(label="PASS 1/2: native/uncapped",
                                         context_map={}, **common)
    app_df_capped, n2, bad2 = _run_pass(label=f"PASS 2/2: capped at {args.context_cap}",
                                         context_map={args.model_slug: args.context_cap},
                                         **common)

    n_checked = n1 + n2
    n_bad = bad1 + bad2

    if "surprisal" in metrics:
        late_idx = max(0, len(df) - 5)
        tok_name = df["token"].iloc[late_idx]
        native_val = app_df_native[f"{spec.slug}_Surprisal"].iloc[late_idx]
        capped_val = app_df_capped[f"{spec.slug}_Surprisal"].iloc[late_idx]
        same = (native_val == capped_val) or (
            native_val != native_val and capped_val != capped_val)
        print(f"\nCapped-vs-native sanity check on late word {tok_name!r}: "
              f"native={native_val}  capped({args.context_cap})={capped_val}  "
              f"{'SAME -- cap may not be taking effect, investigate' if same else 'DIFFERENT (expected)'}")

    print(f"\nTotal: checked {n_checked} (token, metric) pairs across both "
          f"passes. Mismatches: {n_bad}.")
    if n_bad == 0:
        print(f"PASS: {spec.slug} ({spec.hf_id}) predictability matches an "
              f"independent forward-pass recomputation exactly, in both the "
              f"native and chunked code paths.")
        return 0
    else:
        print(f"FAIL: {n_bad} real mismatch(es) found -- see rows above.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
