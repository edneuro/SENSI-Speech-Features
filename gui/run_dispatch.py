"""Background-thread dispatcher to main.run_word_level_batch / run_doc_level_batch.

The hand-rolled feature→stage gating has been replaced by gui.planner; this
file is now a thin shim that loads prefs, runs the planner, and dispatches
batches in the background.
"""

import argparse
import copy
import datetime as _dt
import threading
from typing import Callable

from gui import planner
from utils.preferences import load_preferences


def build_args_word_level() -> argparse.Namespace:
    """Build the argparse.Namespace for run_word_level_batch via the planner.

    Kept as a public function so tests that exercised the old hand-rolled
    version continue to pass after being repointed."""
    prefs = load_preferences()
    plan = planner.plan_from_prefs(prefs)
    return planner.to_namespace(plan, prefs, level="word")


def build_args_doc_level() -> argparse.Namespace:
    """Build the argparse.Namespace for run_doc_level_batch via the planner."""
    prefs = load_preferences()
    plan = planner.plan_from_prefs(prefs)
    return planner.to_namespace(plan, prefs, level="doc")


def _prepare_runtime_args(ns: argparse.Namespace) -> argparse.Namespace:
    """Copy ns; coerce edu_bands list → comma-joined string.

    No-op for planner-produced namespaces (to_namespace already emits a
    string), retained as a safety net for hand-built namespaces in tests
    or future CLI callers that pass a list.
    """
    rt = copy.copy(ns)
    if isinstance(rt.edu_bands, list):
        rt.edu_bands = ",".join(rt.edu_bands) if rt.edu_bands else None
    return rt


def run(progress_cb: Callable[[int, int, str], None],
        done_cb: Callable[[str], None],
        session_api_key: str,
        in_thread: bool = True) -> None:
    """Dispatch a run in a background thread (or inline when in_thread=False).

    progress_cb(current, total, stimulus_id) — forwarded to on_pair_done.
    done_cb("done") or done_cb("error: <message>") on completion.

    Word + doc batches in a single dispatch share a `run_ts` (timestamp
    string) so their log files group together in the Logs tab. The doc
    batch is told to skip its no-match.csv when the word batch is also
    running — both batches scan the same input folders, so the no-match
    list would otherwise be a duplicate.
    """

    def _go():
        import os
        if session_api_key:
            os.environ["OPENAI_API_KEY"] = session_api_key
        prefs = load_preferences()
        plan = planner.plan_from_prefs(prefs)
        run_ts = _dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        try:
            word_outputs = {"features", "syllables", "phonemes"} & plan.output_csvs
            if word_outputs:
                from main import run_word_level_batch
                ns = _prepare_runtime_args(planner.to_namespace(plan, prefs, level="word"))
                ns.run_ts = run_ts
                ns.skip_no_match = False
                run_word_level_batch(ns, on_pair_done=progress_cb)
            if "documents_summary" in plan.output_csvs:
                from main import run_doc_level_batch
                ns = _prepare_runtime_args(planner.to_namespace(plan, prefs, level="doc"))
                ns.run_ts = run_ts
                ns.skip_no_match = bool(word_outputs)
                run_doc_level_batch(ns, on_pair_done=progress_cb)
            if "acoustic" in plan.output_csvs:
                from main import run_acoustic_level_batch
                ns = planner.to_acoustic_namespace(prefs)
                ns.run_ts = run_ts
                run_acoustic_level_batch(ns, on_pair_done=progress_cb)
            done_cb("done")
        except Exception as exc:
            done_cb(f"error: {exc}")
        finally:
            if session_api_key:
                os.environ.pop("OPENAI_API_KEY", None)

    if in_thread:
        threading.Thread(target=_go, daemon=True).start()
    else:
        _go()
