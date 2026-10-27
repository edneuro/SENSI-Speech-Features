"""Per-run log + manifest + no-match.csv writer.

Three artifacts per batch run, all sharing a timestamp, written to a
`logs/` subdirectory of the run's output folder:
    <output>/logs/<prefix>_<YYYY-MM-DD_HH-MM-SS>.log              # human-readable log
    <output>/logs/<prefix>_<YYYY-MM-DD_HH-MM-SS>_manifest.json    # structured manifest
    <output>/logs/<prefix>_<YYYY-MM-DD_HH-MM-SS>_no-match.csv     # only if files didn't pair

Used by main.py's batch mode and gui.py.
"""

import csv
import datetime as _dt
import json
import sys
import traceback
from pathlib import Path
from typing import Any

from utils.system_info import gather_system_info

try:
    from __init__ import __version__  # type: ignore
except Exception:
    __version__ = "unknown"


class _StdoutTee:
    """Mirrors writes to both the original stream and the log file handle.

    Installed on sys.stdout / sys.stderr while a RunLogger is active so the
    `[spacy] …` / `[mfa] …` progress prints in main.py reach the .log file
    (which the GUI Logs panel tails live). The original stream still
    receives the writes so an attached terminal keeps showing them.
    """
    def __init__(self, orig, log_fh):
        self._orig = orig
        self._fh = log_fh

    def write(self, s):
        try:
            n = self._orig.write(s)
        except Exception:
            n = len(s) if s else 0
        if s:
            try:
                self._fh.write(s)
                self._fh.flush()
            except Exception:
                pass
        return n if isinstance(n, int) else (len(s) if s else 0)

    def flush(self):
        for stream in (self._orig, self._fh):
            try:
                stream.flush()
            except Exception:
                pass

    def isatty(self):
        return False

    def __getattr__(self, name):
        return getattr(self._orig, name)


class RunLogger:
    def __init__(
        self,
        output_folder: Path,
        analyzer: str,
        config: dict[str, Any],
        input_transcripts_folder: Path,
        input_audio_folder: Path,
        text_extensions: list[str],
        audio_extensions: list[str],
        filename_prefix: str = 'run',
        subdir: str = 'logs',
        run_ts: str | None = None,
        skip_no_match: bool = False,
    ):
        self.output_folder = Path(output_folder)
        # Logs go under <output>/<subdir>/ to keep the main output folder
        # focused on the feature CSVs the user actually cares about.
        self._logs_dir = self.output_folder / subdir if subdir else self.output_folder
        self._logs_dir.mkdir(parents=True, exist_ok=True)

        self._start = _dt.datetime.now()
        # run_ts is shared between the word + doc batches that get dispatched
        # together (see gui/run_dispatch.py), so their log files group cleanly
        # by timestamp in the Logs tab.
        self._run_ts = run_ts if run_ts is not None else self._start.strftime('%Y-%m-%d_%H-%M-%S')
        ts = self._run_ts
        self._log_path = self._logs_dir / f'{filename_prefix}_{ts}.log'
        self._manifest_path = self._logs_dir / f'{filename_prefix}_{ts}_manifest.json'
        self._no_match_path = self._logs_dir / f'{filename_prefix}_{ts}_no-match.csv'
        self._skip_no_match = skip_no_match

        self._analyzer = analyzer
        self._config = config
        self._input_transcripts_folder = Path(input_transcripts_folder)
        self._input_audio_folder = Path(input_audio_folder)
        self._text_extensions = list(text_extensions)
        self._audio_extensions = list(audio_extensions)

        self._pairs: list[dict] = []
        self._warnings: list[str] = []
        self._errors: list[str] = []
        self._unmatched_text: list[Path] = []
        self._unmatched_audio: list[Path] = []

        self._log_fh = self._log_path.open('w', encoding='utf-8')
        self._info(f'Run started by {analyzer} at {self._start.isoformat(timespec="seconds")}')

        # Tee stdout + stderr into the log file so the per-step progress
        # prints from main.py (`[spacy] …`, `[mfa] …`, etc.) reach the
        # Logs panel's live-tail. Restored in finalize().
        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        sys.stdout = _StdoutTee(self._orig_stdout, self._log_fh)
        sys.stderr = _StdoutTee(self._orig_stderr, self._log_fh)

    # ------------- public API -------------

    @property
    def log_path(self) -> Path:
        """Path to the .log file being written. The Logs tab live-tails this."""
        return self._log_path

    @property
    def logs_dir(self) -> Path:
        """Subdirectory where artifacts are written (default <output>/logs)."""
        return self._logs_dir

    @property
    def run_ts(self) -> str:
        """Timestamp used in artifact filenames. Shared across batches when
        the caller passed `run_ts=` so word + doc logs match."""
        return self._run_ts

    def add_warning(self, msg: str) -> None:
        self._warnings.append(msg)
        self._write_log('WARN', msg)

    def add_error(self, msg: str) -> None:
        self._errors.append(msg)
        self._write_log('ERROR', msg)

    def record_pair_success(
        self,
        stimulus_id: str,
        transcript_path: Path,
        audio_path: Path | str,
        runtime_seconds: float,
        output_csv_path: Path | None,
        output_summary_path: Path | None,
    ) -> None:
        self._pairs.append({
            'stimulus_id': stimulus_id,
            'transcript_path': str(transcript_path),
            'audio_path': str(audio_path),
            'runtime_seconds': round(runtime_seconds, 3),
            'status': 'success',
            'output_csv_path': str(output_csv_path) if output_csv_path else None,
            'output_summary_path': str(output_summary_path) if output_summary_path else None,
            'error_message': None,
        })
        self._info(f'  ok {stimulus_id} ({runtime_seconds:.1f}s)')

    def record_pair_error(
        self,
        stimulus_id: str,
        transcript_path: Path,
        audio_path: Path | str,
        runtime_seconds: float,
        error_message: str,
    ) -> None:
        self._pairs.append({
            'stimulus_id': stimulus_id,
            'transcript_path': str(transcript_path),
            'audio_path': str(audio_path),
            'runtime_seconds': round(runtime_seconds, 3),
            'status': 'error',
            'output_csv_path': None,
            'output_summary_path': None,
            'error_message': error_message,
        })
        self._errors.append(f'{stimulus_id}: {error_message}')
        self._write_log('ERROR', f'  err {stimulus_id}: {error_message}')

    def record_unmatched_text(self, path: Path) -> None:
        self._unmatched_text.append(Path(path))

    def record_unmatched_audio(self, path: Path) -> None:
        self._unmatched_audio.append(Path(path))

    def finalize(self) -> None:
        try:
            self._finalize_impl()
        finally:
            # Always restore the streams, even if manifest write fails.
            if getattr(self, '_orig_stdout', None) is not None:
                sys.stdout = self._orig_stdout
                self._orig_stdout = None
            if getattr(self, '_orig_stderr', None) is not None:
                sys.stderr = self._orig_stderr
                self._orig_stderr = None

    def _finalize_impl(self) -> None:
        end = _dt.datetime.now()
        total = (end - self._start).total_seconds()
        self._info(f'Run finished at {end.isoformat(timespec="seconds")} ({total:.1f}s total)')

        manifest = {
            'start_time': self._start.isoformat(timespec='seconds'),
            'end_time': end.isoformat(timespec='seconds'),
            'total_runtime_seconds': round(total, 3),
            'analyzer': self._analyzer,
            'sensi_extractor_version': __version__,
            'system_info': gather_system_info(),
            'config': self._config,
            'input_transcripts_folder': str(self._input_transcripts_folder),
            'input_audio_folder': str(self._input_audio_folder),
            'output_folder': str(self.output_folder),
            'text_extensions_searched': self._text_extensions,
            'audio_extensions_searched': self._audio_extensions,
            'n_pairs_matched': len(self._pairs),
            'n_pairs_processed': sum(1 for p in self._pairs if p['status'] == 'success'),
            'n_unmatched_text': len(self._unmatched_text),
            'n_unmatched_audio': len(self._unmatched_audio),
            'pairs': self._pairs,
            'warnings': self._warnings,
            'errors': self._errors,
        }

        with self._manifest_path.open('w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2)

        # no-match.csv — only emit if (a) we have unmatched files AND
        # (b) this RunLogger wasn't told to skip (the second batch in a
        # combined word+doc GUI run uses skip_no_match=True since its
        # unmatched list is identical to the first batch's).
        has_unmatched = bool(self._unmatched_text or self._unmatched_audio)
        if has_unmatched and not self._skip_no_match:
            with self._no_match_path.open('w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(
                    f, fieldnames=['filename', 'found_in_folder', 'missing_counterpart', 'reason']
                )
                writer.writeheader()
                for p in self._unmatched_text:
                    writer.writerow({
                        'filename': p.name,
                        'found_in_folder': str(p.parent),
                        'missing_counterpart': str(self._input_audio_folder),
                        'reason': 'no audio file with matching base name',
                    })
                for p in self._unmatched_audio:
                    writer.writerow({
                        'filename': p.name,
                        'found_in_folder': str(p.parent),
                        'missing_counterpart': str(self._input_transcripts_folder),
                        'reason': 'no transcript file with matching base name',
                    })

        self._log_fh.close()

    # ------------- internals -------------

    def _info(self, msg: str) -> None:
        self._write_log('INFO', msg)

    def _write_log(self, level: str, msg: str) -> None:
        ts = _dt.datetime.now().strftime('%H:%M:%S')
        line = f'[{ts}] {level}: {msg}\n'
        self._log_fh.write(line)
        self._log_fh.flush()
