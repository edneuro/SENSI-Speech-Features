"""Pair text files and audio files by base filename (case-insensitive).

- Top-level scan only (no recursion).
- Default text extensions: txt, tsv, md.
- Default audio extensions: wav, mp3, flac.
- Ambiguous matches resolved by extension priority order with a warning.
- Unmatched files reported separately for the no-match.csv.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

DEFAULT_TEXT_EXTS = ['txt', 'tsv', 'md']
DEFAULT_AUDIO_EXTS = ['wav', 'flac', 'mp3']


@dataclass
class Pair:
    stimulus_id: str  # base name (lower-cased for canonical id)
    transcript_path: Path
    audio_path: Path


@dataclass
class PairingResult:
    pairs: list[Pair] = field(default_factory=list)
    unmatched_text: list[Path] = field(default_factory=list)
    unmatched_audio: list[Path] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def find_pairs(
    transcripts_dir: Path,
    audio_dir: Path,
    text_exts: Iterable[str] | None = None,
    audio_exts: Iterable[str] | None = None,
) -> PairingResult:
    """Pair files in transcripts_dir with files in audio_dir by base filename."""
    text_exts = list(text_exts) if text_exts is not None else DEFAULT_TEXT_EXTS
    audio_exts = list(audio_exts) if audio_exts is not None else DEFAULT_AUDIO_EXTS

    transcripts_dir = Path(transcripts_dir)
    audio_dir = Path(audio_dir)

    text_by_base = _scan_top_level(transcripts_dir, text_exts)
    audio_by_base = _scan_top_level(audio_dir, audio_exts)

    result = PairingResult()
    for base, txt_candidates in text_by_base.items():
        chosen_text, txt_warning = _resolve_priority(txt_candidates, text_exts, base, 'text')
        if txt_warning:
            result.warnings.append(txt_warning)

        if base in audio_by_base:
            chosen_audio, aud_warning = _resolve_priority(
                audio_by_base[base], audio_exts, base, 'audio'
            )
            if aud_warning:
                result.warnings.append(aud_warning)
            result.pairs.append(Pair(
                stimulus_id=base,
                transcript_path=chosen_text,
                audio_path=chosen_audio,
            ))
        else:
            result.unmatched_text.append(chosen_text)

    for base, aud_candidates in audio_by_base.items():
        if base not in text_by_base:
            chosen_audio, _ = _resolve_priority(aud_candidates, audio_exts, base, 'audio')
            result.unmatched_audio.append(chosen_audio)

    result.pairs.sort(key=lambda p: p.stimulus_id)
    result.unmatched_text.sort(key=lambda p: p.stem.lower())
    result.unmatched_audio.sort(key=lambda p: p.stem.lower())
    return result


def _scan_top_level(folder: Path, exts: list[str]) -> dict[str, list[Path]]:
    """Map base-name (lowercased, no extension) -> list of matching paths."""
    if not folder.is_dir():
        return {}
    ext_set = {e.lower().lstrip('.') for e in exts}
    out: dict[str, list[Path]] = {}
    for entry in folder.iterdir():
        if not entry.is_file():
            continue
        ext = entry.suffix.lower().lstrip('.')
        if ext not in ext_set:
            continue
        base = entry.stem.lower()
        out.setdefault(base, []).append(entry)
    return out


def _resolve_priority(
    candidates: list[Path],
    priority: list[str],
    base: str,
    kind: str,
) -> tuple[Path, str | None]:
    """Pick the highest-priority candidate; return warning if multiple existed."""
    priority_lower = [e.lower().lstrip('.') for e in priority]

    def rank(p: Path) -> int:
        ext = p.suffix.lower().lstrip('.')
        return priority_lower.index(ext) if ext in priority_lower else len(priority_lower)

    candidates_sorted = sorted(candidates, key=rank)
    chosen = candidates_sorted[0]
    if len(candidates_sorted) > 1:
        others = ', '.join(p.name for p in candidates_sorted[1:])
        warn = f"Ambiguous {kind} match for '{base}': chose {chosen.name} (also found: {others})"
        return chosen, warn
    return chosen, None
