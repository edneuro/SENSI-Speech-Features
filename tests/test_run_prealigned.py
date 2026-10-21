"""run() in a pre-aligned batch reads TextGrids from disk, never spawns MFA.

Also covers the `_prealign_batch` fallback: when the batch MFA invocation
fails outright, the helper must swallow the exception and leave
`_batch_prealigned` False so run() falls back to per-document alignment;
and per-pair isolation: an unreadable transcript skips only its own stimulus.
"""
import types

import pandas as pd
import pytest

import main
from extractors import aligner


def test_run_reads_disk_and_never_calls_mfa(tmp_path, monkeypatch):
    out = tmp_path / "out"
    persist = aligner.alignment_dir(out)
    persist.mkdir(parents=True)
    (persist / "stim.TextGrid").write_text("tg", encoding="utf-8")

    transcript = tmp_path / "stim.txt"
    transcript.write_text("hello world", encoding="utf-8")
    audio = tmp_path / "stim.wav"
    audio.write_bytes(b"RIFFfake")
    aligner._write_sidecar(persist, "stim", str(audio), "hello world")

    # Disk parse returns canned frames; MFA must never be invoked.
    monkeypatch.setattr(
        aligner, "_parse_textgrid",
        lambda p: (pd.DataFrame({"Transcript": ["hello", "world"],
                                 "Onset": [0.0, 0.5], "End": [0.4, 0.9]}),
                   pd.DataFrame({"phone": ["HH"], "onset": [0.0],
                                 "offset": [0.1]})))
    monkeypatch.setattr(aligner, "_align_uncached",
                        lambda a, t, **kw: pytest.fail("MFA must not run"))

    args = types.SimpleNamespace(
        transcript=str(transcript), audio=str(audio), output_folder=str(out),
        _batch_prealigned=True, doc_id=1,
        no_parse=True, no_freq=True, no_edu_freq=True, no_lexical_norms=True,
        no_sem_domain=True, no_surprisal=True, no_llm_cloze=True,
        no_align=False, no_prosody=True, sem_categories="",
        save_prefs=False,
    )
    df = main.run(args)
    assert "Onset" in df.columns
    assert list(df["Onset"]) == [0.0, 0.5]


def test_prealign_batch_fallback_on_mfa_failure(tmp_path, monkeypatch):
    """A batch-wide MFA failure must not raise, and must leave the batch
    unmarked so run() falls back to per-document alignment (run_if_missing)."""

    class FakePair:
        def __init__(self, audio_path, transcript_path):
            self.audio_path = audio_path
            self.transcript_path = transcript_path

    transcript = tmp_path / "stim.txt"
    transcript.write_text("hello world", encoding="utf-8")
    audio = tmp_path / "stim.wav"
    audio.write_bytes(b"RIFFfake")

    pairs = [FakePair(audio, transcript)]

    def _boom(items, persist_dir, g2p=False):
        raise RuntimeError("MFA not installed")

    monkeypatch.setattr(aligner, "ensure_alignments", _boom)

    ns = types.SimpleNamespace(no_align=False)
    main._prealign_batch(ns, pairs, tmp_path / "out")

    assert getattr(ns, "_batch_prealigned", False) is False


def test_prealign_batch_skips_unreadable_transcript(tmp_path, monkeypatch):
    """An unreadable transcript skips only its own stimulus; the readable ones
    still get pre-aligned and the batch is not aborted."""

    class FakePair:
        def __init__(self, stimulus_id, audio_path, transcript_path):
            self.stimulus_id = stimulus_id
            self.audio_path = audio_path
            self.transcript_path = transcript_path

    good_txt = tmp_path / "good.txt"
    good_txt.write_text("hello world", encoding="utf-8")
    good_wav = tmp_path / "good.wav"
    good_wav.write_bytes(b"RIFFgood")
    bad_wav = tmp_path / "bad.wav"
    bad_wav.write_bytes(b"RIFFbad")
    missing_txt = tmp_path / "does_not_exist.txt"  # open() will raise

    pairs = [
        FakePair("bad", bad_wav, missing_txt),
        FakePair("good", good_wav, good_txt),
    ]

    captured = {}

    def _capture(items, persist_dir, g2p=False):
        captured["items"] = list(items)
        captured["g2p"] = g2p

    monkeypatch.setattr(aligner, "ensure_alignments", _capture)

    ns = types.SimpleNamespace(no_align=False)
    main._prealign_batch(ns, pairs, tmp_path / "out")  # must not raise

    audios = [a for a, _ in captured["items"]]
    assert audios == [str(good_wav)], "only the readable stimulus is staged"
    assert getattr(ns, "_batch_prealigned", False) is True
    assert captured["g2p"] is False, "g2p_oov defaults off when unset on args"


def _run_args(transcript, audio, *, g2p_oov, align_cache):
    return types.SimpleNamespace(
        transcript=str(transcript), audio=str(audio), output_folder=None,
        doc_id=1,
        no_parse=True, no_freq=True, no_edu_freq=True, no_lexical_norms=True,
        no_sem_domain=True, no_surprisal=True, no_llm_cloze=True,
        no_align=False, no_prosody=True, sem_categories="",
        save_prefs=False, g2p_oov=g2p_oov, _align_cache=align_cache,
    )


def _canned_alignment(a, t, *, g2p=False):
    return (pd.DataFrame({"Transcript": ["hello", "world"],
                          "Onset": [0.0, 0.5], "End": [0.4, 0.9]}),
           pd.DataFrame({"phone": ["HH"], "onset": [0.0], "offset": [0.1]}))


@pytest.mark.parametrize("g2p_oov", [False, True])
def test_run_cache_hit_log_matches_the_real_align_with_phones_key(
        tmp_path, monkeypatch, capsys, g2p_oov):
    """Fix 1: main.py rebuilds the align_with_phones memo key by hand for a
    log-only cache-hit check. It must use extractors.aligner.cache_key (the
    same helper align_with_phones uses to store the entry) — a hand-rolled
    2-tuple that omits `g2p` never matches, so "[mfa] Reusing cached
    alignment…" never fires even on a genuine memo hit. Covers both g2p
    settings, since g2p is exactly the element that was omitted."""
    transcript = tmp_path / "stim.txt"
    transcript.write_text("hello world", encoding="utf-8")
    audio = tmp_path / "stim.wav"
    audio.write_bytes(b"RIFFfake")

    calls = {"n": 0}

    def _counted(a, t, *, g2p=False, **kw):
        calls["n"] += 1
        return _canned_alignment(a, t, g2p=g2p)

    monkeypatch.setattr(aligner, "_align_uncached", _counted)

    align_cache: dict = {}
    main.run(_run_args(transcript, audio, g2p_oov=g2p_oov, align_cache=align_cache))
    first_out = capsys.readouterr().out
    assert "[mfa] Loading alignment" in first_out
    assert calls["n"] == 1

    # Second call: same audio/transcript/g2p, sharing the same cache dict —
    # align_with_phones actually serves this from its memo (assert via the
    # call count), so the log line reporting it must also fire.
    main.run(_run_args(transcript, audio, g2p_oov=g2p_oov, align_cache=align_cache))
    second_out = capsys.readouterr().out
    assert calls["n"] == 1, "MFA must run at most once per document"
    assert "[mfa] Reusing cached alignment" in second_out
    assert "[mfa] Loading alignment" not in second_out


def test_run_cache_hit_log_does_not_false_positive_across_different_g2p(
        tmp_path, monkeypatch, capsys):
    """A cache entry stored under one g2p setting must not be reported (or
    served) as a hit for the opposite setting — same invariant the memo key
    itself enforces (extractors/aligner.py's `g2p` element)."""
    transcript = tmp_path / "stim.txt"
    transcript.write_text("hello world", encoding="utf-8")
    audio = tmp_path / "stim.wav"
    audio.write_bytes(b"RIFFfake")

    calls = {"n": 0}

    def _counted(a, t, *, g2p=False, **kw):
        calls["n"] += 1
        return _canned_alignment(a, t, g2p=g2p)

    monkeypatch.setattr(aligner, "_align_uncached", _counted)

    align_cache: dict = {}
    main.run(_run_args(transcript, audio, g2p_oov=False, align_cache=align_cache))
    capsys.readouterr()
    main.run(_run_args(transcript, audio, g2p_oov=True, align_cache=align_cache))
    out = capsys.readouterr().out
    assert calls["n"] == 2, "different g2p must not be served from the other's cache entry"
    assert "[mfa] Loading alignment" in out
    assert "[mfa] Reusing cached alignment" not in out


def test_run_forwards_g2p_oov_value_to_align_with_phones(tmp_path, monkeypatch):
    """Final-review Fix 4: main.run()'s call site —
    mfa_align(..., g2p=bool(getattr(args, 'g2p_oov', False))) — was unpinned
    by any test (correct by inspection only). Record the value
    _align_uncached actually receives and assert both settings forward."""
    transcript = tmp_path / "stim.txt"
    transcript.write_text("hello world", encoding="utf-8")
    audio = tmp_path / "stim.wav"
    audio.write_bytes(b"RIFFfake")

    received = {}

    def _capture(a, t, *, g2p=False, **kw):
        received["g2p"] = g2p
        return _canned_alignment(a, t, g2p=g2p)

    monkeypatch.setattr(aligner, "_align_uncached", _capture)

    main.run(_run_args(transcript, audio, g2p_oov=True, align_cache={}))
    assert received["g2p"] is True

    main.run(_run_args(transcript, audio, g2p_oov=False, align_cache={}))
    assert received["g2p"] is False


def test_prealign_batch_threads_g2p_oov_true(tmp_path, monkeypatch):
    """When args.g2p_oov is True, _prealign_batch must pass g2p=True through
    to ensure_alignments — this is the first real caller of that plumbing."""

    class FakePair:
        def __init__(self, audio_path, transcript_path):
            self.audio_path = audio_path
            self.transcript_path = transcript_path

    transcript = tmp_path / "stim.txt"
    transcript.write_text("hello world", encoding="utf-8")
    audio = tmp_path / "stim.wav"
    audio.write_bytes(b"RIFFfake")

    pairs = [FakePair(audio, transcript)]

    captured = {}

    def _capture(items, persist_dir, g2p=False):
        captured["g2p"] = g2p

    monkeypatch.setattr(aligner, "ensure_alignments", _capture)

    ns = types.SimpleNamespace(no_align=False, g2p_oov=True)
    main._prealign_batch(ns, pairs, tmp_path / "out")

    assert captured["g2p"] is True


class _Logger:
    def __init__(self):
        self.warnings = []

    def add_warning(self, msg):
        self.warnings.append(msg)


class _Pair:
    def __init__(self, audio_path, transcript_path):
        self.audio_path = audio_path
        self.transcript_path = transcript_path


def _one_pair(tmp_path):
    txt = tmp_path / "stim.txt"
    txt.write_text("hello world", encoding="utf-8")
    wav = tmp_path / "stim.wav"
    wav.write_bytes(b"RIFFstim")
    return _Pair(wav, txt)


def test_prealign_failure_is_recorded_in_manifest(tmp_path, monkeypatch):
    def _boom(items, persist_dir, g2p=False):
        raise RuntimeError("MFA alignment failed.\nSTDERR: corpus.db locked")

    monkeypatch.setattr(aligner, "ensure_alignments", _boom)
    ns = types.SimpleNamespace(no_align=False)
    log = _Logger()
    main._prealign_batch(ns, [_one_pair(tmp_path)], tmp_path / "out", log)
    assert ns._batch_prealigned is False
    assert log.warnings == [
        "MFA batch pre-alignment failed (MFA alignment failed.); fell back to "
        "per-document alignment"]


def test_prealign_per_stimulus_miss_is_recorded(tmp_path, monkeypatch):
    monkeypatch.setattr(
        aligner, "ensure_alignments",
        lambda items, persist_dir, g2p=False:
            {"cached": [], "aligned": [], "failed": ["stim"]})
    log = _Logger()
    main._prealign_batch(types.SimpleNamespace(no_align=False),
                         [_one_pair(tmp_path)], tmp_path / "out", log)
    assert log.warnings == ["MFA produced no alignment for stim"]


def test_prealign_skips_documents_already_aligned_this_dispatch(tmp_path, monkeypatch):
    """After a fallback pass filled the shared in-memory cache, the next pass
    must not send those documents to MFA again."""
    pair = _one_pair(tmp_path)
    cache = {aligner.cache_key(str(pair.audio_path), "hello world", False): "x"}
    monkeypatch.setattr(
        aligner, "ensure_alignments",
        lambda *a, **kw: pytest.fail("already-aligned docs must not re-run MFA"))
    ns = types.SimpleNamespace(no_align=False, _align_cache=cache)
    main._prealign_batch(ns, [pair], tmp_path / "out", _Logger())
