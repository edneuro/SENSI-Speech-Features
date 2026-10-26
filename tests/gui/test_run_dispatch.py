"""Tests for gui.run_dispatch — background-thread orchestration."""


def test_build_args_word_only_includes_correct_flags(monkeypatch, tmp_path):
    monkeypatch.setenv("SENSI_PREFS_PATH", str(tmp_path / "prefs.json"))
    from gui.run_dispatch import build_args_word_level
    args = build_args_word_level()
    # Should always include transcripts/audio/output folders and analyzer
    assert hasattr(args, "transcripts_folder")
    assert hasattr(args, "audio_folder")
    assert hasattr(args, "output_folder")
    assert hasattr(args, "analyzer")


def test_build_args_word_filters_ewfg_bands(monkeypatch, tmp_path):
    monkeypatch.setenv("SENSI_PREFS_PATH", str(tmp_path / "prefs.json"))
    from utils.preferences import save_preferences
    save_preferences({"features": {
        "freq_ewfg_k2": True, "freq_ewfg_3to5": False,
        "freq_ewfg_6to8": False, "freq_ewfg_9to12": False,
        "freq_ewfg_overall": True,
    }})
    from gui.run_dispatch import build_args_word_level
    args = build_args_word_level()
    assert set(args.edu_bands.split(",")) == {"k2", "overall"}


def test_build_args_word_propagates_feature_toggles_dict(monkeypatch, tmp_path):
    """v4: word-level args must include the full feature_toggles dict so
    main.py's CSV writer can drop per-column toggled-off columns. Named
    `feature_toggles` (not `features`) to avoid collision with CLI args."""
    monkeypatch.setenv("SENSI_PREFS_PATH", str(tmp_path / "prefs.json"))
    from utils.preferences import save_preferences
    save_preferences({"features": {
        "feat_struct_doc_id": False,
        "feat_acoustic_onset": False,
    }})
    from gui.run_dispatch import build_args_word_level
    args = build_args_word_level()
    assert hasattr(args, "feature_toggles")
    assert args.feature_toggles["feat_struct_doc_id"] is False
    assert args.feature_toggles["feat_acoustic_onset"] is False


def test_build_args_doc_propagates_feature_and_output_toggles(monkeypatch, tmp_path):
    monkeypatch.setenv("SENSI_PREFS_PATH", str(tmp_path / "prefs.json"))
    from utils.preferences import save_preferences
    save_preferences({
        "output": {"out_count_tokens": False},
        "features": {"feat_psych_tone": False},
    })
    from gui.run_dispatch import build_args_doc_level
    args = build_args_doc_level()
    assert hasattr(args, "feature_toggles")
    assert hasattr(args, "output_toggles")
    assert args.output_toggles["out_count_tokens"] is False
    assert args.feature_toggles["feat_psych_tone"] is False


def test_dispatcher_runs_both_when_both_outputs_checked(monkeypatch, tmp_path):
    monkeypatch.setenv("SENSI_PREFS_PATH", str(tmp_path / "prefs.json"))
    from utils.preferences import save_preferences
    # document_summary needs at least one active doc cluster to actually run
    # (opt-in defaults leave doc features off), so enable doc token counts.
    save_preferences({"output": {"level_word": True, "level_document": True,
                                 "out_count_tokens": True}})

    called = []
    def fake_word(args, **kwargs): called.append("word")
    def fake_doc(args, **kwargs): called.append("doc")

    monkeypatch.setattr("main.run_word_level_batch", fake_word, raising=False)
    monkeypatch.setattr("main.run_doc_level_batch", fake_doc, raising=False)

    from gui.run_dispatch import run
    run(progress_cb=lambda *a: None, done_cb=lambda *a: None,
        session_api_key="", in_thread=False)
    assert "word" in called and "doc" in called


def test_dispatcher_shares_one_alignment_cache_across_passes(monkeypatch, tmp_path):
    """Word and doc passes must receive the SAME _align_cache dict so MFA is
    computed once per document and reused (not re-run) in the second pass."""
    monkeypatch.setenv("SENSI_PREFS_PATH", str(tmp_path / "prefs.json"))
    from utils.preferences import save_preferences
    # Enable a doc feature so the doc batch actually runs (opt-in defaults).
    save_preferences({"output": {"level_word": True, "level_document": True,
                                 "out_count_tokens": True}})

    seen = {}
    def fake_word(args, **kwargs): seen["word"] = getattr(args, "_align_cache", None)
    def fake_doc(args, **kwargs): seen["doc"] = getattr(args, "_align_cache", None)
    monkeypatch.setattr("main.run_word_level_batch", fake_word, raising=False)
    monkeypatch.setattr("main.run_doc_level_batch", fake_doc, raising=False)

    from gui.run_dispatch import run
    run(progress_cb=lambda *a: None, done_cb=lambda *a: None,
        session_api_key="", in_thread=False)
    assert isinstance(seen["word"], dict)
    assert seen["word"] is seen["doc"], "both passes must share one alignment cache"


def test_word_args_includes_prosody_config(prefs_tmp):
    from utils.preferences import save_preferences
    from gui.run_dispatch import build_args_word_level
    save_preferences({
        "prosody": {"pitch_units": "hz", "pitch_floor_hz": 60,
                    "pitch_ceiling_hz": 600, "pause_threshold_s": 0.4},
        "features": {"feat_prosody_pitch_mean": True},
    })
    args = build_args_word_level()
    assert args.pitch_units == "hz"
    assert args.pitch_floor == 60.0
    assert args.pitch_ceiling == 600.0
    assert args.pause_threshold == 0.4
    assert args.no_prosody is False  # any feat_prosody_* on → group runs


def test_word_args_no_prosody_when_all_word_toggles_off(prefs_tmp):
    from utils.preferences import save_preferences
    from gui.run_dispatch import build_args_word_level
    save_preferences({"features": {
        **{f"feat_prosody_{k}": False for k in (
            "pitch_mean", "pitch_min", "pitch_max", "pitch_std", "pitch_delta",
            "volume_mean", "volume_min", "volume_max", "volume_std", "volume_delta",
        )},
        # The 4 syllable-prosody columns also force no_prosody=False (Bug A
        # fix); they default to True in fresh prefs, so turn them off here
        # to isolate the word-level prosody path under test.
        "feat_syllable_pitch_mean": False, "feat_syllable_volume_mean": False,
        "feat_syllable_pitch_delta": False, "feat_syllable_volume_delta": False,
    }})
    args = build_args_word_level()
    assert args.no_prosody is True


def test_doc_args_forces_alignment_when_doc_prosody_on(prefs_tmp):
    from utils.preferences import save_preferences
    from gui.run_dispatch import build_args_doc_level
    save_preferences({
        "output": {"level_document": True},
        "features": {
            "feat_doc_prosody_hnr": True,
            "feat_acoustic_onset": False,
        }})
    args = build_args_doc_level()
    assert args.no_align is False
    assert args.no_doc_prosody is False
    assert args.no_prosody is False


def test_doc_args_no_prosody_when_all_doc_toggles_off(prefs_tmp):
    from utils.preferences import save_preferences
    from gui.run_dispatch import build_args_doc_level
    save_preferences({"features": {
        f"feat_doc_prosody_{k}": False for k in (
            "pitch_mean", "pitch_min", "pitch_max", "pitch_std",
            "volume_mean", "volume_min", "volume_max", "volume_std",
            "pitch_delta_mean", "volume_delta_mean",
            "hnr", "articulation_rate",
            "pause_count", "pause_mean_duration",
            "pause_total_duration", "pause_pct_silence",
        )
    }})
    args = build_args_doc_level()
    assert args.no_doc_prosody is True


def test_doc_args_predictability_dict_mirrors_prefs_regardless_of_level(prefs_tmp):
    """v18: the old no_llm_cloze/no_surprisal word-extractor flags and the
    no_doc_mean_llm_cloze/no_doc_mean_surprisal aggregate flags are gone,
    replaced by a single dynamic ns.predictability = {models, metrics} dict
    (mirrors prefs verbatim — see gui/planner/emit.py) plus one aggregate
    flag, ns.no_doc_predictability, driven purely by the doc_agg_predictability
    output toggle. An empty models list must not crash and must not force
    the aggregate flag on.
    Regression for the 2026-05-12 'OpenAIError: Missing credentials' bug
    (the underlying concern — no accidental paid-extractor invocation — now
    expressed via the empty-models-list case below).
    """
    from utils.preferences import save_preferences
    from gui.run_dispatch import build_args_doc_level
    save_preferences({
        "output": {"level_document": True, "doc_agg_predictability": True},
        "predictability": {"models": [],
                            "metrics": {"surprisal": False, "cloze": False, "topmatch": False}},
    })
    args = build_args_doc_level()
    assert args.predictability == {
        "models": [], "metrics": {"surprisal": False, "cloze": False, "topmatch": False},
        "context": {}}
    # The aggregate is still requested (doc_agg_predictability=True); it's
    # main.run_doc_level_batch's job (not the GUI namespace) to no-op
    # gracefully when no predictability columns exist to aggregate.
    assert args.no_doc_predictability is False


def test_doc_args_no_doc_predictability_off_when_aggregate_toggle_off(prefs_tmp):
    """The doc aggregate flag is driven by the doc_agg_predictability output
    toggle, independent of which models/metrics are selected. With the
    aggregate off the doc pass is not on the predictability doc path, so it
    gets no models and every metric off (no language model is loaded)."""
    from utils.preferences import save_preferences
    from gui.run_dispatch import build_args_doc_level
    save_preferences({
        "output": {"level_document": True, "doc_agg_predictability": False},
        "predictability": {"models": ["gpt2"],
                            "metrics": {"surprisal": True, "cloze": False, "topmatch": False}},
    })
    args = build_args_doc_level()
    assert args.predictability == {
        "models": [], "metrics": {"surprisal": False, "cloze": False, "topmatch": False},
        "context": {}}
    assert args.no_doc_predictability is True


def test_doc_args_predictability_driven_by_doc_aggregate_not_models(prefs_tmp):
    """Mirror of the old llm-vs-word-toggle regression: the doc aggregate
    flag is independently controlled by doc_agg_predictability, not by which
    word-level models/metrics happen to be selected."""
    from utils.preferences import save_preferences
    from gui.run_dispatch import build_args_doc_level
    save_preferences({
        "output": {"level_document": True, "doc_agg_predictability": True},
        "predictability": {"models": ["llama32_3b"],
                            "metrics": {"surprisal": False, "cloze": True, "topmatch": False}},
    })
    args = build_args_doc_level()
    assert args.no_doc_predictability is False
    assert args.predictability["models"] == ["llama32_3b"]


def test_doc_args_passes_categories_from_unified_key(prefs_tmp):
    """v7 unified: both word and doc levels read from categories_<backend>.
    The dispatcher must pass the selection via ns.sem_categories so
    add_semantic_domain receives it — otherwise count_<cat>/prop_<cat>
    columns are missing from the doc summary."""
    from utils.preferences import save_preferences
    from gui.run_dispatch import build_args_doc_level
    save_preferences({
        "semantic": {
            "backend": "empath",
            "categories_empath": ["violence", "love", "money"],
        },
    })
    args = build_args_doc_level()
    # Sorted-or-not is implementation-defined, but it must be these three.
    assert args.sem_categories is not None
    assert set(args.sem_categories.split(",")) == {"violence", "love", "money"}


def test_word_args_passes_categories_from_unified_key(prefs_tmp):
    """v7 unified: word-level args reads from the shared categories_<backend> key."""
    from utils.preferences import save_preferences
    from gui.run_dispatch import build_args_word_level
    save_preferences({
        "semantic": {
            "backend": "empath",
            "categories_empath": ["work", "family"],
        },
    })
    args = build_args_word_level()
    assert args.sem_categories is not None
    assert set(args.sem_categories.split(",")) == {"work", "family"}


def test_word_args_runs_parse_when_only_sem_indicators_on(prefs_tmp):
    """Regression: previously `no_parse` was only driven by structural /
    lex-ling toggles, so enabling just sem_indicators (or freq, or any
    other non-acoustic feature) left no_parse=True. Parse skipped → df
    None → every downstream extractor silently no-ops → RuntimeError
    'all extractors disabled'. The user reported this as 'I have a few
    word-level categories enabled but the word CSV never appears'."""
    from utils.preferences import save_preferences
    from gui.run_dispatch import build_args_word_level
    # Turn everything off except sem_indicators
    all_off = {k: False for k in (
        "lexling_tokens", "lexling_pos", "lexling_morph", "lexling_dep",
        "feat_struct_doc_id", "feat_struct_paragraph_id",
        "feat_struct_sentence_id", "feat_struct_sentence_text",
        "feat_struct_char_start", "feat_struct_char_end",
        "feat_struct_token_id", "feat_struct_token_in_sentence_id",
        "freq_subtlex",
        "freq_ewfg_k2", "freq_ewfg_3to5", "freq_ewfg_6to8",
        "freq_ewfg_9to12", "freq_ewfg_overall",
        "pred_surprisal_gpt2", "pred_surprisal_llm",
        "pred_cloze", "pred_topmatch",
        "feat_acoustic_onset", "feat_acoustic_end",
        "feat_acoustic_word_duration", "feat_acoustic_word_gap",
    )}
    save_preferences({"features": {**all_off, "sem_indicators": True}})
    args = build_args_word_level()
    assert args.no_parse is False, (
        "sem_indicators alone must trigger parse — add_semantic_domain "
        "appends columns to a parsed DataFrame and needs it to exist"
    )
    assert args.no_sem_domain is False


def test_word_args_runs_parse_when_only_freq_on(prefs_tmp):
    """Same regression as sem_indicators, for the freq toggles."""
    from utils.preferences import save_preferences
    from gui.run_dispatch import build_args_word_level
    save_preferences({"features": {
        "lexling_tokens": False, "lexling_pos": False,
        "lexling_morph": False, "lexling_dep": False,
        "feat_struct_doc_id": False, "feat_struct_paragraph_id": False,
        "feat_struct_sentence_id": False, "feat_struct_sentence_text": False,
        "feat_struct_char_start": False, "feat_struct_char_end": False,
        "feat_struct_token_id": False, "feat_struct_token_in_sentence_id": False,
        "freq_subtlex": True,  # just this
        "sem_indicators": False,
    }})
    args = build_args_word_level()
    assert args.no_parse is False
    assert args.no_freq is False


def test_doc_args_skip_freq_when_word_freq_off_and_no_aggregate(prefs_tmp):
    """Doc-level dispatcher must not force the freq extractor on. When the
    user has freq_subtlex off AND no doc aggregate needs Zipf, freq should
    not run. (main.py's per-pair force-on re-enables it only when
    mean_zipf_freq is actually wanted.)"""
    from utils.preferences import save_preferences
    from gui.run_dispatch import build_args_doc_level
    save_preferences({
        "features": {
            "freq_subtlex": False,
            "freq_ewfg_k2": False, "freq_ewfg_3to5": False,
            "freq_ewfg_6to8": False, "freq_ewfg_9to12": False,
            "freq_ewfg_overall": False,
        },
        "output": {"doc_agg_zipf_freq": False, "doc_agg_edu_zipf": False,
                   "doc_agg_predictability": False},  # no mean_* needed
    })
    args = build_args_doc_level()
    assert args.no_freq is True
    assert args.no_edu_freq is True


def test_word_args_propagates_subword_master_toggles(prefs_tmp):
    from utils.preferences import save_preferences
    from gui.run_dispatch import build_args_word_level
    save_preferences({"output": {"level_subword": True},
                      "features": {"feat_phoneme_arpabet": True}})  # activate phoneme cluster
    args = build_args_word_level()
    assert args.no_syllables is True
    assert args.no_phonemes is False


def test_word_args_forces_alignment_on_for_any_subword_feature(prefs_tmp):
    from utils.preferences import save_preferences
    from gui.run_dispatch import build_args_word_level
    save_preferences({
        "output": {"level_subword": True},
        "features": {
            "feat_syllable_onset": True,
            # All acoustic_* toggles off
            "feat_acoustic_onset": False, "feat_acoustic_end": False,
            "feat_acoustic_word_duration": False, "feat_acoustic_word_gap": False,
        },
    })
    args = build_args_word_level()
    assert args.no_align is False, "subword features require alignment"


def test_word_args_forces_parse_when_only_subword_on(prefs_tmp):
    """Regression for Bug A: when ONLY sub-word features are enabled (no
    structural, no lex-ling, no freq, no semantic, no word-prosody),
    no_parse was True. Parse skipped → word_df lacked doc_id/sentence_id/
    token_in_sentence_id → syllabify_pairs crashed reading those columns."""
    from utils.preferences import save_preferences
    from gui.run_dispatch import build_args_word_level
    save_preferences({
        "output": {"level_subword": True},
        "features": {
            # Turn EVERY non-acoustic, non-subword toggle off
            "lexling_tokens": False, "lexling_pos": False,
            "lexling_morph": False, "lexling_dep": False,
            "feat_struct_doc_id": False, "feat_struct_paragraph_id": False,
            "feat_struct_sentence_id": False, "feat_struct_sentence_text": False,
            "feat_struct_char_start": False, "feat_struct_char_end": False,
            "feat_struct_token_id": False, "feat_struct_token_in_sentence_id": False,
            "freq_subtlex": False,
            "freq_ewfg_k2": False, "freq_ewfg_3to5": False,
            "freq_ewfg_6to8": False, "freq_ewfg_9to12": False,
            "freq_ewfg_overall": False,
            "pred_surprisal_gpt2": False, "pred_surprisal_llm": False,
            "pred_cloze": False, "pred_topmatch": False,
            "sem_indicators": False,
            "feat_acoustic_onset": False, "feat_acoustic_end": False,
            "feat_acoustic_word_duration": False, "feat_acoustic_word_gap": False,
            "feat_prosody_pitch_mean": False, "feat_prosody_pitch_min": False,
            "feat_prosody_pitch_max": False, "feat_prosody_pitch_std": False,
            "feat_prosody_pitch_delta": False,
            "feat_prosody_volume_mean": False, "feat_prosody_volume_min": False,
            "feat_prosody_volume_max": False, "feat_prosody_volume_std": False,
            "feat_prosody_volume_delta": False,
            # Just one sub-word feature on
            "feat_syllable_label": True,
        },
    })
    args = build_args_word_level()
    assert args.no_parse is False, (
        "sub-word features alone must trigger parse — syllabify_pairs reads "
        "doc_id/sentence_id/token_in_sentence_id from word_df"
    )
    assert args.no_align is False, "sub-word features also require alignment"


def test_word_args_forces_prosody_when_only_syllable_prosody_on(prefs_tmp):
    """Regression for Bug A part 3: a syllable-prosody-only run must force
    prosody on (so contours are computed) AND alignment on (so Onset/End
    exist). The word-level prosody-forcing block only checked feat_prosody_*
    keys; the 4 feat_syllable_*_mean/_delta keys were silently ignored."""
    from utils.preferences import save_preferences
    from gui.run_dispatch import build_args_word_level
    save_preferences({
        "output": {"level_subword": True},
        "features": {
            # All word-level prosody keys off
            "feat_prosody_pitch_mean": False, "feat_prosody_pitch_min": False,
            "feat_prosody_pitch_max": False, "feat_prosody_pitch_std": False,
            "feat_prosody_pitch_delta": False,
            "feat_prosody_volume_mean": False, "feat_prosody_volume_min": False,
            "feat_prosody_volume_max": False, "feat_prosody_volume_std": False,
            "feat_prosody_volume_delta": False,
            # All acoustic toggles off too
            "feat_acoustic_onset": False, "feat_acoustic_end": False,
            "feat_acoustic_word_duration": False, "feat_acoustic_word_gap": False,
            # Just syllable pitch mean on
            "feat_syllable_pitch_mean": True,
        },
    })
    args = build_args_word_level()
    assert args.no_prosody is False, (
        "syllable-prosody features must force word-level prosody on so "
        "Praat contours are extracted for add_syllable_prosody"
    )
    assert args.no_align is False, "prosody also requires alignment"


def test_word_args_no_word_csv_propagates_through_runtime_args(prefs_tmp):
    """no_word_csv is set in _go(), not build_args_word_level. The runtime-prep
    helper must round-trip arbitrary new attributes unchanged."""
    import argparse
    from gui.run_dispatch import _prepare_runtime_args
    ns = argparse.Namespace()
    ns.edu_bands = None
    ns.no_word_csv = True
    out = _prepare_runtime_args(ns)
    assert out.no_word_csv is True


def test_dispatcher_runs_word_batch_for_sidecars_only(prefs_tmp):
    """Regression for Bug A part 1: with word_level_csv=False but
    syllables_csv=True (or phonemes_csv=True), the word batch must still
    fire — sidecars are emitted INSIDE run_word_level_batch. Pass
    no_word_csv=True so main.py knows to skip the features.csv write."""
    from utils.preferences import save_preferences
    save_preferences({
        "output": {
            "level_word": False, "level_subword": True, "level_document": False},
        "features": {"feat_syllable_onset": True},  # activate syllable cluster
    })

    captured = {}
    def fake_word(args, **kwargs):
        captured["word_called"] = True
        captured["no_word_csv"] = getattr(args, "no_word_csv", "MISSING")
    def fake_doc(args, **kwargs):
        captured["doc_called"] = True

    import main as _main
    orig_word, orig_doc = _main.run_word_level_batch, _main.run_doc_level_batch
    _main.run_word_level_batch = fake_word
    _main.run_doc_level_batch = fake_doc
    try:
        from gui.run_dispatch import run
        run(progress_cb=lambda *a: None, done_cb=lambda *a: None,
            session_api_key="", in_thread=False)
    finally:
        _main.run_word_level_batch = orig_word
        _main.run_doc_level_batch = orig_doc

    assert captured.get("word_called") is True, (
        "word batch must fire when only sidecars are requested"
    )
    assert captured.get("no_word_csv") is True, (
        "no_word_csv must be True so main.py skips the features.csv write"
    )
    assert "doc_called" not in captured


def test_dispatcher_passes_no_word_csv_false_when_word_csv_on(prefs_tmp):
    """Symmetric check: when word_level_csv is on, no_word_csv must be False
    so the features.csv write happens as usual."""
    from utils.preferences import save_preferences
    save_preferences({"output": {
        "level_word": True, "level_subword": True, "level_document": False}})

    captured = {}
    def fake_word(args, **kwargs):
        captured["no_word_csv"] = getattr(args, "no_word_csv", "MISSING")
    def fake_doc(args, **kwargs):
        pass

    import main as _main
    orig_word, orig_doc = _main.run_word_level_batch, _main.run_doc_level_batch
    _main.run_word_level_batch = fake_word
    _main.run_doc_level_batch = fake_doc
    try:
        from gui.run_dispatch import run
        run(progress_cb=lambda *a: None, done_cb=lambda *a: None,
            session_api_key="", in_thread=False)
    finally:
        _main.run_word_level_batch = orig_word
        _main.run_doc_level_batch = orig_doc

    assert captured.get("no_word_csv") is False


def test_word_args_no_subword_no_align_force(prefs_tmp):
    from utils.preferences import save_preferences
    from gui.run_dispatch import build_args_word_level
    save_preferences({
        "output": {"level_subword": False, "level_sentence": False},
        "features": {
            # All acoustic toggles off
            "feat_acoustic_onset": False, "feat_acoustic_end": False,
            "feat_acoustic_word_duration": False, "feat_acoustic_word_gap": False,
            # All word-level prosody toggles off (default prefs have these on,
            # which would force no_align=False via the prosody path)
            "feat_prosody_pitch_mean": False, "feat_prosody_pitch_min": False,
            "feat_prosody_pitch_max": False, "feat_prosody_pitch_std": False,
            "feat_prosody_pitch_delta": False,
            "feat_prosody_volume_mean": False, "feat_prosody_volume_min": False,
            "feat_prosody_volume_max": False, "feat_prosody_volume_std": False,
            "feat_prosody_volume_delta": False,
        },
    })
    args = build_args_word_level()
    # No sub-word, no acoustic, no prosody → alignment correctly off
    assert args.no_align is True


def test_dispatcher_sets_and_pops_hf_token(monkeypatch, tmp_path):
    monkeypatch.setenv("SENSI_PREFS_PATH", str(tmp_path / "prefs.json"))
    import os
    from utils.preferences import save_preferences
    save_preferences({"output": {"level_word": True}})
    monkeypatch.delenv("HF_TOKEN", raising=False)

    seen = {}
    def fake_word(args, **kwargs): seen["during"] = os.environ.get("HF_TOKEN")
    monkeypatch.setattr("main.run_word_level_batch", fake_word, raising=False)

    from gui.run_dispatch import run
    run(progress_cb=lambda *a: None, done_cb=lambda *a: None,
        session_api_key="", session_hf_token="hf_test123", in_thread=False)

    assert seen["during"] == "hf_test123"          # set for the run
    assert os.environ.get("HF_TOKEN") is None       # popped afterward


def test_dispatcher_exports_file_token_when_no_session_token(monkeypatch, tmp_path):
    import os
    import gui.run_dispatch as rd
    monkeypatch.setenv("SENSI_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("SENSI_PREFS_PATH", str(tmp_path / "prefs.json"))
    monkeypatch.delenv("HF_TOKEN", raising=False)
    (tmp_path / "hf_token.txt").write_text("hf_from_file\n", encoding="utf-8")
    seen = {}
    def fake_word(args, **kwargs): seen["during"] = os.environ.get("HF_TOKEN")
    monkeypatch.setattr("main.run_word_level_batch", fake_word, raising=False)
    rd.run(progress_cb=lambda *a: None, done_cb=lambda s: None,
           session_api_key="", session_hf_token="", in_thread=False)
    assert seen["during"] == "hf_from_file"          # exported for the run
    assert os.environ.get("HF_TOKEN") is None        # popped afterward


def test_acoustic_namespace_carries_config():
    from gui.planner.emit import to_acoustic_namespace
    prefs = {
        "shared": {"last_audio_folder": "/aud", "last_output_folder": "/out",
                   "analyzer_name": "me"},
        "features": {"feat_acoustic_ts_envelope": True, "feat_acoustic_ts_onset_env": False,
                     "feat_acoustic_ts_logmel": True, "feat_acoustic_ts_mfcc": False},
        "acoustic_ts": {"analysis_rate": 64, "env_method": "hilbert", "env_comp": 0.5,
                        "n_mels": 20, "n_mfcc": 13, "mel_preset": "music"},
    }
    ns = to_acoustic_namespace(prefs)
    assert ns.audio_folder == "/aud" and ns.output_folder == "/out"
    assert ns.acoustic_ts["envelope"] is True and ns.acoustic_ts["onset_env"] is False
    assert ns.acoustic_ts["analysis_rate"] == 64
    assert ns.acoustic_ts["env_method"] == "hilbert"
    assert ns.acoustic_ts["mel_preset"] == "music"


def test_dispatch_passes_cancel_and_stops_between_passes(monkeypatch, tmp_path):
    """should_cancel reaches the word batch, and a cancel during it skips the
    doc pass and reports 'cancelled'."""
    monkeypatch.setenv("SENSI_PREFS_PATH", str(tmp_path / "prefs.json"))
    import main
    from gui import run_dispatch
    from utils.preferences import save_preferences
    save_preferences({"output": {"level_word": True, "level_document": True}})
    seen = {}

    def fake_word(ns, *, on_pair_done=None, should_cancel=None):
        seen["word"] = should_cancel
        flag["on"] = True

    def fake_doc(*a, **kw):
        seen["doc"] = True

    flag = {"on": False}
    monkeypatch.setattr(main, "run_word_level_batch", fake_word)
    monkeypatch.setattr(main, "run_doc_level_batch", fake_doc)
    statuses = []
    run_dispatch.run(progress_cb=lambda *a: None, done_cb=statuses.append,
                     session_api_key="", in_thread=False,
                     should_cancel=lambda: flag["on"])
    assert callable(seen["word"])
    assert "doc" not in seen
    assert statuses == ["cancelled"]
