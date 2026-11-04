"""Shape checks for the measured Run-tab estimate constants."""
from sensi.extractors.predictability_models import all_specs
from sensi.utils import estimate_constants as K
from sensi.utils.timing_history import STEP_NAMES


def test_every_roster_model_has_ram_for_each_precision():
    for spec in all_specs():
        for prec in ("fp32", *spec.precisions):
            assert K.predictability_ram_gib(spec.slug, prec) > 0, (spec.slug, prec)
    for key in ("baseline", "word2vec", "bert"):
        assert key in K.RAM_GB


def test_ram_rule():
    assert K.predictability_ram_gib("gpt2", "fp32") == 0.70
    want = 6.0 * 1.2 / 1.0737
    assert abs(K.predictability_ram_gib("gpt2_xl", "fp32") - want) < 0.01
    want = 16 * 1.2 / 1.0737
    assert abs(K.predictability_ram_gib("llama3_8b", "fp16") - want) < 0.01
    assert abs(K.predictability_ram_gib("llama3_8b", "8bit") - want / 2) < 0.01


def test_every_roster_model_has_a_first_rate_and_load_time():
    for spec in all_specs():
        key = f"predictability:{spec.slug}"
        assert K.FIRST_ESTIMATE_RATES.get(key, 0) > 0, key
        assert K.FIRST_LOAD_SECONDS.get(key, 0) > 0, key


def test_roster_module_does_not_import_torch():
    import subprocess
    import sys
    code = ("import sys, sensi.extractors.predictability_models;"
            "sys.exit(1 if 'torch' in sys.modules else 0)")
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


def test_every_step_has_a_first_rate():
    for step in STEP_NAMES:
        assert step in K.FIRST_ESTIMATE_RATES, step


def test_constants_are_positive():
    for art in ("syllables", "phonemes"):
        assert K.ROWS_PER_WORD[art] > 0
    assert K.SENTENCES_PER_WORD > 0
    for art in ("features", "syllables", "phonemes", "sentences", "documents_summary"):
        assert K.CSV_BYTES_PER_CELL[art] > 0
    for table in (K.RAM_GB, K.FIRST_ESTIMATE_RATES):
        for key, value in table.items():
            assert value > 0, key
    lo, hi = K.FIRST_ESTIMATE_SPREAD
    assert 0 < lo < 1 < hi
    lo, hi = K.LEARNED_SPREAD
    assert 0 < lo < 1 < hi


def test_load_seconds_positive_and_llama_4bit_below_fp16():
    assert K.FIRST_LOAD_SECONDS["phoneme_surprisal"] > 0
    for key, value in K.FIRST_LOAD_SECONDS.items():
        assert value > 0, key
    for spec in all_specs():
        if spec.gated:
            # 4-bit body + fp16 embeddings is still far below the fp16 load.
            assert (K.predictability_ram_gib(spec.slug, "4bit")
                    < K.predictability_ram_gib(spec.slug, "fp16") * 0.6)


def test_every_roster_model_has_a_first_load_time():
    for spec in all_specs():
        key = f"predictability:{spec.slug}"
        assert K.FIRST_LOAD_SECONDS.get(key, 0) > 0, key
    # Scaled by weight bytes read: every Llama reads more than GPT-2.
    gpt2 = K.FIRST_LOAD_SECONDS["predictability:gpt2"]
    for spec in all_specs():
        if spec.gated:
            assert K.FIRST_LOAD_SECONDS[f"predictability:{spec.slug}"] > gpt2
