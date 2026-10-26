"""Shape checks for the measured Run-tab estimate constants."""
from extractors.predictability_models import all_specs
from utils import estimate_constants as K
from utils.timing_history import STEP_NAMES


def test_every_roster_model_has_ram_entries():
    for spec in all_specs():
        assert spec.slug in K.RAM_GB, spec.slug
        assert f"{spec.slug}@cuda" in K.RAM_GB, spec.slug
    for key in ("baseline", "word2vec", "bert"):
        assert key in K.RAM_GB


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


def test_load_seconds_positive_and_llama_4bit_below_cpu():
    assert K.FIRST_LOAD_SECONDS["phoneme_surprisal"] > 0
    for key, value in K.FIRST_LOAD_SECONDS.items():
        assert value > 0, key
    for spec in all_specs():
        if spec.gated:
            # 4-bit body + fp16 embeddings is still far below the BF16 CPU load.
            assert K.RAM_GB[f"{spec.slug}@cuda"] < K.RAM_GB[spec.slug] / 2


def test_every_roster_model_has_a_first_load_time():
    for spec in all_specs():
        key = f"predictability:{spec.slug}"
        assert K.FIRST_LOAD_SECONDS.get(key, 0) > 0, key
    # Scaled by weight bytes read: every Llama reads more than GPT-2.
    gpt2 = K.FIRST_LOAD_SECONDS["predictability:gpt2"]
    for spec in all_specs():
        if spec.gated:
            assert K.FIRST_LOAD_SECONDS[f"predictability:{spec.slug}"] > gpt2
