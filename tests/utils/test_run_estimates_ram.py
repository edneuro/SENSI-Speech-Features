"""Peak-memory estimate (utils/run_estimates.estimate_ram)."""
from types import SimpleNamespace

from utils import estimate_constants as K
from utils.run_estimates import estimate_ram

B = K.RAM_GB["baseline"]


def _plan(csvs=(), pred=False, pred_doc=False):
    return SimpleNamespace(output_csvs=frozenset(csvs), predictability_word=pred,
                           predictability_doc=pred_doc)


def _prefs(models=()):
    return {"predictability": {"models": list(models)}}


def _sys(ram=64.0, vram=0.0):
    return {"system_ram_gb": ram,
            "gpu": {"name": "GPU" if vram else None, "vram_gb": vram}}


def test_baseline_only_without_heavy_items():
    e = estimate_ram(_plan(), _prefs(["gpt2"]), _sys())
    assert e.peak_gb == B and e.driver == "the pipeline"
    assert e.fits and e.warning is None


def test_heavy_items_add_up():
    e = estimate_ram(_plan(["word2vec", "bert_word"], pred=True),
                     _prefs(["gpt2"]), _sys())
    want = B + K.RAM_GB["word2vec"] + K.RAM_GB["bert"] + K.RAM_GB["gpt2"]
    assert abs(e.peak_gb - want) < 1e-9
    assert e.driver == "Word2Vec"


def test_bert_counted_once_for_word_and_sentence():
    e = estimate_ram(_plan(["bert_word", "bert_sentence"]), _prefs(), _sys())
    assert abs(e.peak_gb - (B + K.RAM_GB["bert"])) < 1e-9


def test_llama8b_on_big_gpu_does_not_warn():
    e = estimate_ram(_plan(pred=True), _prefs(["llama3_8b"]), _sys(16, 24))
    assert e.fits and e.warning is None
    assert abs(e.peak_gb - B) < 1e-9


def test_llama8b_cpu_only_16gb_warns():
    e = estimate_ram(_plan(pred=True), _prefs(["llama3_8b"]), _sys(16, 0))
    assert not e.fits
    assert e.warning.startswith("⚠ Llama")
    assert "needs ≈" in e.warning and "this computer has 16.0" in e.warning
    assert e.warning.endswith("The run will likely fail or be extremely slow.")


def test_llama_too_big_for_vram_warns():
    e = estimate_ram(_plan(pred=True), _prefs(["llama3_70b"]), _sys(64, 8))
    assert not e.fits and e.warning
    assert e.warning.startswith("⚠ Llama")
    assert "GB of GPU memory; this GPU has 8.0 GB." in e.warning
    assert "this computer" not in e.warning


def test_has_cuda_needs_a_named_gpu_with_memory():
    from utils.run_estimates import has_cuda
    assert has_cuda(_sys(64, 8))
    assert not has_cuda(_sys(64, 0))
    assert not has_cuda({"gpu": {"name": "GPU", "vram_gb": 0}})
    assert not has_cuda({})


def test_predictability_off_ignores_selected_models():
    e = estimate_ram(_plan(), _prefs(["llama3_70b"]), _sys(16))
    assert e.peak_gb == B and e.fits


def test_doc_means_alone_count_models():
    e = estimate_ram(_plan(pred_doc=True), _prefs(["gpt2"]), _sys())
    assert abs(e.peak_gb - (B + K.RAM_GB["gpt2"])) < 1e-9


def test_stacked_items_warn_as_this_run():
    e = estimate_ram(_plan(["word2vec", "bert_word"], pred=True),
                     _prefs(["llama3_8b"]), _sys(16, 0))
    assert not e.fits
    assert e.warning.startswith("⚠ This run needs ≈ ")


def test_stacked_vram_items_warn_as_this_run():
    e = estimate_ram(_plan(pred=True), _prefs(["llama3_8b", "llama3_70b"]),
                     _sys(64, 8))
    assert not e.fits
    assert e.warning.startswith("⚠ This run needs ≈ ")
    assert "GB of GPU memory; this GPU has 8.0 GB" in e.warning


def test_unknown_machine_memory_does_not_warn():
    e = estimate_ram(_plan(["word2vec"], pred=True), _prefs(["llama3_70b"]),
                     {"system_ram_gb": 0, "gpu": {"name": None, "vram_gb": 0}})
    assert e.fits and e.warning is None
