def test_estimate_local_model_is_zero():
    from gui.cost_estimate import estimate
    e = estimate(model_id="llama-3.2-3b", n_input_tokens=10000, n_output_tokens=2000)
    assert e.usd_total == 0.0
    assert e.is_local


def test_estimate_openai_uses_pricing_file():
    from gui.cost_estimate import estimate
    e = estimate(model_id="gpt-4o-mini",
                  n_input_tokens=1_000_000, n_output_tokens=1_000_000)
    # gpt-4o-mini @ $0.15 input + $0.60 output per 1M
    assert abs(e.usd_total - 0.75) < 1e-6
    assert not e.is_local


def test_count_input_tokens_with_tiktoken(tmp_path):
    from gui.cost_estimate import count_input_tokens
    p = tmp_path / "t.txt"
    p.write_text("Hello world. " * 1000, encoding="utf-8")
    n = count_input_tokens([p], model_id="gpt-4o-mini")
    assert 1000 < n < 5000
