from extractors.readability import compute_readability


def test_readability_returns_expected_keys():
    out = compute_readability("The cat sat on the mat.")
    assert set(out.keys()) == {'flesch_reading_ease', 'flesch_kincaid_grade'}


def test_simple_text_high_reading_ease_low_grade():
    simple = "The cat ran. The dog jumped. The bird flew."
    out = compute_readability(simple)
    assert out['flesch_reading_ease'] >= 80
    assert out['flesch_kincaid_grade'] <= 5


def test_complex_text_lower_reading_ease_higher_grade():
    simple = "The cat ran. The dog jumped. The bird flew."
    complex_text = (
        "Notwithstanding the aforementioned considerations, "
        "the implementation necessitates contemporaneous reconfiguration "
        "of the underlying infrastructural paradigms."
    )
    simple_out = compute_readability(simple)
    complex_out = compute_readability(complex_text)
    assert complex_out['flesch_reading_ease'] < simple_out['flesch_reading_ease']
    assert complex_out['flesch_kincaid_grade'] > simple_out['flesch_kincaid_grade']


def test_empty_text_returns_nan_values():
    import math
    out = compute_readability("")
    assert math.isnan(out['flesch_reading_ease'])
    assert math.isnan(out['flesch_kincaid_grade'])
