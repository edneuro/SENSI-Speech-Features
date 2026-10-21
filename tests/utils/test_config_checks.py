"""Value checks shared by run.py (validate_features_yaml) and the GUI Run tab."""
import pytest

from utils.config_checks import value_errors
from utils.config_yaml import validate_features_yaml
from utils.preferences import DEFAULT_PREFS


def test_defaults_are_valid():
    assert value_errors(DEFAULT_PREFS) == []


@pytest.mark.parametrize("data,fragment", [
    ({"prosody": {"pitch_floor_hz": 400, "pitch_ceiling_hz": 100}}, "Pitch floor"),
    ({"prosody": {"pitch_floor_hz": 600}}, "Pitch floor"),          # vs default ceiling 500
    ({"prosody": {"pause_threshold_s": 0}}, "Pause threshold"),
    ({"prosody": {"pitch_units": "mel"}}, "prosody.pitch_units"),
    ({"acoustic_ts": {"analysis_rate": 0}}, "Analysis rate"),
    ({"acoustic_ts": {"analysis_rate": -10}}, "Analysis rate"),
    ({"acoustic_ts": {"env_comp": 0}}, "Envelope compression"),
    ({"acoustic_ts": {"n_mels": 0}}, "acoustic_ts.n_mels"),
    ({"acoustic_ts": {"n_mfcc": 40, "n_mels": 16}}, "MFCC count"),
    ({"acoustic_ts": {"env_method": "peak"}}, "acoustic_ts.env_method"),
    ({"phoneme_surprisal": {"analysis_sr": -1}}, "Phoneme analysis rate"),
    ({"predictability": {"models": ["gpt3"]}}, "unknown predictability model"),
    ({"predictability": {"metrics": {"Surprisal": True}}}, "unknown predictability metric"),
    ({"predictability": {"context": {"gpt2": "lots"}}}, "predictability.context.gpt2"),
    ({"semantic": {"backend": "empth"}}, "semantic.backend"),
    ({"semantic": {"categories_empath": ["not_a_category_xyz"]}}, "unknown Empath"),
])
def test_bad_values_are_reported(data, fragment):
    errors = value_errors(data)
    assert any(fragment in e for e in errors), errors


def test_validate_features_yaml_includes_value_errors():
    errors = validate_features_yaml({"prosody": {"pitch_floor_hz": 400,
                                                 "pitch_ceiling_hz": 100}})
    assert any("Pitch floor" in e for e in errors)


def test_non_mapping_for_a_mapping_setting_is_a_type_error():
    errors = validate_features_yaml({"predictability": {"context": "oops"}})
    assert any("predictability.context" in e for e in errors)
