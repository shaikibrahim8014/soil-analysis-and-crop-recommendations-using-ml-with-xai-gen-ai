import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from utils.validation import validate_inputs

def test_valid_input():
    assert validate_inputs({
        "N": 50, "P": 40, "K": 50,
        "temperature": 25, "humidity": 70, "ph": 6.5, "rainfall": 100
    })

def test_negative_n():
    with pytest.raises(ValueError):
        validate_inputs({
            "N": -1, "P": 40, "K": 50,
            "temperature": 25, "humidity": 70, "ph": 6.5, "rainfall": 100
        })

def test_invalid_ph():
    with pytest.raises(ValueError):
        validate_inputs({
            "N": 50, "P": 40, "K": 50,
            "temperature": 25, "humidity": 70, "ph": 15, "rainfall": 100
        })
