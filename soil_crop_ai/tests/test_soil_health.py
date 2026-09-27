import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.soil_health import analyze_soil

def test_score_is_bounded():
    result = analyze_soil(50, 40, 50, 25, 70, 6.5, 100)
    assert 0 <= result["score"] <= 100
