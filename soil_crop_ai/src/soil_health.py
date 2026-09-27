from utils.validation import validate_inputs

# Simplified project thresholds. They are intentionally documented here rather
# than hidden inside the application. They are not universal agronomic standards.
THRESHOLDS = {
    "N": (40, 100),
    "P": (30, 80),
    "K": (30, 100),
}

def _nutrient_status(value, low, high):
    if value < low:
        return "Low"
    if value > high:
        return "High"
    return "Moderate"

def analyze_soil(N, P, K, temperature, humidity, ph, rainfall):
    values = dict(N=N, P=P, K=K, temperature=temperature,
                  humidity=humidity, ph=ph, rainfall=rainfall)
    validate_inputs(values)

    status = {
        "N": _nutrient_status(N, *THRESHOLDS["N"]),
        "P": _nutrient_status(P, *THRESHOLDS["P"]),
        "K": _nutrient_status(K, *THRESHOLDS["K"]),
        "ph": "Acidic" if ph < 6.0 else ("Alkaline" if ph > 7.5 else "Neutral")
    }

    # Transparent project score: 25 points each for N/P/K being moderate
    # and pH being in the simplified neutral band.
    score = 0
    score += 25 if status["N"] == "Moderate" else 12.5
    score += 25 if status["P"] == "Moderate" else 12.5
    score += 25 if status["K"] == "Moderate" else 12.5
    score += 25 if 6.0 <= ph <= 7.5 else 12.5

    return {
        "status": status,
        "score": score,
        "disclaimer": (
            "The thresholds and score are simplified project rules, not universal "
            "soil-quality standards. Actual recommendations should use laboratory "
            "soil testing and local agricultural guidance."
        )
    }
