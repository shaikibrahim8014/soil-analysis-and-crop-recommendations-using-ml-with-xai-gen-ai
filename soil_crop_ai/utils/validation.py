def validate_inputs(values):
    for key in ["N", "P", "K", "rainfall"]:
        if float(values[key]) < 0:
            raise ValueError(f"{key} cannot be negative.")

    if not 0 <= float(values["ph"]) <= 14:
        raise ValueError("Soil pH must be between 0 and 14.")

    if not 0 <= float(values["humidity"]) <= 100:
        raise ValueError("Humidity must be between 0 and 100%.")

    # Broad project input guard, not a universal climate limit.
    if not -50 <= float(values["temperature"]) <= 70:
        raise ValueError("Temperature must be between -50°C and 70°C.")

    return True
