from flask import Flask, render_template, request
import pandas as pd
import joblib
import os
import shap

from gemini_advisor import get_gemini_advice

# =========================================================
# FLASK APPLICATION
# =========================================================

app = Flask(__name__)


# =========================================================
# LOAD TRAINED ML MODEL
# =========================================================

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "crop_model.pkl")

model = joblib.load(MODEL_PATH)


# =========================================================
# FEATURE NAMES
# =========================================================

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]


# =========================================================
# SHAP EXPLAINER
# =========================================================

explainer = shap.TreeExplainer(model)


# =========================================================
# HOME PAGE
# =========================================================


@app.route("/")
def home():
    return render_template("index.html")


# =========================================================
# CROP PREDICTION
# =========================================================


@app.route("/predict", methods=["POST"])
def predict():

    try:

        # -------------------------------------------------
        # GET USER INPUT
        # -------------------------------------------------

        N = float(request.form["N"])
        P = float(request.form["P"])
        K = float(request.form["K"])

        temperature = float(request.form["temperature"])

        humidity = float(request.form["humidity"])

        ph = float(request.form["ph"])

        rainfall = float(request.form["rainfall"])

        # -------------------------------------------------
        # CREATE DATAFRAME
        # -------------------------------------------------

        input_data = pd.DataFrame(
            [[N, P, K, temperature, humidity, ph, rainfall]], columns=FEATURES
        )

        # -------------------------------------------------
        # ML PREDICTION
        # -------------------------------------------------

        prediction = model.predict(input_data)

        crop = prediction[0]

        # -------------------------------------------------
        # PREDICTION PROBABILITY
        # -------------------------------------------------

        confidence = None

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(input_data)[0]

            confidence = round(max(probabilities) * 100, 2)

        # -------------------------------------------------
        # SHAP EXPLANATION
        # -------------------------------------------------

        shap_values = explainer(input_data)

        values = shap_values.values

        # Handle multiclass models
        if len(values.shape) == 3:
            predicted_class = list(model.classes_).index(crop)

            feature_values = values[0, :, predicted_class]

        else:
            feature_values = values[0]

        # -------------------------------------------------
        # GET IMPORTANT FEATURES
        # -------------------------------------------------

        feature_importance = []

        for feature, value in zip(FEATURES, feature_values):

            feature_importance.append((feature, float(value)))

        # Sort by absolute SHAP value

        feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)

        # Top 5 features

        top_features = feature_importance[:5]

        # -------------------------------------------------
        # CREATE EXPLANATION FOR GEMINI
        # -------------------------------------------------

        explanation_parts = []

        for feature, value in top_features:

            direction = (
                "positively influenced" if value > 0 else "negatively influenced"
            )

            explanation_parts.append(
                f"{feature} {direction} "
                f"the prediction "
                f"(SHAP value: {value:.4f})"
            )

        shap_explanation = "\n".join(explanation_parts)

        # -------------------------------------------------
        # SOIL DATA FOR GEMINI
        # -------------------------------------------------

        soil_data = {
            "Nitrogen (N)": N,
            "Phosphorus (P)": P,
            "Potassium (K)": K,
            "Temperature": temperature,
            "Humidity": humidity,
            "pH": ph,
            "Rainfall": rainfall,
        }

        # -------------------------------------------------
        # GEMINI AI
        # -------------------------------------------------

        gemini_advice = get_gemini_advice(
            soil_data=soil_data, crop=crop, explanation=shap_explanation
        )

        # -------------------------------------------------
        # RETURN RESULT PAGE
        # -------------------------------------------------

        return render_template(
            "result.html",
            crop=crop,
            confidence=confidence,
            soil_data=soil_data,
            top_features=top_features,
            gemini_advice=gemini_advice,
        )

    except Exception as e:

        print("ERROR:", e)

        return render_template("index.html", error=str(e))


# =========================================================
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":

    app.run(debug=True, host="127.0.0.1", port=5000)
