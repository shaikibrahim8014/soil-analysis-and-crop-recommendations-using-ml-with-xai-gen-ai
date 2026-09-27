from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import shap

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "crop_model.pkl"
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

def _load():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Train the model first.")
    return joblib.load(MODEL_PATH)

def _get_tree_model(artifact):
    model = artifact["model"]
    if hasattr(model, "named_steps") and "model" in model.named_steps:
        return model.named_steps["model"]
    return model

def global_feature_importance():
    artifact = _load()
    model = _get_tree_model(artifact)
    if not hasattr(model, "feature_importances_"):
        raise ValueError("Global SHAP bar chart requires a tree-based primary model.")
    X = artifact["model"].named_steps["preprocessor"].transform(
        pd.DataFrame([{f: 0 for f in FEATURES}])
    )
    # Feature names are unchanged because the preprocessor has only numeric features.
    return pd.DataFrame({
        "feature": FEATURES,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

def explain_prediction(values):
    artifact = _load()
    model_pipeline = artifact["model"]
    row = pd.DataFrame([values], columns=FEATURES)
    predicted_id = model_pipeline.predict(row)[0]
    crop = artifact["label_encoder"].inverse_transform([predicted_id])[0]

    if not hasattr(model_pipeline, "named_steps"):
        raise ValueError("Expected a scikit-learn pipeline.")

    pre = model_pipeline.named_steps["preprocessor"]
    estimator = model_pipeline.named_steps["model"]
    transformed = pre.transform(row)

    try:
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(transformed)

        if isinstance(shap_values, list):
            class_index = int(predicted_id)
            vals = np.asarray(shap_values[class_index])[0]
        else:
            arr = np.asarray(shap_values)
            if arr.ndim == 3:
                vals = arr[0, :, int(predicted_id)]
            else:
                vals = arr[0]

        contributions = pd.DataFrame({
            "feature": FEATURES,
            "value": [values[f] for f in FEATURES],
            "shap_value": vals
        })
        contributions["absolute_impact"] = contributions["shap_value"].abs()
        contributions = contributions.sort_values("absolute_impact", ascending=False)

        top = contributions.head(3)
        phrases = []
        for _, r in top.iterrows():
            direction = "supported" if r.shap_value >= 0 else "pushed against"
            phrases.append(
                f'{r.feature} {direction} the model output (SHAP={r.shap_value:.4f}).'
            )

        text = (
            f"The model predicted {crop}. The strongest model-level contributors "
            f"for this individual prediction were: " + " ".join(phrases) +
            " These statements describe model behavior, not guaranteed agricultural causation."
        )
        return {"predicted_crop": str(crop),
                "contributions": contributions,
                "plain_english": text}
    except Exception as e:
        raise RuntimeError(f"SHAP could not explain this model: {e}")
