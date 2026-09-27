from pathlib import Path
import pandas as pd
import joblib

from utils.validation import validate_inputs

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "crop_model.pkl"

def model_available():
    return MODEL_PATH.exists()

def _load():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Trained model not found. Run python src/train_model.py")
    return joblib.load(MODEL_PATH)

def predict_top3(N, P, K, temperature, humidity, ph, rainfall):
    values = dict(N=N, P=P, K=K, temperature=temperature,
                  humidity=humidity, ph=ph, rainfall=rainfall)
    validate_inputs(values)

    artifact = _load()
    model = artifact["model"]
    row = pd.DataFrame([values], columns=artifact["features"])

    pred_id = model.predict(row)[0]
    crop = artifact["label_encoder"].inverse_transform([pred_id])[0]

    top3 = []
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(row)[0]
        classes = artifact["label_encoder"].classes_
        ranked = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)[:3]
        top3 = [{"crop": str(c), "probability": float(p)} for c, p in ranked]
    else:
        top3 = [{"crop": str(crop), "probability": 1.0}]

    return {"predicted_crop": str(crop), "top3": top3, "input": values}
