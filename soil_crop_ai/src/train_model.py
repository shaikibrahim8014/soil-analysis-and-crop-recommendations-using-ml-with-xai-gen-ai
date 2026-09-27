from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer

from data_loader import load_dataset
from preprocessing import clean_dataset, FEATURES, TARGET, build_preprocessor

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

def make_models():
    pre = build_preprocessor()
    return {
        "Logistic Regression": Pipeline([
            ("preprocessor", pre),
            ("model", LogisticRegression(max_iter=3000))
        ]),
        "Decision Tree": Pipeline([
            ("preprocessor", pre),
            ("model", DecisionTreeClassifier(random_state=42))
        ]),
        "Random Forest": Pipeline([
            ("preprocessor", pre),
            ("model", RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1))
        ])
    }

def add_xgboost(models):
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("model", XGBClassifier(
                n_estimators=250, max_depth=6, learning_rate=0.08,
                subsample=0.9, colsample_bytree=0.9,
                objective="multi:softprob", eval_metric="mlogloss",
                random_state=42, n_jobs=-1
            ))
        ])
    except Exception as e:
        print(f"XGBoost unavailable; continuing without it: {e}")

def evaluate_models(models, X_train, X_test, y_train, y_test, class_names):
    rows = []
    fitted = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, average="weighted", zero_division=0),
            "Recall": recall_score(y_test, pred, average="weighted", zero_division=0),
            "F1-score": f1_score(y_test, pred, average="weighted", zero_division=0)
        })
        fitted[name] = model
        print(f"\n{name}")
        print(classification_report(y_test, pred, target_names=class_names, zero_division=0))
    return pd.DataFrame(rows), fitted

def main():
    print("Loading dataset...")
    df = clean_dataset(load_dataset())

    if df[TARGET].nunique() < 2:
        raise ValueError("At least two crop classes are required.")

    X = df[FEATURES]
    encoder = LabelEncoder()
    y = encoder.fit_transform(df[TARGET].astype(str))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = make_models()
    add_xgboost(models)

    comparison, fitted = evaluate_models(
        models, X_train, X_test, y_train, y_test, encoder.classes_
    )

    best_name = comparison.sort_values(
        ["F1-score", "Accuracy"], ascending=False
    ).iloc[0]["Model"]
    print(f"\nSelected model: {best_name}")

    # Tune Random Forest when it is selected or available.
    if best_name == "Random Forest":
        search = RandomizedSearchCV(
            estimator=Pipeline([
                ("preprocessor", build_preprocessor()),
                ("model", RandomForestClassifier(random_state=42, n_jobs=-1))
            ]),
            param_distributions={
                "model__n_estimators": [150, 250, 400],
                "model__max_depth": [None, 8, 12, 20],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
                "model__max_features": ["sqrt", "log2", None]
            },
            n_iter=12, cv=3, scoring="f1_weighted",
            random_state=42, n_jobs=-1
        )
        search.fit(X_train, y_train)
        final_model = search.best_estimator_
        best_params = search.best_params_
    else:
        final_model = fitted[best_name]
        best_params = {}

    final_model.fit(X_train, y_train)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(final_model, X, y, cv=cv, scoring="f1_weighted", n_jobs=-1)

    test_pred = final_model.predict(X_test)
    cm = confusion_matrix(y_test, test_pred).tolist()

    artifact = {
        "model": final_model,
        "label_encoder": encoder,
        "features": FEATURES
    }
    joblib.dump(artifact, MODEL_DIR / "crop_model.pkl")

    meta = {
        "selected_model": best_name,
        "features": FEATURES,
        "classes": encoder.classes_.tolist(),
        "num_classes": int(len(encoder.classes_)),
        "training_samples": int(len(X_train)),
        "testing_samples": int(len(X_test)),
        "cross_validation_f1_weighted_mean": float(np.mean(cv_scores)),
        "cross_validation_f1_weighted_std": float(np.std(cv_scores)),
        "best_params": best_params,
        "model_comparison": comparison.to_dict(orient="records"),
        "confusion_matrix": cm,
        "class_names": encoder.classes_.tolist()
    }
    (MODEL_DIR / "model_metadata.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    print("\nTraining complete.")
    print(comparison.to_string(index=False))
    print(f"\nSaved: {MODEL_DIR / 'crop_model.pkl'}")
    print(f"Saved: {MODEL_DIR / 'model_metadata.json'}")

if __name__ == "__main__":
    main()
