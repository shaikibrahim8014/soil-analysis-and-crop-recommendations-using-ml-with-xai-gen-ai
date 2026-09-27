import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET = "label"

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data = data.drop_duplicates()
    for col in FEATURES:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data["label"] = data["label"].astype(str).str.strip()
    data = data[data["label"].ne("") & data["label"].ne("nan")]
    data = data.dropna(subset=FEATURES + ["label"])
    return data

def build_preprocessor():
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipeline, FEATURES)
    ], remainder="drop")
