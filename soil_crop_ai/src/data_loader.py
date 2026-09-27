from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "soil_crop_dataset.csv"

REQUIRED_COLUMNS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall", "label"]

def validate_dataset_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

def load_dataset(path=DATA_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. Put your dataset at data/soil_crop_dataset.csv"
        )
    df = pd.read_csv(path)
    validate_dataset_columns(df)
    return df
