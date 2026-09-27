from pathlib import Path
import sqlite3
import json
import pandas as pd
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "soil_crop_history.db"

def initialize_database():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            N REAL, P REAL, K REAL,
            temperature REAL, humidity REAL, ph REAL, rainfall REAL,
            predicted_crop TEXT NOT NULL,
            top3 TEXT NOT NULL
        )
        """)
        conn.commit()

def save_prediction(values, predicted_crop, top3):
    initialize_database()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
        INSERT INTO predictions
        (timestamp,N,P,K,temperature,humidity,ph,rainfall,predicted_crop,top3)
        VALUES (?,?,?,?,?,?,?,?,?,?)
        """, (
            datetime.now().isoformat(timespec="seconds"),
            values["N"], values["P"], values["K"],
            values["temperature"], values["humidity"], values["ph"], values["rainfall"],
            predicted_crop, json.dumps(top3)
        ))
        conn.commit()

def get_prediction_history():
    initialize_database()
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(
            "SELECT * FROM predictions ORDER BY id DESC", conn
        )
