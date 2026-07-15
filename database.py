"""
database.py
------------
SQLite schema and data-access helpers for the OptiCrop system, implementing
the seven-entity ER diagram exactly as specified:

    User(user_id PK)
    SoilData(soil_id PK, user_id FK -> User)
    Crop(crop_id PK)
    Dataset(dataset_id PK)
    MLModel(model_id PK, dataset_id FK -> Dataset)
    Prediction(prediction_id PK, soil_id FK -> SoilData [1:1],
               crop_id FK -> Crop, model_id FK -> MLModel)
    Report(report_id PK, prediction_id FK -> Prediction)
"""

import os
import sqlite3
from datetime import datetime, timezone

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "instance", "opticrop.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS User (
    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS Crop (
    crop_id INTEGER PRIMARY KEY AUTOINCREMENT,
    crop_name TEXT UNIQUE NOT NULL,
    description TEXT
);

CREATE TABLE IF NOT EXISTS Dataset (
    dataset_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    description TEXT,
    file_path TEXT,
    upload_date TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS MLModel (
    model_id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id INTEGER NOT NULL,
    model_name TEXT NOT NULL,
    algorithm TEXT NOT NULL,
    accuracy REAL,
    file_path TEXT,
    trained_date TEXT NOT NULL,
    FOREIGN KEY (dataset_id) REFERENCES Dataset(dataset_id)
);

-- 1 User : Many SoilData
CREATE TABLE IF NOT EXISTS SoilData (
    soil_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    N REAL NOT NULL,
    P REAL NOT NULL,
    K REAL NOT NULL,
    temperature REAL NOT NULL,
    humidity REAL NOT NULL,
    ph REAL NOT NULL,
    rainfall REAL NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES User(user_id)
);

-- 1 SoilData : 1 Prediction, Many Crop->Prediction, Many MLModel->Prediction
CREATE TABLE IF NOT EXISTS Prediction (
    prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    soil_id INTEGER NOT NULL UNIQUE,
    crop_id INTEGER NOT NULL,
    model_id INTEGER NOT NULL,
    confidence REAL NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (soil_id) REFERENCES SoilData(soil_id),
    FOREIGN KEY (crop_id) REFERENCES Crop(crop_id),
    FOREIGN KEY (model_id) REFERENCES MLModel(model_id)
);

-- 1 Prediction : Many Report
CREATE TABLE IF NOT EXISTS Report (
    report_id INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id INTEGER NOT NULL,
    summary TEXT NOT NULL,
    recommendations TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (prediction_id) REFERENCES Prediction(prediction_id)
);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    conn = get_db()
    conn.executescript(SCHEMA)
    conn.commit()
    conn.close()


def seed_reference_data(crop_descriptions: dict, dataset_meta: dict, model_meta: dict):
    """Seed Crop, Dataset, and MLModel reference rows if not already present."""
    conn = get_db()
    cur = conn.cursor()

    for crop_name, desc in crop_descriptions.items():
        cur.execute(
            "INSERT OR IGNORE INTO Crop (crop_name, description) VALUES (?, ?)",
            (crop_name, desc),
        )

    cur.execute("SELECT dataset_id FROM Dataset WHERE name = ?", (dataset_meta["name"],))
    row = cur.fetchone()
    if row is None:
        cur.execute(
            "INSERT INTO Dataset (name, description, file_path, upload_date) "
            "VALUES (?, ?, ?, ?)",
            (dataset_meta["name"], dataset_meta["description"],
             dataset_meta["file_path"], now()),
        )
        dataset_id = cur.lastrowid
    else:
        dataset_id = row["dataset_id"]

    cur.execute("SELECT model_id FROM MLModel WHERE model_name = ?", (model_meta["model_name"],))
    row = cur.fetchone()
    if row is None:
        cur.execute(
            "INSERT INTO MLModel (dataset_id, model_name, algorithm, accuracy, "
            "file_path, trained_date) VALUES (?, ?, ?, ?, ?, ?)",
            (dataset_id, model_meta["model_name"], model_meta["algorithm"],
             model_meta["accuracy"], model_meta["file_path"], now()),
        )
        model_id = cur.lastrowid
    else:
        model_id = row["model_id"]

    conn.commit()
    conn.close()
    return dataset_id, model_id


# ---------------------------------------------------------------------------
# User
# ---------------------------------------------------------------------------
def create_user(username, email, password_hash):
    conn = get_db()
    cur = conn.execute(
        "INSERT INTO User (username, email, password_hash, created_at) VALUES (?, ?, ?, ?)",
        (username, email, password_hash, now()),
    )
    conn.commit()
    user_id = cur.lastrowid
    conn.close()
    return user_id


def get_user_by_username(username):
    conn = get_db()
    row = conn.execute("SELECT * FROM User WHERE username = ?", (username,)).fetchone()
    conn.close()
    return row


def get_user_by_id(user_id):
    conn = get_db()
    row = conn.execute("SELECT * FROM User WHERE user_id = ?", (user_id,)).fetchone()
    conn.close()
    return row


# ---------------------------------------------------------------------------
# Crop
# ---------------------------------------------------------------------------
def get_crop_by_name(crop_name):
    conn = get_db()
    row = conn.execute("SELECT * FROM Crop WHERE crop_name = ?", (crop_name,)).fetchone()
    conn.close()
    return row


def get_crop_by_id(crop_id):
    conn = get_db()
    row = conn.execute("SELECT * FROM Crop WHERE crop_id = ?", (crop_id,)).fetchone()
    conn.close()
    return row


# ---------------------------------------------------------------------------
# SoilData
# ---------------------------------------------------------------------------
def create_soil_data(user_id, N, P, K, temperature, humidity, ph, rainfall):
    conn = get_db()
    cur = conn.execute(
        "INSERT INTO SoilData (user_id, N, P, K, temperature, humidity, ph, "
        "rainfall, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (user_id, N, P, K, temperature, humidity, ph, rainfall, now()),
    )
    conn.commit()
    soil_id = cur.lastrowid
    conn.close()
    return soil_id


def get_soil_data(soil_id):
    conn = get_db()
    row = conn.execute("SELECT * FROM SoilData WHERE soil_id = ?", (soil_id,)).fetchone()
    conn.close()
    return row


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
def create_prediction(soil_id, crop_id, model_id, confidence):
    conn = get_db()
    cur = conn.execute(
        "INSERT INTO Prediction (soil_id, crop_id, model_id, confidence, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (soil_id, crop_id, model_id, confidence, now()),
    )
    conn.commit()
    prediction_id = cur.lastrowid
    conn.close()
    return prediction_id


def get_prediction(prediction_id):
    conn = get_db()
    row = conn.execute(
        "SELECT p.*, c.crop_name, c.description AS crop_description, "
        "s.N, s.P, s.K, s.temperature, s.humidity, s.ph, s.rainfall, s.user_id, "
        "m.model_name, m.algorithm, m.accuracy AS model_accuracy "
        "FROM Prediction p "
        "JOIN Crop c ON p.crop_id = c.crop_id "
        "JOIN SoilData s ON p.soil_id = s.soil_id "
        "JOIN MLModel m ON p.model_id = m.model_id "
        "WHERE p.prediction_id = ?",
        (prediction_id,),
    ).fetchone()
    conn.close()
    return row


def get_predictions_for_user(user_id):
    conn = get_db()
    rows = conn.execute(
        "SELECT p.prediction_id, p.confidence, p.created_at, c.crop_name, s.soil_id "
        "FROM Prediction p "
        "JOIN SoilData s ON p.soil_id = s.soil_id "
        "JOIN Crop c ON p.crop_id = c.crop_id "
        "WHERE s.user_id = ? ORDER BY p.created_at DESC",
        (user_id,),
    ).fetchall()
    conn.close()
    return rows


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def create_report(prediction_id, summary, recommendations):
    conn = get_db()
    cur = conn.execute(
        "INSERT INTO Report (prediction_id, summary, recommendations, created_at) "
        "VALUES (?, ?, ?, ?)",
        (prediction_id, summary, recommendations, now()),
    )
    conn.commit()
    report_id = cur.lastrowid
    conn.close()
    return report_id


def get_report_by_prediction(prediction_id):
    conn = get_db()
    row = conn.execute(
        "SELECT * FROM Report WHERE prediction_id = ?", (prediction_id,)
    ).fetchone()
    conn.close()
    return row


def get_report(report_id):
    conn = get_db()
    row = conn.execute(
        "SELECT r.*, p.crop_id, c.crop_name, s.user_id "
        "FROM Report r "
        "JOIN Prediction p ON r.prediction_id = p.prediction_id "
        "JOIN Crop c ON p.crop_id = c.crop_id "
        "JOIN SoilData s ON p.soil_id = s.soil_id "
        "WHERE r.report_id = ?",
        (report_id,),
    ).fetchone()
    conn.close()
    return row


def get_reports_for_user(user_id):
    conn = get_db()
    rows = conn.execute(
        "SELECT r.report_id, r.created_at, r.summary, c.crop_name "
        "FROM Report r "
        "JOIN Prediction p ON r.prediction_id = p.prediction_id "
        "JOIN SoilData s ON p.soil_id = s.soil_id "
        "JOIN Crop c ON p.crop_id = c.crop_id "
        "WHERE s.user_id = ? ORDER BY r.created_at DESC",
        (user_id,),
    ).fetchall()
    conn.close()
    return rows


# ---------------------------------------------------------------------------
# Dataset / MLModel (admin/info views)
# ---------------------------------------------------------------------------
def get_active_model():
    conn = get_db()
    row = conn.execute(
        "SELECT * FROM MLModel ORDER BY trained_date DESC LIMIT 1"
    ).fetchone()
    conn.close()
    return row


def get_all_datasets():
    conn = get_db()
    rows = conn.execute("SELECT * FROM Dataset ORDER BY upload_date DESC").fetchall()
    conn.close()
    return rows


def get_all_crops():
    conn = get_db()
    rows = conn.execute("SELECT * FROM Crop ORDER BY crop_name ASC").fetchall()
    conn.close()
    return rows
