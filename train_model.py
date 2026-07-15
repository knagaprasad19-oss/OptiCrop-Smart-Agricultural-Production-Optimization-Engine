"""
train_model.py
----------------
OptiCrop - Smart Agricultural Production Optimization System

Implements Epics 2-4 of the project plan:
  Epic 2: Data Collection and Analysis
  Epic 3: Data Pre-Processing
  Epic 4: Model Building (KMeans exploration + Logistic Regression / RandomForest,
          evaluation, and saving the best-performing model)

Because this environment has no network access, a realistic *synthetic*
agronomic dataset is generated from published-style nutrient/climate ranges
for 22 common crops (N, P, K in kg/ha, temperature in C, relative humidity %,
soil pH, rainfall in mm). This mirrors the structure of the classic Crop
Recommendation dataset used in the OptiCrop brief without redistributing any
third-party file.
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib

RNG = np.random.default_rng(42)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Epic 2 / Story 1: "Collect" the agricultural dataset.
# Approximate agronomic requirement ranges (N, P, K kg/ha, temp C, humidity %,
# ph, rainfall mm) per crop. Ranges are illustrative/synthetic, drawn from
# general agronomy knowledge, and used only to generate training samples.
# ---------------------------------------------------------------------------
CROP_PROFILES = {
    "rice":        dict(N=(70, 105), P=(35, 65),  K=(35, 45),  temp=(21, 27), hum=(78, 90), ph=(5.0, 7.0), rain=(180, 300)),
    "maize":       dict(N=(60, 100), P=(35, 65),  K=(15, 25),  temp=(18, 27), hum=(55, 75), ph=(5.5, 7.0), rain=(60, 110)),
    "chickpea":    dict(N=(20, 60),  P=(55, 80),  K=(75, 85),  temp=(17, 22), hum=(14, 20), ph=(5.5, 7.5), rain=(65, 95)),
    "kidneybeans": dict(N=(15, 40),  P=(55, 75),  K=(15, 25),  temp=(15, 22), hum=(18, 24), ph=(5.5, 6.0), rain=(60, 150)),
    "pigeonpeas":  dict(N=(15, 40),  P=(55, 80),  K=(15, 25),  temp=(18, 37), hum=(30, 70), ph=(4.5, 7.0), rain=(120, 220)),
    "mothbeans":   dict(N=(15, 45),  P=(35, 60),  K=(15, 25),  temp=(24, 32), hum=(40, 65), ph=(3.5, 10.0), rain=(35, 65)),
    "mungbean":    dict(N=(15, 40),  P=(35, 60),  K=(15, 25),  temp=(27, 32), hum=(75, 90), ph=(6.2, 7.2), rain=(45, 65)),
    "blackgram":   dict(N=(30, 60),  P=(55, 75),  K=(15, 25),  temp=(24, 35), hum=(60, 70), ph=(6.0, 7.5), rain=(60, 80)),
    "lentil":      dict(N=(15, 30),  P=(55, 75),  K=(15, 25),  temp=(18, 26), hum=(60, 70), ph=(5.5, 7.0), rain=(35, 55)),
    "pomegranate": dict(N=(15, 25),  P=(10, 20),  K=(35, 45),  temp=(18, 25), hum=(85, 95), ph=(6.0, 7.0), rain=(35, 55)),
    "banana":      dict(N=(90, 120), P=(70, 95),  K=(45, 55),  temp=(24, 31), hum=(75, 85), ph=(5.5, 6.5), rain=(90, 120)),
    "mango":       dict(N=(15, 25),  P=(15, 25),  K=(25, 35),  temp=(27, 37), hum=(45, 55), ph=(4.5, 7.0), rain=(85, 105)),
    "grapes":      dict(N=(15, 25),  P=(120,145), K=(195,205), temp=(8, 25),  hum=(80, 85), ph=(5.5, 6.5), rain=(65, 75)),
    "watermelon":  dict(N=(90, 110), P=(10, 20),  K=(45, 55),  temp=(24, 27), hum=(80, 90), ph=(6.0, 7.0), rain=(40, 55)),
    "muskmelon":   dict(N=(90, 110), P=(10, 20),  K=(45, 55),  temp=(27, 30), hum=(90, 95), ph=(6.0, 7.0), rain=(20, 30)),
    "apple":       dict(N=(15, 25),  P=(120,145), K=(195,205), temp=(20, 24), hum=(90, 95), ph=(5.5, 6.5), rain=(100, 120)),
    "orange":      dict(N=(15, 25),  P=(10, 20),  K=(8, 12),   temp=(10, 34), hum=(90, 95), ph=(6.0, 8.0), rain=(100, 120)),
    "papaya":      dict(N=(45, 70),  P=(55, 70),  K=(45, 55),  temp=(23, 44), hum=(90, 95), ph=(6.0, 7.0), rain=(40, 250)),
    "coconut":     dict(N=(15, 25),  P=(10, 20),  K=(25, 35),  temp=(25, 30), hum=(90, 100),ph=(5.2, 6.0), rain=(120, 230)),
    "cotton":      dict(N=(100,140), P=(35, 60),  K=(15, 25),  temp=(21, 26), hum=(75, 85), ph=(5.8, 8.0), rain=(60, 100)),
    "jute":        dict(N=(60, 100), P=(35, 60),  K=(35, 45),  temp=(23, 27), hum=(70, 90), ph=(6.0, 7.5), rain=(150, 200)),
    "coffee":      dict(N=(80, 120), P=(15, 30),  K=(25, 35),  temp=(23, 28), hum=(50, 70), ph=(6.0, 7.5), rain=(150, 220)),
}

SAMPLES_PER_CROP = 120


def generate_dataset() -> pd.DataFrame:
    rows = []
    for crop, r in CROP_PROFILES.items():
        for _ in range(SAMPLES_PER_CROP):
            rows.append({
                "N": RNG.uniform(*r["N"]),
                "P": RNG.uniform(*r["P"]),
                "K": RNG.uniform(*r["K"]),
                "temperature": RNG.uniform(*r["temp"]),
                "humidity": RNG.uniform(*r["hum"]),
                "ph": RNG.uniform(*r["ph"]),
                "rainfall": RNG.uniform(*r["rain"]),
                "label": crop,
            })
    df = pd.DataFrame(rows)
    # Epic 3 / Story 1: inject a few missing values, then clean them,
    # to exercise the "handle missing data" story realistically.
    mask = RNG.random(len(df)) < 0.01
    df.loc[mask, "humidity"] = np.nan
    return df


def preprocess(df: pd.DataFrame):
    # Epic 3 / Story 1: null handling
    df = df.copy()
    df["humidity"] = df["humidity"].fillna(df["humidity"].median())

    # Epic 3 / Story 2: simple outlier clipping (IQR-style) on numeric cols
    numeric_cols = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    for col in numeric_cols:
        q1, q3 = df[col].quantile(0.01), df[col].quantile(0.99)
        df[col] = df[col].clip(q1, q3)

    return df, numeric_cols


def main():
    df = generate_dataset()
    df, feature_cols = preprocess(df)

    dataset_path = os.path.join(DATA_DIR, "crop_dataset.csv")
    df.to_csv(dataset_path, index=False)

    X = df[feature_cols].values
    y_raw = df["label"].values

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Epic 3 / Story 4: train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # Epic 4 / Story 1: KMeans exploration of natural groupings (unsupervised
    # sanity check that agronomic conditions cluster sensibly).
    kmeans = KMeans(n_clusters=len(CROP_PROFILES), n_init=10, random_state=42)
    kmeans.fit(X_scaled)

    # Epic 4 / Story 2 & 3: train and compare Logistic Regression and Random Forest
    candidates = {
        "logistic_regression": LogisticRegression(max_iter=2000),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=42),
    }

    results = {}
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="weighted")
        results[name] = {"model": model, "accuracy": acc, "f1": f1}
        print(f"{name}: accuracy={acc:.4f} f1={f1:.4f}")

    # Epic 4 / Story 4: pick and save the best-performing model
    best_name = max(results, key=lambda k: results[k]["accuracy"])
    best_model = results[best_name]["model"]
    best_metrics = {
        "algorithm": best_name,
        "accuracy": results[best_name]["accuracy"],
        "f1_score": results[best_name]["f1"],
    }
    print(f"\nBest model: {best_name} ({best_metrics['accuracy']:.4f} accuracy)")

    print("\nClassification report for best model:")
    print(classification_report(y_test, results[best_name]["model"].predict(X_test),
                                 target_names=encoder.classes_))

    bundle = {
        "model": best_model,
        "scaler": scaler,
        "encoder": encoder,
        "feature_cols": feature_cols,
        "metrics": best_metrics,
    }
    model_path = os.path.join(MODEL_DIR, "crop_model.pkl")
    joblib.dump(bundle, model_path)

    metrics_path = os.path.join(MODEL_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({
            "algorithm": best_metrics["algorithm"],
            "accuracy": round(best_metrics["accuracy"], 4),
            "f1_score": round(best_metrics["f1_score"], 4),
            "trained_samples": len(X_train),
            "test_samples": len(X_test),
            "num_crops": len(CROP_PROFILES),
            "dataset_rows": len(df),
        }, f, indent=2)

    print(f"\nSaved model bundle to {model_path}")
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved dataset to {dataset_path}")


if __name__ == "__main__":
    main()
