# OptiCrop — Smart Agricultural Production Optimization System

A full-stack Flask web app that recommends the best crop to plant given soil
nutrient levels (N, P, K) and climate readings (temperature, humidity, pH,
rainfall), using a trained scikit-learn classifier. Built around the
seven-entity ER diagram: **User, SoilData, Crop, Dataset, MLModel,
Prediction, Report**.

## Features

- Account registration / login (hashed passwords, session auth)
- Soil & climate data entry form (7 parameters)
- Instant crop prediction with confidence score and top-3 alternatives
- Auto-generated recommendation report per prediction
- Prediction history and report archive per user
- Model/dataset transparency page (accuracy, algorithm, training data)

## Project structure

```
app.py              Flask routes and request handling
database.py          SQLite schema (7 entities) + data access functions
train_model.py        Generates the training dataset and trains/saves the model
templates/            Jinja2 HTML templates
static/style.css       Design system (soil-strata visual language)
data/crop_dataset.csv  Generated training data (created by train_model.py)
models/crop_model.pkl  Trained model bundle (created by train_model.py)
models/metrics.json    Accuracy/F1 metrics for the trained model
instance/opticrop.db   SQLite database (created on first run)
```

## Setup

```bash
pip install -r requirements.txt

# 1. Generate the dataset and train the model (Epics 2-4)
python train_model.py

# 2. Run the web app (Epic 5)
python app.py
```

Then open **http://localhost:5000** in your browser.

## How the ML pipeline works

`train_model.py` builds a synthetic-but-realistic agronomic dataset covering
22 crops (rice, maize, cotton, coffee, various fruits, pulses, etc.), each
with characteristic ranges for N, P, K, temperature, humidity, pH, and
rainfall. It then:

1. Cleans missing values and clips outliers (Epic 3)
2. Splits into train/test sets
3. Runs a KMeans pass to sanity-check natural groupings
4. Trains both Logistic Regression and Random Forest, compares accuracy/F1
5. Saves the best-performing model (typically Random Forest, ~98-99%
   accuracy on the held-out test set) as `models/crop_model.pkl`

`app.py` loads that model bundle at startup, seeds the `Crop`, `Dataset`, and
`MLModel` tables, and uses the model live whenever a user submits a soil
sample — writing a `Prediction` row and a generated `Report` row for it.

## Notes

- The dataset is synthetically generated (no external download required),
  since ranges are drawn from general agronomy knowledge rather than a
  redistributed third-party file. Swap in your own labeled CSV in
  `train_model.py` if you have real field data.
- `app.secret_key` should be overridden via the `OPTICROP_SECRET_KEY`
  environment variable in any real deployment.
