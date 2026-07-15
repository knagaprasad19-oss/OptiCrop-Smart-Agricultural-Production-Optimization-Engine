"""
app.py
-------
OptiCrop - Smart Agricultural Production Optimization System (Flask app)

Implements Epic 5 (Application Building):
  Story 1: HTML pages for an interactive, user-friendly interface (templates/)
  Story 2: Python backend integrated with the trained ML model (this file)
  Story 3: Runnable end-to-end flow: register -> login -> submit soil data
           -> get crop prediction -> view generated report -> history
"""

import os
import json
from functools import wraps

from flask import (
    Flask, render_template, request, redirect, url_for, session, flash, abort
)
from werkzeug.security import generate_password_hash, check_password_hash
import joblib
import numpy as np

import database as db

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "crop_model.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "models", "metrics.json")
DATASET_PATH = os.path.join(BASE_DIR, "data", "crop_dataset.csv")

app = Flask(__name__)
app.secret_key = os.environ.get("OPTICROP_SECRET_KEY", "dev-secret-key-change-in-production")

# ---------------------------------------------------------------------------
# Crop knowledge base: short description + agronomic recommendation notes,
# used to seed the Crop table and to compose Report text.
# ---------------------------------------------------------------------------
CROP_INFO = {
    "rice": ("A staple cereal thriving in flooded or high-rainfall paddies.",
             "Maintain standing water during the vegetative stage; apply nitrogen in split doses."),
    "maize": ("A versatile cereal used for food, feed, and fodder.",
              "Ensure good drainage; side-dress nitrogen at the knee-high stage."),
    "chickpea": ("A cool-season legume grown for protein-rich seeds.",
                 "Avoid waterlogging; inoculate seed with Rhizobium for better nodulation."),
    "kidneybeans": ("A warm-season legume valued for its protein-dense beans.",
                    "Provide well-drained loam soil and moderate, even watering."),
    "pigeonpeas": ("A drought-tolerant legume common in intercropping systems.",
                   "Deep-rooted; tolerates dry spells once established."),
    "mothbeans": ("A hardy, drought-resistant legume suited to arid regions.",
                  "Needs minimal irrigation; ideal for sandy, low-fertility soils."),
    "mungbean": ("A fast-maturing legume popular in rotation systems.",
                 "Sow after residual monsoon moisture; avoid excess nitrogen."),
    "blackgram": ("A short-duration pulse crop grown in warm climates.",
                  "Well-drained loamy soil with moderate rainfall gives the best yield."),
    "lentil": ("A cool-climate pulse crop rich in protein.",
               "Prefers well-drained soils; sensitive to waterlogging."),
    "pomegranate": ("A drought-tolerant fruit shrub suited to semi-arid climates.",
                    "Deep, well-drained soil; drip irrigation improves fruit quality."),
    "banana": ("A high-water-demand tropical fruit crop.",
               "Requires consistent moisture and potassium-rich fertilization."),
    "mango": ("A tropical fruit tree tolerant of dry spells once mature.",
              "Avoid excess nitrogen; ensure a distinct dry period before flowering."),
    "grapes": ("A vine fruit crop needing high potassium and phosphorus.",
               "Well-drained soil and controlled irrigation improve sugar content."),
    "watermelon": ("A warm-season vine crop needing sandy, well-drained soil.",
                   "High nitrogen early on, reduced nitrogen as fruit develops."),
    "muskmelon": ("A warm-season melon requiring low humidity for the best flavor.",
                  "Avoid overhead irrigation late in the season to reduce disease."),
    "apple": ("A temperate fruit tree needing a distinct winter chill period.",
              "Rich, well-drained loam with balanced phosphorus and potassium."),
    "orange": ("A citrus crop adaptable to a wide temperature range.",
               "Maintain consistent soil moisture; watch for micronutrient deficiency."),
    "papaya": ("A fast-growing tropical fruit tree sensitive to waterlogging.",
               "Raised beds with rich organic matter and good drainage."),
    "coconut": ("A tropical palm requiring high humidity and rainfall.",
                "Coastal sandy soils with good drainage and regular irrigation."),
    "cotton": ("A fiber crop needing a long, warm, frost-free season.",
               "Balanced NPK with attention to potassium during boll formation."),
    "jute": ("A fiber crop grown in warm, humid, high-rainfall regions.",
             "Needs standing-water tolerance and rich alluvial soil."),
    "coffee": ("A shade-loving perennial crop grown in cool tropical highlands.",
               "Consistent rainfall and shade cover improve bean quality."),
}

DATASET_META = {
    "name": "Synthetic Crop Recommendation Dataset v1",
    "description": (
        "Synthetically generated agronomic dataset (N, P, K, temperature, "
        "humidity, ph, rainfall) covering 22 crops, used to train the "
        "OptiCrop recommendation model."
    ),
    "file_path": "data/crop_dataset.csv",
}


def load_model_bundle():
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            "Model file not found. Run `python train_model.py` first to "
            "generate the dataset and train the model."
        )
    return joblib.load(MODEL_PATH)


def bootstrap():
    """Initialize DB schema and seed Crop / Dataset / MLModel reference rows."""
    db.init_db()
    bundle = load_model_bundle()
    metrics = {}
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            metrics = json.load(f)

    crop_descriptions = {name: info[0] for name, info in CROP_INFO.items()}
    model_meta = {
        "model_name": f"OptiCrop {metrics.get('algorithm', 'model')}",
        "algorithm": metrics.get("algorithm", "unknown"),
        "accuracy": metrics.get("accuracy", 0.0),
        "file_path": "models/crop_model.pkl",
    }
    db.seed_reference_data(crop_descriptions, DATASET_META, model_meta)
    return bundle, metrics


MODEL_BUNDLE, MODEL_METRICS = bootstrap()


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------
def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if "user_id" not in session:
            flash("Please sign in to continue.", "warning")
            return redirect(url_for("login"))
        return view(*args, **kwargs)
    return wrapped


@app.context_processor
def inject_user():
    user = None
    if "user_id" in session:
        row = db.get_user_by_id(session["user_id"])
        if row:
            user = {"user_id": row["user_id"], "username": row["username"]}
    return {"current_user": user}


# ---------------------------------------------------------------------------
# Public routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html", metrics=MODEL_METRICS, num_crops=len(CROP_INFO))


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")

        if not username or not email or not password:
            flash("All fields are required.", "error")
            return render_template("register.html")

        if db.get_user_by_username(username):
            flash("That username is already taken.", "error")
            return render_template("register.html")

        password_hash = generate_password_hash(password)
        user_id = db.create_user(username, email, password_hash)
        session["user_id"] = user_id
        flash("Account created. Welcome to OptiCrop!", "success")
        return redirect(url_for("dashboard"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        row = db.get_user_by_username(username)
        if row and check_password_hash(row["password_hash"], password):
            session["user_id"] = row["user_id"]
            flash(f"Welcome back, {row['username']}.", "success")
            return redirect(url_for("dashboard"))
        flash("Invalid username or password.", "error")
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been signed out.", "info")
    return redirect(url_for("index"))


# ---------------------------------------------------------------------------
# Core application routes
# ---------------------------------------------------------------------------
@app.route("/dashboard")
@login_required
def dashboard():
    history = db.get_predictions_for_user(session["user_id"])
    return render_template("dashboard.html", history=history[:5], total=len(history))


@app.route("/soil/add", methods=["GET", "POST"])
@login_required
def add_soil_data():
    if request.method == "POST":
        try:
            fields = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
            values = {f: float(request.form.get(f)) for f in fields}
        except (TypeError, ValueError):
            flash("Please enter valid numeric values for all fields.", "error")
            return render_template("soil_form.html", form=request.form)

        soil_id = db.create_soil_data(
            session["user_id"], values["N"], values["P"], values["K"],
            values["temperature"], values["humidity"], values["ph"], values["rainfall"],
        )

        prediction_id = run_prediction(soil_id, values)
        return redirect(url_for("prediction_result", prediction_id=prediction_id))

    return render_template("soil_form.html", form={})


def run_prediction(soil_id, values):
    """Run the trained ML model on a SoilData row and persist Prediction + Report."""
    feature_cols = MODEL_BUNDLE["feature_cols"]
    scaler = MODEL_BUNDLE["scaler"]
    model = MODEL_BUNDLE["model"]
    encoder = MODEL_BUNDLE["encoder"]

    x = np.array([[values[col] for col in feature_cols]])
    x_scaled = scaler.transform(x)

    proba = model.predict_proba(x_scaled)[0]
    pred_idx = int(np.argmax(proba))
    crop_name = encoder.inverse_transform([pred_idx])[0]
    confidence = float(proba[pred_idx])

    crop_row = db.get_crop_by_name(crop_name)
    active_model = db.get_active_model()

    prediction_id = db.create_prediction(
        soil_id, crop_row["crop_id"], active_model["model_id"], confidence
    )

    # Compose a human-readable report from the crop knowledge base and the
    # submitted soil parameters.
    _, tips = CROP_INFO.get(crop_name, ("", "Follow standard best practices for this crop."))
    summary = (
        f"Based on the submitted soil and climate readings, {crop_name.title()} is the "
        f"recommended crop with {confidence * 100:.1f}% model confidence."
    )
    recommendations = (
        f"{tips} Submitted readings \u2014 N: {values['N']:.1f}, P: {values['P']:.1f}, "
        f"K: {values['K']:.1f}, temperature: {values['temperature']:.1f}C, "
        f"humidity: {values['humidity']:.1f}%, pH: {values['ph']:.2f}, "
        f"rainfall: {values['rainfall']:.1f}mm."
    )
    db.create_report(prediction_id, summary, recommendations)

    # Also compute top-3 alternatives for display purposes only (not persisted,
    # since the ER model stores the single top prediction per SoilData row).
    top_idx = np.argsort(proba)[::-1][:3]
    session["last_alternatives"] = [
        {"crop": encoder.inverse_transform([i])[0], "confidence": float(proba[i])}
        for i in top_idx
    ]

    return prediction_id


@app.route("/prediction/<int:prediction_id>")
@login_required
def prediction_result(prediction_id):
    prediction = db.get_prediction(prediction_id)
    if prediction is None or prediction["user_id"] != session["user_id"]:
        abort(404)
    report = db.get_report_by_prediction(prediction_id)
    alternatives = session.pop("last_alternatives", None)
    return render_template(
        "prediction_result.html", prediction=prediction, report=report,
        alternatives=alternatives,
    )


@app.route("/history")
@login_required
def history():
    rows = db.get_predictions_for_user(session["user_id"])
    return render_template("history.html", history=rows)


@app.route("/reports")
@login_required
def reports():
    rows = db.get_reports_for_user(session["user_id"])
    return render_template("reports.html", reports=rows)


@app.route("/report/<int:report_id>")
@login_required
def report_detail(report_id):
    report = db.get_report(report_id)
    if report is None or report["user_id"] != session["user_id"]:
        abort(404)
    return render_template("report_detail.html", report=report)


@app.route("/about-model")
def about_model():
    active_model = db.get_active_model()
    datasets = db.get_all_datasets()
    crops = db.get_all_crops()
    return render_template(
        "about_model.html", model=active_model, metrics=MODEL_METRICS,
        datasets=datasets, crops=crops,
    )


@app.errorhandler(404)
def not_found(e):
    return render_template("404.html"), 404


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
