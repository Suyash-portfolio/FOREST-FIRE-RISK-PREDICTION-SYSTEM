from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model.pkl"
FEATURE_ORDER = ["temp", "RH", "wind", "rain"]

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "model.pkl not found in project root. Run 'python scripts/train_and_save_model.py' first."
    )

model = joblib.load(MODEL_PATH)

app = Flask(__name__)


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    try:
        values = [float(payload[name]) for name in FEATURE_ORDER]
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "Invalid input. Provide temp, RH, wind, rain as numbers."}), 400

    features = pd.DataFrame([values], columns=FEATURE_ORDER)
    prediction = int(model.predict(features)[0])
    return jsonify({"prediction": prediction})


if __name__ == "__main__":
    app.run(debug=True)
