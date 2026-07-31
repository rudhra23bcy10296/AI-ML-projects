"""
House Price Prediction - Flask API backend
--------------------------------------------
Trains a Linear Regression model on house_data.csv at startup and exposes:

  POST /api/predict   -> JSON prediction endpoint used by the React frontend
  GET  /*              -> serves the built React app (frontend/dist)

Local development:
    1. Build the frontend once:
         cd frontend
         npm install
         npm run build
    2. From the project root:
         pip install -r requirements.txt
         python app.py
    3. Open http://127.0.0.1:5000
"""

import os
from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
from sklearn.linear_model import LinearRegression

FRONTEND_DIST = os.path.join(os.path.dirname(__file__), "frontend", "dist")

app = Flask(__name__, static_folder=FRONTEND_DIST, static_url_path="")

# ---------------------------------------------------------------------
# 1. LOAD + CLEAN THE DATASET, TRAIN THE MODEL (once, at startup)
# ---------------------------------------------------------------------
DATA_PATH = os.path.join(os.path.dirname(__file__), "house_data.csv")

raw = pd.read_csv(DATA_PATH)
data = raw[(raw["price"] > 0) & (raw["sqft_living"] > 0)].copy()

FEATURES = ["bedrooms", "bathrooms", "sqft_living", "yr_built"]
X = data[FEATURES]
y = data["price"]

model = LinearRegression()
model.fit(X, y)


def predict_price(bedrooms, bathrooms, sqft_living, yr_built):
    features = pd.DataFrame(
        [[bedrooms, bathrooms, sqft_living, yr_built]], columns=FEATURES
    )
    prediction = model.predict(features)[0]
    return round(max(prediction, 0), 2)


# ---------------------------------------------------------------------
# 2. API ROUTE — used by the React frontend
# ---------------------------------------------------------------------
@app.route("/api/predict", methods=["POST"])
def api_predict():
    payload = request.get_json(silent=True) or {}
    try:
        bedrooms = float(payload["bedrooms"])
        bathrooms = float(payload["bathrooms"])
        sqft_living = float(payload["sqft_living"])
        yr_built = float(payload["yr_built"])
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "Please provide valid numeric values."}), 400

    price = predict_price(bedrooms, bathrooms, sqft_living, yr_built)
    return jsonify({"price": price})


# ---------------------------------------------------------------------
# 3. SERVE THE BUILT REACT APP
# ---------------------------------------------------------------------
@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_frontend(path):
    full_path = os.path.join(FRONTEND_DIST, path)
    if path and os.path.exists(full_path):
        return send_from_directory(FRONTEND_DIST, path)
    return send_from_directory(FRONTEND_DIST, "index.html")


if __name__ == "__main__":
    app.run(debug=True)
