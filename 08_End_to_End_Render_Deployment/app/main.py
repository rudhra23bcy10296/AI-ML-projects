"""
End-to-End Render Deployment - FastAPI Web Application & REST API Server
Author: Rudhra Sitholey (Reg No: 23BCY10296 | App No: IN26012560)
"""

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI(
    title="ML Deployment Web Service - Rudhra Sitholey",
    description="End-to-End Machine Learning REST API deployed on Render cloud platform.",
    version="1.0.0"
)

templates = Jinja2Templates(directory="app/templates")

MODEL_PATH = os.path.join("models", "deployed_model.joblib")
model = None

@app.on_event("startup")
def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
    else:
        # Fallback build
        from train_save_model import build_and_save_model
        model = build_and_save_model()


class PredictionRequest(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    feature4: float


@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "student_name": "Rudhra Sitholey",
        "reg_no": "23BCY10296",
        "app_no": "IN26012560",
        "email": "rudhra.23bcy10296@vitbhopal.ac.in"
    })


@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "ML Render Deployment API", "student": "Rudhra Sitholey"}


@app.post("/predict")
def predict(data: PredictionRequest):
    features = np.array([[data.feature1, data.feature2, data.feature3, data.feature4]])
    prediction = int(model.predict(features)[0])
    proba = model.predict_proba(features)[0].tolist() if hasattr(model, "predict_proba") else [0.5, 0.5]
    
    return {
        "prediction": prediction,
        "class_label": "High Risk / Positive" if prediction == 1 else "Low Risk / Negative",
        "probabilities": proba,
        "student_metadata": {
            "name": "Rudhra Sitholey",
            "reg_no": "23BCY10296"
        }
    }
