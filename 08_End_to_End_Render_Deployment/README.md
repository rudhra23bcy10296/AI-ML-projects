# Project 08 — End-to-End House Price Prediction (Render Deployment)

**Author:** Rudhra Sitholey
**Registration No:** 23BCY10296 | **Application No:** IN26012560
**Email:** rudhra.23bcy10296@vitbhopal.ac.in

---

## Overview

A full-stack machine learning application demonstrating end-to-end deployment. A Linear Regression model trained on King County housing data predicts home prices based on bedrooms, bathrooms, square footage, and year built. The backend is a Flask REST API, the frontend is a React + Vite single-page application, and the entire stack is containerized with Docker and configured for one-click deployment on Render.

**Live Deployment URL:** [https://house-price-prediction-webapp-b4jv.onrender.com/](https://house-price-prediction-webapp-b4jv.onrender.com/)

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ (backend), JavaScript (frontend) |
| ML Model | Scikit-learn Linear Regression |
| Data Processing | Pandas |
| Backend | Flask REST API |
| Frontend | React + Vite |
| Containerization | Docker |
| Deployment | Render (render.yaml) |
| Process Manager | Gunicorn |

## Project Structure

```
08_End_to_End_Render_Deployment/
├── app.py                    # Flask API — trains model at startup, serves predictions
├── train_save_model.py       # Standalone training script
├── house_data.csv            # King County house sales dataset
├── main.py                   # CLI entry point
├── requirements.txt          # Python dependencies
├── Dockerfile                # Container build instructions
├── render.yaml               # Render deployment configuration
├── .python-version           # Python version pinning
├── frontend/
│   ├── package.json          # Node.js dependencies
│   ├── vite.config.js        # Vite build configuration
│   ├── index.html            # HTML entry point
│   ├── src/                  # React source code
│   │   ├── App.jsx           # Main application component
│   │   └── App.css           # Application styles
│   └── dist/                 # Production build output
└── app/
    └── ...                   # Additional backend modules
```

## How It Works

1. **Data Loading** — `house_data.csv` (King County) is loaded and cleaned at server startup. Records with zero price or zero square footage are filtered out.
2. **Model Training** — A Linear Regression model is fit on 4 features: `bedrooms`, `bathrooms`, `sqft_living`, `yr_built`.
3. **REST API** — `POST /api/predict` accepts JSON with the 4 feature values and returns a predicted price.
4. **React Frontend** — A Vite-built SPA sends user inputs to the API and displays the predicted price with a polished UI.
5. **Deployment** — Docker + `render.yaml` enable one-click deployment to Render cloud.

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/predict` | Predict house price from features |
| `GET` | `/*` | Serve the React frontend |

## Getting Started

```bash
# 1. Build the frontend
cd frontend
npm install
npm run build
cd ..

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Launch the application
python app.py
# → Open http://localhost:5000
```

### Docker Deployment

```bash
docker build -t house-price-predictor .
docker run -p 5000:5000 house-price-predictor
```

### Render Deployment

Push to a GitHub repository connected to Render. The `render.yaml` handles automatic deployment configuration.

## Dataset

- **Source:** King County House Sales (Kaggle)
- **Records:** ~21,000 house sale records
- **Features Used:** Bedrooms, Bathrooms, Square Footage (living area), Year Built
- **Target:** Sale Price (USD)

