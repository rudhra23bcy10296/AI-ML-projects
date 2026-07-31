# Project 01 — Adult Census Income Classification

**Author:** Rudhra Sitholey
**Registration No:** 23BCY10296 | **Application No:** IN26012560
**Email:** rudhra.23bcy10296@vitbhopal.ac.in

---

## Overview

A supervised machine learning pipeline that predicts whether an individual earns more than $50K/year using the UCI Adult Census Income dataset. The project compares multiple classifiers — Logistic Regression, Random Forest, XGBoost, and LightGBM — through a unified preprocessing and evaluation framework, and serves predictions via an interactive Flask web application.

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| ML Models | Logistic Regression, Random Forest, XGBoost, LightGBM |
| Data Processing | Pandas, NumPy, Scikit-learn (ColumnTransformer, StandardScaler, OneHotEncoder) |
| Visualization | Matplotlib, Seaborn |
| Web Framework | Flask |
| Serialization | Joblib |

## Project Structure

```
01_Adult_Census_Income_Classification/
├── app.py                    # Flask web server with prediction endpoint
├── main.py                   # CLI entry point — trains and evaluates all models
├── requirements.txt          # Python dependencies
├── adult_census_income.csv   # UCI Adult dataset (32,561 records)
├── model_results.csv         # Saved evaluation metrics
├── assignment_results.png    # Comparison chart of model performances
├── assignment_report.pdf     # Written analysis report
├── src/
│   ├── preprocessing.py      # Feature engineering, encoding, scaling
│   ├── train.py              # Model training pipeline
│   └── evaluate.py           # Accuracy, precision, recall, F1 evaluation
└── templates/
    └── index.html            # Interactive prediction UI
```

## How It Works

1. **Preprocessing** — Numerical features (`age`, `hours_per_week`, `capital_gain`, etc.) are standardized; categorical features (`workclass`, `education`, `occupation`, etc.) are one-hot encoded via Scikit-learn's `ColumnTransformer`.
2. **Training** — Four classifiers are trained on the processed data with default hyperparameters.
3. **Evaluation** — Models are compared on accuracy, precision, recall, and F1-score. Results are saved to `model_results.csv`.
4. **Serving** — The best model (Random Forest) is loaded at startup and exposed via a `/predict` endpoint.

## Getting Started

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the CLI training pipeline
python main.py

# 3. Launch the web application
python app.py
# → Open http://localhost:5000
```

## Dataset

- **Source:** UCI Machine Learning Repository — Adult Census Income
- **Records:** 32,561 samples, 14 features + 1 binary target (`<=50K` / `>50K`)
- **File:** `adult_census_income.csv`

## Key Results

| Model | Accuracy | F1-Score |
|---|---|---|
| Logistic Regression | ~82% | ~0.60 |
| Random Forest | ~85% | ~0.68 |
| XGBoost | ~86% | ~0.70 |
| LightGBM | ~86% | ~0.70 |
