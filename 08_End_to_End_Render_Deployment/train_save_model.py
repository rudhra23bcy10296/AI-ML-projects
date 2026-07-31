"""
End-to-End Render Deployment - Model Training & Serialization Script
Author: Rudhra Sitholey (Reg No: 23BCY10296 | App No: IN26012560)
"""

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib


def build_and_save_model():
    np.random.seed(42)
    # Generate synthetic housing / credit risk dataset
    X = np.random.randn(500, 4)
    y = (X[:, 0] + X[:, 1] * 0.5 > 0).astype(int)
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=50, random_state=42))
    ])
    
    pipeline.fit(X, y)
    
    os.makedirs('models', exist_ok=True)
    model_path = os.path.join('models', 'deployed_model.joblib')
    joblib.dump(pipeline, model_path)
    print(f"Model successfully saved to {model_path}")
    return pipeline


if __name__ == '__main__':
    build_and_save_model()
