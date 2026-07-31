"""
Adult Census Income Classification - Model Training Module
Author: Rudhra Sitholey (Reg No: 23BCY10296 | App No: IN26012560)
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False


def train_models(X_train, y_train):
    """
    Trains multiple models for evaluation and comparison.
    """
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    if HAS_XGBOOST:
        models['XGBoost'] = XGBClassifier(n_estimators=100, learning_rate=0.1, eval_metric='logloss', random_state=42)
    if HAS_LIGHTGBM:
        models['LightGBM'] = LGBMClassifier(n_estimators=100, learning_rate=0.1, random_state=42, verbose=-1)
    
    trained_models = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        
    return trained_models

