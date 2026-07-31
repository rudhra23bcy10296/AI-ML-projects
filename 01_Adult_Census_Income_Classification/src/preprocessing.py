"""
Adult Census Income Classification - Data Preprocessing Pipeline
Author: Rudhra Sitholey (Reg No: 23BCY10296 | App No: IN26012560)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def generate_synthetic_census_data(n_samples=2000, random_state=42):
    """
    Generates a realistic synthetic Adult Census Income dataset for demonstration
    and reliable execution without requiring external network downloads.
    """
    np.random.seed(random_state)
    
    age = np.random.randint(18, 75, size=n_samples)
    workclass = np.random.choice(['Private', 'Self-emp-not-inc', 'Local-gov', 'State-gov', 'Federal-gov', 'Private', '?'], size=n_samples, p=[0.6, 0.1, 0.1, 0.05, 0.05, 0.08, 0.02])
    education = np.random.choice(['Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college', 'Assoc-acdm', 'Doctorate'], size=n_samples)
    education_num = np.random.randint(5, 16, size=n_samples)
    marital_status = np.random.choice(['Never-married', 'Married-civ-spouse', 'Divorced', 'Separated', 'Widowed'], size=n_samples)
    occupation = np.random.choice(['Tech-support', 'Craft-repair', 'Other-service', 'Exec-managerial', 'Prof-specialty', 'Sales', '?'], size=n_samples)
    relationship = np.random.choice(['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'], size=n_samples)
    race = np.random.choice(['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Black', 'Other'], size=n_samples)
    sex = np.random.choice(['Female', 'Male'], size=n_samples)
    capital_gain = np.random.choice([0, 1000, 2500, 5000, 14084, 99999], size=n_samples, p=[0.85, 0.05, 0.04, 0.03, 0.02, 0.01])
    capital_loss = np.random.choice([0, 1500, 1900, 2200], size=n_samples, p=[0.9, 0.04, 0.03, 0.03])
    hours_per_week = np.random.randint(20, 65, size=n_samples)
    native_country = np.random.choice(['United-States', 'Mexico', 'India', 'Canada', 'Germany', '?'], size=n_samples, p=[0.85, 0.05, 0.04, 0.03, 0.02, 0.01])
    
    # Target logic based on demographic predictors
    income_prob = (
        0.02 * age + 
        0.05 * education_num + 
        0.03 * (hours_per_week > 40) + 
        0.00002 * capital_gain + 
        (marital_status == 'Married-civ-spouse') * 0.2 + 
        (workclass == 'Federal-gov') * 0.1
    )
    income_prob = 1 / (1 + np.exp(- (income_prob - 1.5)))
    income = np.where(np.random.rand(n_samples) < income_prob, '>50K', '<=50K')

    df = pd.DataFrame({
        'age': age,
        'workclass': workclass,
        'education': education,
        'education_num': education_num,
        'marital_status': marital_status,
        'occupation': occupation,
        'relationship': relationship,
        'race': race,
        'sex': sex,
        'capital_gain': capital_gain,
        'capital_loss': capital_loss,
        'hours_per_week': hours_per_week,
        'native_country': native_country,
        'income': income
    })
    
    # Replace '?' with NaN to simulate real missing values
    df.replace('?', np.nan, inplace=True)
    return df


def get_preprocessor(numerical_cols, categorical_cols):
    """
    Builds a scikit-learn ColumnTransformer pipeline for data preprocessing.
    """
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', num_pipeline, numerical_cols),
        ('cat', cat_pipeline, categorical_cols)
    ])
    
    return preprocessor
