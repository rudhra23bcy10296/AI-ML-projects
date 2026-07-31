"""
Adult Census Income Classification - Main Driver Script
Author: Rudhra Sitholey (Reg No: 23BCY10296 | App No: IN26012560)
"""

from src.preprocessing import generate_synthetic_census_data, get_preprocessor
from src.train import train_models
from src.evaluate import evaluate_models
from sklearn.model_selection import train_test_split


def main():
    print("=" * 65)
    print(" Project 1: Adult Census Income Classification")
    print(" Student: Rudhra Sitholey | Reg: 23BCY10296 | App: IN26012560")
    print("=" * 65)
    
    # 1. Load / Generate Data
    print("\n[1] Generating Census Dataset...")
    df = generate_synthetic_census_data(n_samples=2500)
    print(f"    Dataset Shape: {df.shape}")
    print(f"    Target Distribution:\n{df['income'].value_counts(normalize=True)}")
    
    # 2. Features and Target
    X = df.drop(columns=['income'])
    y = (df['income'] == '>50K').astype(int)
    
    num_cols = ['age', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
    cat_cols = ['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country']
    
    # 3. Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 4. Data Preprocessing
    print("\n[2] Preprocessing Features (Imputation, Scaling, One-Hot Encoding)...")
    preprocessor = get_preprocessor(num_cols, cat_cols)
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    
    # 5. Train Models
    print("\n[3] Training ML Classification Models...")
    models = train_models(X_train_proc, y_train)
    
    # 6. Evaluation
    print("\n[4] Model Comparison & Performance Matrix:")
    results_df = evaluate_models(models, X_test_proc, y_test)
    print(results_df.to_string(index=False))
    print("\nPipeline Execution Successfully Completed!")


if __name__ == '__main__':
    main()
