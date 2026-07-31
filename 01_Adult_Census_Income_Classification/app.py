"""
Adult Census Income Classification - Interactive Flask Web Application
Author: Rudhra Sitholey (Reg No: 23BCY10296 | App No: IN26012560 | Email: rudhra.23bcy10296@vitbhopal.ac.in)
"""

import os
import pandas as pd
from flask import Flask, render_template, request, jsonify
from src.preprocessing import generate_synthetic_census_data, get_preprocessor
from src.train import train_models

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"))

# Train baseline model at server startup
df = generate_synthetic_census_data(n_samples=2000)
X = df.drop(columns=['income'])
y = (df['income'] == '>50K').astype(int)

num_cols = ['age', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
cat_cols = ['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country']

preprocessor = get_preprocessor(num_cols, cat_cols)
X_proc = preprocessor.fit_transform(X)
models = train_models(X_proc, y)
model = models['Random Forest']


@app.route('/')
def index():
    return render_template(
        'index.html',
        student_name="Rudhra Sitholey",
        reg_no="23BCY10296",
        app_no="IN26012560",
        email="rudhra.23bcy10296@vitbhopal.ac.in"
    )


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(silent=True) or {}
    try:
        input_data = pd.DataFrame([{
            'age': float(data.get('age', 35)),
            'workclass': str(data.get('workclass', 'Private')),
            'education': str(data.get('education', 'Bachelors')),
            'education_num': float(data.get('education_num', 13)),
            'marital_status': str(data.get('marital_status', 'Married-civ-spouse')),
            'occupation': str(data.get('occupation', 'Exec-managerial')),
            'relationship': str(data.get('relationship', 'Husband')),
            'race': str(data.get('race', 'White')),
            'sex': str(data.get('sex', 'Male')),
            'capital_gain': float(data.get('capital_gain', 0)),
            'capital_loss': float(data.get('capital_loss', 0)),
            'hours_per_week': float(data.get('hours_per_week', 40)),
            'native_country': str(data.get('native_country', 'United-States'))
        }])
        
        proc_input = preprocessor.transform(input_data)
        pred = int(model.predict(proc_input)[0])
        proba = model.predict_proba(proc_input)[0].tolist() if hasattr(model, "predict_proba") else [0.5, 0.5]
        
        return jsonify({
            'status': 'success',
            'prediction': pred,
            'label': '>50K USD / High Income' if pred == 1 else '<=50K USD / Standard Income',
            'probability': round(proba[pred] * 100, 2)
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


if __name__ == '__main__':
    print("\n  Adult Census Income Web App starting at http://localhost:5000\n")
    app.run(host='0.0.0.0', port=5000, debug=True)
