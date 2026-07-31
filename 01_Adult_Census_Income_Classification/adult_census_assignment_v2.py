import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, classification_report,
                              confusion_matrix, roc_curve)


columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
           'marital_status', 'occupation', 'relationship', 'race', 'sex',
           'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income']

csv_path = 'adult_census_income.csv' if os.path.exists('adult_census_income.csv') else 'adult.csv'
df = pd.read_csv(csv_path, header=None, names=columns)


# ---- Task 1: Dataset Understanding ----

print("Shape:", df.shape)
print("\nData types:\n", df.dtypes)
print("\nFirst 5 rows:\n", df.head())
print("\nBasic stats:\n", df.describe())
print("\nTarget distribution:\n", df['income'].value_counts())
print("\nTarget % split:\n", df['income'].value_counts(normalize=True).mul(100).round(2))


# ---- Task 2: Data Cleaning ----

df = df.apply(lambda x: x.str.strip() if x.dtype == 'object' else x)
df.replace('?', np.nan, inplace=True)

print("\nMissing values:\n", df.isnull().sum())

for col in ['workclass', 'occupation', 'native_country']:
    df[col].fillna(df[col].mode()[0], inplace=True)

before = len(df)
df.drop_duplicates(inplace=True)
print(f"\nDuplicates removed: {before - len(df)}")
print("Clean dataset shape:", df.shape)

num_cols = ['age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)].shape[0]
    print(f"{col}: {outliers} outliers (retained)")


# ---- Task 3: Feature Engineering ----

df['income_binary'] = (df['income'] == '>50K').astype(int)

cat_cols = ['workclass', 'education', 'marital_status', 'occupation',
            'relationship', 'race', 'sex', 'native_country']

le = LabelEncoder()
for col in cat_cols:
    df[col + '_enc'] = le.fit_transform(df[col].astype(str))

df['capital_net'] = df['capital_gain'] - df['capital_loss']
df['age_group'] = pd.cut(df['age'], bins=[0, 25, 35, 50, 65, 100], labels=[0, 1, 2, 3, 4]).astype(int)
df['is_married'] = df['marital_status'].str.contains('Married').astype(int)
df['high_education'] = (df['education_num'] >= 13).astype(int)

feature_cols = (
    ['age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss',
     'hours_per_week', 'capital_net', 'age_group', 'is_married', 'high_education']
    + [c + '_enc' for c in cat_cols]
)

X = df[feature_cols]
y = df['income_binary']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain: {X_train.shape[0]} | Test: {X_test.shape[0]}")


# ---- Task 4: Model Building ----

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree':       DecisionTreeClassifier(max_depth=10, random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'KNN':                 KNeighborsClassifier(n_neighbors=5),
    'SVM':                 SVC(kernel='rbf', probability=True, random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    results[name] = {
        'model':     model,
        'y_pred':    y_pred,
        'y_proba':   y_proba,
        'accuracy':  accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall':    recall_score(y_test, y_pred),
        'f1':        f1_score(y_test, y_pred),
        'roc_auc':   roc_auc_score(y_test, y_proba),
    }
    print(f"{name}: Acc={results[name]['accuracy']:.4f}  F1={results[name]['f1']:.4f}  AUC={results[name]['roc_auc']:.4f}")


# ---- Task 5: Performance Evaluation ----

print(f"\n{'Algorithm':<22} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8} {'ROC-AUC':>9}")
print("-" * 70)
for name, r in results.items():
    print(f"{name:<22} {r['accuracy']:>9.4f} {r['precision']:>10.4f} {r['recall']:>8.4f} {r['f1']:>8.4f} {r['roc_auc']:>9.4f}")

for name, r in results.items():
    print(f"\n--- {name} ---")
    print(classification_report(y_test, r['y_pred'], target_names=['<=50K', '>50K']))


# ---- Visualizations ----

fig = plt.figure(figsize=(22, 26))
fig.patch.set_facecolor('#0f0f0f')

COLORS = ['#00d4ff', '#ff6b6b', '#ffd93d', '#6bcb77', '#c77dff']

def style_ax(ax, title):
    ax.set_facecolor('#1a1a2e')
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.set_title(title, color='#00d4ff', fontsize=13, fontweight='bold', pad=10)
    for sp in ax.spines.values():
        sp.set_edgecolor('#333355')

ax1 = fig.add_subplot(4, 3, 1)
counts = df['income'].value_counts()
bars = ax1.bar(counts.index, counts.values, color=COLORS[:2], edgecolor='white', linewidth=0.5)
for b in bars:
    ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 300,
             f'{b.get_height():,}', ha='center', color='white', fontsize=9)
style_ax(ax1, 'Target Distribution')
ax1.set_ylabel('Count', color='white')

ax2 = fig.add_subplot(4, 3, 2)
for income, color in zip(['<=50K', '>50K'], COLORS[:2]):
    ax2.hist(df[df['income'] == income]['age'], bins=30, alpha=0.6, color=color, label=income)
ax2.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=8)
style_ax(ax2, 'Age Distribution by Income')
ax2.set_xlabel('Age')
ax2.set_ylabel('Count')

ax3 = fig.add_subplot(4, 3, 3)
edu = df.groupby('education_num')['income_binary'].mean()
ax3.plot(edu.index, edu.values, color='#ffd93d', linewidth=2, marker='o', markersize=5)
ax3.fill_between(edu.index, edu.values, alpha=0.15, color='#ffd93d')
style_ax(ax3, 'Education Level vs P(>50K)')
ax3.set_xlabel('Education Num')
ax3.set_ylabel('P(>50K)')

ax4 = fig.add_subplot(4, 3, 4)
for income, color in zip(['<=50K', '>50K'], COLORS[:2]):
    ax4.hist(df[df['income'] == income]['hours_per_week'], bins=30, alpha=0.6, color=color, label=income)
ax4.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=8)
style_ax(ax4, 'Hours/Week by Income')
ax4.set_xlabel('Hours Per Week')
ax4.set_ylabel('Count')

ax5 = fig.add_subplot(4, 3, 5)
sex_income = df.groupby('sex')['income_binary'].mean() * 100
bars = ax5.bar(sex_income.index, sex_income.values, color=COLORS[:2], edgecolor='white', linewidth=0.5)
for b in bars:
    ax5.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5,
             f'{b.get_height():.1f}%', ha='center', color='white', fontsize=9)
style_ax(ax5, 'Gender vs P(>50K)')
ax5.set_ylabel('%')

ax6 = fig.add_subplot(4, 3, 6)
corr_cols = ['age', 'education_num', 'hours_per_week', 'capital_gain', 'capital_net', 'income_binary']
sns.heatmap(df[corr_cols].corr(), annot=True, fmt='.2f', cmap='coolwarm', ax=ax6,
            linewidths=0.4, linecolor='#0f0f0f', annot_kws={'size': 7, 'color': 'white'},
            cbar_kws={'shrink': 0.8})
ax6.set_facecolor('#1a1a2e')
ax6.tick_params(colors='white', labelsize=7)
ax6.set_title('Feature Correlation Heatmap', color='#00d4ff', fontsize=13, fontweight='bold')

ax7 = fig.add_subplot(4, 3, 7)
metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
x = np.arange(len(metrics))
w = 0.15
for i, (name, r) in enumerate(results.items()):
    ax7.bar(x + i * w, [r[m] for m in metrics], width=w, label=name, color=COLORS[i], alpha=0.85)
ax7.set_xticks(x + w * 2)
ax7.set_xticklabels(['Acc', 'Prec', 'Rec', 'F1', 'AUC'], color='white')
ax7.legend(fontsize=7, facecolor='#1a1a2e', labelcolor='white', loc='lower right')
ax7.set_ylim(0.6, 1.0)
style_ax(ax7, 'Model Metric Comparison')
ax7.set_ylabel('Score')

ax8 = fig.add_subplot(4, 3, 8)
ax8.set_facecolor('#1a1a2e')
for i, (name, r) in enumerate(results.items()):
    fpr, tpr, _ = roc_curve(y_test, r['y_proba'])
    ax8.plot(fpr, tpr, color=COLORS[i], lw=1.8, label=f"{name} ({r['roc_auc']:.3f})")
ax8.plot([0, 1], [0, 1], 'w--', lw=1)
ax8.legend(fontsize=7, facecolor='#1a1a2e', labelcolor='white')
style_ax(ax8, 'ROC Curves')
ax8.set_xlabel('FPR')
ax8.set_ylabel('TPR')

best = max(results, key=lambda x: results[x]['f1'])
ax9 = fig.add_subplot(4, 3, 9)
cm = confusion_matrix(y_test, results[best]['y_pred'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax9,
            xticklabels=['<=50K', '>50K'], yticklabels=['<=50K', '>50K'],
            annot_kws={'size': 11, 'color': 'white'})
ax9.set_facecolor('#1a1a2e')
ax9.tick_params(colors='white')
ax9.set_title(f'Confusion Matrix - {best}', color='#00d4ff', fontsize=11, fontweight='bold')
ax9.set_xlabel('Predicted', color='white')
ax9.set_ylabel('Actual', color='white')

ax10 = fig.add_subplot(4, 3, 10)
rf_importances = pd.Series(
    results['Random Forest']['model'].feature_importances_, index=feature_cols
).nlargest(10)
rf_importances[::-1].plot(kind='barh', ax=ax10, color='#00d4ff', edgecolor='none')
style_ax(ax10, 'Top 10 Feature Importances (RF)')
ax10.set_xlabel('Importance')

ax11 = fig.add_subplot(4, 3, (11, 12))
ax11.set_facecolor('#1a1a2e')
ax11.axis('off')
table_data = [[n, f"{r['accuracy']:.4f}", f"{r['precision']:.4f}",
               f"{r['recall']:.4f}", f"{r['f1']:.4f}", f"{r['roc_auc']:.4f}"]
              for n, r in results.items()]
tbl = ax11.table(
    cellText=table_data,
    colLabels=['Algorithm', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC'],
    loc='center', cellLoc='center'
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
for (row, col), cell in tbl.get_celld().items():
    cell.set_facecolor('#16213e' if row % 2 == 0 else '#0f3460')
    cell.set_text_props(color='white')
    cell.set_edgecolor('#333355')
    if row == 0:
        cell.set_facecolor('#00d4ff')
        cell.set_text_props(color='#0f0f0f', fontweight='bold')
tbl.scale(1, 2.2)
ax11.set_title('Performance Summary Table', color='#00d4ff', fontsize=13, fontweight='bold', pad=15)

plt.suptitle('Adult Census Income - ML Assignment Results',
             fontsize=18, fontweight='bold', color='white', y=1.01)
plt.tight_layout(pad=2.5)
plt.savefig('assignment_results.png', dpi=180, bbox_inches='tight', facecolor='#0f0f0f')
print("\nVisualizations saved.")

results_df = pd.DataFrame([{
    'Algorithm': name,
    'Accuracy':  round(r['accuracy'],  4),
    'Precision': round(r['precision'], 4),
    'Recall':    round(r['recall'],    4),
    'F1 Score':  round(r['f1'],        4),
    'ROC-AUC':   round(r['roc_auc'],   4),
} for name, r in results.items()])

results_df.to_csv('model_results.csv', index=False)
print("Results saved to model_results.csv")
