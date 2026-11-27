# Program 70: PCA classification comparison
# Question: A financial organization develops a classification model using numerous customer attributes. Compare classification performance before and after PCA transformation and determine whether dimensionality reduction improves the model. (BL5 – Evaluate)

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

np.random.seed(42)

# 1. Generate Financial Customer Credit Risk Dataset (500 customers, 20 attributes with redundant collinear features)
X_fin, y_fin = make_classification(
    n_samples=500, n_features=20, n_informative=8, n_redundant=8, n_repeated=2,
    flip_y=0.05, random_state=42
)

X_tr, X_te, y_tr, y_te = train_test_split(
    X_fin, y_fin, test_size=0.25, random_state=42, stratify=y_fin
)

# 2. Standardize Features
scaler_fin = StandardScaler()
X_tr_std = scaler_fin.fit_transform(X_tr)
X_te_std = scaler_fin.transform(X_te)

# 3. Model 1: Classifier Before PCA (All 20 Original Features)
clf_orig = LogisticRegression(random_state=42)
clf_orig.fit(X_tr_std, y_tr)
train_acc_orig = clf_orig.score(X_tr_std, y_tr)
test_acc_orig = clf_orig.score(X_te_std, y_te)
test_f1_orig = f1_score(y_te, clf_orig.predict(X_te_std))

# 4. Model 2: Classifier After PCA (Reduced to 6 Principal Components)
pca_fin = PCA(n_components=6)
X_tr_pca = pca_fin.fit_transform(X_tr_std)
X_te_pca = pca_fin.transform(X_te_std)

clf_pca_fin = LogisticRegression(random_state=42)
clf_pca_fin.fit(X_tr_pca, y_tr)
train_acc_pca = clf_pca_fin.score(X_tr_pca, y_tr)
test_acc_pca = clf_pca_fin.score(X_te_pca, y_te)
test_f1_pca = f1_score(y_te, clf_pca_fin.predict(X_te_pca))

# 5. Comparative Evaluation Table
print(f"{'Performance Metric':<25} | {'Before PCA (20 Features)':<24} | {'After PCA (6 Features)':<22}")
print("-" * 75)
print(f"{'Number of Features':<25} | {20:<24} | {6:<22}")
print(f"{'Training Accuracy':<25} | {train_acc_orig*100:>22.2f}% | {train_acc_pca*100:>20.2f}%")
print(f"{'Testing Accuracy':<25} | {test_acc_orig*100:>22.2f}% | {test_acc_pca*100:>20.2f}%")
print(f"{'Testing F1-Score':<25} | {test_f1_orig:>23.4f} | {test_f1_pca:>21.4f}")
print(f"{'Generalization Gap':<25} | {(train_acc_orig - test_acc_orig)*100:>22.2f}% | {(train_acc_pca - test_acc_pca)*100:>20.2f}%")
print("Evaluation: Dimensionality reduction eliminates multicollinear noise, reduces overfitting gap, and maintains test accuracy.")

# Output:
# Performance Metric        | Before PCA (20 Features) | After PCA (6 Features)
# ---------------------------------------------------------------------------
# Number of Features        | 20                       | 6                     
# Training Accuracy         |                  72.80% |                72.80%
# Testing Accuracy          |                  71.20% |                72.80%
# Testing F1-Score          |                  0.7049 |                0.7069
# Generalization Gap        |                   1.60% |                 0.00%
# Evaluation: Dimensionality reduction eliminates multicollinear noise, reduces overfitting gap, and maintains test accuracy.

# ======================================================================
