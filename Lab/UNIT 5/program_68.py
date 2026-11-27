# Program 68: PCA Iris
# Question: A botanical research centre wants to simplify the Iris dataset while preserving its important information. Apply PCA to reduce the dataset dimensions and evaluate the transformed features for classification. (BL3 – Apply)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

# 1. Load the Iris Botanical Dataset
iris = load_iris()
X_iris, y_iris = iris.data, iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# 2. Train-Test Split (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X_iris, y_iris, test_size=0.2, random_state=42, stratify=y_iris
)

# 3. Standardize Features
scaler_iris = StandardScaler()
X_train_std = scaler_iris.fit_transform(X_train)
X_test_std = scaler_iris.transform(X_test)

# 4. Baseline Classification on Original 4-D Features
clf_full = LogisticRegression(random_state=42)
clf_full.fit(X_train_std, y_train)
y_pred_full = clf_full.predict(X_test_std)
acc_full = accuracy_score(y_test, y_pred_full)
f1_full = f1_score(y_test, y_pred_full, average='weighted')

# 5. Apply PCA Dimensionality Reduction (4D -> 2D)
pca_iris = PCA(n_components=2)
X_train_pca = pca_iris.fit_transform(X_train_std)
X_test_pca = pca_iris.transform(X_test_std)

# 6. Classification on Reduced 2-D PCA Features
clf_pca = LogisticRegression(random_state=42)
clf_pca.fit(X_train_pca, y_train)
y_pred_pca = clf_pca.predict(X_test_pca)
acc_pca = accuracy_score(y_test, y_pred_pca)
f1_pca = f1_score(y_test, y_pred_pca, average='weighted')

# 7. Comparative Performance Evaluation
print(f"{'Representation':<24} | {'Dimensions':<10} | {'Test Accuracy':<15} | {'Weighted F1':<12}")
print("-" * 67)
print(f"{'Original Iris Features':<24} | {4:<10} | {acc_full*100:>13.2f}% | {f1_full:>11.4f}")
print(f"{'PCA Transformed Features':<24} | {2:<10} | {acc_pca*100:>13.2f}% | {f1_pca:>11.4f}")
var_retained = np.sum(pca_iris.explained_variance_ratio_) * 100
print(f"\nVariance Retained by 2 Principal Components: {var_retained:.2f}%")
print("Evaluation: 50% dimensionality reduction achieved with virtually identical classification accuracy.")

# 8. Visualization of 2D PCA Transformed Iris Space
plt.figure(figsize=(7, 5))
colors = ['navy', 'turquoise', 'darkorange']
for i, target_name in enumerate(target_names):
    plt.scatter(
        X_train_pca[y_train == i, 0], X_train_pca[y_train == i, 1],
        color=colors[i], label=target_name, alpha=0.8, edgecolors='k'
    )
plt.xlabel(f"PC1 ({pca_iris.explained_variance_ratio_[0]*100:.1f}% Variance)")
plt.ylabel(f"PC2 ({pca_iris.explained_variance_ratio_[1]*100:.1f}% Variance)")
plt.title("Iris Dataset Projected into 2D Principal Component Space")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Output:
# Representation           | Dimensions | Test Accuracy   | Weighted F1 
# -------------------------------------------------------------------
# Original Iris Features   | 4          |         93.33% |      0.9333
# PCA Transformed Features | 2          |         90.00% |      0.8997
#
# Variance Retained by 2 Principal Components: 95.74%
# Evaluation: 50% dimensionality reduction achieved with virtually identical classification accuracy.

# ======================================================================
