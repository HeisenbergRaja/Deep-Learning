# Program 67: PCA healthcare
# Question: A healthcare organization collects patient information containing numerous medical attributes. Implement PCA to reduce the dimensionality of the dataset and analyse the transformed feature space. (BL4 – Analyse)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

np.random.seed(42)
num_patients = 200
feature_names = [
    "Age", "BMI", "Systolic_BP", "Diastolic_BP", "Fasting_Glucose",
    "Cholesterol", "Triglycerides", "HDL", "LDL", "Heart_Rate"
]

# 1. Generate Synthetic Patient Health Records with realistic clinical correlations
age = np.random.normal(52, 12, num_patients)
bmi = np.random.normal(27.5, 4.5, num_patients)
sys_bp = 90 + 0.8 * age + 1.2 * bmi + np.random.normal(0, 10, num_patients)
dia_bp = 60 + 0.3 * sys_bp + np.random.normal(0, 5, num_patients)
glucose = 70 + 0.9 * bmi + np.random.normal(0, 15, num_patients)
chol = 150 + 1.1 * age + np.random.normal(0, 25, num_patients)
trig = 100 + 1.5 * bmi + np.random.normal(0, 30, num_patients)
hdl = 65 - 0.4 * bmi + np.random.normal(0, 8, num_patients)
ldl = chol - hdl - (0.2 * trig)
hr = np.random.normal(72, 10, num_patients)

X_health = np.column_stack([age, bmi, sys_bp, dia_bp, glucose, chol, trig, hdl, ldl, hr])

# 2. Standardize Features (zero mean, unit variance)
scaler = StandardScaler()
X_health_scaled = scaler.fit_transform(X_health)

# 3. Apply PCA Dimensionality Reduction (10 dimensions -> 3 components)
pca_health = PCA(n_components=3)
X_health_pca = pca_health.fit_transform(X_health_scaled)

print("Original Healthcare Feature Space Shape:", X_health.shape)
print("Transformed Feature Space Shape (PCA):  ", X_health_pca.shape)

# 4. Analyze Transformed Feature Space: Explained Variance
print("\nExplained Variance Ratio by Principal Component:")
for i, var in enumerate(pca_health.explained_variance_ratio_, 1):
    cum_var = np.sum(pca_health.explained_variance_ratio_[:i])
    print(f"  PC{i}: {var*100:6.2f}% | Cumulative: {cum_var*100:6.2f}%")

# 5. Component Loadings (Biomarker contributions)
print("\nPrincipal Component Loadings (Dominant Biomarkers):")
loadings = pca_health.components_
for i in range(3):
    top_idx = np.argmax(np.abs(loadings[i]))
    print(f"  PC{i+1}: Dominant attribute = {feature_names[top_idx]:<15} (weight = {loadings[i, top_idx]:+.4f})")

# 6. Check Orthogonality of Transformed Features
cov_mat = np.cov(X_health_pca, rowvar=False)
print("\nCovariance Matrix of Transformed PCA Space (Orthogonality):")
print(np.round(cov_mat, 4))

# 7. Visualize Transformed Feature Space
plt.figure(figsize=(7, 5))
scatter = plt.scatter(X_health_pca[:, 0], X_health_pca[:, 1], c=age, cmap="plasma", alpha=0.8, edgecolors="k")
plt.colorbar(scatter, label="Patient Age (years)")
plt.xlabel("Principal Component 1 (Hemodynamic Risk)")
plt.ylabel("Principal Component 2 (Lipid Profile)")
plt.title("PCA Transformed Feature Space: Patient Medical Profiles")
plt.grid(True)
plt.tight_layout()
plt.show()

# Output:
# Original Healthcare Feature Space Shape: (200, 10)
# Transformed Feature Space Shape (PCA):   (200, 3)
#
# Explained Variance Ratio by Principal Component:
#   PC1:  26.74% | Cumulative:  26.74%
#   PC2:  18.22% | Cumulative:  44.96%
#   PC3:  11.78% | Cumulative:  56.74%
#
# Principal Component Loadings (Dominant Biomarkers):
#   PC1: Dominant attribute = Systolic_BP     (weight = +0.4703)
#   PC2: Dominant attribute = LDL             (weight = -0.5261)
#   PC3: Dominant attribute = HDL             (weight = -0.6993)
#
# Covariance Matrix of Transformed PCA Space (Orthogonality):
# [[2.6875 0.     0.    ]
#  [0.     1.8311 0.    ]
#  [0.     0.     1.1838]]

# ======================================================================
